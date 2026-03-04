"""Multi-snapshot Sionna RT channel data generation.

Usage:
    python -m src.data.generate --num_snapshots 100 --data_dir data/channels
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_scene(cfg):
    """Build Sionna RT scene with BSs configured."""
    import sionna.rt
    from sionna.rt import (
        load_scene, PlanarArray, Transmitter, Camera,
        PathSolver, RadioMapSolver, subcarrier_frequencies,
    )
    import mitsuba as mi
    import drjit as dr

    mi.set_variant("cuda_ad_mono_polarized", "llvm_ad_mono_polarized")

    scene = load_scene(sionna.rt.scene.munich, merge_shapes=True)
    scene.frequency = cfg.frequency
    scene.bandwidth = cfg.bandwidth
    scene.temperature = cfg.temperature

    # TX array (BS)
    tx_array = PlanarArray(
        num_rows=cfg.tx_rows, num_cols=cfg.tx_cols,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization=cfg.tx_polarization,
    )
    scene.tx_array = tx_array

    # RX array (UE)
    rx_array = PlanarArray(
        num_rows=cfg.rx_rows, num_cols=cfg.rx_cols,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="dipole", polarization=cfg.rx_polarization,
    )
    scene.rx_array = rx_array

    # Add BSs
    for i, pos in enumerate(cfg.bs_positions[:cfg.num_bs]):
        ori = cfg.bs_orientations[i] if cfg.bs_orientations[i] else [
            pos[0] + float(np.random.uniform(-5, 5)),
            pos[1] + float(np.random.uniform(-5, 5)),
            1.5,
        ]
        bs = Transmitter(
            name=f"bs_{i}",
            position=list(pos),
            power_dbm=cfg.power_dbm,
            orientation=ori,
            display_radius=10,
        )
        scene.add(bs)

    return scene


def sample_ue_positions(radio_map, num_ue, cfg):
    """Sample UE positions from radio map with balanced BS distribution."""
    positions, cells = radio_map.sample_positions(
        num_pos=num_ue,
        metric="sinr",
        min_val_db=cfg.sinr_min_db,
        max_val_db=cfg.sinr_max_db,
        min_dist=cfg.dist_min,
        max_dist=cfg.dist_max,
        tx_association=True,
    )
    positions = positions.numpy()  # (num_tx, num_ue, 3)
    num_tx = positions.shape[0]

    # Random split across BSs
    split_pts = np.sort(np.random.choice(range(1, num_ue), num_tx - 1, replace=False))
    split_pts = np.concatenate([[0], split_pts, [num_ue]])
    counts = [int(split_pts[i + 1] - split_pts[i]) for i in range(num_tx)]

    # Assign device types to UEs
    device_types = cfg.ue_device_types
    num_device_types = len(device_types)

    ue_infos = []
    for tx_id in range(num_tx):
        for idx in range(counts[tx_id]):
            dev_type_idx = np.random.randint(0, num_device_types)
            dev = device_types[dev_type_idx]
            ue_infos.append({
                "bs_id": tx_id,
                "idx_in_bs": idx,
                "x": float(positions[tx_id, idx, 0]),
                "y": float(positions[tx_id, idx, 1]),
                "z": float(positions[tx_id, idx, 2]),
                # F_UE: Device features
                "ue_device_type": dev_type_idx,
                "ue_rx_rows": dev[0],
                "ue_rx_cols": dev[1],
                "ue_polarization": dev[2],
                # F_UE: Mobility (velocity = 0 for static; computed across snapshots later)
                "vx": 0.0,
                "vy": 0.0,
            })
    return ue_infos, counts


def generate_snapshot(scene, cfg, snapshot_id: int, seed: int, data_dir: Path):
    """Generate one snapshot: sample UEs, compute paths, extract CIR/CFR, save."""
    from sionna.rt import Receiver, PathSolver, RadioMapSolver, subcarrier_frequencies
    import drjit as dr

    np.random.seed(seed)

    # Remove old receivers
    rx_names = list(scene.receivers.keys())
    for name in rx_names:
        scene.remove(name)

    # Compute radio map
    rm_solver = RadioMapSolver()
    radio_map = rm_solver(
        scene=scene, cell_size=(1.0, 1.0),
        samples_per_tx=10_000_000, max_depth=cfg.max_depth,
        los=True, specular_reflection=True, diffuse_reflection=True,
        refraction=True, diffraction=True, edge_diffraction=True,
    )

    # Sample UE positions
    ue_infos, counts = sample_ue_positions(radio_map, cfg.num_ue, cfg)
    print(f"  Snapshot {snapshot_id}: UEs per BS = {counts}")

    # Add receivers
    for i, info in enumerate(ue_infos):
        scene.add(Receiver(
            name=f"ue_{i}",
            position=[info["x"], info["y"], info["z"]],
            orientation=[0, 0, 0],
        ))

    # Compute paths
    p_solver = PathSolver()
    paths = p_solver(
        scene=scene, max_depth=cfg.max_depth,
        max_num_paths_per_src=cfg.max_num_paths_per_src,
        samples_per_src=cfg.samples_per_src,
        los=True, specular_reflection=True, diffuse_reflection=True,
        refraction=True, synthetic_array=False, seed=seed,
    )

    associated_tx_idxs = [info["bs_id"] for info in ue_infos]

    # CIR
    a, tau = paths.cir(
        normalize_delays=True,
        associated_tx_idxs=associated_tx_idxs,
        out_type="numpy",
    )
    # a: (N_UE, n_rx_ant, 1, n_tx_ant, n_paths, 1) complex
    # tau: (N_UE, n_rx_ant, 1, n_tx_ant, n_paths)

    # CFR
    frequencies = subcarrier_frequencies(cfg.num_subcarriers, cfg.subcarrier_spacing)
    h_freq = paths.cfr(
        frequencies=frequencies,
        associated_tx_idxs=associated_tx_idxs,
        normalize=True,
        normalize_delays=True,
        out_type="numpy",
    )
    # h_freq: (N_UE, n_rx_ant, 1, n_tx_ant, 1, n_subcarriers) complex

    # Save
    snap_dir = data_dir / f"snapshot_{snapshot_id:04d}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    # Squeeze singleton dims: tx=1, time=1
    cfr = h_freq[:, :, 0, :, 0, :]  # (N_UE, n_rx_ant, n_tx_ant, n_sc) complex
    cir_a = a[:, :, 0, :, :, 0]     # (N_UE, n_rx_ant, n_tx_ant, n_paths) complex
    cir_tau = tau[:, :, 0, :, :]     # (N_UE, n_rx_ant, n_tx_ant, n_paths)

    np.savez_compressed(
        snap_dir / "channels.npz",
        cfr=cfr,
        cir_a=cir_a,
        cir_tau=cir_tau,
    )

    # Clean up GPU memory
    dr.flush_malloc_cache()
    dr.flush_kernel_cache()

    return ue_infos


def generate_dataset(cfg, data_dir: str, num_snapshots: int, seed_offset: int = 0):
    """Generate full multi-snapshot dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    scene = build_scene(cfg)

    all_records = []
    for snap_id in range(num_snapshots):
        seed = seed_offset + snap_id * 17 + 41  # deterministic seeds
        ue_infos = generate_snapshot(scene, cfg, snap_id, seed, data_dir)

        for ue_id, info in enumerate(ue_infos):
            all_records.append({
                "snapshot_id": snap_id,
                "ue_id": ue_id,
                "bs_id": info["bs_id"],
                "idx_in_bs": info["idx_in_bs"],
                "x": info["x"],
                "y": info["y"],
                "z": info["z"],
                "ue_device_type": info["ue_device_type"],
                "ue_rx_rows": info["ue_rx_rows"],
                "ue_rx_cols": info["ue_rx_cols"],
                "ue_polarization": info["ue_polarization"],
                "vx": info["vx"],
                "vy": info["vy"],
            })

    # Save metadata
    df = pd.DataFrame(all_records)
    df.to_parquet(data_dir / "metadata.parquet", index=False)

    # Save BS positions
    bs_info = {
        "positions": [list(p) for p in cfg.bs_positions[:cfg.num_bs]],
        "num_bs": cfg.num_bs,
        "num_snapshots": num_snapshots,
        "frequency_hz": cfg.frequency,
        "bandwidth_hz": cfg.bandwidth,
        "num_subcarriers": cfg.num_subcarriers,
        "num_tx_ant": cfg.num_tx_ant,
        "num_rx_ant": cfg.num_rx_ant,
    }
    with open(data_dir / "bs_info.json", "w") as f:
        json.dump(bs_info, f, indent=2)

    print(f"\nDataset saved to {data_dir}")
    print(f"  {num_snapshots} snapshots, {len(all_records)} total samples")
    print(f"  Metadata: {data_dir / 'metadata.parquet'}")
    return df


if __name__ == "__main__":
    from src.config import SceneConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_snapshots", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default="data/channels")
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--num_ue", type=int, default=100)
    args = parser.parse_args()

    cfg = SceneConfig(num_ue=args.num_ue)
    generate_dataset(cfg, args.data_dir, args.num_snapshots, args.seed_offset)
