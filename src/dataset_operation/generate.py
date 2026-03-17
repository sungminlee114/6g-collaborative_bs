"""Multi-snapshot Sionna RT channel data generation.

Supports two modes:
  - Independent: each snapshot samples new UE positions (default, for CE accuracy)
  - Temporal: UEs follow mobility trajectories across snapshots (for CE scheduling)

Usage:
    # Independent (default)
    python -m src.dataset_operation.generate --preset munich_elaa_s_1k_15g

    # Temporal trajectory
    python -m src.dataset_operation.generate --preset munich_elaa_s_1k_15g \
        --mode temporal --dt_ms 10 --velocities 0,1,8.3
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
                "ue_device_type": dev_type_idx,
                "ue_rx_rows": dev[0],
                "ue_rx_cols": dev[1],
                "ue_polarization": dev[2],
                "vx": 0.0,
                "vy": 0.0,
            })
    return ue_infos, counts


def compute_radio_map(scene, cfg):
    """Compute radio map (expensive — call once, reuse for temporal mode)."""
    from sionna.rt import RadioMapSolver

    rm_solver = RadioMapSolver()
    return rm_solver(
        scene=scene, cell_size=(1.0, 1.0),
        samples_per_tx=10_000_000, max_depth=cfg.max_depth,
        los=True, specular_reflection=True, diffuse_reflection=True,
        refraction=True, diffraction=True, edge_diffraction=True,
    )


def generate_snapshot(scene, cfg, snapshot_id: int, seed: int, data_dir: Path,
                      ue_infos=None, radio_map=None):
    """Generate one snapshot: place UEs, compute paths, extract CIR/CFR, save.

    Args:
        ue_infos: Pre-defined UE positions (temporal mode). If None, samples new
                  positions from radio_map (independent mode).
        radio_map: Pre-computed radio map. If None and ue_infos is None, computes one.

    Returns:
        ue_infos: list of UE info dicts with positions and metadata.
    """
    from sionna.rt import Receiver, PathSolver, RadioMapSolver, subcarrier_frequencies
    import drjit as dr

    np.random.seed(seed)

    # Remove old receivers
    rx_names = list(scene.receivers.keys())
    for name in rx_names:
        scene.remove(name)

    # Get UE positions
    if ue_infos is None:
        # Independent mode: sample new positions each snapshot
        if radio_map is None:
            radio_map = compute_radio_map(scene, cfg)
        ue_infos, counts = sample_ue_positions(radio_map, cfg.num_ue, cfg)
        print(f"  Snapshot {snapshot_id}: UEs per BS = {counts}")
    else:
        bs_counts = {}
        for info in ue_infos:
            bs_counts[info["bs_id"]] = bs_counts.get(info["bs_id"], 0) + 1
        print(f"  Snapshot {snapshot_id}: UEs per BS = {[bs_counts.get(i, 0) for i in range(cfg.num_bs)]}")

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
        refraction=True, synthetic_array=cfg.synthetic_array, seed=seed,
    )

    associated_tx_idxs = [info["bs_id"] for info in ue_infos]

    # CIR
    a, tau = paths.cir(
        normalize_delays=True,
        associated_tx_idxs=associated_tx_idxs,
        out_type="numpy",
    )
    cir_a = np.squeeze(a)
    cir_tau = np.squeeze(tau)

    # CFR — compute on GPU via torch (drjit has 2^32 limit, numpy eats CPU RAM)
    # H(f) = sum_paths[ a * exp(-j*2*pi*f*tau) ]
    import torch
    frequencies = subcarrier_frequencies(cfg.num_subcarriers, cfg.subcarrier_spacing)
    freqs_t = torch.tensor(frequencies.numpy(), dtype=torch.float32, device="cuda")

    tau_for_cfr = cir_tau
    while tau_for_cfr.ndim < cir_a.ndim:
        tau_for_cfr = np.expand_dims(tau_for_cfr, axis=1)

    a_t = torch.tensor(cir_a, dtype=torch.complex64, device="cuda")
    tau_t = torch.tensor(tau_for_cfr, dtype=torch.float32, device="cuda")

    # Per-UE on GPU to stay within VRAM (~40GB)
    # Per UE: (n_rx, n_tx, n_paths, n_sc) complex64 = small on GPU
    per_ue_elems = int(np.prod(cir_a.shape[1:])) * len(freqs_t)
    chunk = max(1, int(30e9 / (per_ue_elems * 8)))  # 30GB budget on 40GB GPU
    n_ue = cir_a.shape[0]

    if chunk >= n_ue:
        phase = -2j * torch.pi * tau_t[..., None] * freqs_t
        cfr = torch.sum(a_t[..., None] * torch.exp(phase), dim=-2).cpu().numpy()
    else:
        cfr_chunks = []
        for i in range(0, n_ue, chunk):
            j = min(i + chunk, n_ue)
            phase = -2j * torch.pi * tau_t[i:j, ..., None] * freqs_t
            cfr_chunks.append(
                torch.sum(a_t[i:j, ..., None] * torch.exp(phase), dim=-2).cpu().numpy()
            )
        cfr = np.concatenate(cfr_chunks, axis=0)

    del a_t, tau_t
    torch.cuda.empty_cache()

    # Save
    snap_dir = data_dir / f"snapshot_{snapshot_id:04d}"
    snap_dir.mkdir(parents=True, exist_ok=True)

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


# ── Mobility models ────────────────────────────────────────────────

def init_trajectories(ue_infos, velocities, rng):
    """Assign velocity and random direction to each UE.

    Args:
        ue_infos: list of UE info dicts (with x, y, z).
        velocities: list of speeds [m/s]. Each UE randomly gets one.
                    e.g. [0, 1.0, 8.3] for static/pedestrian/vehicle mix.
        rng: numpy random generator.

    Returns:
        trajectories: list of dicts with velocity, direction for each UE.
    """
    trajectories = []
    for info in ue_infos:
        v = rng.choice(velocities)
        angle = rng.uniform(0, 2 * np.pi)
        trajectories.append({
            "vx": float(v * np.cos(angle)),
            "vy": float(v * np.sin(angle)),
            "speed": float(v),
        })
        info["vx"] = trajectories[-1]["vx"]
        info["vy"] = trajectories[-1]["vy"]
    return trajectories


def advance_positions(ue_infos, trajectories, dt):
    """Move UEs by dt seconds along their trajectories.

    Simple linear model — UEs move in straight lines.
    z (height) stays fixed at 1.5m.

    Returns:
        Updated ue_infos with new x, y positions.
    """
    for info, traj in zip(ue_infos, trajectories):
        info["x"] += traj["vx"] * dt
        info["y"] += traj["vy"] * dt
    return ue_infos


# ── Dataset generation ─────────────────────────────────────────────

def generate_dataset(cfg, data_dir: str, num_snapshots: int, seed_offset: int = 0,
                     mode: str = "independent", dt: float = 0.01,
                     velocities: list = None):
    """Generate multi-snapshot dataset.

    Args:
        mode: "independent" (random UE positions each snapshot) or
              "temporal" (fixed UEs with mobility trajectories).
        dt: Time step between snapshots [seconds]. Only for temporal mode.
            e.g. 0.01 = 10ms, 0.001 = 1ms.
        velocities: List of UE speeds [m/s] for temporal mode.
                    e.g. [0, 1.0, 8.3] for static/pedestrian/vehicle mix.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if velocities is None:
        velocities = [0.0, 1.0, 8.3]  # static, pedestrian, 30km/h

    scene = build_scene(cfg)

    all_records = []
    ue_infos = None
    trajectories = None
    radio_map = None

    if mode == "temporal":
        # Compute radio map once, sample initial positions
        radio_map = compute_radio_map(scene, cfg)
        rng = np.random.default_rng(seed_offset)
        init_seed = seed_offset + 41
        np.random.seed(init_seed)
        ue_infos, counts = sample_ue_positions(radio_map, cfg.num_ue, cfg)
        trajectories = init_trajectories(ue_infos, velocities, rng)
        print(f"  Initial UEs per BS = {counts}")
        print(f"  Velocities: {[f'{t['speed']:.1f}' for t in trajectories[:5]]}... (m/s)")
        print(f"  dt = {dt*1000:.1f}ms, total duration = {dt*num_snapshots*1000:.1f}ms")

    for snap_id in range(num_snapshots):
        seed = seed_offset + snap_id * 17 + 41

        if mode == "temporal" and snap_id > 0:
            ue_infos = advance_positions(ue_infos, trajectories, dt)

        snap_ue_infos = generate_snapshot(
            scene, cfg, snap_id, seed, data_dir,
            ue_infos=ue_infos if mode == "temporal" else None,
            radio_map=radio_map,
        )

        for ue_id, info in enumerate(snap_ue_infos):
            record = {
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
            }
            if mode == "temporal":
                record["t_ms"] = snap_id * dt * 1000
            all_records.append(record)

    # Save metadata
    df = pd.DataFrame(all_records)
    df.to_parquet(data_dir / "metadata.parquet", index=False)

    # Save dataset info
    ds_info = {
        "positions": [list(p) for p in cfg.bs_positions[:cfg.num_bs]],
        "num_bs": cfg.num_bs,
        "num_snapshots": num_snapshots,
        "frequency_hz": cfg.frequency,
        "bandwidth_hz": cfg.bandwidth,
        "num_subcarriers": cfg.num_subcarriers,
        "num_tx_ant": cfg.num_tx_ant,
        "num_rx_ant": cfg.num_rx_ant,
        "mode": mode,
    }
    if mode == "temporal":
        ds_info["dt_s"] = dt
        ds_info["velocities_ms"] = velocities
        ds_info["total_duration_ms"] = dt * num_snapshots * 1000
    with open(data_dir / "bs_info.json", "w") as f:
        json.dump(ds_info, f, indent=2)

    print(f"\nDataset saved to {data_dir}")
    print(f"  {num_snapshots} snapshots, {len(all_records)} total samples")
    print(f"  Mode: {mode}" + (f", dt={dt*1000:.1f}ms" if mode == "temporal" else ""))
    return df


if __name__ == "__main__":
    from src.config import SceneConfig, DatasetConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default="munich_umi16",
                        help=f"Scene preset name. Available: {SceneConfig.list_presets()}")
    parser.add_argument("--num_snapshots", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--num_ue", type=int, default=100)
    parser.add_argument("--mode", type=str, default="independent",
                        choices=["independent", "temporal"])
    parser.add_argument("--dt_ms", type=float, default=10.0,
                        help="Snapshot interval in ms (temporal mode)")
    parser.add_argument("--velocities", type=str, default="0,1,8.3",
                        help="Comma-separated UE speeds in m/s (temporal mode)")
    args = parser.parse_args()

    cfg = SceneConfig.from_preset(args.preset, num_ue=args.num_ue)

    # Read data_dir from YAML
    if args.data_dir:
        data_dir = args.data_dir
    else:
        import yaml
        yaml_path = Path("assets/configs") / f"{args.preset}.yaml"
        if yaml_path.exists():
            with open(yaml_path) as f:
                raw = yaml.safe_load(f)
            data_dir = raw.get("data_dir", f"assets/data/channels_{args.preset}")
        else:
            data_dir = DatasetConfig.from_scene(cfg).data_dir

    # Temporal mode: append suffix to data_dir
    if args.mode == "temporal":
        data_dir = f"{data_dir}_temporal"

    velocities = [float(v) for v in args.velocities.split(",")]

    print(f"Preset: {args.preset}")
    print(f"  TX array: {cfg.tx_rows}x{cfg.tx_cols} = {cfg.num_tx_ant} ant")
    print(f"  Freq: {cfg.frequency/1e9:.1f} GHz, BW: {cfg.bandwidth/1e6:.0f} MHz")
    print(f"  Subcarriers: {cfg.num_subcarriers}")
    print(f"  BS: {cfg.num_bs}, UE: {cfg.num_ue}")
    print(f"  Synthetic array: {cfg.synthetic_array}")
    print(f"  Mode: {args.mode}")
    if args.mode == "temporal":
        print(f"  dt: {args.dt_ms}ms, velocities: {velocities} m/s")
    print(f"  Output: {data_dir}")
    print(f"  Snapshots: {args.num_snapshots}")

    generate_dataset(cfg, data_dir, args.num_snapshots, args.seed_offset,
                     mode=args.mode, dt=args.dt_ms / 1000, velocities=velocities)
