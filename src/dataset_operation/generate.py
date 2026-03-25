"""Core Sionna RT channel generation functions.

Used by generate_worker.py (per-GPU) and generate_parallel.py (orchestrator).
Key functions: build_scene(), sample_ue_positions(), compute_radio_map(), generate_snapshot().
"""
from pathlib import Path

import numpy as np


def build_scene(cfg):
    """Build Sionna RT scene with BSs configured."""
    import sionna.rt
    from sionna.rt import load_scene, PlanarArray, Transmitter
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

    # Soft-balanced split across BSs: each BS gets at least min_per_bs,
    # remainder distributed with some randomness (Dirichlet).
    min_per_bs = max(3, num_ue // (num_tx * 3))  # at least 3 or 1/3 of uniform
    remainder = num_ue - min_per_bs * num_tx
    if remainder < 0:
        # Not enough UEs for minimum — just distribute uniformly
        counts = [num_ue // num_tx] * num_tx
        counts[0] += num_ue - sum(counts)
    else:
        # Dirichlet-distributed remainder for natural imbalance
        weights = np.random.dirichlet(np.ones(num_tx) * 2.0)
        extra = np.round(weights * remainder).astype(int)
        extra[-1] = remainder - extra[:-1].sum()  # fix rounding
        counts = [min_per_bs + int(e) for e in extra]

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
                      ue_infos=None, radio_map=None, h5_file=None,
                      p_solver=None, freqs_torch=None):
    """Generate one snapshot: place UEs, compute paths, extract CIR + CFR, save.

    Args:
        ue_infos: Pre-defined UE positions (temporal mode). If None, samples new
                  positions from radio_map (independent mode).
        radio_map: Pre-computed radio map. If None and ue_infos is None, computes one.
        h5_file: HDF5 file handle — writes CIR + CFR (if 'cfr' dataset exists).
        p_solver: Reusable PathSolver instance (avoids re-creation per snapshot).
        freqs_torch: Pre-computed subcarrier frequencies on GPU (torch.Tensor).

    Returns:
        ue_infos: list of UE info dicts with positions and metadata.
    """
    from sionna.rt import Receiver, PathSolver, subcarrier_frequencies
    import drjit as dr
    import torch

    np.random.seed(seed)

    # Remove old receivers
    rx_names = list(scene.receivers.keys())
    for name in rx_names:
        scene.remove(name)

    # Get UE positions
    if ue_infos is None:
        if radio_map is None:
            radio_map = compute_radio_map(scene, cfg)
        # Oversample then filter (ray_test + SINR)
        oversample = cfg.num_ue * 4
        ue_infos_raw, _ = sample_ue_positions(radio_map, oversample, cfg)

        # Ray_test: shoot upward, hit roof = inside building
        import mitsuba as mi
        mi_scene = scene.mi_scene
        cand_pos = np.array([[info["x"], info["y"], info["z"]] for info in ue_infos_raw])
        n_cand = len(cand_pos)
        origins = mi.Point3f(cand_pos[:, 0], cand_pos[:, 1], np.full(n_cand, 1.5))
        directions = mi.Vector3f(np.zeros(n_cand), np.zeros(n_cand), np.ones(n_cand))
        hit = np.array(mi_scene.ray_test(mi.Ray3f(o=origins, d=directions)))
        ue_infos_raw = [info for info, h in zip(ue_infos_raw, hit) if not h]

        # SINR filter
        sinr_min_db = 5.0
        sinr_lin = np.array(radio_map.sinr).max(axis=0)
        sinr_db_map = 10 * np.log10(np.maximum(sinr_lin, 1e-10))
        cc = np.array(radio_map.cell_centers)
        x_c, y_c = cc[0, :, 0], cc[:, 0, 1]
        ue_infos_filtered = []
        for info in ue_infos_raw:
            xi = np.argmin(np.abs(x_c - info["x"]))
            yi = np.argmin(np.abs(y_c - info["y"]))
            if sinr_db_map[yi, xi] >= sinr_min_db:
                ue_infos_filtered.append(info)

        # Truncate to requested num_ue
        ue_infos = ue_infos_filtered[:cfg.num_ue]
        bs_counts = {}
        for info in ue_infos:
            bs_counts[info["bs_id"]] = bs_counts.get(info["bs_id"], 0) + 1
        counts = [bs_counts.get(i, 0) for i in range(cfg.num_bs)]
        print(f"  Snapshot {snapshot_id}: UEs per BS = {counts} "
              f"(filtered {len(ue_infos)}/{n_cand})")
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

    # Reuse PathSolver if provided, else create once
    if p_solver is None:
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
    if cir_tau.ndim > 2:
        while cir_tau.ndim > 2:
            cir_tau = cir_tau[:, 0] if cir_tau.shape[1] > 1 else cir_tau.squeeze(1)

    # CFR — GPU-accelerated CIR→CFR via batched matmul
    # H(f) = sum_paths[ a * exp(-j*2*pi*f*tau) ]
    if freqs_torch is None:
        frequencies = subcarrier_frequencies(cfg.num_subcarriers, cfg.subcarrier_spacing)
        freqs_torch = torch.tensor(frequencies.numpy(), dtype=torch.float32, device="cuda")

    n_ue = cir_a.shape[0]
    n_sc = len(freqs_torch)

    # Batched bmm: reshape to (n_ue, n_rx*n_tx, n_paths) @ (n_ue, n_paths, n_sc)
    n_rx, n_tx, n_paths = cir_a.shape[1], cir_a.shape[2], cir_a.shape[3]
    free_vram = torch.cuda.mem_get_info()[0]
    # Per-UE peak: a(rx*tx, paths) + exp_phase(paths, sc) + result(rx*tx, sc) in complex64
    per_ue_bytes = (n_rx * n_tx * n_paths + n_paths * n_sc + n_rx * n_tx * n_sc) * 8
    chunk = max(1, min(n_ue, int(free_vram * 0.5) // max(per_ue_bytes, 1)))

    cfr_chunks = []
    for i in range(0, n_ue, chunk):
        j = min(i + chunk, n_ue)
        a_t = torch.tensor(cir_a[i:j], dtype=torch.complex64, device="cuda")
        tau_t = torch.tensor(cir_tau[i:j], dtype=torch.float32, device="cuda")
        # exp_phase: (batch, n_paths, n_sc)
        exp_phase = torch.exp(-2j * np.pi * tau_t[:, :, None] * freqs_torch[None, None, :])
        # bmm: (batch, rx*tx, paths) @ (batch, paths, sc) → (batch, rx*tx, sc)
        cfr_batch = torch.bmm(
            a_t.reshape(j - i, n_rx * n_tx, n_paths),
            exp_phase,
        ).reshape(j - i, n_rx, n_tx, n_sc)
        cfr_chunks.append(cfr_batch.cpu().numpy())
        del a_t, tau_t, exp_phase, cfr_batch
    cfr = np.concatenate(cfr_chunks, axis=0) if len(cfr_chunks) > 1 else cfr_chunks[0]
    torch.cuda.empty_cache()

    # Save CIR + CFR
    if h5_file is not None:
        n_p = cir_a.shape[-1]
        max_paths = h5_file["cir_a"].shape[-1]
        local_idx = snapshot_id - h5_file.attrs["snapshot_start"]
        h5_file["cir_a"][local_idx, :, :, :, :n_p] = cir_a
        h5_file["cir_tau"][local_idx, :, :n_p] = cir_tau
        if "cfr" in h5_file:
            h5_file["cfr"][local_idx] = cfr
    else:
        snap_dir = data_dir / f"snapshot_{snapshot_id:04d}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            snap_dir / "channels.npz",
            cir_a=cir_a, cir_tau=cir_tau, cfr=cfr,
        )

    # Clean up GPU memory
    dr.flush_malloc_cache()
    dr.flush_kernel_cache()

    return ue_infos
