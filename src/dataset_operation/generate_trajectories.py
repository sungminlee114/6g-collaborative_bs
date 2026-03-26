"""Pre-compute UE trajectories and save to data_dir/trajectories.npz.

This decouples position computation from channel generation,
enabling multi-GPU parallel temporal dataset generation.

Mobility models:
  - gauss_markov (default): Smooth, realistic movement with tunable
    temporal correlation (alpha). Building-aware collision avoidance.
  - linear: Legacy constant-velocity straight-line (no collision check).

Usage:
    # Gauss-Markov with building avoidance (recommended)
    python -m src.dataset_operation.generate_trajectories \
        --preset munich_elaa_s_1k_15g --num_snapshots 1000 \
        --dt_ms 10 --velocities 0,1,8.3 --mobility gauss_markov

    # Then generate channels in parallel using pre-computed trajectories
    python -m src.dataset_operation.generate_parallel \
        --preset munich_elaa_s_1k_15g --num_snapshots 1000 \
        --gpus 0 1 2 3 4 5 7 --trajectories assets/data/channels_elaa_s_1k_15g_temporal/trajectories.npz

Output:
    trajectories.npz:
        positions: (num_snapshots, num_ue, 3)  — x,y,z per UE per snapshot
        velocities: (num_snapshots, num_ue, 2) — vx, vy per UE per snapshot (time-varying for GM)
        speeds: (num_ue,)                      — mean speed per UE
        bs_ids: (num_ue,)                      — serving BS per UE
        device_types: (num_ue,)                — device type index per UE
    metadata.json:
        UE info (device, BS assignment) — static across snapshots
"""
import argparse
import json
from pathlib import Path

import numpy as np


# ── Building collision detection ───────────────────────────────────

def load_buildings(buildings_path="assets/data/munich_buildings.json"):
    """Load building bounding boxes. Returns (N, 4) array of [x_min, y_min, x_max, y_max]."""
    path = Path(buildings_path)
    if not path.exists():
        return None
    with open(path) as f:
        buildings = json.load(f)
    bboxes = np.array(
        [[b["x_min"], b["y_min"], b["x_max"], b["y_max"]] for b in buildings],
        dtype=np.float32,
    )
    return bboxes


def is_inside_building(x, y, bboxes):
    """Check if (x, y) is inside any building bbox. Returns bool."""
    if bboxes is None:
        return False
    return np.any(
        (bboxes[:, 0] <= x) & (x <= bboxes[:, 2]) &
        (bboxes[:, 1] <= y) & (y <= bboxes[:, 3])
    )


def reflect_off_building(pos_old, pos_new, vel, bboxes, rng=None):
    """If pos_new is inside a building, reflect velocity and return corrected position.

    Tries 3 reflection strategies (x, y, both). If all fail, pushes UE to the
    nearest building edge. Falls back to staying at pos_old with randomized direction.
    """
    if bboxes is None or not is_inside_building(pos_new[0], pos_new[1], bboxes):
        return pos_new, vel

    dx = pos_new[0] - pos_old[0]
    dy = pos_new[1] - pos_old[1]
    vel_new = vel.copy()

    # Strategy 1: Reflect x-component
    cand = pos_old.copy()
    cand[0] = pos_old[0] - dx
    cand[1] = pos_new[1]
    if not is_inside_building(cand[0], cand[1], bboxes):
        vel_new[0] = -vel[0]
        return cand, vel_new

    # Strategy 2: Reflect y-component
    cand = pos_old.copy()
    cand[0] = pos_new[0]
    cand[1] = pos_old[1] - dy
    if not is_inside_building(cand[0], cand[1], bboxes):
        vel_new[1] = -vel[1]
        return cand, vel_new

    # Strategy 3: Reflect both
    cand = pos_old.copy()
    cand[0] = pos_old[0] - dx
    cand[1] = pos_old[1] - dy
    if not is_inside_building(cand[0], cand[1], bboxes):
        return cand, -vel

    # Strategy 4: Try 8 random directions from pos_old
    speed = np.linalg.norm(vel)
    step = max(abs(dx), abs(dy), 2.0)  # minimum 2m step to escape buildings
    _rng = rng if rng is not None else np.random.default_rng()
    for _ in range(8):
        angle = _rng.uniform(0, 2 * np.pi)
        cand = pos_old.copy()
        cand[0] += step * np.cos(angle)
        cand[1] += step * np.sin(angle)
        if not is_inside_building(cand[0], cand[1], bboxes):
            vel_new = np.array([speed * np.cos(angle), speed * np.sin(angle)],
                               dtype=np.float32)
            return cand, vel_new

    # All strategies failed — stay at pos_old, randomize direction
    angle = _rng.uniform(0, 2 * np.pi)
    vel_new = np.array([speed * np.cos(angle), speed * np.sin(angle)], dtype=np.float32)
    return pos_old, vel_new


# ── Gauss-Markov mobility model ───────────────────────────────────

def propagate_gauss_markov(positions, vel_array, speed_array, dt, num_snapshots,
                           rng, bboxes=None, alpha=0.75, speed_sigma_ratio=0.3,
                           direction_sigma=np.pi / 6):
    """Gauss-Markov mobility with building collision avoidance.

    v(t) = alpha * v(t-1) + (1-alpha) * v_mean + sqrt(1-alpha^2) * noise

    Args:
        positions: (num_snapshots, num_ue, 3) — positions[0] already filled
        vel_array: (num_ue, 2) — initial velocities
        speed_array: (num_ue,) — mean speeds per UE
        dt: time step in seconds
        num_snapshots: total snapshots
        rng: numpy random generator
        bboxes: (N, 4) building bounding boxes or None
        alpha: temporal correlation (0=random, 1=straight line)
        speed_sigma_ratio: speed noise as fraction of mean speed
        direction_sigma: direction noise std in radians
    """
    num_ue = len(speed_array)
    # Store time-varying velocities
    vel_history = np.zeros((num_snapshots, num_ue, 2), dtype=np.float32)
    vel_history[0] = vel_array.copy()

    # Current velocities (will be updated each step)
    cur_vel = vel_array.copy()

    from tqdm.auto import tqdm
    for t in tqdm(range(1, num_snapshots), desc="Propagating", unit="snap"):
        for u in range(num_ue):
            mean_speed = speed_array[u]
            if mean_speed < 0.01:
                # Static UE
                positions[t, u] = positions[t - 1, u]
                vel_history[t, u] = [0.0, 0.0]
                continue

            # Current speed and direction
            cur_speed = np.linalg.norm(cur_vel[u])
            if cur_speed < 1e-6:
                cur_speed = mean_speed
                cur_dir = rng.uniform(0, 2 * np.pi)
            else:
                cur_dir = np.arctan2(cur_vel[u, 1], cur_vel[u, 0])

            # Gauss-Markov update
            speed_noise = rng.normal(0, speed_sigma_ratio * mean_speed)
            dir_noise = rng.normal(0, direction_sigma)

            mean_dir = np.arctan2(cur_vel[u, 1], cur_vel[u, 0])

            new_speed = alpha * cur_speed + (1 - alpha) * mean_speed + \
                np.sqrt(1 - alpha**2) * speed_noise
            new_speed = max(new_speed, 0.1 * mean_speed)  # don't go below 10% of mean
            new_speed = min(new_speed, 3.0 * mean_speed)  # don't exceed 3x mean

            new_dir = alpha * cur_dir + (1 - alpha) * mean_dir + \
                np.sqrt(1 - alpha**2) * dir_noise

            new_vel = np.array([
                new_speed * np.cos(new_dir),
                new_speed * np.sin(new_dir),
            ], dtype=np.float32)

            # Propose new position
            new_pos = positions[t - 1, u].copy()
            new_pos[0] += new_vel[0] * dt
            new_pos[1] += new_vel[1] * dt

            # Building collision check + reflection
            new_pos, new_vel = reflect_off_building(
                positions[t - 1, u], new_pos, new_vel, bboxes, rng=rng
            )

            positions[t, u] = new_pos
            cur_vel[u] = new_vel
            vel_history[t, u] = new_vel

    return vel_history


def propagate_linear(positions, vel_array, dt, num_snapshots, bboxes=None):
    """Constant-velocity straight-line propagation with building collision check."""
    num_ue = positions.shape[1]
    vel_history = np.zeros((num_snapshots, num_ue, 2), dtype=np.float32)
    vel_history[0] = vel_array.copy()
    cur_vel = vel_array.copy()

    for t in range(1, num_snapshots):
        for u in range(num_ue):
            new_pos = positions[t - 1, u].copy()
            new_pos[0] += cur_vel[u, 0] * dt
            new_pos[1] += cur_vel[u, 1] * dt
            new_pos, new_vel = reflect_off_building(
                positions[t - 1, u], new_pos, cur_vel[u], bboxes,
            )
            positions[t, u] = new_pos
            cur_vel[u] = new_vel
            vel_history[t, u] = new_vel

    return vel_history


# ── Main trajectory generation ─────────────────────────────────────

def generate_trajectories_gpu(preset, num_snapshots, num_ue, dt, velocities,
                              seed=42, data_dir=None, mobility="gauss_markov",
                              alpha=0.75, target_bs=None):
    """Sample initial positions from radio map (needs GPU), then compute trajectories (CPU).

    Args:
        target_bs: If set, place ALL UEs near this BS only (single-cell mode).

    Returns data_dir where trajectories.npz was saved.
    """
    from src.config import SceneConfig
    from src.dataset_operation.generate import build_scene, compute_radio_map, sample_ue_positions

    cfg = SceneConfig.from_preset(preset, num_ue=num_ue)

    if data_dir is None:
        import yaml
        yaml_path = Path("assets/configs") / f"{preset}.yaml"
        if yaml_path.exists():
            with open(yaml_path) as f:
                raw = yaml.safe_load(f)
            data_dir = raw.get("data_dir", f"assets/data/channels_{preset}")
        else:
            data_dir = f"assets/data/channels_{preset}"
        data_dir = f"{data_dir}_temporal"

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Build scene and compute radio map (needs GPU)
    print(f"Building scene for {preset}...")
    scene = build_scene(cfg)
    print("Computing radio map...")
    radio_map = compute_radio_map(scene, cfg)

    # Sample initial UE positions
    np.random.seed(seed)
    if target_bs is not None:
        # Single-cell mode: oversample then filter to target BS
        # Extra margin for path quality filtering downstream
        oversample = num_ue * cfg.num_bs * 4
        all_infos, _ = sample_ue_positions(radio_map, oversample, cfg)
        ue_infos = [info for info in all_infos if info["bs_id"] == target_bs]
        print(f"  BS{target_bs} candidates: {len(ue_infos)} (need {num_ue} after path filter)")
    else:
        ue_infos, counts = sample_ue_positions(radio_map, num_ue, cfg)
        print(f"Initial UEs per BS = {counts}")

    # ── Filter 1: Building interior (ray_test — shoot ray upward, hit roof = inside) ──
    import mitsuba as mi
    mi_scene = scene.mi_scene
    cand_pos = np.array([[info["x"], info["y"], info["z"]] for info in ue_infos])
    n_cand = len(cand_pos)
    origins = mi.Point3f(cand_pos[:, 0], cand_pos[:, 1], np.full(n_cand, 1.5))
    directions = mi.Vector3f(np.zeros(n_cand), np.zeros(n_cand), np.ones(n_cand))
    hit = np.array(mi_scene.ray_test(mi.Ray3f(o=origins, d=directions)))
    n_inside = int(hit.sum())
    ue_infos = [info for info, h in zip(ue_infos, hit) if not h]
    print(f"  Building ray_test: {n_cand} → {len(ue_infos)} UEs "
          f"(removed {n_inside} under roofs)")

    # ── Filter 2: SINR coverage (radio map) ──
    sinr_min_db = 5.0
    sinr_lin = np.array(radio_map.sinr)     # (num_tx, W, H)
    best_sinr = sinr_lin.max(axis=0)        # (W, H)
    sinr_db_map = 10 * np.log10(np.maximum(best_sinr, 1e-10))
    cc = np.array(radio_map.cell_centers)   # (nY, nX, 3)
    x_centers = cc[0, :, 0]
    y_centers = cc[:, 0, 1]

    n_before_sinr = len(ue_infos)
    sinr_kept = []
    for info in ue_infos:
        xi = np.argmin(np.abs(x_centers - info["x"]))
        yi = np.argmin(np.abs(y_centers - info["y"]))
        snr = sinr_db_map[yi, xi]  # (nY, nX) indexing
        if snr >= sinr_min_db:
            sinr_kept.append(info)
    ue_infos = sinr_kept
    print(f"  SINR filter (>= {sinr_min_db} dB): {n_before_sinr} → {len(ue_infos)} UEs")

    # ── Filter 3: Path quality (PathSolver probe) ──
    min_significant_paths = 5
    print(f"Path quality probe: filtering UEs with >= {min_significant_paths} significant paths...")
    from sionna.rt import Receiver, PathSolver
    rx_names = list(scene.receivers.keys())
    for name in rx_names:
        scene.remove(name)
    for i, info in enumerate(ue_infos):
        scene.add(Receiver(
            name=f"ue_{i}",
            position=[info["x"], info["y"], info["z"]],
            orientation=[0, 0, 0],
        ))
    probe_solver = PathSolver()
    probe_paths = probe_solver(
        scene=scene, max_depth=cfg.max_depth,
        max_num_paths_per_src=cfg.max_num_paths_per_src,
        samples_per_src=cfg.samples_per_src,
        los=True, specular_reflection=True, diffuse_reflection=True,
        refraction=True, synthetic_array=cfg.synthetic_array, seed=seed,
    )
    probe_a, _ = probe_paths.cir(
        normalize_delays=True,
        associated_tx_idxs=[info["bs_id"] for info in ue_infos],
        out_type="numpy",
    )
    probe_a = np.squeeze(probe_a)  # (n_ue, n_rx, n_tx, n_paths)

    keep_mask = []
    path_counts = []
    for u in range(len(ue_infos)):
        power = np.abs(probe_a[u]).flatten()
        peak = power.max()
        if peak < 1e-12:
            keep_mask.append(False)
            path_counts.append(0)
            continue
        threshold = peak * 0.01  # -20dB
        n_sig = int(np.sum(power > threshold))
        path_counts.append(n_sig)
        keep_mask.append(n_sig >= min_significant_paths)

    n_before = len(ue_infos)
    ue_infos = [info for info, keep in zip(ue_infos, keep_mask) if keep]
    path_counts_kept = [c for c, keep in zip(path_counts, keep_mask) if keep]
    print(f"  Path filter: {n_before} → {len(ue_infos)} UEs "
          f"(removed {n_before - len(ue_infos)}, min_paths={min_significant_paths})")
    if path_counts_kept:
        print(f"  Significant paths: min={min(path_counts_kept)}, "
              f"median={int(np.median(path_counts_kept))}, max={max(path_counts_kept)}")

    # Cleanup
    rx_names = list(scene.receivers.keys())
    for name in rx_names:
        scene.remove(name)

    import drjit as dr
    dr.flush_malloc_cache()
    dr.flush_kernel_cache()

    if len(ue_infos) < num_ue:
        print(f"  Warning: only {len(ue_infos)} UEs passed filters (requested {num_ue})")
        num_ue = len(ue_infos)
    elif len(ue_infos) > num_ue:
        # Sort by distance to target BS — prefer closer UEs with better connectivity
        if target_bs is not None:
            bs_pos = np.array(cfg.bs_positions[target_bs])[:2]
            ue_infos.sort(key=lambda u: (u["x"] - bs_pos[0])**2 + (u["y"] - bs_pos[1])**2)
            max_dist = np.sqrt((ue_infos[num_ue - 1]["x"] - bs_pos[0])**2 +
                               (ue_infos[num_ue - 1]["y"] - bs_pos[1])**2)
            print(f"  Sorted by distance to BS{target_bs}: max_dist={max_dist:.1f}m")
        ue_infos = ue_infos[:num_ue]
        print(f"  Truncated to {num_ue} UEs")

    # Shuffle UE order so speed groups are spatially dispersed
    # (sample_ue_positions returns spatially clustered UEs)
    rng_shuffle = np.random.default_rng(seed)
    shuffle_idx = rng_shuffle.permutation(len(ue_infos))
    ue_infos = [ue_infos[i] for i in shuffle_idx]
    print(f"  Shuffled UE order for spatial dispersion")

    # Load buildings for collision avoidance
    bboxes = load_buildings()
    if bboxes is not None:
        print(f"Loaded {len(bboxes)} building bounding boxes for collision avoidance")
    else:
        print("Warning: No building data found, skipping collision avoidance")

    # Compute all trajectory positions
    rng = np.random.default_rng(seed)
    positions = np.zeros((num_snapshots, num_ue, 3), dtype=np.float32)
    vel_array = np.zeros((num_ue, 2), dtype=np.float32)
    speed_array = np.zeros(num_ue, dtype=np.float32)
    bs_ids = np.zeros(num_ue, dtype=np.int32)
    device_types = np.zeros(num_ue, dtype=np.int32)

    # Initial positions and velocities
    # Deterministic speed group assignment: UEs are assigned to speed groups
    # round-robin so each speed gets exactly ue_per_speed UEs.
    # e.g. velocities=[0,1,2,5,8.3], num_ue=25 → 5 UEs per speed
    ue_per_speed = max(1, num_ue // len(velocities))
    for u, info in enumerate(ue_infos):
        positions[0, u] = [info["x"], info["y"], info["z"]]
        bs_ids[u] = info["bs_id"]
        device_types[u] = info["ue_device_type"]

        speed_group = min(u // ue_per_speed, len(velocities) - 1)
        v = velocities[speed_group]
        angle = rng.uniform(0, 2 * np.pi)
        vel_array[u] = [v * np.cos(angle), v * np.sin(angle)]
        speed_array[u] = v

    # Verify initial positions are outside buildings (bbox check only — no relocation).
    # Radio map sampling uses actual RT geometry which is more accurate than bbox.
    # We only warn here; positions from sample_ue_positions are trusted.
    if bboxes is not None:
        n_inside = 0
        for u in range(num_ue):
            x, y = positions[0, u, 0], positions[0, u, 1]
            if is_inside_building(x, y, bboxes):
                n_inside += 1
        if n_inside > 0:
            print(f"  Note: {n_inside}/{num_ue} UEs inside building bbox "
                  f"(bbox is approximate, RT geometry is authoritative)")

    # Propagate — coarse Gauss-Markov at dt_coarse, then interpolate to target dt.
    # This ensures building avoidance works (step size big enough to escape buildings)
    # while providing fine temporal resolution for channel measurement.
    coarse_dt = 0.01  # 10ms — proven to work with building avoidance
    upsample = max(1, round(coarse_dt / dt))
    coarse_steps = max(2, num_snapshots // upsample + 1)

    print(f"Mobility model: {mobility} (alpha={alpha})")
    if upsample > 1:
        print(f"  Coarse propagation: {coarse_steps} steps @ dt={coarse_dt*1000:.1f}ms, "
              f"then {upsample}x interpolation → {num_snapshots} @ dt={dt*1000:.4f}ms")

    # bbox collision avoidance: only use for gauss_markov (long trajectories).
    # For linear mobility with short duration (<1s), bbox is too inaccurate
    # (overestimates buildings) and positions from RT radio map are authoritative.
    use_bbox_collision = (mobility == "gauss_markov") and (dt * num_snapshots > 0.5)
    if not use_bbox_collision:
        print(f"  bbox collision: OFF (short linear trajectory, RT positions trusted)")
    collision_bboxes = bboxes if use_bbox_collision else None

    if upsample > 1:
        # Step 1: Coarse trajectory
        max_retries = 3 if use_bbox_collision else 0
        for attempt in range(max_retries + 1):
            coarse_pos = np.zeros((coarse_steps, num_ue, 3), dtype=np.float32)
            coarse_pos[0] = positions[0]
            if mobility == "gauss_markov":
                propagate_gauss_markov(
                    coarse_pos, vel_array, speed_array, coarse_dt, coarse_steps,
                    rng, bboxes=collision_bboxes, alpha=alpha,
                )
            else:
                propagate_linear(coarse_pos, vel_array, coarse_dt, coarse_steps, bboxes=collision_bboxes)

            if not use_bbox_collision:
                break

            # Detect stuck UEs: moving UEs with <1% expected displacement
            stuck_ues = []
            for u in range(num_ue):
                if speed_array[u] < 0.01:
                    continue
                disp = np.linalg.norm(coarse_pos[-1, u, :2] - coarse_pos[0, u, :2])
                expected = speed_array[u] * coarse_dt * coarse_steps
                if disp < expected * 0.01:
                    stuck_ues.append(u)

            if not stuck_ues or attempt == max_retries:
                if stuck_ues:
                    print(f"  ⚠ {len(stuck_ues)} UEs still stuck after {max_retries} retries: {stuck_ues}")
                break

            print(f"  Attempt {attempt+1}: relocating {len(stuck_ues)} stuck UEs")
            for u in stuck_ues:
                for radius in [20, 50, 100, 200]:
                    relocated = False
                    for _ in range(30):
                        angle = rng.uniform(0, 2 * np.pi)
                        cx = positions[0, u, 0] + radius * np.cos(angle)
                        cy = positions[0, u, 1] + radius * np.sin(angle)
                        if bboxes is not None and not is_inside_building(cx, cy, bboxes):
                            expanded = np.column_stack([
                                bboxes[:, 0] - 5, bboxes[:, 1] - 5,
                                bboxes[:, 2] + 5, bboxes[:, 3] + 5,
                            ])
                            if not np.any(
                                (expanded[:, 0] <= cx) & (cx <= expanded[:, 2]) &
                                (expanded[:, 1] <= cy) & (cy <= expanded[:, 3])
                            ):
                                positions[0, u, 0] = cx
                                positions[0, u, 1] = cy
                                relocated = True
                                break
                    if relocated:
                        break

        # Step 2: Linear interpolation to fine resolution
        from tqdm.auto import tqdm as _tqdm
        for c in _tqdm(range(coarse_steps - 1), desc="Interpolating", unit="seg"):
            t_start = c * upsample
            t_end = min((c + 1) * upsample, num_snapshots)
            for t in range(t_start, t_end):
                frac = (t - t_start) / upsample
                positions[t] = (1 - frac) * coarse_pos[c] + frac * coarse_pos[min(c + 1, coarse_steps - 1)]

        vel_history = np.zeros((num_snapshots, num_ue, 2), dtype=np.float32)
        for t in range(1, num_snapshots):
            vel_history[t, :, 0] = (positions[t, :, 0] - positions[t-1, :, 0]) / dt
            vel_history[t, :, 1] = (positions[t, :, 1] - positions[t-1, :, 1]) / dt
        vel_history[0] = vel_history[1]
    else:
        # No upsampling needed — direct propagation
        if mobility == "gauss_markov":
            vel_history = propagate_gauss_markov(
                positions, vel_array, speed_array, dt, num_snapshots,
                rng, bboxes=collision_bboxes, alpha=alpha,
            )
        elif mobility == "linear":
            vel_history = propagate_linear(positions, vel_array, dt, num_snapshots, bboxes=collision_bboxes)
        else:
            raise ValueError(f"Unknown mobility model: {mobility}")

    # Save
    np.savez_compressed(
        data_dir / "trajectories.npz",
        positions=positions,
        velocities=vel_history,  # (num_snapshots, num_ue, 2) for GM, same shape for linear
        speeds=speed_array,
        bs_ids=bs_ids,
        device_types=device_types,
    )

    # Save UE metadata (static info, same across snapshots)
    ue_meta = []
    for u, info in enumerate(ue_infos):
        ue_meta.append({
            "ue_id": u,
            "bs_id": int(bs_ids[u]),
            "ue_device_type": int(device_types[u]),
            "ue_rx_rows": info["ue_rx_rows"],
            "ue_rx_cols": info["ue_rx_cols"],
            "ue_polarization": info["ue_polarization"],
            "vx": float(vel_array[u, 0]),
            "vy": float(vel_array[u, 1]),
            "speed": float(speed_array[u]),
        })
    with open(data_dir / "ue_meta.json", "w") as f:
        json.dump(ue_meta, f, indent=2)

    # Save dataset config
    ds_info = {
        "preset": preset,
        "mode": "temporal",
        "mobility_model": mobility,
        "alpha": alpha if mobility == "gauss_markov" else None,
        "num_snapshots": num_snapshots,
        "num_ue": num_ue,
        "dt_s": dt,
        "dt_ms": dt * 1000,
        "total_duration_ms": dt * num_snapshots * 1000,
        "velocities_ms": velocities,
        "seed": seed,
        "building_collision": bboxes is not None,
        "num_buildings": len(bboxes) if bboxes is not None else 0,
        "bs_positions": [list(p) for p in cfg.bs_positions[:cfg.num_bs]],
        "target_bs": target_bs,
    }
    with open(data_dir / "trajectory_info.json", "w") as f:
        json.dump(ds_info, f, indent=2)

    total_ms = dt * num_snapshots * 1000
    max_displacement = max(speed_array) * dt * num_snapshots if max(speed_array) > 0 else 0
    print(f"\nTrajectories saved to {data_dir}/trajectories.npz")
    print(f"  {num_snapshots} snapshots × {num_ue} UEs")
    print(f"  dt={dt*1000:.4f}ms, total={total_ms:.0f}ms ({total_ms/1000:.1f}s)")
    print(f"  Speed distribution: {np.unique(speed_array, return_counts=True)}")
    print(f"  Max displacement: {max_displacement:.1f}m")
    return str(data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, required=True)
    parser.add_argument("--num_snapshots", type=int, default=1000)
    parser.add_argument("--num_ue", type=int, default=100)
    # dt is derived from config's numerology_mu (slot duration)
    parser.add_argument("--velocities", type=str, default="0,1,8.3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--mobility", type=str, default="gauss_markov",
                        choices=["gauss_markov", "linear"],
                        help="Mobility model (default: gauss_markov)")
    parser.add_argument("--alpha", type=float, default=0.75,
                        help="Gauss-Markov temporal correlation (0=random, 1=straight)")
    parser.add_argument("--target_bs", type=int, default=None,
                        help="Place all UEs near this BS only (single-cell mode)")
    args = parser.parse_args()

    velocities = [float(v) for v in args.velocities.split(",")]

    from src.config import SceneConfig
    cfg = SceneConfig.from_preset(args.preset)
    assert cfg.numerology_mu >= 0, f"numerology_mu not set in preset '{args.preset}'"
    dt = cfg.slot_duration_s
    print(f"dt = {dt*1000:.4f} ms (mu={cfg.numerology_mu}, SCS={cfg.scs/1e3:.0f} kHz)")

    generate_trajectories_gpu(
        args.preset, args.num_snapshots, args.num_ue,
        dt=dt, velocities=velocities,
        seed=args.seed, data_dir=args.data_dir,
        mobility=args.mobility, alpha=args.alpha,
        target_bs=args.target_bs,
    )
