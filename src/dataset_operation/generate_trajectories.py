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


def propagate_linear(positions, vel_array, dt, num_snapshots):
    """Legacy constant-velocity straight-line propagation (no collision check)."""
    vel_history = np.zeros((num_snapshots, positions.shape[1], 2), dtype=np.float32)
    vel_history[:] = vel_array[None, :]  # constant across time

    for t in range(1, num_snapshots):
        positions[t] = positions[t - 1]
        positions[t, :, 0] += vel_array[:, 0] * dt
        positions[t, :, 1] += vel_array[:, 1] * dt

    return vel_history


# ── Main trajectory generation ─────────────────────────────────────

def generate_trajectories_gpu(preset, num_snapshots, num_ue, dt, velocities,
                              seed=42, data_dir=None, mobility="gauss_markov",
                              alpha=0.75):
    """Sample initial positions from radio map (needs GPU), then compute trajectories (CPU).

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
    ue_infos, counts = sample_ue_positions(radio_map, num_ue, cfg)
    print(f"Initial UEs per BS = {counts}")

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
    for u, info in enumerate(ue_infos):
        positions[0, u] = [info["x"], info["y"], info["z"]]
        bs_ids[u] = info["bs_id"]
        device_types[u] = info["ue_device_type"]

        v = rng.choice(velocities)
        angle = rng.uniform(0, 2 * np.pi)
        vel_array[u] = [v * np.cos(angle), v * np.sin(angle)]
        speed_array[u] = v

    # Fix initial positions inside buildings — push to nearest building edge
    if bboxes is not None:
        n_fixed = 0
        for u in range(num_ue):
            if is_inside_building(positions[0, u, 0], positions[0, u, 1], bboxes):
                # Find which building(s) contain this point
                x, y = positions[0, u, 0], positions[0, u, 1]
                inside = (bboxes[:, 0] <= x) & (x <= bboxes[:, 2]) & \
                         (bboxes[:, 1] <= y) & (y <= bboxes[:, 3])
                for bidx in np.where(inside)[0]:
                    # Push to nearest edge + 1m margin
                    bx0, by0, bx2, by2 = bboxes[bidx]
                    dists = [abs(x - bx0), abs(x - bx2), abs(y - by0), abs(y - by2)]
                    edge = np.argmin(dists)
                    margin = 1.0
                    if edge == 0:
                        positions[0, u, 0] = bx0 - margin
                    elif edge == 1:
                        positions[0, u, 0] = bx2 + margin
                    elif edge == 2:
                        positions[0, u, 1] = by0 - margin
                    else:
                        positions[0, u, 1] = by2 + margin
                    break  # fix for first containing building
                if not is_inside_building(positions[0, u, 0], positions[0, u, 1], bboxes):
                    n_fixed += 1
        if n_fixed > 0:
            print(f"  Fixed {n_fixed} UEs initially inside buildings (pushed to nearest edge)")

    # Propagate
    print(f"Mobility model: {mobility} (alpha={alpha})")
    if mobility == "gauss_markov":
        vel_history = propagate_gauss_markov(
            positions, vel_array, speed_array, dt, num_snapshots,
            rng, bboxes=bboxes, alpha=alpha,
        )
    elif mobility == "linear":
        vel_history = propagate_linear(positions, vel_array, dt, num_snapshots)
    else:
        raise ValueError(f"Unknown mobility model: {mobility}")

    # Count building collisions avoided
    if bboxes is not None and mobility == "gauss_markov":
        from tqdm.auto import tqdm as _tqdm
        inside_count = 0
        for t in _tqdm(range(num_snapshots), desc="Verifying positions", unit="snap"):
            for u in range(num_ue):
                if is_inside_building(positions[t, u, 0], positions[t, u, 1], bboxes):
                    inside_count += 1
        if inside_count > 0:
            print(f"  Warning: {inside_count} positions still inside buildings")
        else:
            print(f"  All {num_snapshots * num_ue} positions verified outside buildings")

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
    parser.add_argument("--dt_ms", type=float, default=10.0)
    parser.add_argument("--velocities", type=str, default="0,1,8.3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--mobility", type=str, default="gauss_markov",
                        choices=["gauss_markov", "linear"],
                        help="Mobility model (default: gauss_markov)")
    parser.add_argument("--alpha", type=float, default=0.75,
                        help="Gauss-Markov temporal correlation (0=random, 1=straight)")
    args = parser.parse_args()

    velocities = [float(v) for v in args.velocities.split(",")]

    generate_trajectories_gpu(
        args.preset, args.num_snapshots, args.num_ue,
        dt=args.dt_ms / 1000, velocities=velocities,
        seed=args.seed, data_dir=args.data_dir,
        mobility=args.mobility, alpha=args.alpha,
    )
