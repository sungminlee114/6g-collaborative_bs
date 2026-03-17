"""Pre-compute UE trajectories and save to data_dir/trajectories.npz.

This decouples position computation from channel generation,
enabling multi-GPU parallel temporal dataset generation.

Usage:
    # Generate trajectories (CPU only, instant)
    python -m src.dataset_operation.generate_trajectories \
        --preset munich_elaa_s_1k_15g --num_snapshots 1000 \
        --dt_ms 10 --velocities 0,1,8.3

    # Then generate channels in parallel using pre-computed trajectories
    python -m src.dataset_operation.generate_parallel \
        --preset munich_elaa_s_1k_15g --num_snapshots 1000 \
        --gpus 0 1 2 3 4 5 7 --trajectories assets/data/channels_elaa_s_1k_15g_temporal/trajectories.npz

Output:
    trajectories.npz:
        positions: (num_snapshots, num_ue, 3)  — x,y,z per UE per snapshot
        velocities: (num_ue, 2)                — vx, vy per UE
        speeds: (num_ue,)                      — scalar speed per UE
        bs_ids: (num_ue,)                      — serving BS per UE
        device_types: (num_ue,)                — device type index per UE
    metadata.json:
        UE info (device, BS assignment) — static across snapshots
"""
import argparse
import json
from pathlib import Path

import numpy as np


def generate_trajectories_gpu(preset, num_snapshots, num_ue, dt, velocities,
                              seed=42, data_dir=None):
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

    # Compute all trajectory positions
    rng = np.random.default_rng(seed)
    positions = np.zeros((num_snapshots, num_ue, 3), dtype=np.float32)
    vel_array = np.zeros((num_ue, 2), dtype=np.float32)
    speed_array = np.zeros(num_ue, dtype=np.float32)
    bs_ids = np.zeros(num_ue, dtype=np.int32)
    device_types = np.zeros(num_ue, dtype=np.int32)

    # Initial positions
    for u, info in enumerate(ue_infos):
        positions[0, u] = [info["x"], info["y"], info["z"]]
        bs_ids[u] = info["bs_id"]
        device_types[u] = info["ue_device_type"]

        # Assign velocity
        v = rng.choice(velocities)
        angle = rng.uniform(0, 2 * np.pi)
        vel_array[u] = [v * np.cos(angle), v * np.sin(angle)]
        speed_array[u] = v

    # Propagate positions
    for t in range(1, num_snapshots):
        positions[t] = positions[t - 1]
        positions[t, :, 0] += vel_array[:, 0] * dt
        positions[t, :, 1] += vel_array[:, 1] * dt
        # z stays fixed

    # Save
    np.savez_compressed(
        data_dir / "trajectories.npz",
        positions=positions,
        velocities=vel_array,
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
        "num_snapshots": num_snapshots,
        "num_ue": num_ue,
        "dt_s": dt,
        "dt_ms": dt * 1000,
        "total_duration_ms": dt * num_snapshots * 1000,
        "velocities_ms": velocities,
        "seed": seed,
    }
    with open(data_dir / "trajectory_info.json", "w") as f:
        json.dump(ds_info, f, indent=2)

    total_ms = dt * num_snapshots * 1000
    max_displacement = max(speed_array) * dt * num_snapshots
    print(f"\nTrajectories saved to {data_dir}/trajectories.npz")
    print(f"  {num_snapshots} snapshots × {num_ue} UEs")
    print(f"  dt={dt*1000:.1f}ms, total={total_ms:.0f}ms ({total_ms/1000:.1f}s)")
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
    args = parser.parse_args()

    velocities = [float(v) for v in args.velocities.split(",")]

    generate_trajectories_gpu(
        args.preset, args.num_snapshots, args.num_ue,
        dt=args.dt_ms / 1000, velocities=velocities,
        seed=args.seed, data_dir=args.data_dir,
    )
