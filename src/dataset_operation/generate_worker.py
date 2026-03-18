"""Worker for parallel dataset generation — handles a slice of snapshots.

Called by generate_parallel.py, not directly.
Supports both independent and temporal (pre-computed trajectories) modes.
Also handles --merge_only to combine per-worker metadata.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def resolve_data_dir(preset, explicit_dir=None):
    """Resolve data_dir from args or YAML config."""
    if explicit_dir:
        return explicit_dir
    import yaml
    yaml_path = Path("assets/configs") / f"{preset}.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        return raw.get("data_dir", f"assets/data/channels_{preset}")
    return f"assets/data/channels_{preset}"


def merge_metadata(data_dir, preset):
    """Merge per-snapshot metadata and create unified metadata.parquet + bs_info.json."""
    from src.config import SceneConfig
    data_dir = Path(data_dir)
    cfg = SceneConfig.from_preset(preset)

    # Collect all per-snapshot metadata
    all_records = []
    snap_dirs = sorted(data_dir.glob("snapshot_*"))
    for snap_dir in snap_dirs:
        meta_path = snap_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                records = json.load(f)
            all_records.extend(records)

    if not all_records:
        print("No snapshot metadata found to merge.")
        return

    df = pd.DataFrame(all_records)
    df.to_parquet(data_dir / "metadata.parquet", index=False)

    # Save BS info
    traj_info_path = data_dir / "trajectory_info.json"
    mode = "independent"
    if traj_info_path.exists():
        with open(traj_info_path) as f:
            mode = json.load(f).get("mode", "temporal")

    bs_info = {
        "positions": [list(p) for p in cfg.bs_positions[:cfg.num_bs]],
        "num_bs": cfg.num_bs,
        "num_snapshots": len(snap_dirs),
        "frequency_hz": cfg.frequency,
        "bandwidth_hz": cfg.bandwidth,
        "num_subcarriers": cfg.num_subcarriers,
        "num_tx_ant": cfg.num_tx_ant,
        "num_rx_ant": cfg.num_rx_ant,
        "synthetic_array": cfg.synthetic_array,
        "mode": mode,
    }
    with open(data_dir / "bs_info.json", "w") as f:
        json.dump(bs_info, f, indent=2)

    print(f"Merged {len(snap_dirs)} snapshots, {len(all_records)} samples → {data_dir / 'metadata.parquet'}")


def run_worker(preset, snapshot_start, snapshot_end, num_ue, data_dir,
               trajectories_path=None):
    """Generate snapshots [start, end) for a single GPU.

    If trajectories_path is given, reads pre-computed UE positions (temporal mode).
    Otherwise samples new positions each snapshot (independent mode).
    """
    from src.config import SceneConfig
    from src.dataset_operation.generate import build_scene, generate_snapshot

    cfg = SceneConfig.from_preset(preset, num_ue=num_ue)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load trajectories if temporal mode
    traj_data = None
    ue_meta = None
    if trajectories_path:
        traj_data = np.load(trajectories_path)
        ue_meta_path = Path(trajectories_path).parent / "ue_meta.json"
        if ue_meta_path.exists():
            with open(ue_meta_path) as f:
                ue_meta = json.load(f)

    mode = "temporal" if traj_data is not None else "independent"
    print(f"Worker: preset={preset}, snapshots=[{snapshot_start}, {snapshot_end}), mode={mode}")
    print(f"  TX: {cfg.tx_rows}x{cfg.tx_cols}={cfg.num_tx_ant}, "
          f"Freq: {cfg.frequency/1e9:.1f}GHz, SC: {cfg.num_subcarriers}, "
          f"synthetic_array={cfg.synthetic_array}")

    scene = build_scene(cfg)

    for snap_id in range(snapshot_start, snapshot_end):
        # Resume: skip if valid snapshot already exists
        snap_dir = data_dir / f"snapshot_{snap_id:04d}"
        npz_path = snap_dir / "channels.npz"
        if npz_path.exists():
            try:
                d = np.load(npz_path)
                if "cfr" in d and d["cfr"].shape[0] > 0:
                    print(f"  ⏭ Snapshot {snap_id} exists ({d['cfr'].shape}), skipping")
                    continue
            except Exception:
                print(f"  ⚠ Snapshot {snap_id} corrupt, regenerating")

        seed = snap_id * 17 + 41

        # Build ue_infos from trajectories (temporal) or let generate_snapshot sample (independent)
        ue_infos = None
        if traj_data is not None:
            positions = traj_data["positions"][snap_id]  # (num_ue, 3)
            bs_ids = traj_data["bs_ids"]
            device_types = traj_data["device_types"]
            velocities = traj_data["velocities"]

            # velocities shape: (num_snapshots, num_ue, 2) for Gauss-Markov
            #                or (num_ue, 2) for legacy linear
            if velocities.ndim == 3:
                snap_vel = velocities[snap_id]  # (num_ue, 2)
            else:
                snap_vel = velocities  # (num_ue, 2) constant

            ue_infos = []
            for u in range(len(bs_ids)):
                meta = ue_meta[u] if ue_meta else {}
                ue_infos.append({
                    "bs_id": int(bs_ids[u]),
                    "idx_in_bs": u,
                    "x": float(positions[u, 0]),
                    "y": float(positions[u, 1]),
                    "z": float(positions[u, 2]),
                    "ue_device_type": int(device_types[u]),
                    "ue_rx_rows": meta.get("ue_rx_rows", 1),
                    "ue_rx_cols": meta.get("ue_rx_cols", 1),
                    "ue_polarization": meta.get("ue_polarization", "cross"),
                    "vx": float(snap_vel[u, 0]),
                    "vy": float(snap_vel[u, 1]),
                })

        snap_ue_infos = generate_snapshot(
            scene, cfg, snap_id, seed, data_dir,
            ue_infos=ue_infos,
        )

        # Save per-snapshot metadata as JSON (merged later)
        records = []
        for ue_id, info in enumerate(snap_ue_infos):
            record = {
                "snapshot_id": snap_id,
                "ue_id": ue_id,
                "bs_id": info["bs_id"],
                "idx_in_bs": info.get("idx_in_bs", ue_id),
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
            records.append(record)

        snap_dir = data_dir / f"snapshot_{snap_id:04d}"
        with open(snap_dir / "metadata.json", "w") as f:
            json.dump(records, f)

        print(f"  ✓ Snapshot {snap_id}/{snapshot_end-1} done ({len(snap_ue_infos)} UEs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, required=True)
    parser.add_argument("--snapshot_start", type=int, default=0)
    parser.add_argument("--snapshot_end", type=int, default=1)
    parser.add_argument("--num_ue", type=int, default=100)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--merge_only", action="store_true")
    parser.add_argument("--trajectories", type=str, default=None,
                        help="Path to trajectories.npz for temporal mode")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.preset, args.data_dir)

    if args.merge_only:
        merge_metadata(data_dir, args.preset)
    else:
        run_worker(args.preset, args.snapshot_start, args.snapshot_end,
                   args.num_ue, data_dir, trajectories_path=args.trajectories)
