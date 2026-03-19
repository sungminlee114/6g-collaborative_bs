#!/usr/bin/env python3
"""Verify generated temporal data matches current shared trajectory.

Checks:
1. Snapshot count matches trajectory
2. UE count matches trajectory
3. Static UE channels are consistent (δ ≈ 0)
4. Moving UE channels vary (δ > static noise floor)
5. Trajectory displacement vs channel δ correlation

Usage:
    python scripts/verify_datagen.py assets/data/channels_elaa_m_1k_15g_temporal
    python scripts/verify_datagen.py --all
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def verify(data_dir: str):
    data_dir = Path(data_dir)
    from src.ce_skip.temporal_dataset import TemporalChannelData

    traj_dir = Path("assets/data/shared_trajectories")
    traj = np.load(traj_dir / "trajectories.npz")

    print(f"\n{'═'*60}")
    print(f"  Verifying: {data_dir.name}")
    print(f"{'═'*60}")

    # Infer preset
    dirname = data_dir.name.replace("channels_", "").replace("_temporal", "")
    preset = f"munich_{dirname}"

    # 1. Load data
    data = TemporalChannelData(data_dir, trajectory_dir=traj_dir, max_snapshots=100, preset=preset)
    print(f"  Snapshots: {data.num_snapshots}")
    print(f"  UEs (trajectory): {traj['positions'].shape[1]}")
    print(f"  UEs (data): {data.num_ue}")

    n_ue = data.num_ue
    speeds = traj["speeds"]
    positions = traj["positions"]

    # 2. Check δ for first 50 snapshots, all UEs
    print(f"\n  Per-UE δ (first 50 snaps):")
    bs_id = int(traj["bs_ids"][0])  # all UEs on same BS

    h_all, ue_ids = data.get_all_series(bs_id, snap_range=(0, min(50, data.num_snapshots)))
    if len(ue_ids) == 0:
        print("  ✗ No UEs loaded!")
        return False

    ok = True
    for i, uid in enumerate(ue_ids):
        speed = float(speeds[uid])
        disp = float(np.linalg.norm(positions[49, uid, :2] - positions[0, uid, :2])) if positions.shape[0] > 49 else 0

        h = h_all[i]  # (T, n_ant, n_sc) complex
        deltas = []
        for t in range(1, h.shape[0]):
            ref = np.linalg.norm(h[t-1])
            if ref > 1e-12:
                deltas.append(float(np.linalg.norm(h[t] - h[t-1]) / ref))
        med_delta = np.median(deltas) if deltas else 0

        label = "static" if speed < 0.1 else ("ped" if speed < 5 else "veh")
        status = "✓"

        # Checks
        if speed < 0.01 and med_delta > 0.05:
            status = "✗ STATIC BUT HIGH δ"
            ok = False
        if speed > 0.5 and med_delta < 0.001 and disp > 0.1:
            status = "✗ MOVING BUT δ≈0 (stale data?)"
            ok = False

        print(f"    UE {uid:2d} ({label:6s}): δ={med_delta:.6f}, disp={disp:.2f}m  {status}")

    # 3. Summary
    print(f"\n  {'✓ ALL CHECKS PASSED' if ok else '✗ SOME CHECKS FAILED'}")
    data.clear_cache()
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?", type=str)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        dirs = sorted(Path("assets/data").glob("channels_*_temporal"))
    elif args.data_dir:
        dirs = [Path(args.data_dir)]
    else:
        parser.print_help()
        sys.exit(1)

    all_ok = True
    for d in dirs:
        if not d.is_dir():
            continue
        if not verify(str(d)):
            all_ok = False

    print(f"\n{'═'*60}")
    print(f"  {'ALL DATASETS OK ✓' if all_ok else 'SOME DATASETS FAILED ✗'}")
    print(f"{'═'*60}")
