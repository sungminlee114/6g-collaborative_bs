"""S0-controlled: Speed vs δ analysis with distance binning.

Groups UEs by distance bins, then compares δ across speeds within each bin.
This controls for distance confounding in the speed-δ relationship.

Not a perfect controlled experiment (would need same position, different speed),
but the best we can do with existing data.

Usage:
    python -m src.experiments.S0_persistence.controlled_speed --preset munich_elaa_m_1k_15g
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

from src.ce_skip import ExperimentConfig, PRIMARY_PRESETS
from src.ce_skip.temporal_dataset import TemporalChannelData

SPEED_LABELS = {0.0: "static", 1.0: "ped", 8.3: "veh"}
DIST_BINS = [(10, 30), (30, 50), (50, 80), (80, 150)]


def run_controlled(preset: str, max_snapshots: int = 200):
    cfg = ExperimentConfig.from_preset(preset)
    if not cfg.temporal_dir.exists():
        print(f"  ⚠ No temporal data: {cfg.temporal_dir}")
        return None

    data = TemporalChannelData(cfg.temporal_dir, trajectory_dir=cfg.trajectory_dir, max_snapshots=max_snapshots)
    test_bs = cfg.test_bs_ids[0]

    # Collect per-UE: (distance, speed, median_delta)
    ue_stats = []
    h_all, ue_ids = data.get_all_series(test_bs, snap_range=(0, min(data.num_snapshots, max_snapshots)))
    if len(ue_ids) == 0:
        return None

    N_ue, T, n_ant, n_sc = h_all.shape
    for i, uid in enumerate(ue_ids):
        dist = data.get_ue_distance(uid, test_bs, snap_idx=0)
        speed = data.get_ue_speed(uid)
        # Compute all deltas for this UE
        deltas = []
        for t in range(1, T):
            ref_norm = np.linalg.norm(h_all[i, t - 1])
            if ref_norm > 1e-12:
                deltas.append(float(np.linalg.norm(h_all[i, t] - h_all[i, t - 1]) / ref_norm))
        if deltas:
            ue_stats.append({
                "uid": int(uid), "dist": float(dist), "speed": float(speed),
                "median_delta": float(np.median(deltas)),
                "mean_delta": float(np.mean(deltas)),
                "p90_delta": float(np.percentile(deltas, 90)),
            })

    # Distance-binned speed comparison
    results = {"preset": preset, "ue_stats": ue_stats, "binned": {}}
    print(f"\n  {'Dist bin':>12} | {'Speed':>6} | {'N':>3} | {'Median δ':>10} | {'P90 δ':>10}")
    print(f"  {'─'*55}")

    for d_lo, d_hi in DIST_BINS:
        bin_key = f"{d_lo}-{d_hi}m"
        bin_ues = [u for u in ue_stats if d_lo <= u["dist"] < d_hi]
        by_speed = defaultdict(list)
        for u in bin_ues:
            label = SPEED_LABELS.get(u["speed"], f"v{u['speed']:.1f}")
            by_speed[label].append(u)

        results["binned"][bin_key] = {}
        for label in ["static", "ped", "veh"]:
            ues = by_speed.get(label, [])
            if not ues:
                continue
            med = np.median([u["median_delta"] for u in ues])
            p90 = np.median([u["p90_delta"] for u in ues])
            results["binned"][bin_key][label] = {
                "n": len(ues), "median_delta": float(med), "p90_delta": float(p90),
            }
            print(f"  {bin_key:>12} | {label:>6} | {len(ues):>3} | {med:>10.4f} | {p90:>10.4f}")

    data.clear_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-snapshots", type=int, default=200)
    args = parser.parse_args()

    presets = PRIMARY_PRESETS if args.all else [args.preset or PRIMARY_PRESETS[0]]
    out_dir = Path("assets/results/S0_persistence")
    out_dir.mkdir(parents=True, exist_ok=True)

    for preset in presets:
        print(f"{'═'*60}")
        print(f"  {preset}")
        print(f"{'═'*60}")
        r = run_controlled(preset, args.max_snapshots)
        if r:
            with open(out_dir / f"controlled_speed_{preset}.json", "w") as f:
                json.dump(r, f, indent=2)


if __name__ == "__main__":
    main()
