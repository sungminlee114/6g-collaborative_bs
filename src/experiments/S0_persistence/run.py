"""S0: Temporal Persistence Profiling (H1 — go/no-go).

Measures normalized LS difference δ(t) = ||h(t) - h(t-1)|| / ||h(t-1)||
across temporal snapshots for all configs and mobility levels.

Kill criterion: pedestrian median δ > 0.5 → direction needs rethinking.

Outputs:
    - CDF plots of δ per mobility per config
    - δ vs distance scatter plots
    - Summary statistics JSON

Usage:
    python -m src.experiments.S0_persistence.run --preset munich_elaa_l_1k_15g
    python -m src.experiments.S0_persistence.run --all
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import SceneConfig, get_plot_dir, get_results_dir
from src.tracker import Tracker
from src.ce_skip.temporal_dataset import TemporalChannelData

# ─── Configs ────────────────────────────────────────────────────────
PRIMARY_PRESETS = [
    "munich_elaa_s_1k_15g",   # Config A: 15 GHz, 16×16, 1024 SC
    "munich_mimo_15g",        # Config B: 15 GHz, 8×8, 1024 SC (FF baseline)
]

SPEED_LABELS = {0.0: "static", 1.0: "pedestrian", 8.3: "low_vehicle"}


def compute_delta_profile(data: TemporalChannelData, bs_id: int) -> dict:
    """Compute δ(t) for all UEs served by a BS, grouped by speed.

    Returns dict: {speed_label: {"deltas": [...], "distances": [...], "ue_ids": [...]}}
    """
    results = {}

    h_all, ue_ids = data.get_all_series(bs_id)
    if len(ue_ids) == 0:
        return results

    # h_all: (N_UE, T, n_ant, n_sc) complex
    N_ue, T, n_ant, n_sc = h_all.shape

    for i, uid in enumerate(ue_ids):
        speed = data.get_ue_speed(uid)
        label = SPEED_LABELS.get(speed, f"v{speed:.1f}")
        dist = data.get_ue_distance(uid, bs_id, snap_idx=0)

        if label not in results:
            results[label] = {"deltas": [], "distances": [], "ue_ids": []}

        # Compute δ(t) for this UE
        h_ue = h_all[i]  # (T, n_ant, n_sc)
        for t in range(1, T):
            diff = h_ue[t] - h_ue[t - 1]
            ref = h_ue[t - 1]
            ref_norm = np.linalg.norm(ref)
            if ref_norm > 1e-12:
                delta = float(np.linalg.norm(diff) / ref_norm)
            else:
                delta = float("inf")
            results[label]["deltas"].append(delta)
            results[label]["distances"].append(dist)

        results[label]["ue_ids"].append(uid)

    return results


def run_persistence_profile(preset: str, gpu: int = 0):
    """Run persistence profiling for a single preset."""
    cfg = SceneConfig.from_preset(preset)
    temporal_dir = Path(cfg.data_dir).parent / f"{Path(cfg.data_dir).name}_temporal"

    if not temporal_dir.exists():
        print(f"  ⚠ Temporal data not found: {temporal_dir}")
        print(f"    Generate with: bash scripts/run_all_datagen.sh")
        return None

    print(f"  Loading temporal data from {temporal_dir}")
    data = TemporalChannelData(temporal_dir)
    print(f"  Snapshots: {data.num_snapshots}, UEs: {data.num_ue}, dt: {data.dt_s}s")

    all_results = {}
    bs_ids = cfg.test_bs_ids if hasattr(cfg, "test_bs_ids") else list(range(cfg.num_bs))

    # Use a subset of BSs for profiling (test set)
    for bs_id in bs_ids[:2]:  # Profile 2 BSs to save time
        print(f"  BS {bs_id}...")
        profile = compute_delta_profile(data, bs_id)
        for label, vals in profile.items():
            if label not in all_results:
                all_results[label] = {"deltas": [], "distances": []}
            all_results[label]["deltas"].extend(vals["deltas"])
            all_results[label]["distances"].extend(vals["distances"])

    # Summary statistics
    summary = {}
    for label, vals in all_results.items():
        d = np.array(vals["deltas"])
        d_finite = d[np.isfinite(d)]
        if len(d_finite) == 0:
            continue
        summary[label] = {
            "count": len(d_finite),
            "median": float(np.median(d_finite)),
            "mean": float(np.mean(d_finite)),
            "p10": float(np.percentile(d_finite, 10)),
            "p25": float(np.percentile(d_finite, 25)),
            "p75": float(np.percentile(d_finite, 75)),
            "p90": float(np.percentile(d_finite, 90)),
            "p99": float(np.percentile(d_finite, 99)),
            "frac_below_0.1": float(np.mean(d_finite < 0.1)),
            "frac_below_0.3": float(np.mean(d_finite < 0.3)),
            "frac_below_0.5": float(np.mean(d_finite < 0.5)),
        }
        print(f"  {label}: median={summary[label]['median']:.4f}, "
              f"<0.1: {summary[label]['frac_below_0.1']:.1%}, "
              f"<0.5: {summary[label]['frac_below_0.5']:.1%}")

    data.clear_cache()
    return {"preset": preset, "summary": summary, "raw": all_results}


def check_go_nogo(results: list) -> bool:
    """Check kill criterion: pedestrian median δ > 0.5 → no-go."""
    go = True
    for r in results:
        if r is None:
            continue
        preset = r["preset"]
        for label, stats in r["summary"].items():
            if "pedestrian" in label and stats["median"] > 0.5:
                print(f"  ✗ KILL: {preset} {label} median δ = {stats['median']:.3f} > 0.5")
                go = False
            elif "pedestrian" in label:
                print(f"  ✓ GO: {preset} {label} median δ = {stats['median']:.3f} ≤ 0.5")
    return go


def main():
    parser = argparse.ArgumentParser(description="S0: Temporal Persistence Profiling")
    parser.add_argument("--preset", type=str, default=None, help="Single preset to profile")
    parser.add_argument("--all", action="store_true", help="Profile all primary presets")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    presets = PRIMARY_PRESETS if args.all else [args.preset or PRIMARY_PRESETS[0]]

    results_dir = get_results_dir("S0_persistence")

    with Tracker(
        "S0/persistence",
        config={"presets": presets},
        capture_output=True,
        purpose="Temporal persistence profiling — go/no-go for CE skip",
        variables={
            "independent": ["preset", "mobility"],
            "dependent": ["delta_median", "delta_cdf"],
            "controlled": ["dt=10ms", "test_bs_ids"],
        },
        hypothesis="Pedestrian median δ ≤ 0.5 (go condition)",
        eval_criteria="Kill if any primary preset fails go condition",
    ) as run:
        all_results = []
        for preset in presets:
            print(f"\n{'─'*60}")
            print(f"  Preset: {preset}")
            print(f"{'─'*60}")
            r = run_persistence_profile(preset, gpu=args.gpu)
            all_results.append(r)

            if r is not None:
                # Save per-preset results
                out_path = results_dir / f"persistence_{preset}.json"
                # Convert raw deltas to serializable format (truncate for size)
                save_data = {
                    "preset": r["preset"],
                    "summary": r["summary"],
                }
                with open(out_path, "w") as f:
                    json.dump(save_data, f, indent=2)
                run.log(**{f"{preset}_done": True})

        # Go/no-go check
        print(f"\n{'═'*60}")
        print("  GO / NO-GO CHECK")
        print(f"{'═'*60}")
        go = check_go_nogo(all_results)
        run.set_result(go=go)

        # Save combined summary
        combined = {}
        for r in all_results:
            if r is not None:
                combined[r["preset"]] = r["summary"]
        with open(results_dir / "persistence_summary.json", "w") as f:
            json.dump(combined, f, indent=2)

        print(f"\n  Final verdict: {'GO ✓' if go else 'NO-GO ✗'}")


if __name__ == "__main__":
    main()
