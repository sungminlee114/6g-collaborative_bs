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

import numpy as np
import torch

from src.config import get_results_dir
from src.tracker import Tracker
from src.ce_skip import ExperimentConfig
from src.ce_skip.temporal_dataset import TemporalChannelData

# ─── Configs ────────────────────────────────────────────────────────
from src.ce_skip import PRIMARY_PRESETS  # noqa: E402 (defined in ce_skip/__init__)
# Presets imported from ce_skip — change there to update all experiments

SPEED_LABELS = {0.0: "static", 1.0: "pedestrian", 8.3: "low_vehicle"}


def compute_delta_profile(data: TemporalChannelData, bs_id: int) -> dict:
    """Compute δ_oracle(t) for all UEs served by a BS.

    Computes ground truth channel variation (no noise) and
    the NMSE that results from reusing the previous channel estimate.

    Returns dict with:
        - per_ue: list of {uid, speed, dist, median_delta, nmse_reuse_db}
        - overall: aggregate statistics
        - static_sanity: δ for static UEs (should be ~0 if simulator is deterministic)
    """
    from src.ce_skip.helpers import iter_ue_tensors

    per_ue = []

    for h_ue_gpu, uid, dist, speed in iter_ue_tensors(data, bs_id, "cpu", data.num_snapshots, max_ue=50, desc=f"S0 BS{bs_id}"):
        # h_ue_gpu: (T, 2, n_ant, n_sc) float — ground truth (no noise added)
        T = h_ue_gpu.shape[0]

        deltas = []
        nmse_reuse = []  # NMSE when reusing h(t-1) as estimate for h(t)
        for t in range(1, T):
            h_t = h_ue_gpu[t]
            h_prev = h_ue_gpu[t - 1]
            ref_norm = h_prev.norm()
            if ref_norm > 1e-12:
                delta = float((h_t - h_prev).norm() / ref_norm)
                # NMSE of reuse: ‖h(t) - h(t-1)‖² / ‖h(t)‖²
                h_t_norm = h_t.norm()
                if h_t_norm > 1e-12:
                    reuse_nmse = float((h_t - h_prev).pow(2).sum() / h_t.pow(2).sum())
                    nmse_reuse.append(reuse_nmse)
            else:
                delta = float("inf")
            deltas.append(delta)

        d_arr = np.array([d for d in deltas if np.isfinite(d)])
        nm_arr = np.array(nmse_reuse)

        if len(d_arr) == 0:
            continue

        per_ue.append({
            "uid": int(uid),
            "speed": float(speed),
            "dist": float(dist),
            "median_delta": float(np.median(d_arr)),
            "mean_delta": float(np.mean(d_arr)),
            "p90_delta": float(np.percentile(d_arr, 90)),
            "nmse_reuse_db": float(10 * np.log10(max(np.mean(nm_arr), 1e-30))) if len(nm_arr) > 0 else float("nan"),
        })

    # Aggregate
    all_deltas = [u["median_delta"] for u in per_ue]
    static_deltas = [u["median_delta"] for u in per_ue if u["speed"] == 0.0]

    overall = {
        "n_ues": len(per_ue),
        "median_delta_all": float(np.median(all_deltas)) if all_deltas else float("nan"),
        "mean_delta_all": float(np.mean(all_deltas)) if all_deltas else float("nan"),
        "static_median_delta": float(np.median(static_deltas)) if static_deltas else float("nan"),
        "static_n": len(static_deltas),
        "frac_below_0.1": float(np.mean(np.array(all_deltas) < 0.1)) if all_deltas else 0.0,
        "frac_below_0.3": float(np.mean(np.array(all_deltas) < 0.3)) if all_deltas else 0.0,
    }

    return {"per_ue": per_ue, "overall": overall}


def run_persistence_profile(preset: str, gpu: int = 0, max_snapshots: int = 200):
    """Run persistence profiling for a single preset.

    Computes:
    - δ_oracle: ground truth channel variation (no noise)
    - NMSE of reusing previous estimate (skip quality impact)
    - Static UE sanity check (should be ~0 if simulator is deterministic)
    """
    cfg = ExperimentConfig.from_preset(preset)

    if not cfg.temporal_dir.exists():
        print(f"  ⚠ Temporal data not found: {cfg.temporal_dir}")
        return None

    print(f"  Loading temporal data from {cfg.temporal_dir}")
    data = TemporalChannelData(cfg.temporal_dir, trajectory_dir=cfg.trajectory_dir, max_snapshots=max_snapshots)
    print(f"  Snapshots: {data.num_snapshots}, UEs: {data.num_ue}, dt: {data.dt_s}s")

    all_per_ue = []
    for bs_id in cfg.test_bs_ids[:2]:
        print(f"  BS {bs_id}...")
        profile = compute_delta_profile(data, bs_id)
        all_per_ue.extend(profile["per_ue"])
        ov = profile["overall"]
        print(f"    {ov['n_ues']} UEs, median δ={ov['median_delta_all']:.4f}, "
              f"static δ={ov['static_median_delta']:.4f} (n={ov['static_n']})")

    # Aggregate
    all_deltas = [u["median_delta"] for u in all_per_ue]
    all_nmse = [u["nmse_reuse_db"] for u in all_per_ue if not np.isnan(u["nmse_reuse_db"])]
    static_deltas = [u["median_delta"] for u in all_per_ue if u["speed"] == 0.0]

    summary = {
        "n_ues": len(all_per_ue),
        "median_delta": float(np.median(all_deltas)) if all_deltas else float("nan"),
        "mean_delta": float(np.mean(all_deltas)) if all_deltas else float("nan"),
        "frac_below_0.1": float(np.mean(np.array(all_deltas) < 0.1)) if all_deltas else 0.0,
        "frac_below_0.3": float(np.mean(np.array(all_deltas) < 0.3)) if all_deltas else 0.0,
        "frac_below_0.5": float(np.mean(np.array(all_deltas) < 0.5)) if all_deltas else 0.0,
        "median_nmse_reuse_db": float(np.median(all_nmse)) if all_nmse else float("nan"),
        "static_median_delta": float(np.median(static_deltas)) if static_deltas else float("nan"),
        "static_n": len(static_deltas),
    }

    print(f"  Overall: median δ={summary['median_delta']:.4f}, "
          f"<0.1: {summary['frac_below_0.1']:.0%}, "
          f"NMSE(reuse)={summary['median_nmse_reuse_db']:.1f}dB, "
          f"static δ={summary['static_median_delta']:.4f}")

    data.clear_cache()
    return {"preset": preset, "summary": summary, "per_ue": all_per_ue}


def check_go_nogo(results: list) -> bool:
    """Check kill criterion: overall median δ > 0.5 → no-go."""
    go = True
    for r in results:
        if r is None:
            continue
        s = r["summary"]
        med = s["median_delta"]
        static = s["static_median_delta"]
        if med > 0.5:
            print(f"  ✗ KILL: {r['preset']} median δ = {med:.3f} > 0.5")
            go = False
        else:
            print(f"  ✓ GO: {r['preset']} median δ = {med:.3f} ≤ 0.5")
        if static > 0.01:
            print(f"    ⚠ Static δ = {static:.4f} (simulator noise — should be ~0)")
    return go


def main():
    parser = argparse.ArgumentParser(description="S0: Temporal Persistence Profiling")
    parser.add_argument("--preset", type=str, default=None, help="Single preset to profile")
    parser.add_argument("--all", action="store_true", help="Profile all primary presets")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-snapshots", type=int, default=200)
    args = parser.parse_args()

    presets = PRIMARY_PRESETS if args.all else [args.preset or PRIMARY_PRESETS[0]]

    results_dir = get_results_dir("S0_persistence")

    with Tracker(
        "S0/persistence",
        config={"presets": presets},
        capture_output=True,
        purpose="Temporal persistence: δ_oracle + NMSE degradation + static sanity check",
        variables={
            "independent": ["preset"],
            "dependent": ["delta_oracle", "nmse_reuse_db", "static_delta"],
            "controlled": ["dt=10ms", "test_bs_ids"],
        },
        hypothesis="Overall median δ_oracle ≤ 0.5 and static δ ≈ 0",
        eval_criteria="Kill if median δ > 0.5 or static δ > 0.01 (simulator noise)",
    ) as run:
        all_results = []
        for preset in presets:
            print(f"\n{'─'*60}")
            print(f"  Preset: {preset}")
            print(f"{'─'*60}")
            r = run_persistence_profile(preset, gpu=args.gpu, max_snapshots=args.max_snapshots)
            all_results.append(r)

            if r is not None:
                out_path = results_dir / f"persistence_{preset}.json"
                with open(out_path, "w") as f:
                    json.dump({"preset": r["preset"], "summary": r["summary"], "per_ue": r["per_ue"]}, f, indent=2)
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
