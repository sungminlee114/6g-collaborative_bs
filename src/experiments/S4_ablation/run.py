"""S4: Delta Update Ablation.

Compares 3 Tier-1 (delta) update strategies:
  (a) Pure skip:  ĥ(t) = ĥ(t-1)
  (b) EMA:        ĥ(t) = α·h_LS(t) + (1-α)·ĥ(t-1)
  (c) LS-delta:   ĥ(t) = ĥ(t-1) + β·(h_LS(t) - h_LS(t-1))

At a fixed τ (from S2 knee point or default), measures NMSE across
mobility levels and configs to determine which update mode is best.

Usage:
    python -m src.experiments.S4_ablation.run --preset munich_elaa_l_1k_15g --gpu 0
    python -m src.experiments.S4_ablation.run --all
"""
import argparse
import json

import numpy as np
import torch

from src.config import get_results_dir
from src.tracker import Tracker
from src.ce_skip import ExperimentConfig
from src.ce_skip.temporal_dataset import TemporalChannelData
from src.ce_skip.ce_algorithms import LSEstimator
from src.ce_skip.scheduling import adaptive_ce_scheduling
from src.ce_skip.metrics import nmse_per_slot

from src.ce_skip import PRIMARY_PRESETS  # noqa: E402 (defined in ce_skip/__init__)
# Presets imported from ce_skip — change there to update all experiments

DELTA_MODES = ["skip", "ema", "ls_delta"]
ALPHA_VALUES = [0.3, 0.5, 0.7]  # sweep α for EMA and LS-delta
SPEED_LABELS = {0.0: "static", 1.0: "pedestrian", 8.3: "low_vehicle"}


def run_ablation(preset: str, gpu: int = 0, tau_low: float = 0.2, max_snapshots: int = 200):
    """Run delta update ablation for one preset."""
    device = f"cuda:{gpu}"
    torch.cuda.set_device(device)
    cfg = ExperimentConfig.from_preset(preset)

    if not cfg.temporal_dir.exists():
        print(f"  ⚠ Temporal data not found: {cfg.temporal_dir}")
        return None

    from src.ce_skip.helpers import iter_ue_tensors

    data = TemporalChannelData(cfg.temporal_dir, trajectory_dir=cfg.trajectory_dir, max_snapshots=max_snapshots, preset=preset)
    ce = LSEstimator()
    test_bs = cfg.test_bs_ids[0]

    # Collect results per (mode, alpha, speed) — UE-by-UE for memory safety
    results = {}
    for mode in DELTA_MODES:
        for alpha in (ALPHA_VALUES if mode != "skip" else [0.0]):
            key = f"{mode}_a{alpha:.1f}" if mode != "skip" else "skip"

            per_speed = {}
            for h_ue, uid, dist, speed in iter_ue_tensors(data, test_bs, device, max_snapshots, max_ue=20):
                label = SPEED_LABELS.get(speed, f"v{speed:.1f}")
                if label not in per_speed:
                    per_speed[label] = {"nmse": [], "skip_rates": []}

                h_hat, stats = adaptive_ce_scheduling(
                    h_ue, ce,
                    tau_low=tau_low, tau_high=2 * tau_low,
                    alpha=alpha, snr_db=20.0,
                    delta_mode=mode,
                )
                nm = nmse_per_slot(h_hat, h_ue).cpu().numpy()
                per_speed[label]["nmse"].extend(nm.tolist())
                per_speed[label]["skip_rates"].append(stats.skip_rate)

            # Aggregate
            agg = {}
            for sl, vals in per_speed.items():
                agg[sl] = {
                    "nmse_db": 10 * np.log10(max(np.mean(vals["nmse"]), 1e-30)),
                    "skip_rate": float(np.mean(vals["skip_rates"])),
                }
            results[key] = agg

            for sl, st in agg.items():
                print(f"    {key:<15} {sl:<15} NMSE={st['nmse_db']:.1f}dB SR={st['skip_rate']:.1%}")

    data.clear_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="S4: Delta Update Ablation")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tau", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5],
                        help="τ_low values to sweep")
    parser.add_argument("--max-snapshots", type=int, default=200)
    args = parser.parse_args()

    presets = PRIMARY_PRESETS if args.all else [args.preset or PRIMARY_PRESETS[0]]
    results_dir = get_results_dir("S4_ablation")

    with Tracker(
        "S4/ablation",
        config={"presets": presets, "tau_values": args.tau, "alphas": ALPHA_VALUES},
        capture_output=True,
        purpose="Delta update ablation: skip vs EMA vs LS-delta",
        variables={
            "independent": ["delta_mode", "alpha", "mobility"],
            "dependent": ["nmse_db", "skip_rate"],
            "controlled": [f"tau_low={args.tau}", "snr=20dB"],
        },
        hypothesis="LS-delta with α=0.5 gives best NMSE-skip tradeoff across mobility levels",
        eval_criteria="Compare NMSE across modes; best mode = lowest NMSE at same skip rate",
    ) as run:
        all_results = {}
        for preset in presets:
            print(f"\n{'═'*60}")
            print(f"  Preset: {preset}")
            print(f"{'═'*60}")
            preset_results = {}
            for tau in args.tau:
                print(f"  τ = {tau}")
                r = run_ablation(preset, gpu=args.gpu, tau_low=tau, max_snapshots=args.max_snapshots)
                if r is not None:
                    preset_results[f"tau_{tau}"] = r
            if preset_results:
                all_results[preset] = preset_results

        with open(results_dir / "delta_ablation.json", "w") as f:
            json.dump(all_results, f, indent=2)
        run.set_result(n_presets=len(all_results))


if __name__ == "__main__":
    main()
