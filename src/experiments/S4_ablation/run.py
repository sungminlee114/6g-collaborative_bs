"""S4: Alpha-Scheduling Ablation.

Compares 4 scheduling configurations:
  (a) Ramp + EMA:        alpha_mode="ramp", delta_mode="ema"   (proposed continuous)
  (b) Ramp + LS-delta:   alpha_mode="ramp", delta_mode="ls_delta"
  (c) Step + fixed α=0.7: alpha_mode="step", delta_mode="ema"  (legacy 3-tier)
  (d) Skip-or-full only: alpha_mode="step", tau_low=tau_high   (no delta tier)

At a fixed τ (from S2 knee point or default), measures NMSE across
mobility levels and configs to determine which scheduling mode is best.

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

# Each config is a dict of kwargs passed to adaptive_ce_scheduling
ABLATION_CONFIGS = {
    "ramp_ema": {
        "label": "Ramp + EMA (proposed)",
        "alpha_mode": "ramp",
        "delta_mode": "ema",
    },
    "ramp_ls_delta": {
        "label": "Ramp + LS-delta",
        "alpha_mode": "ramp",
        "delta_mode": "ls_delta",
    },
    "step_fixed07": {
        "label": "Step α=0.7 + EMA (legacy 3-tier)",
        "alpha_mode": "step",
        "alpha": 0.7,
        "delta_mode": "ema",
    },
    "skip_or_full": {
        "label": "Skip-or-full only (no delta tier)",
        "alpha_mode": "step",
        "alpha": 0.7,
        "delta_mode": "ema",
        # tau_low = tau_high is set dynamically in run_ablation
        "_skip_or_full": True,
    },
}

SPEED_LABELS = {0.0: "static", 1.0: "pedestrian", 8.3: "low_vehicle"}


def run_ablation(preset: str, gpu: int = 0, tau_low: float = 0.2, max_snapshots: int = 200):
    """Run alpha-scheduling ablation for one preset."""
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

    # Collect results per (config_name, speed) — UE-by-UE for memory safety
    results = {}
    for cfg_name, cfg_dict in ABLATION_CONFIGS.items():
        # Build kwargs for adaptive_ce_scheduling
        sched_kwargs = {
            "tau_low": tau_low,
            "tau_high": 2 * tau_low,
            "snr_db": 20.0,
        }
        for k in ("alpha_mode", "delta_mode", "alpha"):
            if k in cfg_dict:
                sched_kwargs[k] = cfg_dict[k]

        # Skip-or-full: collapse tiers by setting tau_low = tau_high
        if cfg_dict.get("_skip_or_full"):
            sched_kwargs["tau_low"] = sched_kwargs["tau_high"]

        per_speed = {}
        for h_ue, uid, dist, speed in iter_ue_tensors(data, test_bs, device, max_snapshots, max_ue=20):
            label = SPEED_LABELS.get(speed, f"v{speed:.1f}")
            if label not in per_speed:
                per_speed[label] = {"nmse": [], "skip_rates": []}

            h_hat, stats = adaptive_ce_scheduling(
                h_ue, ce, **sched_kwargs,
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
        results[cfg_name] = agg

        for sl, st in agg.items():
            print(f"    {cfg_name:<20} {sl:<15} NMSE={st['nmse_db']:.1f}dB SR={st['skip_rate']:.1%}")

    data.clear_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="S4: Alpha-Scheduling Ablation")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tau", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5],
                        help="τ_low values to sweep")
    parser.add_argument("--max-snapshots", type=int, default=20000)
    args = parser.parse_args()

    presets = PRIMARY_PRESETS if args.all else [args.preset or PRIMARY_PRESETS[0]]
    results_dir = get_results_dir("S4_ablation")

    config_names = list(ABLATION_CONFIGS.keys())
    with Tracker(
        "S4/ablation",
        config={"presets": presets, "tau_values": args.tau, "configs": config_names},
        capture_output=True,
        purpose="Alpha-scheduling ablation: ramp vs step, EMA vs LS-delta, skip-or-full",
        variables={
            "independent": ["alpha_mode", "delta_mode", "mobility"],
            "dependent": ["nmse_db", "skip_rate"],
            "controlled": [f"tau_low={args.tau}", "snr=20dB"],
        },
        hypothesis="Ramp + EMA (continuous alpha) gives best NMSE-skip tradeoff across mobility levels",
        eval_criteria="Compare NMSE across configs; best = lowest NMSE at same skip rate",
    ) as run:
        all_results = {}
        for preset in presets:
            print(f"\n{'='*60}")
            print(f"  Preset: {preset}")
            print(f"{'='*60}")
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
