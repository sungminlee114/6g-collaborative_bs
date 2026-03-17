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
from pathlib import Path

import numpy as np
import torch

from src.config import SceneConfig, get_results_dir
from src.tracker import Tracker
from src.ce_skip.temporal_dataset import TemporalChannelData
from src.ce_skip.ce_algorithms import LSEstimator
from src.ce_skip.scheduling import adaptive_ce_scheduling
from src.ce_skip.metrics import nmse_per_slot

PRIMARY_PRESETS = [
    "munich_elaa_l_1k_15g",
    "munich_elaa_l_4k_28g",
    "munich_5g_mimo_3g5",
]

DELTA_MODES = ["skip", "ema", "ls_delta"]
ALPHA_VALUES = [0.3, 0.5, 0.7]  # sweep α for EMA and LS-delta
SPEED_LABELS = {0.0: "static", 1.0: "pedestrian", 8.3: "low_vehicle"}


def run_ablation(preset: str, gpu: int = 0, tau_low: float = 0.2, max_snapshots: int = 200):
    """Run delta update ablation for one preset."""
    device = f"cuda:{gpu}"
    torch.cuda.set_device(device)
    cfg = SceneConfig.from_preset(preset)

    temporal_dir = Path(cfg.data_dir).parent / f"{Path(cfg.data_dir).name}_temporal"
    if not temporal_dir.exists():
        print(f"  ⚠ Temporal data not found: {temporal_dir}")
        return None

    data = TemporalChannelData(temporal_dir, max_snapshots=max_snapshots)
    ce = LSEstimator()

    test_bs = cfg.test_bs_ids[0] if cfg.test_bs_ids else 1
    h_all, ue_ids = data.get_all_series(test_bs, snap_range=(0, min(data.num_snapshots, max_snapshots)))
    if len(ue_ids) == 0:
        return None

    N_ue, T, n_ant, n_sc = h_all.shape
    h_real = np.stack([h_all.real, h_all.imag], axis=2).astype(np.float32)
    h_tensor = torch.from_numpy(h_real).to(device)

    # Group UEs by speed
    speed_groups = {}
    for i, uid in enumerate(ue_ids):
        speed = data.get_ue_speed(uid)
        label = SPEED_LABELS.get(speed, f"v{speed:.1f}")
        if label not in speed_groups:
            speed_groups[label] = []
        speed_groups[label].append(i)

    results = {}
    for mode in DELTA_MODES:
        for alpha in (ALPHA_VALUES if mode != "skip" else [0.0]):
            key = f"{mode}_a{alpha:.1f}" if mode != "skip" else "skip"

            per_speed = {}
            for speed_label, ue_indices in speed_groups.items():
                nmse_list = []
                skip_rates = []

                for ue_idx in ue_indices[:20]:
                    h_hat, stats = adaptive_ce_scheduling(
                        h_tensor[ue_idx], ce,
                        tau_low=tau_low, tau_high=2 * tau_low,
                        alpha=alpha, snr_db=20.0,
                        delta_mode=mode,
                    )
                    nm = nmse_per_slot(h_hat, h_tensor[ue_idx]).cpu().numpy()
                    nmse_list.extend(nm.tolist())
                    skip_rates.append(stats.skip_rate)

                avg_nmse_db = 10 * np.log10(max(np.mean(nmse_list), 1e-30))
                avg_skip_rate = np.mean(skip_rates)
                per_speed[speed_label] = {
                    "nmse_db": avg_nmse_db,
                    "skip_rate": avg_skip_rate,
                }

            results[key] = per_speed

            # Print
            for sl, stats in per_speed.items():
                print(f"    {key:<15} {sl:<15} NMSE={stats['nmse_db']:.1f}dB SR={stats['skip_rate']:.1%}")

    data.clear_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="S4: Delta Update Ablation")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.2, help="Fixed τ_low for ablation")
    parser.add_argument("--max-snapshots", type=int, default=200)
    args = parser.parse_args()

    presets = PRIMARY_PRESETS if args.all else [args.preset or PRIMARY_PRESETS[0]]
    results_dir = get_results_dir("S4_ablation")

    with Tracker(
        "S4/ablation",
        config={"presets": presets, "tau_low": args.tau, "alphas": ALPHA_VALUES},
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
            r = run_ablation(preset, gpu=args.gpu, tau_low=args.tau, max_snapshots=args.max_snapshots)
            if r is not None:
                all_results[preset] = r

        with open(results_dir / "delta_ablation.json", "w") as f:
            json.dump(all_results, f, indent=2)
        run.set_result(n_presets=len(all_results))


if __name__ == "__main__":
    main()
