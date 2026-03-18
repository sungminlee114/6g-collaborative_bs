"""S5: Beamforming Rate Impact (H4).

Measures achievable rate loss when using stale channel estimates (from skip)
for MRT beamforming. Target: P(rate_loss > 5%) < 10%.

CDF of rate loss across time and UEs.

Usage:
    python -m src.experiments.S5_beamforming.run --preset munich_elaa_l_1k_15g --gpu 0
    python -m src.experiments.S5_beamforming.run --all
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
from src.ce_skip.metrics import (
    rate_loss_per_slot, achievable_rate, oracle_rate,
    rate_preservation_ratio, skip_miss_rate,
)

PRIMARY_PRESETS = [
    "munich_elaa_s_1k_15g",   # 15 GHz, 16×16, 1024 SC
    "munich_mimo_15g",        # 15 GHz, 8×8, 1024 SC (FF baseline)
]

SNR_LIST = [10.0, 15.0, 20.0, 25.0]
TAU_LIST = [0.1, 0.2, 0.3, 0.5]


def run_beamforming_analysis(
    preset: str,
    gpu: int = 0,
    max_snapshots: int = 200,
):
    """Run beamforming rate impact analysis."""
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

    results = {}

    for snr_db in SNR_LIST:
        snr_linear = 10 ** (snr_db / 10)
        snr_key = f"snr{int(snr_db)}"
        results[snr_key] = {}

        for tau in TAU_LIST:
            all_rate_loss = []
            all_rpr = []
            all_smr = []

            for ue_idx in range(min(N_ue, 20)):
                h_hat, stats = adaptive_ce_scheduling(
                    h_tensor[ue_idx], ce,
                    tau_low=tau, tau_high=2 * tau,
                    snr_db=snr_db,
                )

                rl = rate_loss_per_slot(h_hat, h_tensor[ue_idx], snr_linear)
                all_rate_loss.extend(rl.cpu().numpy().tolist())

                rpr = rate_preservation_ratio(h_hat, h_tensor[ue_idx], snr_linear)
                all_rpr.append(rpr)

                smr = skip_miss_rate(
                    h_hat, h_tensor[ue_idx], stats.tiers,
                    snr_linear, rate_loss_threshold=0.05,
                )
                all_smr.append(smr)

            rl_arr = np.array(all_rate_loss)
            results[snr_key][f"tau{tau}"] = {
                "mean_rate_loss": float(np.mean(rl_arr)),
                "median_rate_loss": float(np.median(rl_arr)),
                "p95_rate_loss": float(np.percentile(rl_arr, 95)),
                "p99_rate_loss": float(np.percentile(rl_arr, 99)),
                "frac_above_5pct": float(np.mean(rl_arr > 0.05)),
                "frac_above_10pct": float(np.mean(rl_arr > 0.10)),
                "avg_rpr": float(np.mean(all_rpr)),
                "avg_smr": float(np.mean(all_smr)),
            }

            r = results[snr_key][f"tau{tau}"]
            print(f"    SNR={int(snr_db)}dB τ={tau:.1f}: "
                  f"RPR={r['avg_rpr']:.3f}, "
                  f"P(rl>5%)={r['frac_above_5pct']:.1%}, "
                  f"SMR={r['avg_smr']:.3f}")

    data.clear_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="S5: Beamforming Rate Impact")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-snapshots", type=int, default=200)
    args = parser.parse_args()

    presets = PRIMARY_PRESETS if args.all else [args.preset or PRIMARY_PRESETS[0]]
    results_dir = get_results_dir("S5_beamforming")

    with Tracker(
        "S5/beamforming",
        config={"presets": presets, "snr_list": SNR_LIST, "tau_list": TAU_LIST},
        capture_output=True,
        purpose="Beamforming rate impact from stale channel estimates",
        variables={
            "independent": ["tau", "snr_db"],
            "dependent": ["rate_loss", "rpr", "smr"],
            "controlled": ["delta_mode=ls_delta"],
        },
        hypothesis="Operating point exists where P(rate_loss > 5%) < 10%",
        eval_criteria="CDF of rate_loss; find (τ, SNR) where target is met",
    ) as run:
        all_results = {}
        for preset in presets:
            print(f"\n{'═'*60}")
            print(f"  Preset: {preset}")
            print(f"{'═'*60}")
            r = run_beamforming_analysis(preset, gpu=args.gpu, max_snapshots=args.max_snapshots)
            if r is not None:
                all_results[preset] = r

        with open(results_dir / "beamforming_impact.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # Find sweet spots
        print(f"\n{'═'*60}")
        print("  Sweet Spots: P(rate_loss > 5%) < 10%")
        print(f"{'═'*60}")
        for preset, snr_results in all_results.items():
            short = preset.replace("munich_", "")
            for snr_key, tau_results in snr_results.items():
                for tau_key, stats in tau_results.items():
                    if stats["frac_above_5pct"] < 0.10:
                        print(f"  ✓ {short} {snr_key} {tau_key}: "
                              f"RPR={stats['avg_rpr']:.3f}")

        run.set_result(n_presets=len(all_results))


if __name__ == "__main__":
    main()
