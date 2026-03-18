"""S6: Multi-BS Generalization.

Tests whether τ* learned on train BSs {0,5} generalizes to test BSs {1,3,6,7}.
Measures generalization gap in skip rate and NMSE.

Usage:
    python -m src.experiments.S6_generalization.run --preset munich_elaa_l_1k_15g --gpu 0
    python -m src.experiments.S6_generalization.run --all
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
from src.ce_skip.metrics import nmse_per_slot, rate_preservation_ratio

PRIMARY_PRESETS = [
    "munich_elaa_s_1k_15g",   # 15 GHz, 16×16, 1024 SC
    "munich_mimo_15g",        # 15 GHz, 8×8, 1024 SC (FF baseline)
]


def find_optimal_tau(
    h_tensor: torch.Tensor,
    tau_values: np.ndarray,
    snr_db: float = 20.0,
    target_nmse_db: float = -10.0,
) -> float:
    """Find optimal τ* on given data (largest τ meeting target NMSE)."""
    ce = LSEstimator()
    N_ue = h_tensor.shape[0]
    optimal = tau_values[0]

    for tau in tau_values:
        nmse_list = []
        for ue_idx in range(min(N_ue, 15)):
            h_hat, stats = adaptive_ce_scheduling(
                h_tensor[ue_idx], ce,
                tau_low=tau, tau_high=2 * tau,
                snr_db=snr_db,
            )
            nm = nmse_per_slot(h_hat, h_tensor[ue_idx]).cpu().numpy()
            nmse_list.extend(nm.tolist())

        avg_nmse_db = 10 * np.log10(max(np.mean(nmse_list), 1e-30))
        if avg_nmse_db <= target_nmse_db:
            optimal = tau
        else:
            break

    return float(optimal)


def evaluate_tau(
    h_tensor: torch.Tensor,
    tau: float,
    snr_db: float = 20.0,
) -> dict:
    """Evaluate a fixed τ on given data."""
    ce = LSEstimator()
    N_ue = h_tensor.shape[0]
    snr_linear = 10 ** (snr_db / 10)

    nmse_list = []
    skip_rates = []
    rpr_list = []

    for ue_idx in range(min(N_ue, 20)):
        h_hat, stats = adaptive_ce_scheduling(
            h_tensor[ue_idx], ce,
            tau_low=tau, tau_high=2 * tau,
            snr_db=snr_db,
        )
        nm = nmse_per_slot(h_hat, h_tensor[ue_idx]).cpu().numpy()
        nmse_list.extend(nm.tolist())
        skip_rates.append(stats.skip_rate)
        rpr = rate_preservation_ratio(h_hat, h_tensor[ue_idx], snr_linear)
        rpr_list.append(rpr)

    return {
        "nmse_db": 10 * np.log10(max(np.mean(nmse_list), 1e-30)),
        "skip_rate": float(np.mean(skip_rates)),
        "rpr": float(np.mean(rpr_list)),
    }


def run_generalization(preset: str, gpu: int = 0, max_snapshots: int = 200):
    """Run multi-BS generalization test."""
    device = f"cuda:{gpu}"
    torch.cuda.set_device(device)
    cfg = SceneConfig.from_preset(preset)

    temporal_dir = Path(cfg.data_dir).parent / f"{Path(cfg.data_dir).name}_temporal"
    if not temporal_dir.exists():
        print(f"  ⚠ Temporal data not found: {temporal_dir}")
        return None

    data = TemporalChannelData(temporal_dir, max_snapshots=max_snapshots)
    tau_values = np.linspace(0.01, 0.8, 30)
    T_max = min(data.num_snapshots, max_snapshots)

    def load_bs_data(bs_id):
        h_all, ue_ids = data.get_all_series(bs_id, snap_range=(0, T_max))
        if len(ue_ids) == 0:
            return None
        N_ue, T, n_ant, n_sc = h_all.shape
        h_real = np.stack([h_all.real, h_all.imag], axis=2).astype(np.float32)
        return torch.from_numpy(h_real).to(device)

    # Step 1: Find τ* on train BSs
    print("  Phase 1: Finding τ* on train BSs...")
    train_taus = []
    for bs_id in cfg.train_bs_ids:
        h_train = load_bs_data(bs_id)
        if h_train is None:
            continue
        tau_star = find_optimal_tau(h_train, tau_values)
        train_taus.append(tau_star)
        print(f"    BS {bs_id}: τ* = {tau_star:.3f}")

    if not train_taus:
        print("  ⚠ No train BS data available")
        return None

    avg_tau_star = float(np.mean(train_taus))
    print(f"  Average τ* from train BSs: {avg_tau_star:.3f}")

    # Step 2: Evaluate on all BSs with this τ*
    print("\n  Phase 2: Evaluating τ* on all BSs...")
    results = {
        "train_taus": {str(bs): tau for bs, tau in zip(cfg.train_bs_ids, train_taus)},
        "avg_tau_star": avg_tau_star,
        "per_bs": {},
    }

    all_bs = cfg.train_bs_ids + cfg.val_bs_ids + cfg.test_bs_ids
    for bs_id in all_bs:
        h_bs = load_bs_data(bs_id)
        if h_bs is None:
            continue

        # Evaluate with transferred τ*
        eval_transferred = evaluate_tau(h_bs, avg_tau_star)

        # Also find BS-specific optimal τ* for comparison
        tau_local = find_optimal_tau(h_bs, tau_values)
        eval_local = evaluate_tau(h_bs, tau_local)

        split = ("train" if bs_id in cfg.train_bs_ids
                 else "val" if bs_id in cfg.val_bs_ids
                 else "test")

        results["per_bs"][str(bs_id)] = {
            "split": split,
            "tau_transferred": avg_tau_star,
            "tau_local": tau_local,
            "transferred": eval_transferred,
            "local_optimal": eval_local,
            "nmse_gap_db": eval_transferred["nmse_db"] - eval_local["nmse_db"],
        }

        print(f"    BS {bs_id} ({split}): "
              f"τ*_local={tau_local:.3f}, "
              f"transferred NMSE={eval_transferred['nmse_db']:.1f}dB, "
              f"local NMSE={eval_local['nmse_db']:.1f}dB, "
              f"gap={eval_transferred['nmse_db'] - eval_local['nmse_db']:.1f}dB")

    data.clear_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="S6: Multi-BS Generalization")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-snapshots", type=int, default=200)
    args = parser.parse_args()

    presets = PRIMARY_PRESETS if args.all else [args.preset or PRIMARY_PRESETS[0]]
    results_dir = get_results_dir("S6_generalization")

    with Tracker(
        "S6/generalization",
        config={"presets": presets},
        capture_output=True,
        purpose="Multi-BS generalization: train τ* on BS{0,5} → test on BS{1,3,6,7}",
        variables={
            "independent": ["bs_id", "split"],
            "dependent": ["nmse_gap_db", "skip_rate", "rpr"],
            "controlled": ["snr=20dB", "target_nmse=-10dB"],
        },
        hypothesis="τ* generalizes across BSs with < 2dB NMSE gap",
        eval_criteria="Compare transferred vs local-optimal τ* performance per BS",
    ) as run:
        all_results = {}
        for preset in presets:
            print(f"\n{'═'*60}")
            print(f"  Preset: {preset}")
            print(f"{'═'*60}")
            r = run_generalization(preset, gpu=args.gpu, max_snapshots=args.max_snapshots)
            if r is not None:
                all_results[preset] = r

        with open(results_dir / "multi_bs_generalization.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # Summary
        print(f"\n{'═'*60}")
        print("  Generalization Gap Summary")
        print(f"{'═'*60}")
        for preset, r in all_results.items():
            short = preset.replace("munich_", "")
            test_gaps = [
                v["nmse_gap_db"]
                for v in r["per_bs"].values()
                if v["split"] == "test"
            ]
            if test_gaps:
                avg_gap = np.mean(test_gaps)
                max_gap = np.max(test_gaps)
                print(f"  {short}: avg test gap = {avg_gap:.1f}dB, "
                      f"max = {max_gap:.1f}dB")

        run.set_result(n_presets=len(all_results))


if __name__ == "__main__":
    main()
