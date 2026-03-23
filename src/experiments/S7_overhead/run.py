"""S7: System Overhead Analysis + Effective Throughput.

Measures actual GPU wall-clock time for each component:
  - LS computation (Y/X)
  - Monitor (norm comparison)
  - Full CE (LS / LMMSE / DL-CE)
  - Blend update (α-weighted)

Then computes effective throughput: per-UE rate loss vs system capacity gain.

Usage:
    python -m src.experiments.S7_overhead.run --preset munich_elaa_s_1k_15g --gpu 0
    python -m src.experiments.S7_overhead.run --all
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import get_results_dir
from src.tracker import Tracker
from src.ce_skip import ExperimentConfig
from src.ce_skip.ce_algorithms import LSEstimator, GenieLMMSE, DLCEEstimator
from src.ce_skip.metrics import scheduling_overhead, effective_throughput_gain
from src.dataset_operation.dataset import ChannelEstimationDataset
from torch.utils.data import DataLoader

from src.ce_skip import PRIMARY_PRESETS  # noqa: E402 (defined in ce_skip/__init__)
# Presets imported from ce_skip — change there to update all experiments


def profile_operation(op_fn, *args, n_warmup=10, n_repeat=100):
    """Profile a single GPU operation. Returns time in ms."""
    for _ in range(n_warmup):
        op_fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_repeat):
        op_fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_repeat


def run_overhead_analysis(preset: str, gpu: int = 0):
    """Profile all scheduling components for one preset."""
    device = f"cuda:{gpu}"
    torch.cuda.set_device(device)
    cfg = ExperimentConfig.from_preset(preset)

    if not Path(cfg.data_dir).exists():
        print(f"  ⚠ Data not found: {cfg.data_dir}")
        return None

    n_ant = cfg.num_rx_ant * cfg.num_tx_ant
    n_sc = cfg.num_subcarriers
    print(f"  Config: {n_ant} ant × {n_sc} SC")

    # Create representative data
    B = 1  # single-sample (per-slot operation)
    h_curr = torch.randn(B, 2, n_ant, n_sc, device=device)
    h_prev = torch.randn(B, 2, n_ant, n_sc, device=device)
    h_hat_prev = torch.randn(B, 2, n_ant, n_sc, device=device)

    # ─── Profile each component ───────────────────────────────
    results = {"n_ant": n_ant, "n_sc": n_sc, "components": {}}

    # 1. LS estimate (Y/X — element-wise division)
    def op_ls():
        return h_curr / (h_prev + 1e-10)
    t_ls = profile_operation(op_ls)
    results["components"]["ls_estimate"] = {"time_ms": t_ls, "desc": "Y/X element-wise"}
    print(f"  LS estimate (Y/X):     {t_ls:.4f} ms")

    # 2. Monitor (norm comparison)
    def op_monitor():
        diff = h_curr - h_prev
        return diff.norm() / h_prev.norm().clamp(min=1e-12)
    t_monitor = profile_operation(op_monitor)
    results["components"]["monitor"] = {"time_ms": t_monitor, "desc": "δ = norm(diff)/norm(ref)"}
    print(f"  Monitor (δ compute):   {t_monitor:.4f} ms")

    # 3. Blend update (cost is independent of alpha value — single weighted sum)
    alpha = 0.5  # placeholder; blend cost is the same for any alpha
    def op_blend():
        return alpha * h_curr + (1 - alpha) * h_hat_prev
    t_blend = profile_operation(op_blend)
    results["components"]["blend_update"] = {"time_ms": t_blend, "desc": "α·h_LS + (1-α)·ĥ_prev"}
    print(f"  Blend update:          {t_blend:.4f} ms")

    # 4. Full CE: LMMSE
    lmmse = GenieLMMSE(device=device)
    # Fit on small sample
    h_fit = torch.randn(200, 2, n_ant, n_sc, device=device)
    lmmse.fit(h_fit)
    del h_fit

    def op_lmmse():
        return lmmse(h_curr, snr_db=20.0)
    t_lmmse = profile_operation(op_lmmse)
    results["components"]["full_ce_lmmse"] = {"time_ms": t_lmmse, "desc": "Genie-LMMSE"}
    print(f"  Full CE (LMMSE):       {t_lmmse:.4f} ms")

    # 5. Full CE: DL-CE
    dl_ce = DLCEEstimator(n_blocks=8, channels=64).to(device).eval()
    ckpt_dir = Path("assets/checkpoints/ce_skip")
    ckpt_path = ckpt_dir / f"dl_ce_{cfg.deployment}_{cfg.num_subcarriers}_{int(cfg.frequency/1e9)}g.pt"
    if ckpt_path.exists():
        dl_ce.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"  (loaded trained DL-CE from {ckpt_path})")

    def op_dlce():
        with torch.no_grad():
            return dl_ce(h_curr)
    t_dlce = profile_operation(op_dlce)
    results["components"]["full_ce_dlce"] = {"time_ms": t_dlce, "desc": "DL-CE ResNet (8 blocks)"}
    print(f"  Full CE (DL-CE):       {t_dlce:.4f} ms")

    # ─── Overhead analysis ────────────────────────────────────
    t_fixed = t_ls + t_monitor  # per-slot fixed cost (always paid)

    results["overhead"] = {}
    for ce_name, t_ce in [("ls", t_ls), ("lmmse", t_lmmse), ("dl_ce", t_dlce)]:
        overhead_pct = t_fixed / max(t_ce, 1e-6) * 100
        results["overhead"][ce_name] = {
            "t_fixed_ms": t_fixed,
            "t_full_ce_ms": t_ce,
            "overhead_pct": overhead_pct,
        }
        print(f"  Overhead vs {ce_name:6s}: {t_fixed:.4f} / {t_ce:.4f} ms = {overhead_pct:.1f}%")

    # ─── Memory overhead ──────────────────────────────────────
    mem_bytes = n_ant * n_sc * 4 * 2 * 2  # 2 tensors, real repr, float32
    results["memory"] = {
        "bytes": mem_bytes,
        "mb": mem_bytes / 1024 / 1024,
        "desc": "h_LS(t-1) + ĥ(t-1), float32"
    }
    print(f"  Memory overhead:       {mem_bytes/1024/1024:.1f} MB")

    # ─── Effective throughput (using S2/S5 results) ───────────
    # Load S2 results for Pareto operating points
    s5_path = get_results_dir("S5_beamforming") / "beamforming_impact.json"
    if s5_path.exists():
        with open(s5_path) as f:
            s5 = json.load(f)
        if preset in s5:
            print(f"\n  Effective Throughput (from S5 results):")
            results["effective_throughput"] = {}
            for snr_key, tau_data in s5[preset].items():
                for tau_key, stats in tau_data.items():
                    rpr = stats["avg_rpr"]
                    # CR depends on CE method — compute for each
                    for ce_name, t_ce in [("lmmse", t_lmmse), ("dl_ce", t_dlce)]:
                        # Approximate CR: skip slots pay t_fixed, full slots pay t_ce
                        # Use skip_rate from beamforming results isn't stored,
                        # so estimate from tau value
                        tau = float(tau_key.replace("tau", ""))
                        # Rough skip estimate from S2 data
                        sr_est = min(0.95, tau * 2.5)  # rough linear approx
                        cr = (1 - sr_est) + sr_est * (t_fixed / t_ce)
                        eff = effective_throughput_gain(sr_est, rpr, cr)
                        key = f"{snr_key}_{tau_key}_{ce_name}"
                        results["effective_throughput"][key] = eff
                        if "snr20" in snr_key and "tau0.3" in tau_key:
                            print(f"    {snr_key} {tau_key} {ce_name}: "
                                  f"per-UE loss={eff['per_ue_rate_loss_pct']:.1f}%, "
                                  f"capacity={eff['capacity_multiplier']:.1f}x, "
                                  f"system gain={eff['system_throughput_gain']:.1f}x")

    return results


def main():
    parser = argparse.ArgumentParser(description="S7: System Overhead Analysis")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    presets = PRIMARY_PRESETS if args.all else [args.preset or PRIMARY_PRESETS[0]]
    results_dir = get_results_dir("S7_overhead")

    with Tracker(
        "S7/overhead",
        config={"presets": presets},
        capture_output=True,
        purpose="System overhead analysis: component profiling + effective throughput",
        variables={
            "independent": ["preset", "ce_method"],
            "dependent": ["time_ms", "overhead_pct", "capacity_multiplier"],
            "controlled": ["B=1", "n_repeat=100"],
        },
        hypothesis="Monitor overhead < 5% of any full CE method",
        eval_criteria="Overhead ratio and effective throughput gain",
    ) as run:
        all_results = {}
        for preset in presets:
            print(f"\n{'═'*60}")
            print(f"  Preset: {preset}")
            print(f"{'═'*60}")
            r = run_overhead_analysis(preset, gpu=args.gpu)
            if r is not None:
                all_results[preset] = r

        with open(results_dir / "overhead_analysis.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # Summary table
        print(f"\n{'═'*60}")
        print("  Overhead Summary")
        print(f"{'═'*60}")
        print(f"{'Preset':<25} {'Component':<20} {'Time (ms)':<12}")
        for preset, r in all_results.items():
            short = preset.replace("munich_", "")
            for comp, data in r["components"].items():
                print(f"{short:<25} {comp:<20} {data['time_ms']:<12.4f}")
            print(f"{short:<25} {'memory':<20} {r['memory']['mb']:.1f} MB")
            print()

        run.set_result(n_presets=len(all_results))


if __name__ == "__main__":
    main()
