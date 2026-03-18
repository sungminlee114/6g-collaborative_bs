"""S2: Threshold Sweep + Pareto Front (H2 — CE-agnostic effectiveness).

Sweeps τ_low (with τ_high = 2*τ_low) across a range of values and measures:
- Skip Rate, NMSE, Rate, Computation Ratio for each (config, CE, τ) triple.

Core figure: X=Computation Ratio, Y=Rate Preservation Ratio, 3 CE × 3 configs.

Usage:
    python -m src.experiments.S2_pareto.run --preset munich_elaa_l_1k_15g --gpu 0
    python -m src.experiments.S2_pareto.run --all
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import SceneConfig, get_results_dir
from src.tracker import Tracker
from src.ce_skip.temporal_dataset import TemporalChannelData
from src.ce_skip.ce_algorithms import LSEstimator, GenieLMMSE, DLCEEstimator
from src.ce_skip.scheduling import adaptive_ce_scheduling, compute_scheduling_cost
from src.ce_skip.metrics import (
    nmse_per_slot, nmse_db_per_slot, rate_preservation_ratio,
)
from src.dataset_operation.dataset import ChannelEstimationDataset
from src.dataset_operation.utils import complex_to_real

PRIMARY_PRESETS = [
    "munich_elaa_s_1k_15g",   # 15 GHz, 16×16, 1024 SC
    "munich_mimo_15g",        # 15 GHz, 8×8, 1024 SC (FF baseline)
]

# CE profiling times (ms) — will be overridden by S1 results if available
DEFAULT_CE_TIMES = {
    "ls": 0.01,
    "lmmse": 0.5,
    "dl_ce": 2.0,
}


def load_ce_times(preset: str) -> dict:
    """Load CE profiling times from S1 results, fallback to defaults."""
    s1_path = get_results_dir("S1_ce_profiling") / "ce_profiling.json"
    if s1_path.exists():
        with open(s1_path) as f:
            s1 = json.load(f)
        if preset in s1:
            return {m: s1[preset][m]["time_ms"] for m in s1[preset]}
    return DEFAULT_CE_TIMES.copy()


def prepare_ce_methods(
    data_dir: str,
    cfg: SceneConfig,
    device: str,
    load_dl_checkpoint: bool = True,
) -> dict:
    """Prepare all 3 CE methods."""
    from torch.utils.data import DataLoader

    methods = {}

    # LS
    methods["ls"] = LSEstimator()

    # Genie-LMMSE
    lmmse = GenieLMMSE(device=device)
    if Path(data_dir).exists():
        ds = ChannelEstimationDataset(data_dir, bs_ids=cfg.train_bs_ids, snr_range_db=(20, 20))
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        h_samples = []
        for batch in loader:
            h_samples.append(batch["target"])
            if len(h_samples) * 256 >= 2000:
                break
        lmmse.fit(torch.cat(h_samples)[:2000].to(device))
    methods["lmmse"] = lmmse

    # DL-CE
    dl_ce = DLCEEstimator(n_blocks=8, channels=64).to(device)
    # Try to load trained checkpoint from S1
    ckpt_dir = Path("assets/checkpoints/ce_skip")
    ckpt_path = ckpt_dir / f"dl_ce_{cfg.deployment}_{cfg.num_subcarriers}_{int(cfg.frequency/1e9)}g.pt"
    if ckpt_path.exists() and load_dl_checkpoint:
        dl_ce.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"  Loaded DL-CE checkpoint: {ckpt_path}")
    else:
        print(f"  DL-CE: using untrained model (no checkpoint at {ckpt_path})")
    dl_ce.eval()
    methods["dl_ce"] = dl_ce

    return methods


def run_threshold_sweep(
    preset: str,
    gpu: int = 0,
    n_tau: int = 50,
    snr_db: float = 20.0,
    max_snapshots: int = 200,
):
    """Run threshold sweep for one preset."""
    device = f"cuda:{gpu}"
    torch.cuda.set_device(device)
    cfg = SceneConfig.from_preset(preset)

    temporal_dir = Path(cfg.data_dir).parent / f"{Path(cfg.data_dir).name}_temporal"
    if not temporal_dir.exists():
        print(f"  ⚠ Temporal data not found: {temporal_dir}")
        return None

    data = TemporalChannelData(temporal_dir, max_snapshots=max_snapshots)
    ce_methods = prepare_ce_methods(cfg.data_dir, cfg, device)
    ce_times = load_ce_times(preset)

    snr_linear = 10 ** (snr_db / 10)
    tau_values = np.linspace(0.01, 1.0, n_tau)

    results = {}
    # Test on one BS from test set
    test_bs = cfg.test_bs_ids[0] if hasattr(cfg, "test_bs_ids") and cfg.test_bs_ids else 1

    h_all, ue_ids = data.get_all_series(test_bs, snap_range=(0, min(data.num_snapshots, max_snapshots)))
    if len(ue_ids) == 0:
        print(f"  ⚠ No UEs for BS {test_bs}")
        return None

    # Convert to torch real: (N_UE, T, 2, n_ant, n_sc)
    N_ue, T, n_ant, n_sc = h_all.shape
    h_real = np.stack([h_all.real, h_all.imag], axis=2).astype(np.float32)
    h_tensor = torch.from_numpy(h_real).to(device)

    for ce_name, ce_method in ce_methods.items():
        print(f"  CE: {ce_name}")

        # Wrap LMMSE to include snr_db
        if ce_name == "lmmse":
            _ce = lambda x, _m=ce_method, _s=snr_db: _m(x, snr_db=_s)
        else:
            _ce = ce_method

        sweep = []
        for tau_low in tau_values:
            tau_high = 2 * tau_low  # T1 band is [tau_low, 2*tau_low]

            all_nmse = []
            all_stats_agg = {"n_skip": 0, "n_delta": 0, "n_full": 0, "total": 0}

            for ue_idx in range(min(N_ue, 20)):  # Profile up to 20 UEs
                h_ue = h_tensor[ue_idx]  # (T, 2, n_ant, n_sc)

                h_hat, stats = adaptive_ce_scheduling(
                    h_ue, _ce,
                    tau_low=tau_low, tau_high=tau_high,
                    snr_db=snr_db,
                )

                per_slot_nmse = nmse_per_slot(h_hat, h_ue).cpu().numpy()
                all_nmse.extend(per_slot_nmse.tolist())

                all_stats_agg["n_skip"] += stats.n_skip
                all_stats_agg["n_delta"] += stats.n_delta
                all_stats_agg["n_full"] += stats.n_full
                all_stats_agg["total"] += stats.total

            avg_nmse = np.mean(all_nmse)
            avg_nmse_db = 10 * np.log10(max(avg_nmse, 1e-30))

            total = all_stats_agg["total"]
            skip_rate = all_stats_agg["n_skip"] / max(total, 1)

            cost_adaptive = (
                all_stats_agg["n_full"] * ce_times.get(ce_name, 1.0)
                + all_stats_agg["n_delta"] * 0.005
                + all_stats_agg["n_skip"] * 0.005
            )
            cost_full = total * ce_times.get(ce_name, 1.0)
            cr = cost_adaptive / max(cost_full, 1e-12)

            sweep.append({
                "tau_low": float(tau_low),
                "tau_high": float(tau_high),
                "skip_rate": skip_rate,
                "avg_nmse_db": avg_nmse_db,
                "computation_ratio": cr,
                "n_skip": all_stats_agg["n_skip"],
                "n_delta": all_stats_agg["n_delta"],
                "n_full": all_stats_agg["n_full"],
            })

        results[ce_name] = sweep
        # Print summary at a few key points
        for idx in [0, n_tau // 4, n_tau // 2, 3 * n_tau // 4, n_tau - 1]:
            s = sweep[idx]
            print(f"    τ={s['tau_low']:.2f}: SR={s['skip_rate']:.1%}, "
                  f"CR={s['computation_ratio']:.3f}, NMSE={s['avg_nmse_db']:.1f}dB")

    data.clear_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="S2: Threshold Sweep + Pareto Front")
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-tau", type=int, default=50)
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--max-snapshots", type=int, default=200)
    args = parser.parse_args()

    presets = PRIMARY_PRESETS if args.all else [args.preset or PRIMARY_PRESETS[0]]
    results_dir = get_results_dir("S2_pareto")

    with Tracker(
        "S2/pareto",
        config={"presets": presets, "n_tau": args.n_tau, "snr_db": args.snr},
        capture_output=True,
        purpose="Threshold sweep and Pareto front analysis",
        variables={
            "independent": ["tau_low", "ce_method", "preset"],
            "dependent": ["skip_rate", "computation_ratio", "nmse_db"],
            "controlled": [f"snr={args.snr}dB", f"max_snap={args.max_snapshots}"],
        },
        hypothesis="CE-agnostic: skip scheduling works across all 3 CE methods. "
                   "More expensive CE → larger absolute skip benefit.",
        eval_criteria="Pareto front: computation_ratio vs NMSE for each (CE, preset)",
    ) as run:
        all_results = {}
        for preset in presets:
            print(f"\n{'═'*60}")
            print(f"  Preset: {preset}")
            print(f"{'═'*60}")
            r = run_threshold_sweep(
                preset, gpu=args.gpu, n_tau=args.n_tau,
                snr_db=args.snr, max_snapshots=args.max_snapshots,
            )
            if r is not None:
                all_results[preset] = r
                run.log(**{f"{preset}_done": True})

        with open(results_dir / "pareto_sweep.json", "w") as f:
            json.dump(all_results, f, indent=2)
        run.set_result(n_presets=len(all_results))


if __name__ == "__main__":
    main()
