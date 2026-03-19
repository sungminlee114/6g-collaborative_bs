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

from src.config import get_results_dir
from src.tracker import Tracker
from src.ce_skip import ExperimentConfig
from src.ce_skip.temporal_dataset import TemporalChannelData
from src.ce_skip.ce_algorithms import LSEstimator, GenieLMMSE
from src.ce_skip.scheduling import adaptive_ce_scheduling, compute_scheduling_cost
from src.ce_skip.metrics import (
    nmse_per_slot, nmse_db_per_slot, rate_preservation_ratio,
)
from src.dataset_operation.dataset import ChannelEstimationDataset
from src.dataset_operation.utils import complex_to_real

from src.ce_skip import PRIMARY_PRESETS  # noqa: E402 (defined in ce_skip/__init__)
# Presets imported from ce_skip — change there to update all experiments

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
    cfg: ExperimentConfig,
    device: str,
    load_dl_checkpoint: bool = True,
) -> dict:
    """Prepare all 3 CE methods."""
    from torch.utils.data import DataLoader

    methods = {}

    # LS
    methods["ls"] = LSEstimator()

    # Genie-LMMSE (oracle covariance from training data)
    # Need N > D for well-conditioned covariance (D = 2 * n_ant)
    lmmse = GenieLMMSE(device=device)
    if Path(data_dir).exists():
        n_ant = cfg.num_rx_ant * cfg.num_tx_ant
        D = 2 * n_ant
        # Need at least 2*D samples for reasonable covariance estimate
        min_samples = max(2 * D, 500)
        # Estimate snapshots needed: ~25 UEs/BS × 2 train BSs = ~50 UE/snap
        max_snap = max(min_samples // 30, 5)  # conservative
        # But cap memory: each snapshot × n_ant cache
        max_snap = min(max_snap, 50 if n_ant <= 256 else 10)
        ds = ChannelEstimationDataset(
            data_dir, bs_ids=cfg.train_bs_ids, snapshot_ids=list(range(max_snap)),
            snr_range_db=(20, 20),
        )
        n_fit = min(min_samples, len(ds))
        loader = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=False)
        h_samples = []
        for batch in loader:
            h_samples.append(batch["target"])
            if sum(s.shape[0] for s in h_samples) >= n_fit:
                break
        h_fit = torch.cat(h_samples)[:n_fit].to(device)
        print(f"  LMMSE fit: N={len(h_fit)}, D={D} (N/D={len(h_fit)/D:.1f})")
        lmmse.fit(h_fit)
        del h_fit
    methods["lmmse"] = lmmse

    # DL-CE — uses SiteAwareEstimator (same as E0) with per-sample RMS normalization
    from src.models.estimator import create_model as create_estimator
    from src.training.trainer import load_checkpoint

    ckpt_name = f"ce_skip/dl_ce_{cfg.deployment}_{cfg.num_subcarriers}_{int(cfg.frequency/1e9)}g"
    ckpt_path = Path("assets/checkpoints") / f"{ckpt_name}.pt"
    if ckpt_path.exists() and load_dl_checkpoint:
        dl_ce = create_estimator(site_integration="none").to(device)
        meta = load_checkpoint(dl_ce, ckpt_name, device=device)
        dl_ce.eval()
        # Wrap with per-sample RMS normalization (training used normalized data)
        class NormalizedDLCE:
            def __init__(self, model):
                self.model = model
            def __call__(self, h_ls):
                rms = h_ls.flatten(1).pow(2).mean(1, keepdim=True).sqrt().unsqueeze(-1).unsqueeze(-1).clamp(min=1e-12)
                h_norm = h_ls / rms
                h_hat_norm = self.model(h_norm)
                return h_hat_norm * rms
            def to(self, device): self.model.to(device); return self
            def eval(self): self.model.eval(); return self
        methods["dl_ce"] = NormalizedDLCE(dl_ce)
        print(f"  Loaded DL-CE: {ckpt_path}")
    else:
        print(f"  DL-CE: skipped (no checkpoint at {ckpt_path})")

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
    cfg = ExperimentConfig.from_preset(preset)

    if not cfg.temporal_dir.exists():
        print(f"  ⚠ Temporal data not found: {cfg.temporal_dir}")
        return None

    data = TemporalChannelData(cfg.temporal_dir, trajectory_dir=cfg.trajectory_dir, max_snapshots=max_snapshots, preset=preset)
    ce_methods = prepare_ce_methods(cfg.data_dir, cfg, device)
    ce_times = load_ce_times(preset)

    snr_linear = 10 ** (snr_db / 10)
    tau_values = np.linspace(0.01, 1.0, n_tau)

    from src.ce_skip.helpers import iter_ue_tensors

    results = {}
    test_bs = cfg.test_bs_ids[0]

    for ce_name, ce_method in ce_methods.items():
        print(f"  CE: {ce_name}")

        # Wrap LMMSE to include snr_db
        if ce_name == "lmmse":
            _ce = lambda x, _m=ce_method, _s=snr_db: _m(x, snr_db=_s)
        else:
            _ce = ce_method

        # Collect per-UE results across all taus (UE-by-UE to save GPU memory)
        per_tau = {tau: {"nmse": [], "n_skip": 0, "n_delta": 0, "n_full": 0, "total": 0}
                   for tau in tau_values}

        for h_ue, uid, dist, speed in iter_ue_tensors(data, test_bs, device, max_snapshots, max_ue=20):
            for tau_low in tau_values:

                tau_high = 2 * tau_low
                h_hat, stats = adaptive_ce_scheduling(
                    h_ue, _ce,
                    tau_low=tau_low, tau_high=tau_high,
                    snr_db=snr_db,
                )
                nm = nmse_per_slot(h_hat, h_ue).cpu().numpy()
                pt = per_tau[tau_low]
                pt["nmse"].extend(nm.tolist())
                pt["n_skip"] += stats.n_skip
                pt["n_delta"] += stats.n_delta
                pt["n_full"] += stats.n_full
                pt["total"] += stats.total

        sweep = []
        for tau_low in tau_values:
            pt = per_tau[tau_low]
            avg_nmse = np.mean(pt["nmse"]) if pt["nmse"] else 1.0
            avg_nmse_db = 10 * np.log10(max(avg_nmse, 1e-30))
            total = pt["total"]
            skip_rate = pt["n_skip"] / max(total, 1)

            cost_adaptive = (
                pt["n_full"] * ce_times.get(ce_name, 1.0)
                + pt["n_delta"] * 0.005
                + pt["n_skip"] * 0.005
            )
            cost_full = total * ce_times.get(ce_name, 1.0)
            cr = cost_adaptive / max(cost_full, 1e-12)

            sweep.append({
                "tau_low": float(tau_low),
                "tau_high": float(2 * tau_low),
                "skip_rate": skip_rate,
                "avg_nmse_db": avg_nmse_db,
                "computation_ratio": cr,
                "n_skip": pt["n_skip"],
                "n_delta": pt["n_delta"],
                "n_full": pt["n_full"],
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
