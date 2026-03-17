"""3-Tier adaptive CE scheduling algorithm.

Tier 0 (Skip):   reuse previous estimate
Tier 1 (Delta):  lightweight update from LS difference
Tier 2 (Full):   run full CE inference

The scheduler monitors normalized LS difference δ(t) between consecutive
slots and decides which tier to execute.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import numpy as np


@dataclass
class SchedulingStats:
    """Statistics from a scheduling run."""
    n_skip: int = 0       # T0: reused previous
    n_delta: int = 0      # T1: delta update
    n_full: int = 0       # T2: full CE
    n_safety: int = 0     # forced full CE (safety counter)
    total: int = 0
    deltas: list = field(default_factory=list)  # δ(t) values
    tiers: list = field(default_factory=list)   # tier per slot

    @property
    def skip_rate(self) -> float:
        return self.n_skip / max(self.total, 1)

    @property
    def delta_rate(self) -> float:
        return self.n_delta / max(self.total, 1)

    @property
    def full_rate(self) -> float:
        return self.n_full / max(self.total, 1)


def compute_normalized_delta(h_ls_t: torch.Tensor, h_ls_prev: torch.Tensor) -> float:
    """Compute normalized LS difference: ||h(t) - h(t-1)|| / ||h(t-1)||.

    Args:
        h_ls_t: (n_ant, n_sc) or (2, n_ant, n_sc) — current LS estimate
        h_ls_prev: same shape — previous LS estimate
    """
    diff = (h_ls_t - h_ls_prev).flatten()
    ref = h_ls_prev.flatten()
    ref_norm = ref.norm()
    if ref_norm < 1e-12:
        return float("inf")
    return float(diff.norm() / ref_norm)


def adaptive_ce_scheduling(
    h_true_series: torch.Tensor,
    ce_method: Callable,
    tau_low: float = 0.1,
    tau_high: float = 0.5,
    alpha: float = 0.5,
    n_max: int = 50,
    snr_db: float = 20.0,
    delta_mode: str = "ls_delta",
) -> tuple:
    """Run 3-tier adaptive CE scheduling on a temporal channel series.

    Args:
        h_true_series: (T, 2, n_ant, n_sc) float — ground truth channels
        ce_method: callable that takes (batch, 2, n_ant, n_sc) → same shape
        tau_low: threshold for T0 (skip)
        tau_high: threshold for T2 (full CE). Between tau_low and tau_high → T1
        alpha: EMA/delta coefficient for T1 updates
        n_max: max consecutive skip slots before forced full CE
        snr_db: SNR for LS estimate noise
        delta_mode: "skip" (pure reuse), "ema", or "ls_delta"

    Returns:
        h_hat_series: (T, 2, n_ant, n_sc) — estimated channels
        stats: SchedulingStats
    """
    T = h_true_series.shape[0]
    device = h_true_series.device
    stats = SchedulingStats(total=T)

    h_hat_series = torch.zeros_like(h_true_series)

    # Generate noisy LS estimates
    signal_power = h_true_series.flatten(1).pow(2).mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
    noise_std = torch.sqrt(signal_power / (10 ** (snr_db / 10)))
    noise = torch.randn_like(h_true_series) * noise_std
    h_ls_series = h_true_series + noise

    skip_counter = 0  # consecutive skip count

    for t in range(T):
        h_ls_t = h_ls_series[t]  # (2, n_ant, n_sc)

        if t == 0:
            # First slot: always full CE
            h_hat = _run_full_ce(ce_method, h_ls_t.unsqueeze(0)).squeeze(0)
            h_hat_series[t] = h_hat
            stats.n_full += 1
            stats.tiers.append(2)
            stats.deltas.append(0.0)
            skip_counter = 0
            continue

        # Monitor: compute δ(t)
        delta = compute_normalized_delta(h_ls_t, h_ls_series[t - 1])
        stats.deltas.append(delta)

        # Safety check
        force_full = (skip_counter >= n_max)

        if force_full or delta > tau_high:
            # T2: Full CE
            h_hat = _run_full_ce(ce_method, h_ls_t.unsqueeze(0)).squeeze(0)
            h_hat_series[t] = h_hat
            stats.n_full += 1
            if force_full:
                stats.n_safety += 1
            stats.tiers.append(2)
            skip_counter = 0

        elif delta <= tau_low:
            # T0: Skip — reuse previous
            h_hat_series[t] = h_hat_series[t - 1]
            stats.n_skip += 1
            stats.tiers.append(0)
            skip_counter += 1

        else:
            # T1: Delta update
            if delta_mode == "skip":
                h_hat_series[t] = h_hat_series[t - 1]
            elif delta_mode == "ema":
                h_hat_series[t] = alpha * h_ls_t + (1 - alpha) * h_hat_series[t - 1]
            elif delta_mode == "ls_delta":
                ls_diff = h_ls_t - h_ls_series[t - 1]
                h_hat_series[t] = h_hat_series[t - 1] + alpha * ls_diff
            else:
                raise ValueError(f"Unknown delta_mode: {delta_mode}")
            stats.n_delta += 1
            stats.tiers.append(1)
            skip_counter += 1

    return h_hat_series, stats


def _run_full_ce(ce_method: Callable, h_ls_batch: torch.Tensor) -> torch.Tensor:
    """Run full CE method on a batch of LS estimates."""
    with torch.no_grad():
        return ce_method(h_ls_batch)


def compute_scheduling_cost(
    stats: SchedulingStats,
    t_full_ms: float,
    t_delta_ms: float = 0.005,
    t_monitor_ms: float = 0.005,
) -> dict:
    """Compute scheduling cost metrics.

    Returns dict with:
        cost_adaptive: total time (ms)
        cost_full_ce: total time if full CE every slot (ms)
        computation_ratio: adaptive / full
        skip_rate: fraction of skipped slots
    """
    cost_adaptive = (
        stats.n_full * t_full_ms
        + stats.n_delta * t_delta_ms
        + stats.n_skip * t_monitor_ms
    )
    cost_full = stats.total * t_full_ms

    return {
        "cost_adaptive_ms": cost_adaptive,
        "cost_full_ce_ms": cost_full,
        "computation_ratio": cost_adaptive / max(cost_full, 1e-12),
        "skip_rate": stats.skip_rate,
        "delta_rate": stats.delta_rate,
        "full_rate": stats.full_rate,
    }
