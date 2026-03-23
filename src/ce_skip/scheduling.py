"""Adaptive CE inference scheduling algorithm.

Two operating modes:
  Blend: reuse/interpolate with continuous alpha derived from channel variation
  Full:  run full CE inference when variation exceeds threshold

The scheduler monitors normalized LS difference delta(t) between consecutive
slots and computes a per-slot blending weight alpha(t) that varies continuously
from 0 (pure reuse) to 1 (full LS weight). When delta exceeds tau_full, the
full CE method is launched.

Backward compatibility: alpha_mode="step" reproduces the legacy 3-tier behavior.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import numpy as np


def delta_to_alpha(
    deltas: np.ndarray, delta_min: float, tau_full: float
) -> np.ndarray:
    """Map normalized delta to continuous blending weight via linear ramp.

    alpha = clamp((delta - delta_min) / (tau_full - delta_min), 0, 1)

    Args:
        deltas: array of normalized LS differences
        delta_min: noise floor below which alpha=0 (typically sqrt(2/SNR))
        tau_full: threshold above which Full CE is triggered (alpha not used)
    """
    denom = max(tau_full - delta_min, 1e-12)
    return np.clip((deltas - delta_min) / denom, 0.0, 1.0)


@dataclass
class SchedulingStats:
    """Statistics from a scheduling run."""
    n_skip: int = 0       # alpha=0 slots (pure reuse)
    n_delta: int = 0      # 0 < alpha <= 1 slots (blend)
    n_full: int = 0       # Full CE slots
    n_safety: int = 0     # forced full CE (safety counter)
    total: int = 0
    deltas: list = field(default_factory=list)   # delta(t) values
    tiers: list = field(default_factory=list)    # legacy tier per slot (0/1/2)
    alphas: list = field(default_factory=list)   # per-slot alpha values

    @property
    def skip_rate(self) -> float:
        """Fraction of slots with alpha=0 (pure reuse)."""
        return self.n_skip / max(self.total, 1)

    @property
    def delta_rate(self) -> float:
        """Fraction of slots with 0 < alpha <= 1 (blend)."""
        return self.n_delta / max(self.total, 1)

    @property
    def blend_rate(self) -> float:
        """Fraction of slots with 0 < alpha <= 1 (partial blend only)."""
        return self.n_delta / max(self.total, 1)

    @property
    def non_full_rate(self) -> float:
        """Fraction of all non-Full slots (skip + blend)."""
        return (self.n_skip + self.n_delta) / max(self.total, 1)

    @property
    def full_rate(self) -> float:
        return self.n_full / max(self.total, 1)


def compute_normalized_delta(h_ls_t: torch.Tensor, h_ls_prev: torch.Tensor) -> float:
    """Compute normalized LS difference: ||h(t) - h(t-1)|| / ||h(t-1)||.

    Args:
        h_ls_t: (n_ant, n_sc) or (2, n_ant, n_sc) -- current LS estimate
        h_ls_prev: same shape -- previous LS estimate
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
    # New interface (continuous alpha)
    tau_full: Optional[float] = None,
    delta_min: Optional[float] = None,
    alpha_mode: str = "ramp",
    # Legacy interface (discrete 3-tier)
    tau_low: float = 0.1,
    tau_high: float = 0.5,
    alpha: float = 0.5,
    # Common
    n_max: int = 50,
    snr_db: float = 20.0,
    delta_mode: str = "ema",
) -> tuple:
    """Run adaptive CE scheduling on a temporal channel series.

    Supports two modes via alpha_mode parameter.
    "ramp" (default): continuous alpha from delta via linear ramp.
    "step" (legacy): discrete 3-tier (Skip/Delta/Full) with fixed alpha.

    Args:
        h_true_series: (T, 2, n_ant, n_sc) float -- ground truth channels
        ce_method: callable that takes (batch, 2, n_ant, n_sc) -> same shape
        tau_full: threshold for Full CE trigger (ramp mode). Falls back to tau_high.
        delta_min: noise floor for alpha=0 (ramp mode). Auto-computed if None.
        alpha_mode: "ramp" (continuous) or "step" (legacy 3-tier)
        tau_low: legacy T0 threshold (step mode only)
        tau_high: legacy T2 threshold (step mode only)
        alpha: legacy fixed EMA weight (step mode only)
        n_max: max consecutive non-Full slots before forced Full CE
        snr_db: SNR for LS estimate noise
        delta_mode: "ema" or "ls_delta" for blend update formula

    Returns:
        h_hat_series: (T, 2, n_ant, n_sc) -- estimated channels
        stats: SchedulingStats
    """
    T = h_true_series.shape[0]
    device = h_true_series.device
    stats = SchedulingStats(total=T)

    # Resolve parameters
    if tau_full is None:
        tau_full = tau_high
    if delta_min is None:
        snr_linear = 10 ** (snr_db / 10)
        delta_min = float(np.sqrt(2.0 / snr_linear))

    h_hat_series = torch.zeros_like(h_true_series)

    # Generate noisy LS estimates
    signal_power = h_true_series.flatten(1).pow(2).mean(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
    noise_std = torch.sqrt(signal_power / (10 ** (snr_db / 10)))
    noise = torch.randn_like(h_true_series) * noise_std
    h_ls_series = h_true_series + noise

    skip_counter = 0

    for t in range(T):
        h_ls_t = h_ls_series[t]

        if t == 0:
            h_hat = _run_full_ce(ce_method, h_ls_t.unsqueeze(0)).squeeze(0)
            h_hat_series[t] = h_hat
            stats.n_full += 1
            stats.tiers.append(2)
            stats.alphas.append(0.0)
            stats.deltas.append(0.0)
            skip_counter = 0
            continue

        delta = compute_normalized_delta(h_ls_t, h_ls_series[t - 1])
        stats.deltas.append(delta)

        force_full = (skip_counter >= n_max)

        if force_full or delta > tau_full:
            # Full CE
            h_hat = _run_full_ce(ce_method, h_ls_t.unsqueeze(0)).squeeze(0)
            h_hat_series[t] = h_hat
            stats.n_full += 1
            if force_full:
                stats.n_safety += 1
            stats.tiers.append(2)
            stats.alphas.append(0.0)
            skip_counter = 0

        else:
            # Blend mode -- compute alpha
            if alpha_mode == "ramp":
                alpha_t = float(np.clip(
                    (delta - delta_min) / max(tau_full - delta_min, 1e-12),
                    0.0, 1.0
                ))
            elif alpha_mode == "step":
                # Legacy 3-tier behavior
                if delta <= tau_low:
                    alpha_t = 0.0
                else:
                    alpha_t = alpha
            else:
                raise ValueError(f"Unknown alpha_mode: {alpha_mode}")

            stats.alphas.append(alpha_t)

            if alpha_t < 1e-8:
                # Pure reuse (equivalent to old Skip)
                h_hat_series[t] = h_hat_series[t - 1]
                stats.n_skip += 1
                stats.tiers.append(0)
            else:
                # Blend update
                if delta_mode == "ema":
                    h_hat_series[t] = alpha_t * h_ls_t + (1 - alpha_t) * h_hat_series[t - 1]
                elif delta_mode == "ls_delta":
                    ls_diff = h_ls_t - h_ls_series[t - 1]
                    h_hat_series[t] = h_hat_series[t - 1] + alpha_t * ls_diff
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
    t_blend_ms: float = 0.005,
    t_monitor_ms: float = 0.005,
) -> dict:
    """Compute scheduling cost metrics.

    Returns dict with cost breakdown and ratios.
    """
    n_blend = stats.n_skip + stats.n_delta  # all non-Full slots
    cost_adaptive = (
        stats.n_full * t_full_ms
        + n_blend * t_monitor_ms  # blend cost = monitor cost regardless of alpha
    )
    cost_full = stats.total * t_full_ms

    return {
        "cost_adaptive_ms": cost_adaptive,
        "cost_full_ce_ms": cost_full,
        "computation_ratio": cost_adaptive / max(cost_full, 1e-12),
        "skip_rate": stats.skip_rate,
        "blend_rate": stats.blend_rate,
        "non_full_rate": stats.non_full_rate,
        "full_rate": stats.full_rate,
    }
