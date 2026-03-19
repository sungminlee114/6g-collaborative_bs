"""Metrics for CE skip scheduling evaluation.

Extends standard NMSE with scheduling-specific metrics:
- Skip Rate (SR), Rate Preservation Ratio (RPR), Computation Ratio (CR)
- Efficiency Score (ES), Skip Miss Rate (SMR)
- Achievable rate (for beamforming impact)
"""
import torch
import numpy as np


def nmse_per_slot(h_hat: torch.Tensor, h_true: torch.Tensor) -> torch.Tensor:
    """Per-slot NMSE: ||h_hat - h_true||^2 / ||h_true||^2.

    Args:
        h_hat: (T, 2, n_ant, n_sc) or (T, n_ant, n_sc) complex
        h_true: same shape

    Returns: (T,) NMSE per slot
    """
    diff = (h_hat - h_true).flatten(1)
    ref = h_true.flatten(1)
    return diff.pow(2).sum(1) / ref.pow(2).sum(1).clamp(min=1e-12)


def nmse_db_per_slot(h_hat: torch.Tensor, h_true: torch.Tensor) -> torch.Tensor:
    """Per-slot NMSE in dB."""
    return 10 * torch.log10(nmse_per_slot(h_hat, h_true).clamp(min=1e-30))


def achievable_rate(
    h_hat: torch.Tensor,
    h_true: torch.Tensor,
    snr_linear: float,
) -> torch.Tensor:
    """Per-slot achievable rate with MRT beamforming.

    Rate = log2(1 + SNR * |w^H h_true|^2)
    where w = h_hat / ||h_hat|| (MRT beamformer from estimated channel)

    Args:
        h_hat: (T, 2, n_ant, n_sc) float — estimated channel (real repr)
        h_true: (T, 2, n_ant, n_sc) float — true channel (real repr)
        snr_linear: linear SNR

    Returns: (T,) rate in bits/s/Hz (averaged over subcarriers)
    """
    # Convert to complex: (T, n_ant, n_sc)
    h_hat_c = h_hat[:, 0] + 1j * h_hat[:, 1]
    h_true_c = h_true[:, 0] + 1j * h_true[:, 1]

    # MRT beamformer per subcarrier: w = h_hat / ||h_hat||
    h_hat_norm = torch.abs(h_hat_c).pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-12)
    w = h_hat_c / h_hat_norm  # (T, n_ant, n_sc)

    # Effective channel gain: |w^H h_true|^2 per subcarrier
    gain = torch.abs((w.conj() * h_true_c).sum(dim=1)).pow(2)  # (T, n_sc)

    # Rate per subcarrier, averaged
    rate = torch.log2(1 + snr_linear * gain)  # (T, n_sc)
    return rate.mean(dim=1)  # (T,) avg rate across subcarriers


def oracle_rate(h_true: torch.Tensor, snr_linear: float) -> torch.Tensor:
    """Oracle rate with perfect CSI (MRT with true channel)."""
    return achievable_rate(h_true, h_true, snr_linear)


def rate_preservation_ratio(
    h_hat: torch.Tensor,
    h_true: torch.Tensor,
    snr_linear: float,
    h_fullce: torch.Tensor = None,
) -> dict:
    """Rate preservation ratios vs oracle and vs full-CE.

    Args:
        h_hat: estimated channel (from scheduling)
        h_true: ground truth channel
        snr_linear: linear SNR
        h_fullce: full-CE estimate (if None, uses h_true as proxy)

    Returns dict with:
        rpr_oracle: R_adaptive / R_oracle (perfect CSI upper bound)
        rpr_fullce: R_adaptive / R_fullCE (practical baseline)
    """
    r_adaptive = achievable_rate(h_hat, h_true, snr_linear).mean()
    r_oracle = oracle_rate(h_true, snr_linear).mean()

    result = {"rpr_oracle": float(r_adaptive / r_oracle.clamp(min=1e-12))}

    if h_fullce is not None:
        r_fullce = achievable_rate(h_fullce, h_true, snr_linear).mean()
        result["rpr_fullce"] = float(r_adaptive / r_fullce.clamp(min=1e-12))
    else:
        result["rpr_fullce"] = result["rpr_oracle"]  # fallback

    return result


def rate_loss_per_slot(
    h_hat: torch.Tensor,
    h_true: torch.Tensor,
    snr_linear: float,
) -> torch.Tensor:
    """Per-slot rate loss: 1 - R_hat / R_oracle."""
    r_hat = achievable_rate(h_hat, h_true, snr_linear)
    r_oracle = oracle_rate(h_true, snr_linear)
    return 1.0 - r_hat / r_oracle.clamp(min=1e-12)


def skip_miss_rate(
    h_hat: torch.Tensor,
    h_true: torch.Tensor,
    tiers: list,
    snr_linear: float,
    rate_loss_threshold: float = 0.10,
) -> float:
    """SMR = P(skip AND rate_loss > threshold).

    Fraction of skipped slots where rate loss exceeds threshold.
    """
    rl = rate_loss_per_slot(h_hat, h_true, snr_linear)
    skip_mask = torch.tensor([t == 0 for t in tiers], dtype=torch.bool, device=rl.device)
    if skip_mask.sum() == 0:
        return 0.0
    bad_skips = (skip_mask & (rl > rate_loss_threshold)).sum()
    return float(bad_skips / skip_mask.sum())


# ─── System overhead & effective throughput metrics ─────────────────

def scheduling_overhead(
    n_ant: int,
    n_sc: int,
    t_full_ms: float,
) -> dict:
    """Compute scheduling overhead breakdown.

    Monitor cost: 2 norms + 1 subtraction = O(n_ant * n_sc) flops.
    Negligible compared to full CE but quantified for completeness.

    Args:
        n_ant: number of antenna elements (rx * tx)
        n_sc: number of subcarriers
        t_full_ms: full CE wall-clock time in ms

    Returns dict with overhead breakdown.
    """
    # Monitor: norm(diff) + norm(ref) + subtraction
    # ~3 * n_ant * n_sc FLOPs (add + mul + sqrt negligible)
    monitor_flops = 3 * n_ant * n_sc

    # LS: element-wise Y/X = n_ant * n_sc complex divisions
    ls_flops = n_ant * n_sc * 6  # complex div ≈ 6 real ops

    # A100: ~312 TFLOPS FP32 → ~312e9 FLOPS/ms
    a100_flops_per_ms = 312e9

    t_monitor_ms = monitor_flops / a100_flops_per_ms
    t_ls_ms = ls_flops / a100_flops_per_ms
    t_overhead_ms = t_monitor_ms + t_ls_ms  # total per-slot fixed cost

    return {
        "n_ant": n_ant,
        "n_sc": n_sc,
        "monitor_flops": monitor_flops,
        "ls_flops": ls_flops,
        "t_monitor_ms": t_monitor_ms,
        "t_ls_ms": t_ls_ms,
        "t_overhead_ms": t_overhead_ms,
        "t_full_ce_ms": t_full_ms,
        "overhead_ratio": t_overhead_ms / max(t_full_ms, 1e-12),
        "memory_bytes": n_ant * n_sc * 4 * 2 * 2,  # 2 tensors (prev LS + prev est), real, float32
    }


def effective_throughput_gain(
    skip_rate: float,
    rpr: float,
    computation_ratio: float,
) -> dict:
    """Compute effective throughput metrics.

    Per-UE: rate is RPR * R_full (slight loss per UE).
    System: GPU freed by skipping can serve more UEs.

    Args:
        skip_rate: fraction of skipped slots
        rpr: rate preservation ratio (0-1)
        computation_ratio: CR = cost_adaptive / cost_full

    Returns dict with per-UE and system-level metrics.
    """
    # Per-UE rate factor
    per_ue_rate_factor = rpr  # e.g., 0.976

    # System capacity multiplier: 1/CR UEs can be served with same GPU budget
    capacity_multiplier = 1.0 / max(computation_ratio, 1e-6)

    # System effective throughput = capacity_multiplier * per_ue_rate_factor
    # Normalized to full-CE baseline (1 UE at full rate)
    system_throughput_gain = capacity_multiplier * per_ue_rate_factor

    return {
        "per_ue_rate_factor": per_ue_rate_factor,
        "per_ue_rate_loss_pct": (1 - per_ue_rate_factor) * 100,
        "capacity_multiplier": capacity_multiplier,
        "system_throughput_gain": system_throughput_gain,
        "skip_rate": skip_rate,
        "computation_ratio": computation_ratio,
    }
