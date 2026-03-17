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
) -> float:
    """RPR = R_adaptive / R_fullCE (scalar, averaged over time)."""
    r_adaptive = achievable_rate(h_hat, h_true, snr_linear).mean()
    r_oracle = oracle_rate(h_true, snr_linear).mean()
    return float(r_adaptive / r_oracle.clamp(min=1e-12))


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
