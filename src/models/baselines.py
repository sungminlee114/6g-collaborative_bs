"""Baseline models for comparison.

- IndependentEstimator: No site embedding, trained per-BS independently
- PlainEstimator: No site embedding, single model (FedAvg/centralized)
- FedPerEstimator: 2-way split — shared encoder, local task head (FedPer)
"""
import torch
import torch.nn as nn
from .estimator import SharedEncoder, TaskHead, ResBlock2D


class PlainEstimator(nn.Module):
    """Vanilla ReEsNet without site embedding. Used for:
    - FedAvg baseline (all params shared)
    - Centralized baseline (single model on all data)
    - Independent baseline (trained per-BS separately)
    """

    def __init__(self, encoder_channels: int = 64, encoder_blocks: int = 3,
                 task_head_blocks: int = 2, kernel_size: int = 3):
        super().__init__()
        self.encoder = SharedEncoder(2, encoder_channels, encoder_blocks, kernel_size)
        self.task_head = TaskHead(encoder_channels, 2, task_head_blocks, kernel_size)

    def forward(self, h_ls):
        feat = self.encoder(h_ls)
        residual = self.task_head(feat)
        return h_ls + residual


class FedPerEstimator(nn.Module):
    """FedPer: 2-way split — shared encoder + local task head.

    In FL, only encoder is aggregated. Task head stays local per BS.
    This is the 2-way baseline for comparing against our 3-way decomposition.
    """

    def __init__(self, encoder_channels: int = 64, encoder_blocks: int = 3,
                 task_head_blocks: int = 2, kernel_size: int = 3):
        super().__init__()
        self.encoder = SharedEncoder(2, encoder_channels, encoder_blocks, kernel_size)
        self.task_head = TaskHead(encoder_channels, 2, task_head_blocks, kernel_size)

    def forward(self, h_ls):
        feat = self.encoder(h_ls)
        residual = self.task_head(feat)
        return h_ls + residual

    def shared_parameters(self):
        return list(self.encoder.parameters())

    def local_parameters(self):
        return list(self.task_head.parameters())

    def shared_state_dict(self):
        full = self.state_dict()
        return {k: v for k, v in full.items() if k.startswith("encoder.")}

    def local_state_dict(self):
        full = self.state_dict()
        return {k: v for k, v in full.items() if k.startswith("task_head.")}

    def load_shared_state_dict(self, state_dict: dict):
        current = self.state_dict()
        current.update(state_dict)
        self.load_state_dict(current)


class LSEstimator:
    """LS estimator baseline — just returns the noisy input (identity)."""

    def __call__(self, h_ls):
        return h_ls

    def eval(self):
        return self

    def parameters(self):
        return iter([])


class LMMSEEstimator(nn.Module):
    """Simplified LMMSE estimator.

    Uses learned channel covariance from training data.
    H_lmmse = R_hh @ (R_hh + sigma^2 I)^{-1} @ H_ls
    """

    def __init__(self, n_ant_pairs: int = 8, n_subcarriers: int = 1024):
        super().__init__()
        self.n_feat = n_ant_pairs * n_subcarriers
        # Learnable diagonal approximation of covariance
        self.log_diag = nn.Parameter(torch.zeros(2, n_ant_pairs, n_subcarriers))
        self.log_noise = nn.Parameter(torch.tensor(0.0))

    def forward(self, h_ls):
        # Diagonal LMMSE: element-wise Wiener filter
        r = torch.exp(self.log_diag)  # signal variance per element
        sigma2 = torch.exp(self.log_noise)
        wiener = r / (r + sigma2)  # (2, n_ant, n_sc)
        return h_ls * wiener[None]
