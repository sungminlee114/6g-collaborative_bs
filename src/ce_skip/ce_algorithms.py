"""CE algorithms: LS, Genie-LMMSE, DL-CE (ResNet).

All CE methods follow the same interface:
    Input:  h_ls — (batch, n_ant, n_sc) complex or (batch, 2, n_ant, n_sc) real
    Output: h_hat — same shape as input

For scheduling experiments, these are the "Full CE" methods that may be skipped.
"""
import torch
import torch.nn as nn
import numpy as np


# ─── LS Estimator ───────────────────────────────────────────────────
class LSEstimator:
    """LS estimator — identity (LS estimate is the input itself).

    Included as a lower-bound baseline. Cost is effectively zero since
    the LS estimate is already computed from DMRS.
    """

    def __call__(self, h_ls: torch.Tensor) -> torch.Tensor:
        return h_ls

    def to(self, device):
        return self

    def eval(self):
        return self


# ─── Genie-LMMSE Estimator ─────────────────────────────────────────
class GenieLMMSE:
    """Genie-aided LMMSE using sample covariance from ground truth.

    This is an oracle upper-bound: it uses the true channel covariance
    matrix R_hh computed from training data. In practice, R_hh is unknown.

    LMMSE: h_hat = R_hh @ (R_hh + sigma^2 I)^{-1} @ h_ls

    Operates in real-valued domain: (batch, 2, n_ant, n_sc) float.
    Covariance is computed per-subcarrier across antenna dimension.
    """

    def __init__(self, device: str = "cuda"):
        self.R_hh = None  # (n_sc, n_ant*2, n_ant*2) or simplified
        self.device = device

    def fit(self, h_true_samples: torch.Tensor):
        """Compute sample covariance from ground truth channels.

        For large antenna arrays (D > 256), uses diagonal covariance to avoid
        O(n_sc * D^2) memory. Full covariance for small arrays.

        Args:
            h_true_samples: (N, 2, n_ant, n_sc) float — training channels
        """
        N, _, n_ant, n_sc = h_true_samples.shape
        D = 2 * n_ant
        # (N, n_sc, D)
        h = h_true_samples.permute(0, 3, 1, 2).reshape(N, n_sc, D)
        h = h.to(self.device)

        h_mean = h.mean(dim=0, keepdim=True)  # (1, n_sc, D)
        h_centered = h - h_mean

        if D > 256:
            # Diagonal covariance for large arrays (memory-safe)
            # R_diag[sc, d] = var of d-th element across samples
            self.R_diag = h_centered.pow(2).mean(dim=0)  # (n_sc, D)
            self.R_hh = None
            self._diagonal_mode = True
        else:
            # Full per-subcarrier covariance
            self.R_hh = torch.einsum("nsd,nse->sde", h_centered, h_centered) / (N - 1)
            self.R_diag = None
            self._diagonal_mode = False
        self.h_mean = h_mean.squeeze(0)  # (n_sc, D)

    def __call__(self, h_ls: torch.Tensor, snr_db: float = 20.0) -> torch.Tensor:
        """Apply LMMSE estimation.

        Args:
            h_ls: (batch, 2, n_ant, n_sc) float — noisy LS estimate
            snr_db: SNR in dB (used to compute noise variance)
        """
        assert self.R_hh is not None or self.R_diag is not None, "Must call fit() first"

        B, _, n_ant, n_sc = h_ls.shape
        device = h_ls.device
        D = 2 * n_ant

        # Reshape: (B, n_sc, D)
        y = h_ls.permute(0, 3, 1, 2).reshape(B, n_sc, D).to(device)

        snr_linear = 10 ** (snr_db / 10)

        if self._diagonal_mode:
            # Diagonal LMMSE: w_d = r_d / (r_d + σ²)  element-wise
            r = self.R_diag.to(device)  # (n_sc, D)
            sigma2 = float(r.mean() / snr_linear)
            w = r / (r + sigma2)  # (n_sc, D)
            h_hat = w.unsqueeze(0) * y  # (B, n_sc, D)
        else:
            R = self.R_hh.to(device)  # (n_sc, D, D)
            I = torch.eye(D, device=device).unsqueeze(0)

            avg_signal_power = R.diagonal(dim1=-2, dim2=-1).mean()
            sigma2 = float(avg_signal_power / snr_linear)

            # W = R(R + σ²I)^{-1}
            W = torch.linalg.solve(R + sigma2 * I, R).transpose(-1, -2)
            h_hat = torch.einsum("sde,bse->bsd", W, y)  # (B, n_sc, D)

        # Reshape back: (B, 2, n_ant, n_sc)
        h_hat = h_hat.reshape(B, n_sc, 2, n_ant).permute(0, 2, 3, 1)
        return h_hat

    def to(self, device):
        self.device = device
        if self.R_hh is not None:
            self.R_hh = self.R_hh.to(device)
        if self.R_diag is not None:
            self.R_diag = self.R_diag.to(device)
        return self

    def eval(self):
        return self


# ─── DL-CE: ResNet Channel Estimator ───────────────────────────────
class ResBlock(nn.Module):
    """Residual block for channel estimation."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class DLCEEstimator(nn.Module):
    """DL-based channel estimator (ResNet denoiser).

    Takes noisy LS estimate as input and outputs refined estimate.
    Residual learning: output = input + refinement.

    Input/Output: (batch, 2, n_ant, n_sc) float — real/imag stacked.
    """

    def __init__(self, n_blocks: int = 8, channels: int = 64):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])
        self.output_proj = nn.Conv2d(channels, 2, 3, padding=1)
        # Init output_proj near zero so initial refinement ≈ 0 (identity at start)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, h_ls: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual learning."""
        x = self.input_proj(h_ls)
        x = self.blocks(x)
        refinement = self.output_proj(x)
        return h_ls + refinement


def create_ce_estimator(method: str, device: str = "cuda", **kwargs):
    """Factory function for CE estimators.

    Args:
        method: "ls", "lmmse", or "dl_ce"
        device: target device
        **kwargs: passed to constructor (e.g., n_blocks for dl_ce)
    """
    if method == "ls":
        return LSEstimator()
    elif method == "lmmse":
        return GenieLMMSE(device=device)
    elif method == "dl_ce":
        model = DLCEEstimator(**kwargs).to(device)
        return model
    else:
        raise ValueError(f"Unknown CE method: {method}")
