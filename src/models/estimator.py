"""3-way decomposition channel estimator: E (encoder) + theta_task (task head) + theta_BS (site embedding).

Based on ReEsNet residual learning approach (Li et al., 2020).
Site embedding integration variants: FiLM, concat, add, none.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class ResBlock2D(nn.Module):
    """Residual block with 2D convolution."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class SEResBlock2D(nn.Module):
    """ResBlock + Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018).

    Standard SE: global avg pool → FC → ReLU → FC → Sigmoid → channel reweighting.
    Same interface as ResBlock2D — drop-in replacement.
    """

    def __init__(self, channels: int, kernel_size: int = 3, reduction: int = 4):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm2d(channels),
        )
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        w = self.se(out).unsqueeze(-1).unsqueeze(-1)
        return self.relu(x + out * w)


class PolarizedSelfAttention(nn.Module):
    """Polarized Self-Attention — parallel layout (Liu et al., arXiv:2107.00782).

    Two orthogonal branches:
      - Channel-only SA: collapses spatial dims via softmax, produces (C,1,1) weight
      - Spatial-only SA: collapses channel dims via global pool, produces (1,H,W) weight
    Output = channel_branch + spatial_branch (element-wise sum).
    """

    def __init__(self, channels: int):
        super().__init__()
        mid = max(channels // 2, 1)
        # Channel-only branch
        self.ch_q = nn.Conv2d(channels, mid, 1)
        self.ch_k = nn.Conv2d(channels, 1, 1)
        self.ch_v = nn.Conv2d(mid, channels, 1)
        self.ch_ln = nn.LayerNorm(channels)
        # Spatial-only branch
        self.sp_q = nn.Conv2d(channels, mid, 1)
        self.sp_k = nn.Conv2d(channels, mid, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        mid = self.ch_q.out_channels

        # === Channel-only Self-Attention ===
        q = self.ch_q(x).reshape(B, mid, H * W)           # B, C/2, HW
        k = F.softmax(self.ch_k(x).reshape(B, H * W, 1), dim=1)  # B, HW, 1
        a = torch.bmm(q, k)                                # B, C/2, 1
        a = self.ch_v(a.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # B, C
        ch_attn = torch.sigmoid(self.ch_ln(a)).unsqueeze(-1).unsqueeze(-1)  # B, C, 1, 1
        ch_out = x * ch_attn

        # === Spatial-only Self-Attention ===
        q = self.sp_q(x).reshape(B, mid, H * W)            # B, C/2, HW
        k = F.adaptive_avg_pool2d(self.sp_k(x), 1)         # B, C/2, 1, 1
        k = F.softmax(k.reshape(B, 1, mid), dim=2)         # B, 1, C/2
        a = torch.bmm(k, q).reshape(B, 1, H, W)            # B, 1, HW → B, 1, H, W
        sp_out = x * torch.sigmoid(a)

        return ch_out + sp_out


class PACENetBlock2D(nn.Module):
    """PACE-Net block (Yang et al., Entropy 2025) for channel estimation.

    Faithful to the paper architecture (Figure 3):
      - Asymmetric convolutions: 1×K then K×1 (factorized, paper's hidden layers)
      - LeakyReLU (α=0.08): preserves negative channel values (paper Section 4.2)
      - Dropout instead of BatchNorm: paper shows ~3 dB gain (Figure 14)
      - PSA (Polarized Self-Attention): channel + spatial dual attention (Figures 5-6)
    Same interface as ResBlock2D — drop-in replacement.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, (1, kernel_size), padding=(0, kernel_size // 2)),
            nn.LeakyReLU(0.08, inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, (kernel_size, 1), padding=(kernel_size // 2, 0)),
            nn.LeakyReLU(0.08, inplace=True),
            nn.Dropout2d(dropout),
        )
        self.psa = PolarizedSelfAttention(channels)
        self.act = nn.LeakyReLU(0.08, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.psa(self.block(x))
        return self.act(x + out)


class DWSResBlock2D(nn.Module):
    """Depthwise-separable ResBlock (MobileNet-style, on-device friendly).

    Replaces standard conv with depthwise conv + pointwise conv.
    ~C/kernel² times fewer params per conv layer. Same interface as ResBlock2D.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            # Depthwise + pointwise (1st layer)
            nn.Conv2d(channels, channels, kernel_size, padding=pad, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # Depthwise + pointwise (2nd layer)
            nn.Conv2d(channels, channels, kernel_size, padding=pad, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


BLOCK_REGISTRY = {
    "resnet": ResBlock2D,
    "se_resnet": SEResBlock2D,
    "pacenet": PACENetBlock2D,
    "dws_resnet": DWSResBlock2D,
}


class SharedEncoder(nn.Module):
    """E: Shared encoder — extracts features while preserving spatial dims.

    Input:  (B, 2, n_ant_pairs, n_sc)
    Output: (B, C, n_ant_pairs, n_sc)
    """

    def __init__(self, in_channels: int = 2, hidden_channels: int = 64,
                 num_blocks: int = 3, kernel_size: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_blocks):
            layers.append(ResBlock2D(hidden_channels, kernel_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TaskHead(nn.Module):
    """theta_task: Shared task head — maps features back to channel estimate.

    Input:  (B, C, n_ant_pairs, n_sc)
    Output: (B, 2, n_ant_pairs, n_sc)
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 2,
                 num_blocks: int = 2, kernel_size: int = 3):
        super().__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(ResBlock2D(in_channels, kernel_size))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SiteEmbedding(nn.Module):
    """theta_BS: Site-specific learnable embedding.

    Zero-initialized (like LoRA). Stays local, not aggregated in FL.
    """

    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros(embed_dim))

    def reset(self):
        """Reset to zero for new site adaptation."""
        nn.init.zeros_(self.embedding)


class FiLMSiteInjection(nn.Module):
    """FiLM: Feature-wise Linear Modulation from site embedding.

    theta_BS → (scale, shift) applied channel-wise to feature maps.
    """

    def __init__(self, embed_dim: int, feature_channels: int):
        super().__init__()
        self.scale_proj = nn.Linear(embed_dim, feature_channels)
        self.shift_proj = nn.Linear(embed_dim, feature_channels)
        # Init so that FiLM(zeros) = identity transform (scale=1, shift=0)
        # When embedding=0: scale = scale_proj(0) = bias = 1, shift = shift_proj(0) = bias = 0
        nn.init.zeros_(self.scale_proj.weight)
        nn.init.ones_(self.scale_proj.bias)
        nn.init.zeros_(self.shift_proj.weight)
        nn.init.zeros_(self.shift_proj.bias)

    def forward(self, features: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W)
            embedding: (D,) site embedding vector
        Returns:
            (B, C, H, W) modulated features
        """
        scale = self.scale_proj(embedding)  # (C,)
        shift = self.shift_proj(embedding)  # (C,)
        return features * scale[None, :, None, None] + shift[None, :, None, None]


class ConcatSiteInjection(nn.Module):
    """Concatenate site embedding as additional feature channels."""

    def __init__(self, embed_dim: int, feature_channels: int):
        super().__init__()
        # Project embedding to spatial feature map channels
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.merge = nn.Conv2d(feature_channels + embed_dim, feature_channels, 1)

    def forward(self, features: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape
        site_feat = self.proj(embedding)  # (D,)
        site_map = site_feat[None, :, None, None].expand(B, -1, H, W)  # (B, D, H, W)
        return self.merge(torch.cat([features, site_map], dim=1))


class AddSiteInjection(nn.Module):
    """Add projected site embedding to feature channels."""

    def __init__(self, embed_dim: int, feature_channels: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, feature_channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, features: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        site_bias = self.proj(embedding)  # (C,)
        return features + site_bias[None, :, None, None]


SITE_INJECTION_CLASSES = {
    "film": FiLMSiteInjection,
    "concat": ConcatSiteInjection,
    "add": AddSiteInjection,
}


class SiteAwareEstimator(nn.Module):
    """3-way decomposition channel estimator.

    Architecture:
        Input (H_LS) → Encoder(E) → SiteInjection(theta_BS) → TaskHead(theta_task) → residual → H_est

    The residual learning approach: H_est = H_LS + model_output
    where model_output learns the noise/error correction.

    Parameter groups for FL:
        - shared: encoder + task_head + site_injection (aggregated across BSs)
        - local:  site_embedding (stays at each BS, not aggregated)
    """

    def __init__(
        self,
        n_ant_pairs: int = 8,
        n_subcarriers: int = 1024,
        encoder_channels: int = 32,
        encoder_blocks: int = 2,
        task_head_blocks: int = 1,
        site_embed_dim: int = 64,
        site_integration: Literal["film", "concat", "add", "none"] = "film",
        kernel_size: int = 3,
    ):
        super().__init__()
        self.site_integration_type = site_integration

        # E: Shared Encoder
        self.encoder = SharedEncoder(
            in_channels=2, hidden_channels=encoder_channels,
            num_blocks=encoder_blocks, kernel_size=kernel_size,
        )

        # theta_BS: Site Embedding (zero-initialized, local)
        self.site_embedding = SiteEmbedding(site_embed_dim)

        # Site injection module
        if site_integration != "none":
            InjectionClass = SITE_INJECTION_CLASSES[site_integration]
            self.site_injection = InjectionClass(site_embed_dim, encoder_channels)
        else:
            self.site_injection = None

        # theta_task: Task Head
        self.task_head = TaskHead(
            in_channels=encoder_channels, out_channels=2,
            num_blocks=task_head_blocks, kernel_size=kernel_size,
        )

    def forward(self, h_ls: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_ls: (B, 2, n_ant_pairs, n_subcarriers) — noisy LS estimate
        Returns:
            h_est: (B, 2, n_ant_pairs, n_subcarriers) — estimated channel
        """
        # Encode
        feat = self.encoder(h_ls)  # (B, C, H, W)

        # Inject site embedding
        if self.site_injection is not None:
            feat = self.site_injection(feat, self.site_embedding.embedding)

        # Task head outputs residual correction
        residual = self.task_head(feat)  # (B, 2, H, W)

        # Residual learning
        h_est = h_ls + residual
        return h_est

    def shared_parameters(self):
        """Parameters to aggregate in FL (encoder + task_head + site_injection)."""
        params = list(self.encoder.parameters()) + list(self.task_head.parameters())
        if self.site_injection is not None:
            params += list(self.site_injection.parameters())
        return params

    def local_parameters(self):
        """Parameters that stay local (site_embedding only)."""
        return list(self.site_embedding.parameters())

    def shared_state_dict(self):
        """State dict of shared parameters only."""
        full = self.state_dict()
        return {k: v for k, v in full.items() if not k.startswith("site_embedding.")}

    def local_state_dict(self):
        """State dict of local parameters only."""
        full = self.state_dict()
        return {k: v for k, v in full.items() if k.startswith("site_embedding.")}

    def load_shared_state_dict(self, state_dict: dict):
        """Load only shared parameters (keeps local site embedding unchanged)."""
        current = self.state_dict()
        current.update(state_dict)
        self.load_state_dict(current)

    def freeze_encoder(self):
        """Freeze encoder for few-shot adaptation."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def freeze_task_head(self):
        """Freeze task head for transfer experiments."""
        for p in self.task_head.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True


def create_model(
    site_integration: str = "film",
    site_embed_dim: int = 64,
    encoder_channels: int = 32,
    encoder_blocks: int = 2,
    task_head_blocks: int = 1,
    n_ant_pairs: int = 8,
    n_subcarriers: int = 1024,
) -> SiteAwareEstimator:
    """Factory function to create model with given config."""
    return SiteAwareEstimator(
        n_ant_pairs=n_ant_pairs,
        n_subcarriers=n_subcarriers,
        encoder_channels=encoder_channels,
        encoder_blocks=encoder_blocks,
        task_head_blocks=task_head_blocks,
        site_embed_dim=site_embed_dim,
        site_integration=site_integration,
    )
