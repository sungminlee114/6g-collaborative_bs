"""Site-specific adapter modules (theta_BS) for per-BS personalization.

Two fundamentally different approaches:
- SSF: Scale & Shift Features (per-layer affine transform, mergeable at inference)
- LoRA: Low-Rank Adaptation (low-rank weight modification, mergeable at inference)

Both are zero-initialized so the model starts with identity behavior.
"""
import torch
import torch.nn as nn


class SSFAdapter(nn.Module):
    """Scale & Shift Features — per-channel affine transform.

    Applies: x * scale + shift  (channel-wise)
    At init: scale=1, shift=0 → identity transform.
    Can be merged into preceding Conv2d/BN weights at inference.

    Ref: NeurIPS 2022, "Scaling & Shifting Your Features"
    """

    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels))
        self.shift = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale[None, :, None, None] + self.shift[None, :, None, None]

    def n_params(self) -> int:
        return self.scale.numel() + self.shift.numel()


class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation for Conv2d — additive low-rank weight modification.

    Applies: x + up(down(x))  where down: C→r, up: r→C
    At init: up.weight=0 → output is zero → identity when added residually.
    Can be merged: W_new = W_orig + up.weight @ down.weight (reshaped).

    Ref: Hu et al. 2022, "LoRA: Low-Rank Adaptation"
    """

    def __init__(self, channels: int, kernel_size: int = 3, rank: int = 4):
        super().__init__()
        padding = kernel_size // 2
        self.down = nn.Conv2d(channels, rank, kernel_size, padding=padding, bias=False)
        self.up = nn.Conv2d(rank, channels, 1, bias=False)
        # Kaiming init for down, zero init for up → starts at zero output
        nn.init.kaiming_normal_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def make_adapter(adapter_type: str, channels: int, kernel_size: int = 3,
                 lora_rank: int = 4) -> nn.Module:
    """Factory function for creating adapters."""
    if adapter_type == "ssf":
        return SSFAdapter(channels)
    elif adapter_type == "lora":
        return LoRAAdapter(channels, kernel_size, lora_rank)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
