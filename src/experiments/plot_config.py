"""Plot configuration system.

Base config → plot type configs hierarchy.
Changing base config propagates to all figures for visual consistency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt


# ── Method color palette (fixed across all figures) ──────────────────
METHOD_COLORS: Dict[str, str] = {
    # Ours (3-way)
    "3way":         "#1f77b4",  # blue
    "ours":         "#1f77b4",  # alias
    # FL baselines
    "fedavg":       "#ff7f0e",  # orange
    "fedper":       "#9467bd",  # purple
    "independent":  "#d62728",  # red
    # Classical baselines
    "ls":           "#7f7f7f",  # gray
    "lmmse":        "#8c564b",  # brown
    # Training strategies
    "from_scratch": "#2ca02c",  # green
    "pretrained":   "#1f77b4",  # blue (same as ours)
    "maml":         "#e377c2",  # pink
    # Adapter variants
    "ssf":          "#17becf",  # cyan
    "lora":         "#bcbd22",  # olive
    # Site injection ablation
    "film":         "#1f77b4",  # blue
    "concat":       "#ff7f0e",  # orange
    "add":          "#2ca02c",  # green
    "no_site":      "#7f7f7f",  # gray
}

# Display names for legend labels
METHOD_LABELS: Dict[str, str] = {
    "3way":         "3-Way (Ours)",
    "ours":         "3-Way (Ours)",
    "fedavg":       "FedAvg",
    "fedper":       "FedPer",
    "independent":  "Independent",
    "ls":           "LS",
    "lmmse":        "LMMSE",
    "from_scratch":  "From Scratch",
    "pretrained":   "Pre-trained",
    "maml":         "MAML",
    "ssf":          "SSF",
    "lora":         "LoRA",
    "film":         "FiLM",
    "concat":       "Concat",
    "add":          "Add",
    "no_site":      "No Site Emb.",
}


# ── Base Config ──────────────────────────────────────────────────────

@dataclass
class BaseConfig:
    # Font
    font_family: str = "sans-serif"
    font_sans_serif: List[str] = field(
        default_factory=lambda: ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"]
    )
    mathtext_fontset: str = "custom"
    mathtext_rm: str = "Helvetica"
    mathtext_it: str = "Helvetica:italic"
    mathtext_bf: str = "Helvetica:bold"
    font_size_title: float = 16
    font_size_axis_label: float = 14
    font_size_tick: float = 12
    font_size_legend: float = 10
    font_size_annotation: float = 11
    font_size_bar_label: float = 12

    # Figure sizes
    fig_single: Tuple[float, float] = (4.2, 3)
    fig_wide: Tuple[float, float] = (12, 5.2)
    fig_double: Tuple[float, float] = (8, 3)
    fig_square: Tuple[float, float] = (4, 4)
    fig_1x2: Tuple[float, float] = (14, 4.5)
    fig_2x2: Tuple[float, float] = (14, 9)
    dpi_save: int = 300
    dpi_display: int = 100
    save_formats: List[str] = field(default_factory=lambda: ["pdf", "png"])

    # Grid / spines / ticks
    grid_alpha: float = 0.4
    grid_linewidth: float = 0.6
    spine_visible: bool = True
    tick_direction: str = "out"

    # Legend
    legend_frameon: bool = True
    legend_framealpha: float = 0.8
    legend_edgecolor: str = "0.8"

    def apply_rcparams(self):
        """Apply base config to matplotlib rcParams."""
        mpl.rcParams.update({
            "font.family": self.font_family,
            "font.sans-serif": self.font_sans_serif,
            "mathtext.fontset": self.mathtext_fontset,
            "mathtext.rm": self.mathtext_rm,
            "mathtext.it": self.mathtext_it,
            "mathtext.bf": self.mathtext_bf,
            "font.size": self.font_size_tick,
            "axes.titlesize": self.font_size_title,
            "axes.labelsize": self.font_size_axis_label,
            "xtick.labelsize": self.font_size_tick,
            "ytick.labelsize": self.font_size_tick,
            "legend.fontsize": self.font_size_legend,
            "xtick.direction": self.tick_direction,
            "ytick.direction": self.tick_direction,
            "axes.grid": True,
            "grid.alpha": self.grid_alpha,
            "grid.linewidth": self.grid_linewidth,
            "savefig.dpi": self.dpi_save,
            "figure.dpi": self.dpi_display,
            "legend.frameon": self.legend_frameon,
            "legend.framealpha": self.legend_framealpha,
            "legend.edgecolor": self.legend_edgecolor,
        })


# ── Plot Type Configs ────────────────────────────────────────────────

@dataclass
class LinePlotConfig:
    linewidth: float = 2.5
    marker: str = "o"
    markersize: float = 6.0
    error_band_alpha: float = 0.15


@dataclass
class BarPlotConfig:
    bar_width: float = 0.35
    group_gap: float = 0.05
    edgecolor: str = "black"
    edge_linewidth: float = 0.5
    annotation_fontsize: float = 7.5


@dataclass
class ScatterPlotConfig:
    marker_size: float = 30.0
    alpha: float = 0.6
    cmap: str = "tab10"


@dataclass
class MultiPanelConfig:
    wspace: float = 0.3
    hspace: float = 0.3
    panel_label_fontsize: float = 14
    panel_label_fontweight: str = "bold"
    panel_label_x: float = -0.05
    panel_label_y: float = 1.05


# ── Singletons ───────────────────────────────────────────────────────
BASE = BaseConfig()
LINE = LinePlotConfig()
BAR = BarPlotConfig()
SCATTER = ScatterPlotConfig()
MULTI = MultiPanelConfig()

_REF_WIDTH = 6.0  # reference figure width for font scaling


def setup(fig_width: Optional[float] = None):
    """Call at the start of every plot script / notebook cell.

    Args:
        fig_width: If given and larger than _REF_WIDTH, scale fonts/lines
                   proportionally so text stays readable at wider sizes.
    """
    BASE.apply_rcparams()
    if fig_width and fig_width > _REF_WIDTH:
        s = fig_width / _REF_WIDTH
        mpl.rcParams.update({
            "font.size": BASE.font_size_tick * s,
            "axes.titlesize": BASE.font_size_title * s,
            "axes.labelsize": BASE.font_size_axis_label * s,
            "xtick.labelsize": BASE.font_size_tick * s,
            "ytick.labelsize": BASE.font_size_tick * s,
            "legend.fontsize": BASE.font_size_legend * s,
            "lines.linewidth": LINE.linewidth * s,
            "lines.markersize": LINE.markersize * s,
            "grid.linewidth": BASE.grid_linewidth * s,
        })


def get_style(method: str) -> dict:
    """Return dict with color and label for a method key."""
    return {
        "color": METHOD_COLORS.get(method, "#333333"),
        "label": METHOD_LABELS.get(method, method),
    }


def savefig(fig, name: str, out_dir: Optional[Path] = None, tight: bool = True):
    """Save figure in all configured formats.

    Args:
        fig: matplotlib Figure.
        name: Filename stem (no extension).
        out_dir: Output directory. If None, saves next to the calling script.
        tight: Use bbox_inches='tight'.
    """
    if out_dir is None:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"dpi": BASE.dpi_save}
    if tight:
        kwargs["bbox_inches"] = "tight"
    for fmt in BASE.save_formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path, **kwargs)
        print(f"  Saved: {path}")
