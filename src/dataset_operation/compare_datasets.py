"""Compare UMa 8BS vs UMi 16BS datasets.

Generates side-by-side plots:
  1. BS layout map (positions + heights)
  2. UE distribution per BS (bar chart)
  3. Channel power statistics (CDF of |H|^2)
  4. Summary statistics table

Usage:
    python -m src.experiments.compare_datasets
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#222",
    "text.color": "#222",
    "xtick.color": "#333",
    "ytick.color": "#333",
    "grid.color": "#ccc",
    "legend.facecolor": "white",
    "legend.edgecolor": "#999",
})

DATASETS = {
    "UMa 8BS (old)": "assets/data/channels",
    "UMi 16BS (new)": "assets/data/channels_umi16",
}
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from src.config import get_plot_dir, get_results_dir
OUT_DIR = get_results_dir("comparison")
COLORS = ["#1f77b4", "#e91e63"]
BUILDINGS_CACHE = Path("assets/data/munich_buildings.json")


def load_buildings():
    """Load cached Munich building footprints."""
    with open(BUILDINGS_CACHE) as f:
        return json.load(f)


def draw_buildings(ax, buildings, xlim=None, ylim=None):
    """Draw building footprints as gray rectangles on ax."""
    patches = []
    for b in buildings:
        x0, y0 = b["x_min"], b["y_min"]
        w, h_rect = b["x_max"] - x0, b["y_max"] - y0
        if xlim and (b["x_max"] < xlim[0] or b["x_min"] > xlim[1]):
            continue
        if ylim and (b["y_max"] < ylim[0] or b["y_min"] > ylim[1]):
            continue
        patches.append(Rectangle((x0, y0), w, h_rect))
    # Color by height
    heights = [b["height"] for b in buildings
               if not (xlim and (b["x_max"] < xlim[0] or b["x_min"] > xlim[1]))
               and not (ylim and (b["y_max"] < ylim[0] or b["y_min"] > ylim[1]))]
    if patches:
        pc = PatchCollection(patches, alpha=0.4, edgecolor="#999", linewidth=0.3)
        pc.set_array(np.array(heights))
        pc.set_cmap("YlOrBr")
        pc.set_clim(0, 80)
        ax.add_collection(pc)


def load_dataset_info(data_dir: str):
    """Load metadata + bs_info + sample channel stats."""
    data_dir = Path(data_dir)
    meta = pd.read_parquet(data_dir / "metadata.parquet")
    with open(data_dir / "bs_info.json") as f:
        bs_info = json.load(f)

    # Sample CIR stats from a few snapshots
    delay_spreads = []
    n_paths_list = []
    path_gains_db = []
    for snap_id in range(min(10, bs_info["num_snapshots"])):
        snap_dir = data_dir / f"snapshot_{snap_id:04d}"
        ch = np.load(snap_dir / "channels.npz")
        cir_a = ch["cir_a"]    # (N_UE, n_rx, n_tx, n_paths) complex
        cir_tau = ch["cir_tau"]  # (N_UE, n_rx, n_tx, n_paths)
        n_paths_list.append(cir_a.shape[-1])

        # Per-UE RMS delay spread (averaged over ant pairs)
        pw = np.abs(cir_a) ** 2
        # Replace NaN with 0
        pw = np.nan_to_num(pw, nan=0.0)
        cir_tau_clean = np.nan_to_num(cir_tau, nan=0.0)
        total_pw = pw.sum(axis=-1, keepdims=True) + 1e-30
        mean_tau = (pw * cir_tau_clean).sum(axis=-1, keepdims=True) / total_pw
        rms_ds = np.sqrt(((pw * (cir_tau_clean - mean_tau) ** 2).sum(axis=-1)) / total_pw.squeeze(-1))
        # Average over ant pairs, per UE
        ds_per_ue = np.nanmean(rms_ds, axis=(1, 2))  # (N_UE,)
        delay_spreads.append(ds_per_ue)

        # Path gain (strongest path per UE)
        max_gain = np.nanmax(np.abs(cir_a), axis=(1, 2, 3))
        path_gains_db.append(20 * np.log10(max_gain + 1e-30))

    delay_spreads = np.concatenate(delay_spreads)
    path_gains_db = np.concatenate(path_gains_db)
    return meta, bs_info, delay_spreads, path_gains_db, n_paths_list


def plot_bs_layout(ax, bs_info, meta, title, color, buildings=None):
    """Plot BS positions with UE counts and building footprints."""
    xlim = (-600, 500)
    ylim = (-500, 450)

    # Draw buildings first (background)
    if buildings:
        draw_buildings(ax, buildings, xlim=xlim, ylim=ylim)

    positions = np.array(bs_info["positions"])
    bs_counts = meta.bs_id.value_counts().sort_index()

    ax.scatter(positions[:, 0], positions[:, 1],
               c=color, marker="^", s=120, edgecolors="black",
               linewidths=0.8, zorder=10)

    for i, pos in enumerate(positions):
        count = bs_counts.get(i, 0)
        ax.annotate(f"BS{i}\n{pos[2]:.0f}m\n({count})",
                    (pos[0], pos[1]),
                    fontsize=6, ha="center", va="bottom",
                    color=color, fontweight="bold")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.15)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    infos = {}
    for label, data_dir in DATASETS.items():
        print(f"Loading {label} from {data_dir}...")
        meta, bs_info, delay_spreads, path_gains, n_paths = load_dataset_info(data_dir)
        infos[label] = {
            "meta": meta, "bs_info": bs_info,
            "delay_spreads": delay_spreads,
            "path_gains": path_gains,
            "n_paths": n_paths,
        }

    labels = list(infos.keys())

    # Load building footprints
    buildings = load_buildings() if BUILDINGS_CACHE.exists() else None

    # ── Figure: 2x3 comparison ──
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # (0,0) + (0,1): BS layout maps
    for i, label in enumerate(labels):
        plot_bs_layout(
            axes[0, i], infos[label]["bs_info"],
            infos[label]["meta"], label, COLORS[i],
            buildings=buildings,
        )

    # (0,2): BS height comparison
    ax = axes[0, 2]
    for i, label in enumerate(labels):
        heights = [p[2] for p in infos[label]["bs_info"]["positions"]]
        x = np.arange(len(heights))
        ax.bar(x + i * 0.4, heights, 0.35, label=label, color=COLORS[i], alpha=0.85)
    ax.set_xlabel("BS ID")
    ax.set_ylabel("Height (m)")
    ax.set_title("BS Heights", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15, axis="y")

    # (1,0): UE distribution per BS
    ax = axes[1, 0]
    for i, label in enumerate(labels):
        meta = infos[label]["meta"]
        counts = meta.bs_id.value_counts().sort_index()
        x = np.arange(len(counts))
        ax.bar(x + i * 0.4, counts.values, 0.35,
               label=label, color=COLORS[i], alpha=0.85)
        ax.axhline(counts.mean(), color=COLORS[i], linestyle="--",
                   alpha=0.5, linewidth=1)
    ax.set_xlabel("BS ID")
    ax.set_ylabel("# UE samples")
    ax.set_title("UE Distribution per BS", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15, axis="y")

    # (1,1): RMS delay spread CDF
    ax = axes[1, 1]
    for i, label in enumerate(labels):
        ds = infos[label]["delay_spreads"]
        ds = ds[np.isfinite(ds)]
        sorted_ds = np.sort(ds) * 1e9  # convert to ns
        cdf = np.arange(1, len(sorted_ds) + 1) / len(sorted_ds)
        ax.plot(sorted_ds, cdf, color=COLORS[i], linewidth=2, label=label)
    ax.set_xlabel("RMS delay spread (ns)")
    ax.set_ylabel("CDF")
    ax.set_title("RMS Delay Spread Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    # (1,2): Peak path gain CDF
    ax = axes[1, 2]
    for i, label in enumerate(labels):
        pg = infos[label]["path_gains"]
        pg = pg[np.isfinite(pg)]
        sorted_pg = np.sort(pg)
        cdf = np.arange(1, len(sorted_pg) + 1) / len(sorted_pg)
        ax.plot(sorted_pg, cdf, color=COLORS[i], linewidth=2, label=label)
    ax.set_xlabel("Peak path gain (dB)")
    ax.set_ylabel("CDF")
    ax.set_title("Peak Path Gain Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    fig.suptitle("Dataset Comparison: UMa 8BS vs UMi 16BS",
                 fontsize=15, fontweight="bold", color="#222")
    plt.tight_layout()
    plot_dir = get_plot_dir("comparison")
    out_path = plot_dir / "uma_vs_umi_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # ── Summary stats ──
    summary = {}
    for label in labels:
        meta = infos[label]["meta"]
        bs_info = infos[label]["bs_info"]
        ds = infos[label]["delay_spreads"]
        ds = ds[np.isfinite(ds)]
        pg = infos[label]["path_gains"]
        pg = pg[np.isfinite(pg)]
        counts = meta.bs_id.value_counts()
        heights = [p[2] for p in bs_info["positions"]]

        summary[label] = {
            "num_bs": bs_info["num_bs"],
            "total_samples": len(meta),
            "samples_per_bs_mean": float(counts.mean()),
            "samples_per_bs_std": float(counts.std()),
            "bs_height_mean_m": float(np.mean(heights)),
            "bs_height_range_m": [float(min(heights)), float(max(heights))],
            "delay_spread_mean_ns": float(np.mean(ds) * 1e9),
            "delay_spread_median_ns": float(np.median(ds) * 1e9),
            "peak_path_gain_mean_db": float(np.mean(pg)),
            "n_paths_per_snapshot": infos[label]["n_paths"],
            "frequency_ghz": bs_info["frequency_hz"] / 1e9,
        }

    stats_path = OUT_DIR / "comparison_stats.json"
    with open(stats_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {stats_path}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"{'Metric':<30} {'UMa 8BS':>18} {'UMi 16BS':>18}")
    print("-" * 70)
    for key in ["num_bs", "total_samples", "samples_per_bs_mean",
                "bs_height_mean_m", "delay_spread_mean_ns",
                "delay_spread_median_ns", "peak_path_gain_mean_db"]:
        v1 = summary[labels[0]][key]
        v2 = summary[labels[1]][key]
        fmt = ".1f" if isinstance(v1, float) else "d"
        print(f"{key:<30} {v1:>18{fmt}} {v2:>18{fmt}}")
    print(f"{'n_paths (snap 0)':<30} {summary[labels[0]]['n_paths_per_snapshot'][0]:>18d} {summary[labels[1]]['n_paths_per_snapshot'][0]:>18d}")
    print("=" * 70)


if __name__ == "__main__":
    main()
