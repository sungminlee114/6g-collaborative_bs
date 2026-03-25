"""Generate trajectory PNG with radio map (SINR) background + UE tracks.

Usage:
    python -m src.dataset_operation.plot_trajectories_png [--traj_dir DIR] [--output PATH]

Requires GPU for radio map computation (cached to traj_dir/radio_map_cache.npz).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def load_buildings(path="assets/data/munich_buildings.json"):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def compute_or_load_radio_map(traj_dir, info):
    """Compute radio map SINR grid or load from cache."""
    cache_path = Path(traj_dir) / "radio_map_cache.npz"
    if cache_path.exists():
        d = np.load(cache_path)
        print(f"  Loaded radio map cache: {cache_path}")
        return d["sinr_db"], d["x_edges"], d["y_edges"]

    # Need GPU to compute
    print("  Computing radio map (GPU)...")
    from src.config import SceneConfig
    from src.dataset_operation.generate import build_scene, compute_radio_map

    preset = info.get("preset", "munich_elaa_m_1k_28g")
    cfg = SceneConfig.from_preset(preset)
    scene = build_scene(cfg)
    rm = compute_radio_map(scene, cfg)

    # sinr: (num_tx, W, H) linear scale → best-server max → dB
    sinr_lin = np.array(rm.sinr)       # (num_tx, W, H)
    best_sinr = sinr_lin.max(axis=0)   # (W, H) — best-server SINR
    sinr_db = 10 * np.log10(np.maximum(best_sinr, 1e-10))

    # cell_centers: (nY, nX, 3) — dim0=Y, dim1=X
    cc = np.array(rm.cell_centers)     # (nY, nX, 3)
    x_centers = cc[0, :, 0]            # (nX,) — x varies along dim1
    y_centers = cc[:, 0, 1]            # (nY,) — y varies along dim0
    cell_w = x_centers[1] - x_centers[0] if len(x_centers) > 1 else 1.0
    cell_h = y_centers[1] - y_centers[0] if len(y_centers) > 1 else 1.0
    x_edges = np.append(x_centers - cell_w / 2, x_centers[-1] + cell_w / 2)
    y_edges = np.append(y_centers - cell_h / 2, y_centers[-1] + cell_h / 2)
    # sinr_db: (num_tx, nY, nX) → best-server → (nY, nX) → transpose to (nX, nY) for cache
    sinr_db = sinr_db.T  # now (nX, nY) so sinr_db[xi, yi] works with x_edges, y_edges

    np.savez_compressed(cache_path, sinr_db=sinr_db, x_edges=x_edges, y_edges=y_edges)
    print(f"  Cached radio map: {cache_path} ({sinr_db.shape})")

    # Cleanup GPU
    import drjit as dr
    dr.flush_malloc_cache()
    dr.flush_kernel_cache()

    return sinr_db, x_edges, y_edges


def plot_trajectories(traj_dir="assets/data/shared_trajectories",
                      output="assets/plots/trajectory_15ue.png"):
    traj_dir = Path(traj_dir)
    traj = np.load(traj_dir / "trajectories.npz")
    positions = traj["positions"]  # (T, N_UE, 3)
    speeds = traj["speeds"]       # (N_UE,)
    bs_ids = traj["bs_ids"]       # (N_UE,)

    with open(traj_dir / "trajectory_info.json") as f:
        info = json.load(f)

    T, N_UE, _ = positions.shape
    dt_ms = info["dt_ms"]
    total_ms = dt_ms * T
    freq_ghz = info.get("preset", "").split("_")[-1].replace("g", "")

    # Speed groups
    unique_speeds = sorted(set(float(s) for s in speeds))
    n_speeds = len(unique_speeds)

    # Distinct colors for speed groups
    speed_cmap = plt.cm.tab10
    speed_colors = {s: speed_cmap(i) for i, s in enumerate(unique_speeds)}

    # BS positions
    bs_positions = np.array(info.get("bs_positions", []))
    if len(bs_positions) == 0:
        import yaml
        preset = info.get("preset", "munich_elaa_m_1k_28g")
        yaml_path = Path("assets/configs") / f"{preset}.yaml"
        if yaml_path.exists():
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            bs_positions = np.array(cfg.get("bs_positions", []))

    buildings = load_buildings()

    # Compute view bounds: auto-fit to UE extent with margin
    all_x = positions[:, :, 0].flatten()
    all_y = positions[:, :, 1].flatten()
    margin = 15  # meters padding around UE extent
    xmin, xmax = all_x.min() - margin, all_x.max() + margin
    ymin, ymax = all_y.min() - margin, all_y.max() + margin
    # Ensure aspect ratio is reasonable (at least 50m span)
    span_x = xmax - xmin
    span_y = ymax - ymin
    min_span = 50
    if span_x < min_span:
        cx = (xmin + xmax) / 2
        xmin, xmax = cx - min_span / 2, cx + min_span / 2
    if span_y < min_span:
        cy = (ymin + ymax) / 2
        ymin, ymax = cy - min_span / 2, cy + min_span / 2

    # ── Radio map ──
    try:
        sinr_db, x_edges, y_edges = compute_or_load_radio_map(traj_dir, info)
        has_radio_map = True
    except Exception as e:
        print(f"  Radio map unavailable: {e}")
        has_radio_map = False

    # ── Plot ──
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Layer 1: Radio map heatmap
    if has_radio_map:
        # Clip to view region
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        xi = np.where((x_centers >= xmin) & (x_centers <= xmax))[0]
        yi = np.where((y_centers >= ymin) & (y_centers <= ymax))[0]

        if len(xi) > 0 and len(yi) > 0:
            sinr_view = sinr_db[xi[0]:xi[-1]+1, yi[0]:yi[-1]+1]
            xe = x_edges[xi[0]:xi[-1]+2]
            ye = y_edges[yi[0]:yi[-1]+2]

            # Mask no-coverage areas (buildings/dead zones) as transparent
            masked_sinr = np.ma.masked_where(sinr_view < -10, sinr_view)

            cmap = plt.cm.RdYlGn.copy()
            cmap.set_bad(alpha=0)  # no-coverage → fully transparent

            im = ax.pcolormesh(
                xe, ye, masked_sinr.T,
                cmap=cmap, vmin=-5, vmax=30,
                alpha=0.25, zorder=0, shading="flat",
            )
            cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
            cbar.set_label("Best-server SINR [dB]", fontsize=10)

    # Buildings: use actual mesh triangles (exact 2D projection) if available
    tris_path = Path("assets/data/munich_buildings_tris.json")
    if tris_path.exists():
        from matplotlib.patches import Polygon as MplPolygon
        with open(tris_path) as f:
            all_tris = json.load(f)
        # Filter to view region for performance
        patches = []
        for tri in all_tris:
            t = np.array(tri)
            if (t[:, 0].max() < xmin or t[:, 0].min() > xmax or
                t[:, 1].max() < ymin or t[:, 1].min() > ymax):
                continue
            patches.append(MplPolygon(tri, closed=True))
        pc = PatchCollection(patches, facecolor="#c8c8c8", edgecolor="#999999",
                             linewidth=0.15, alpha=0.85, zorder=2)
        ax.add_collection(pc)
    elif buildings:
        patches = []
        for b in buildings:
            w = b["x_max"] - b["x_min"]
            h = b["y_max"] - b["y_min"]
            patches.append(Rectangle((b["x_min"], b["y_min"]), w, h))
        pc = PatchCollection(patches, facecolor="#d5d5d5", edgecolor="#aaaaaa",
                             linewidth=0.3, alpha=0.7, zorder=2)
        ax.add_collection(pc)

    # Layer 3: UE trajectories
    for u in range(N_UE):
        spd = float(speeds[u])
        color = speed_colors[spd]
        xs = positions[:, u, 0]
        ys = positions[:, u, 1]

        if spd < 0.01:
            ax.plot(xs[0], ys[0], 'o', color=color, markersize=5, zorder=6,
                    markeredgecolor='white', markeredgewidth=0.5)
        else:
            ax.plot(xs, ys, '--', color=color, linewidth=1.2, alpha=0.8, zorder=4)
            ax.plot(xs[0], ys[0], 'o', color=color, markersize=3.5, zorder=6,
                    markeredgecolor='white', markeredgewidth=0.3)

    # Layer 4: BS markers
    if len(bs_positions) > 0:
        target_bs = info.get("target_bs")
        for i, pos in enumerate(bs_positions):
            is_target = (target_bs is not None and i == target_bs)
            size = 200 if is_target else 100
            ec = 'darkred' if is_target else 'gray'
            fc = 'red' if is_target else 'lightcoral'
            alpha = 1.0 if is_target else 0.4
            ax.scatter(pos[0], pos[1], c=fc, marker='^', s=size, zorder=10,
                       edgecolors=ec, linewidths=1.5, alpha=alpha)
            ax.annotate(f'BS{i}', (pos[0], pos[1]), fontsize=9, fontweight='bold',
                        color=ec, ha='center', va='bottom', alpha=alpha,
                        xytext=(0, 10), textcoords='offset points', zorder=11)

    # Legend
    legend_handles = []
    for spd in unique_speeds:
        color = speed_colors[spd]
        label = f"Static" if spd < 0.01 else f"{spd:.1f} m/s"
        ue_count = sum(1 for s in speeds if abs(float(s) - spd) < 0.01)
        if spd < 0.01:
            h = ax.plot([], [], 'o', color=color, markersize=6,
                        label=f"{label} ({ue_count} UEs)")[0]
        else:
            h = ax.plot([], [], '-', color=color, linewidth=2,
                        label=f"{label} ({ue_count} UEs)")[0]
        legend_handles.append(h)

    ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
              framealpha=0.9, title="Speed groups")

    ax.set_xlabel("x [m]", fontsize=11)
    ax.set_ylabel("y [m]", fontsize=11)
    ax.set_title(f"UE Trajectories + Radio Map: {N_UE} UEs, {T} snapshots, "
                 f"dt={dt_ms}ms, total={total_ms:.0f}ms", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", default="assets/data/shared_trajectories")
    parser.add_argument("--output", default="assets/plots/trajectory_15ue.png")
    args = parser.parse_args()
    plot_trajectories(args.traj_dir, args.output)
