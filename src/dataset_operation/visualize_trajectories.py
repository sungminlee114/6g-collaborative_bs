"""Visualize UE trajectories on Munich map as GIF.

Usage:
    python -m src.dataset_operation.visualize_trajectories \
        --data_dir assets/data/channels_elaa_m_1k_15g_temporal \
        --subsample 100 --output assets/plots/trajectory.gif
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_buildings(path="assets/data/munich_buildings.json"):
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        buildings = json.load(f)
    return buildings


def make_gif(data_dir: str, output: str = "assets/plots/trajectory.gif",
             subsample: int = 100, fps: int = 20, trail: int = 50):
    """Create trajectory GIF.

    Args:
        data_dir: temporal dataset directory
        output: output GIF path
        subsample: plot every Nth snapshot
        fps: frames per second
        trail: number of past positions to show as trail
    """
    data_dir = Path(data_dir)
    traj = np.load(data_dir / "trajectories.npz")
    positions = traj["positions"]  # (T, N_UE, 3)
    speeds = traj["speeds"]        # (N_UE,)
    bs_ids = traj["bs_ids"]        # (N_UE,)

    with open(data_dir / "trajectory_info.json") as f:
        info = json.load(f)
    dt_ms = info["dt_ms"]

    T, N_UE, _ = positions.shape
    frame_indices = list(range(0, T, subsample))
    print(f"Trajectories: {T} snapshots, {N_UE} UEs, dt={dt_ms}ms")
    print(f"GIF: {len(frame_indices)} frames (subsample={subsample}), fps={fps}")

    # Load BS positions — try bs_info.json first, fallback to config YAML
    bs_positions = []
    bs_info_path = data_dir / "bs_info.json"
    if bs_info_path.exists():
        with open(bs_info_path) as f:
            bs_positions = json.load(f).get("positions", [])
    if not bs_positions:
        # Read from trajectory_info → preset → YAML
        traj_info_path = data_dir / "trajectory_info.json"
        if traj_info_path.exists():
            with open(traj_info_path) as f:
                preset = json.load(f).get("preset", "")
            if preset:
                import yaml
                yaml_path = Path("assets/configs") / f"{preset}.yaml"
                if yaml_path.exists():
                    with open(yaml_path) as f:
                        bs_positions = yaml.safe_load(f).get("bs_positions", [])

    # Load buildings
    buildings = load_buildings()

    # Speed-based colors
    ue_colors = []
    for u in range(N_UE):
        s = float(speeds[u])
        if s < 0.01:
            ue_colors.append("blue")      # static
        elif s < 5.0:
            ue_colors.append("green")     # pedestrian
        else:
            ue_colors.append("red")       # vehicle

    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Draw buildings
    if buildings:
        for b in buildings:
            rect = plt.Rectangle(
                (b["x_min"], b["y_min"]),
                b["x_max"] - b["x_min"], b["y_max"] - b["y_min"],
                facecolor="#d0d0d0", edgecolor="#a0a0a0", linewidth=0.3, zorder=1,
            )
            ax.add_patch(rect)

    # Draw BS positions
    for i, bp in enumerate(bs_positions):
        ax.plot(bp[0], bp[1], "k^", markersize=10, zorder=5)
        ax.annotate(f"BS{i}", (bp[0], bp[1]), fontsize=7, ha="center", va="bottom")

    # Set axis limits from positions
    margin = 20
    all_x = positions[:, :, 0]
    all_y = positions[:, :, 1]
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Animation elements
    dots = ax.scatter([], [], s=20, zorder=10)
    trails = [ax.plot([], [], linewidth=0.5, alpha=0.4, color=ue_colors[u])[0] for u in range(N_UE)]
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=10,
                        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", label="static (0 m/s)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green", label="ped (1 m/s)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", label="veh (8.3 m/s)"),
        Line2D([0], [0], marker="^", color="black", label="BS", markersize=8),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8)

    def update(frame_idx):
        t = frame_indices[frame_idx]
        x = positions[t, :, 0]
        y = positions[t, :, 1]
        dots.set_offsets(np.column_stack([x, y]))
        dots.set_color(ue_colors)

        # Trails
        trail_start = max(0, t - trail * subsample)
        for u in range(N_UE):
            trail_t = range(trail_start, t + 1, max(subsample, 1))
            trail_t = [tt for tt in trail_t if tt < T]
            if len(trail_t) > 1:
                trails[u].set_data(positions[trail_t, u, 0], positions[trail_t, u, 1])
            else:
                trails[u].set_data([], [])

        elapsed_ms = t * dt_ms
        time_text.set_text(f"t = {elapsed_ms:.1f} ms  (snap {t}/{T})")
        return [dots, time_text] + trails

    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=1000 // fps, blit=False)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    anim.save(output, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved: {output} ({len(frame_indices)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize UE trajectories as GIF")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="assets/plots/trajectory.gif")
    parser.add_argument("--subsample", type=int, default=100, help="Plot every Nth snapshot")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--trail", type=int, default=50, help="Trail length (in subsampled frames)")
    args = parser.parse_args()

    make_gif(args.data_dir, args.output, args.subsample, args.fps, args.trail)
