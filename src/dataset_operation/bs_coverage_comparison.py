"""BS placement + coverage visualization for Munich scene.

Compares different numbers of BSes (8, 12, 16) placed following
3GPP TR 38.901 UMi-street canyon guidelines on the Sionna Munich scene.
Outputs coverage maps (RSS, SINR) as images.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def extract_buildings(scene):
    """Extract building geometry from unmerged scene."""
    buildings = []
    for name, obj in scene.objects.items():
        mesh = obj.mi_mesh
        bbox = mesh.bbox()
        min_pt = [float(bbox.min[0]), float(bbox.min[1]), float(bbox.min[2])]
        max_pt = [float(bbox.max[0]), float(bbox.max[1]), float(bbox.max[2])]
        height = max_pt[2] - min_pt[2]
        footprint_x = max_pt[0] - min_pt[0]
        footprint_y = max_pt[1] - min_pt[1]

        if height > 3.0 and footprint_x < 500 and footprint_y < 500:
            buildings.append({
                "name": name,
                "bbox_min": min_pt,
                "bbox_max": max_pt,
                "cx": (min_pt[0] + max_pt[0]) / 2,
                "cy": (min_pt[1] + max_pt[1]) / 2,
                "height": height,
                "roof_z": max_pt[2],
                "footprint_x": footprint_x,
                "footprint_y": footprint_y,
            })
    return buildings


def place_bs_3gpp_umi(buildings, num_bs, target_isd=200.0, bs_height=10.0,
                       core_half_x=400, core_half_y=300):
    """Place BSes on hex grid snapped to building rooftops, 3GPP UMi style."""
    # Scene center from building centroids
    cx = np.mean([b["cx"] for b in buildings])
    cy = np.mean([b["cy"] for b in buildings])

    # Hex grid
    dx = target_isd
    dy = target_isd * np.sqrt(3) / 2
    candidates = []

    core_min_x, core_max_x = cx - core_half_x, cx + core_half_x
    core_min_y, core_max_y = cy - core_half_y, cy + core_half_y

    n_rows = int((core_max_y - core_min_y) / dy) + 2
    n_cols = int((core_max_x - core_min_x) / dx) + 2

    for row in range(n_rows + 1):
        for col in range(n_cols + 1):
            x = core_min_x + col * dx + (row % 2) * dx / 2
            y = core_min_y + row * dy
            if core_min_x <= x <= core_max_x and core_min_y <= y <= core_max_y:
                candidates.append([x, y])

    # Snap each candidate to nearest suitable building rooftop
    placed = []
    for cx_c, cy_c in candidates:
        best_score = -np.inf
        best_pos = None
        best_info = None

        for b in buildings:
            dist = np.sqrt((b["cx"] - cx_c)**2 + (b["cy"] - cy_c)**2)
            if dist > 80:
                continue
            roof = b["roof_z"]
            if roof < 5 or roof > 25:
                continue

            dx_dir = cx_c - b["cx"]
            dy_dir = cy_c - b["cy"]
            norm = np.sqrt(dx_dir**2 + dy_dir**2) + 1e-6
            dx_dir /= norm
            dy_dir /= norm

            edge_x = b["bbox_max"][0] if dx_dir > 0 else b["bbox_min"][0]
            edge_y = b["bbox_max"][1] if dy_dir > 0 else b["bbox_min"][1]
            pos_x = edge_x + dx_dir * 1.0
            pos_y = edge_y + dy_dir * 1.0
            pos_z = max(roof, bs_height)

            grid_dist = np.sqrt((pos_x - cx_c)**2 + (pos_y - cy_c)**2)
            height_penalty = abs(roof - 12.0)
            score = -grid_dist * 0.5 - height_penalty * 2.0

            if score > best_score:
                best_score = score
                best_pos = [pos_x, pos_y, pos_z]
                best_info = {"building": b["name"], "roof": roof, "type": "rooftop"}

        if best_pos is None:
            # Check if inside a building
            inside = False
            for b in buildings:
                if (b["bbox_min"][0] - 2 <= cx_c <= b["bbox_max"][0] + 2 and
                    b["bbox_min"][1] - 2 <= cy_c <= b["bbox_max"][1] + 2):
                    pos_x = b["bbox_max"][0] + 2.0
                    pos_y = cy_c
                    inside = True
                    break
            if not inside:
                pos_x, pos_y = cx_c, cy_c
            best_pos = [pos_x, pos_y, bs_height]
            best_info = {"building": None, "roof": bs_height, "type": "lamppost"}

        placed.append({"pos": best_pos, "info": best_info})

    # Farthest-point sampling
    positions = np.array([p["pos"][:2] for p in placed])
    center_dists = np.sqrt((positions[:, 0] - cx)**2 + (positions[:, 1] - cy)**2)
    selected = [int(np.argmin(center_dists))]

    for _ in range(min(num_bs - 1, len(placed) - 1)):
        min_dists = np.full(len(placed), np.inf)
        for s in selected:
            d = np.sqrt((positions[:, 0] - positions[s, 0])**2 +
                        (positions[:, 1] - positions[s, 1])**2)
            min_dists = np.minimum(min_dists, d)
        for s in selected:
            min_dists[s] = -1
        selected.append(int(np.argmax(min_dists)))

    result = []
    for idx in selected:
        p = placed[idx]
        result.append(p["pos"])

    # Compute ISD stats
    nn_isds = []
    for i in range(len(result)):
        min_d = float('inf')
        for j in range(len(result)):
            if i != j:
                d = np.sqrt((result[i][0] - result[j][0])**2 +
                            (result[i][1] - result[j][1])**2)
                min_d = min(min_d, d)
        nn_isds.append(min_d)

    print(f"  {num_bs} BSes: NN-ISD min={min(nn_isds):.0f}m, "
          f"max={max(nn_isds):.0f}m, mean={np.mean(nn_isds):.0f}m")

    return result


def setup_scene_with_bs(bs_positions, frequency=28e9):
    """Create a fresh scene with given BS positions."""
    import sionna.rt
    from sionna.rt import load_scene, PlanarArray, Transmitter

    scene = load_scene(sionna.rt.scene.munich, merge_shapes=True)
    scene.frequency = frequency
    scene.bandwidth = 400e6
    scene.temperature = 293.0

    # TX array (BS) - 2x2
    tx_array = PlanarArray(
        num_rows=2, num_cols=2,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="iso", polarization="V",
    )
    scene.tx_array = tx_array

    # RX array (UE) - 1x1 cross-pol
    rx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="dipole", polarization="cross",
    )
    scene.rx_array = rx_array

    # Add BSes
    for i, pos in enumerate(bs_positions):
        bs = Transmitter(
            name=f"bs_{i}",
            position=list(pos),
            power_dbm=40.0,
            orientation=[0, 0, 0],
            display_radius=10,
        )
        scene.add(bs)

    return scene


def compute_and_plot_coverage(scene, bs_positions, title, ax_rss, ax_sinr,
                               samples_per_tx=10_000_000):
    """Compute radio map and plot RSS + SINR coverage."""
    from sionna.rt import RadioMapSolver
    import drjit as dr

    solver = RadioMapSolver()
    radio_map = solver(
        scene=scene,
        cell_size=(2.0, 2.0),
        samples_per_tx=samples_per_tx,
        max_depth=5,
        los=True,
        specular_reflection=True,
        diffuse_reflection=True,
        refraction=True,
        diffraction=True,
        edge_diffraction=True,
    )

    # Plot RSS
    radio_map.show(metric="rss", ax=ax_rss)
    ax_rss.set_title(f"{title}\nRSS (dBm)", fontsize=11)

    # Plot BS positions
    bs_arr = np.array(bs_positions)
    ax_rss.scatter(bs_arr[:, 0], bs_arr[:, 1], c='red', marker='^',
                   s=80, edgecolors='black', linewidths=0.5, zorder=10,
                   label=f'{len(bs_positions)} BSes')
    for i, pos in enumerate(bs_positions):
        ax_rss.annotate(f'{i}', (pos[0], pos[1]), fontsize=6,
                       ha='center', va='bottom', color='red',
                       fontweight='bold')
    ax_rss.legend(fontsize=8, loc='upper right')

    # Plot SINR
    radio_map.show(metric="sinr", ax=ax_sinr)
    ax_sinr.set_title(f"{title}\nSINR (dB)", fontsize=11)
    ax_sinr.scatter(bs_arr[:, 0], bs_arr[:, 1], c='red', marker='^',
                    s=80, edgecolors='black', linewidths=0.5, zorder=10)
    for i, pos in enumerate(bs_positions):
        ax_sinr.annotate(f'{i}', (pos[0], pos[1]), fontsize=6,
                        ha='center', va='bottom', color='red',
                        fontweight='bold')

    # Compute coverage statistics
    rss_map = radio_map.rss().numpy()  # dBm
    sinr_map = radio_map.sinr().numpy()  # dB

    # Coverage rate (RSS > -100 dBm)
    valid = rss_map > -200  # valid cells
    covered = rss_map > -100
    coverage_rate = np.sum(covered) / np.sum(valid) * 100 if np.sum(valid) > 0 else 0

    # Mean SINR for covered cells
    sinr_covered = sinr_map[covered]
    mean_sinr = np.mean(sinr_covered) if len(sinr_covered) > 0 else 0

    stats = {
        "num_bs": len(bs_positions),
        "coverage_rate_pct": float(coverage_rate),
        "mean_sinr_db": float(mean_sinr),
        "median_sinr_db": float(np.median(sinr_covered)) if len(sinr_covered) > 0 else 0,
    }

    ax_rss.text(0.02, 0.02, f"Coverage: {coverage_rate:.1f}%",
                transform=ax_rss.transAxes, fontsize=9,
                verticalalignment='bottom', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_sinr.text(0.02, 0.02, f"Mean SINR: {mean_sinr:.1f} dB",
                 transform=ax_sinr.transAxes, fontsize=9,
                 verticalalignment='bottom', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    dr.flush_malloc_cache()
    dr.flush_kernel_cache()

    return stats


def main():
    import mitsuba as mi
    mi.set_variant("cuda_ad_mono_polarized", "llvm_ad_mono_polarized")
    import sionna.rt
    from sionna.rt import load_scene

    from src.config import get_plot_dir, get_results_dir
    out_dir = get_plot_dir("coverage")

    # Step 1: Extract buildings (unmerged)
    print("Loading unmerged scene for building geometry...")
    scene_unmerged = load_scene(sionna.rt.scene.munich, merge_shapes=False)
    buildings = extract_buildings(scene_unmerged)
    print(f"Extracted {len(buildings)} buildings, "
          f"median roof={np.median([b['roof_z'] for b in buildings]):.1f}m")
    del scene_unmerged

    # Step 2: Compute BS placements for different densities
    configs = [
        {"num_bs": 8,  "target_isd": 200, "label": "8 BS (ISD~200m)"},
        {"num_bs": 12, "target_isd": 160, "label": "12 BS (ISD~160m)"},
        {"num_bs": 16, "target_isd": 130, "label": "16 BS (ISD~130m)"},
    ]

    all_bs_positions = {}
    for cfg in configs:
        positions = place_bs_3gpp_umi(
            buildings, cfg["num_bs"], target_isd=cfg["target_isd"],
            bs_height=10.0, core_half_x=400, core_half_y=300,
        )
        all_bs_positions[cfg["label"]] = positions

    # Step 3: Plot building footprints + BS positions (no radio map needed)
    print("\nPlotting building footprints with BS positions...")
    fig_layout, axes_layout = plt.subplots(1, 3, figsize=(21, 7))
    for ax_idx, (label, bs_pos) in enumerate(all_bs_positions.items()):
        ax = axes_layout[ax_idx]
        # Draw building footprints
        for b in buildings:
            rect = plt.Rectangle(
                (b["bbox_min"][0], b["bbox_min"][1]),
                b["footprint_x"], b["footprint_y"],
                linewidth=0.3, edgecolor='gray', facecolor='lightgray', alpha=0.5
            )
            ax.add_patch(rect)

        # Draw BSes
        bs_arr = np.array(bs_pos)
        ax.scatter(bs_arr[:, 0], bs_arr[:, 1], c='red', marker='^',
                   s=120, edgecolors='black', linewidths=1, zorder=10)
        for i, pos in enumerate(bs_pos):
            ax.annotate(f'BS{i}\n({pos[2]:.0f}m)', (pos[0], pos[1]),
                       fontsize=7, ha='center', va='bottom', color='darkred',
                       fontweight='bold')

        # Draw ISD circles
        for pos in bs_pos:
            circle = plt.Circle((pos[0], pos[1]), 100, fill=False,
                               linestyle='--', color='red', alpha=0.2, linewidth=0.5)
            ax.add_patch(circle)

        ax.set_xlim(-600, 500)
        ax.set_ylim(-500, 400)
        ax.set_aspect('equal')
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True, alpha=0.2)

    fig_layout.suptitle('Munich Scene - 3GPP UMi BS Placement (building rooftops)',
                        fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_layout.savefig(out_dir / "munich_bs_layout.png", dpi=150, bbox_inches='tight')
    print(f"Saved {out_dir / 'munich_bs_layout.png'}")

    # Step 4: Compute coverage maps
    print("\nComputing coverage maps (this takes a while per config)...")
    fig_cov, axes_cov = plt.subplots(2, 3, figsize=(21, 14))

    all_stats = {}
    for col_idx, (label, bs_pos) in enumerate(all_bs_positions.items()):
        print(f"\n--- {label} ---")
        scene = setup_scene_with_bs(bs_pos, frequency=28e9)
        stats = compute_and_plot_coverage(
            scene, bs_pos, label,
            ax_rss=axes_cov[0, col_idx],
            ax_sinr=axes_cov[1, col_idx],
            samples_per_tx=10_000_000,
        )
        all_stats[label] = stats
        print(f"  Coverage: {stats['coverage_rate_pct']:.1f}%, "
              f"Mean SINR: {stats['mean_sinr_db']:.1f} dB")
        del scene

    fig_cov.suptitle('Munich Scene @ 28 GHz - Coverage Comparison (3GPP UMi)',
                     fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_cov.savefig(out_dir / "munich_coverage_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved {out_dir / 'munich_coverage_comparison.png'}")

    # Save stats
    results_dir = get_results_dir("coverage")
    with open(results_dir / "munich_coverage_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Saved {out_dir / 'munich_coverage_stats.json'}")

    print("\n=== Summary ===")
    for label, stats in all_stats.items():
        print(f"  {label}: coverage={stats['coverage_rate_pct']:.1f}%, "
              f"mean_SINR={stats['mean_sinr_db']:.1f}dB, "
              f"median_SINR={stats['median_sinr_db']:.1f}dB")


if __name__ == "__main__":
    main()
