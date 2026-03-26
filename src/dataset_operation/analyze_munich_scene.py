"""Analyze Munich scene geometry and find 3GPP-compliant BS positions.

Extracts building bounding boxes from the Sionna Munich scene (unmerged),
then places BSes following 3GPP TR 38.901 UMi-street canyon guidelines:
  - BS height: 10m
  - Target ISD: ~200m
  - BSes placed on/near building rooftops
"""
import numpy as np


def extract_scene_geometry():
    """Load Munich scene (unmerged) and extract building geometry."""
    import mitsuba as mi
    mi.set_variant("cuda_ad_mono_polarized", "llvm_ad_mono_polarized")

    import sionna.rt
    from sionna.rt import load_scene

    # merge_shapes=False to get individual buildings
    scene = load_scene(sionna.rt.scene.munich, merge_shapes=False)

    buildings = []
    for name, obj in scene.objects.items():
        mesh = obj.mi_mesh
        bbox = mesh.bbox()
        min_pt = [float(bbox.min[0]), float(bbox.min[1]), float(bbox.min[2])]
        max_pt = [float(bbox.max[0]), float(bbox.max[1]), float(bbox.max[2])]
        center_x = (min_pt[0] + max_pt[0]) / 2
        center_y = (min_pt[1] + max_pt[1]) / 2
        height = max_pt[2] - min_pt[2]
        footprint_x = max_pt[0] - min_pt[0]
        footprint_y = max_pt[1] - min_pt[1]

        buildings.append({
            "name": name,
            "bbox_min": min_pt,
            "bbox_max": max_pt,
            "cx": center_x,
            "cy": center_y,
            "height": height,
            "roof_z": max_pt[2],
            "footprint_x": footprint_x,
            "footprint_y": footprint_y,
        })

    # Filter out ground plane and tiny objects
    buildings = [b for b in buildings if b["height"] > 3.0 and b["footprint_x"] < 500]
    buildings.sort(key=lambda b: b["height"], reverse=True)

    # Scene extent from building footprints
    all_min_x = min(b["bbox_min"][0] for b in buildings)
    all_max_x = max(b["bbox_max"][0] for b in buildings)
    all_min_y = min(b["bbox_min"][1] for b in buildings)
    all_max_y = max(b["bbox_max"][1] for b in buildings)

    print(f"\n=== Munich Scene Analysis (unmerged) ===")
    print(f"Building extent: x=[{all_min_x:.1f}, {all_max_x:.1f}], y=[{all_min_y:.1f}, {all_max_y:.1f}]")
    print(f"Scene size: {all_max_x - all_min_x:.0f}m x {all_max_y - all_min_y:.0f}m")
    print(f"Total building objects: {len(buildings)}")

    # Height distribution
    heights = [b["roof_z"] for b in buildings]
    print(f"Roof height stats: min={min(heights):.1f}m, max={max(heights):.1f}m, "
          f"median={np.median(heights):.1f}m, mean={np.mean(heights):.1f}m")

    print(f"\nTop 30 tallest buildings:")
    for b in buildings[:30]:
        print(f"  {b['name']:40s}  center=({b['cx']:7.1f}, {b['cy']:7.1f})  "
              f"roof={b['roof_z']:5.1f}m  size={b['footprint_x']:.0f}x{b['footprint_y']:.0f}m")

    return buildings, [all_min_x, all_min_y], [all_max_x, all_max_y]


def is_inside_building(x, y, buildings, margin=2.0):
    """Check if (x,y) is inside any building footprint."""
    for b in buildings:
        if (b["bbox_min"][0] - margin <= x <= b["bbox_max"][0] + margin and
            b["bbox_min"][1] - margin <= y <= b["bbox_max"][1] + margin):
            return True, b
    return False, None


def place_bs_3gpp_umi(buildings, scene_min, scene_max, num_bs=8, target_isd=200.0, bs_height=10.0):
    """Place BSes following 3GPP UMi guidelines.

    Strategy:
    1. Create hex grid candidates at target ISD
    2. For each candidate, snap to nearest building rooftop edge
    3. Ensure min height = bs_height
    4. Select num_bs with farthest-point sampling for coverage
    """
    cx = (scene_min[0] + scene_max[0]) / 2
    cy = (scene_min[1] + scene_max[1]) / 2

    # Focus on the denser urban core (not the entire scene extent)
    # Use a reasonable urban area
    core_radius_x = 400  # 800m x 600m core area
    core_radius_y = 300
    core_min_x = cx - core_radius_x
    core_max_x = cx + core_radius_x
    core_min_y = cy - core_radius_y
    core_max_y = cy + core_radius_y

    print(f"\n=== BS Placement (3GPP UMi-street canyon) ===")
    print(f"Target ISD: {target_isd}m, BS height: {bs_height}m")
    print(f"Core area: x=[{core_min_x:.0f}, {core_max_x:.0f}], y=[{core_min_y:.0f}, {core_max_y:.0f}]")

    # Generate hex grid
    dx = target_isd
    dy = target_isd * np.sqrt(3) / 2
    candidates = []

    n_rows = int((core_max_y - core_min_y) / dy) + 2
    n_cols = int((core_max_x - core_min_x) / dx) + 2

    for row in range(n_rows + 1):
        for col in range(n_cols + 1):
            x = core_min_x + col * dx + (row % 2) * dx / 2
            y = core_min_y + row * dy
            if core_min_x <= x <= core_max_x and core_min_y <= y <= core_max_y:
                candidates.append([x, y])

    print(f"Hex grid candidates: {len(candidates)}")

    # For each candidate, find best BS position near a building
    # UMi: BS typically at street level (lamppost) or low rooftop
    # We snap to building edge at rooftop height, capped at ~15m for UMi
    placed = []
    for cx_c, cy_c in candidates:
        # Find buildings within search radius
        best_score = -np.inf
        best_pos = None
        best_info = None

        for b in buildings:
            dist = np.sqrt((b["cx"] - cx_c)**2 + (b["cy"] - cy_c)**2)
            if dist > 80:  # search within 80m
                continue

            # Prefer buildings with roof 8-20m (typical UMi)
            roof = b["roof_z"]
            if roof < 5 or roof > 25:
                continue

            # Place on building edge facing outward from building center
            # Direction from building center to candidate
            dx_dir = cx_c - b["cx"]
            dy_dir = cy_c - b["cy"]
            norm = np.sqrt(dx_dir**2 + dy_dir**2) + 1e-6
            dx_dir /= norm
            dy_dir /= norm

            # Snap to building edge + small offset
            edge_x = b["bbox_max"][0] if dx_dir > 0 else b["bbox_min"][0]
            edge_y = b["bbox_max"][1] if dy_dir > 0 else b["bbox_min"][1]
            # Slightly outside building
            pos_x = edge_x + dx_dir * 1.0
            pos_y = edge_y + dy_dir * 1.0
            pos_z = max(roof, bs_height)

            # Score: prefer close to grid point, reasonable height
            grid_dist = np.sqrt((pos_x - cx_c)**2 + (pos_y - cy_c)**2)
            height_penalty = abs(roof - 12.0)  # prefer ~12m roofs
            score = -grid_dist * 0.5 - height_penalty * 2.0

            if score > best_score:
                best_score = score
                best_pos = [pos_x, pos_y, pos_z]
                best_info = {"building": b["name"], "roof": roof, "type": "rooftop"}

        if best_pos is None:
            # No suitable building, use lamppost
            # But check we're not inside a building
            inside, b = is_inside_building(cx_c, cy_c, buildings)
            if inside:
                # Move to building edge
                pos_x = b["bbox_max"][0] + 2.0
                pos_y = cy_c
            else:
                pos_x = cx_c
                pos_y = cy_c
            best_pos = [pos_x, pos_y, bs_height]
            best_info = {"building": None, "roof": bs_height, "type": "lamppost"}

        placed.append({"pos": best_pos, "info": best_info})

    # Farthest-point sampling to select num_bs
    positions = np.array([p["pos"][:2] for p in placed])

    # Start near scene center
    center_dists = np.sqrt((positions[:, 0] - cx)**2 + (positions[:, 1] - cy)**2)
    selected = [int(np.argmin(center_dists))]

    for _ in range(num_bs - 1):
        min_dists = np.full(len(placed), np.inf)
        for s in selected:
            d = np.sqrt((positions[:, 0] - positions[s, 0])**2 +
                        (positions[:, 1] - positions[s, 1])**2)
            min_dists = np.minimum(min_dists, d)
        for s in selected:
            min_dists[s] = -1
        selected.append(int(np.argmax(min_dists)))

    # Results
    final_bs = []
    for i, idx in enumerate(selected):
        p = placed[idx]
        final_bs.append(p)
        pos = p["pos"]
        info = p["info"]
        print(f"  BS {i}: ({pos[0]:8.2f}, {pos[1]:8.2f}, {pos[2]:5.1f}m) "
              f"type={info['type']}, roof={info['roof']:.1f}m, building={info['building']}")

    # ISD statistics
    print(f"\nPairwise ISD:")
    isds = []
    for i in range(len(final_bs)):
        for j in range(i + 1, len(final_bs)):
            pi = final_bs[i]["pos"]
            pj = final_bs[j]["pos"]
            d = np.sqrt((pi[0] - pj[0])**2 + (pi[1] - pj[1])**2)
            isds.append(d)
            print(f"  BS{i}-BS{j}: {d:.0f}m")

    # Nearest-neighbor ISD
    nn_isds = []
    for i in range(len(final_bs)):
        min_d = float('inf')
        for j in range(len(final_bs)):
            if i != j:
                pi = final_bs[i]["pos"]
                pj = final_bs[j]["pos"]
                d = np.sqrt((pi[0] - pj[0])**2 + (pi[1] - pj[1])**2)
                min_d = min(min_d, d)
        nn_isds.append(min_d)

    print(f"\nNearest-neighbor ISD: min={min(nn_isds):.0f}m, max={max(nn_isds):.0f}m, "
          f"mean={np.mean(nn_isds):.0f}m")

    return final_bs


def print_config(final_bs):
    """Print config.py update code."""
    print(f"\n=== Config Update (copy to src/config.py) ===")
    print("bs_positions: List[Tuple[float, float, float]] = field(default_factory=lambda: [")
    for i, bs in enumerate(final_bs):
        pos = bs["pos"]
        info = bs["info"]
        comment = f"  # BS{i}: {info['type']}"
        if info["building"]:
            comment += f", {info['building']}"
        print(f"    [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}],{comment}")
    print("])")

    print("\nbs_orientations: List[List[float]] = field(default_factory=lambda: [")
    for i in range(len(final_bs)):
        print(f"    [],  # BS{i}")
    print("])")


if __name__ == "__main__":
    buildings, scene_min, scene_max = extract_scene_geometry()
    final_bs = place_bs_3gpp_umi(buildings, scene_min, scene_max,
                                  num_bs=8, target_isd=200.0, bs_height=10.0)
    print_config(final_bs)
