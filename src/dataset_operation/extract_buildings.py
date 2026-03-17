"""Extract building footprints from Munich scene PLY meshes and cache as JSON.

Reads binary PLY files, extracts XY bounding boxes for each mesh element,
filters to building-like objects (height > 3m, footprint < 500m).

Usage:
    python -m src.dataset_operation.extract_buildings
"""
import json
import struct
from pathlib import Path

import numpy as np

MUNICH_MESH_DIR = Path("sionna-rt/build/lib/sionna/rt/scenes/munich/meshes")
CACHE_PATH = Path("assets/data/munich_buildings.json")


def parse_ply_vertices(ply_path: Path) -> np.ndarray:
    """Parse binary little-endian PLY and return (N, 3) float32 vertices."""
    with open(ply_path, "rb") as f:
        # Read header
        n_vertices = 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif line == "end_header":
                break

        if n_vertices == 0:
            return np.empty((0, 3), dtype=np.float32)

        # Read vertex data (3 floats = 12 bytes each)
        data = f.read(n_vertices * 12)
        vertices = np.frombuffer(data, dtype="<f4").reshape(n_vertices, 3)
        return vertices


def extract_buildings():
    """Extract building bounding boxes from all Munich PLY meshes."""
    if not MUNICH_MESH_DIR.exists():
        raise FileNotFoundError(f"Munich mesh dir not found: {MUNICH_MESH_DIR}")

    ply_files = sorted(MUNICH_MESH_DIR.glob("*.ply"))
    print(f"Found {len(ply_files)} PLY files")

    buildings = []
    for ply_path in ply_files:
        name = ply_path.stem
        try:
            verts = parse_ply_vertices(ply_path)
        except Exception:
            continue

        if len(verts) < 3:
            continue

        x_min, y_min, z_min = verts.min(axis=0)
        x_max, y_max, z_max = verts.max(axis=0)
        height = float(z_max - z_min)
        footprint_x = float(x_max - x_min)
        footprint_y = float(y_max - y_min)

        # Filter: building-like (height > 3m, not ground plane)
        if height < 3.0 or footprint_x > 500 or footprint_y > 500:
            continue

        buildings.append({
            "name": name,
            "x_min": float(x_min), "y_min": float(y_min),
            "x_max": float(x_max), "y_max": float(y_max),
            "height": height,
        })

    print(f"Extracted {len(buildings)} building footprints")

    # Save cache
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(buildings, f)
    print(f"Saved: {CACHE_PATH}")
    return buildings


if __name__ == "__main__":
    buildings = extract_buildings()
    heights = [b["height"] for b in buildings]
    print(f"Height stats: min={min(heights):.1f}, max={max(heights):.1f}, "
          f"median={np.median(heights):.1f}, mean={np.mean(heights):.1f}")
