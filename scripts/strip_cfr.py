#!/usr/bin/env python3
"""Strip CFR arrays from temporal channel data to save disk space.

CFR can be reconstructed from CIR (cir_a, cir_tau) on-the-fly during training.
The dataset loaders auto-detect missing CFR and reconstruct via path-domain DFT.

Usage:
    # Strip CFR (saves ~95% disk per snapshot)
    python scripts/strip_cfr.py assets/data/channels_elaa_m_1k_28g_temporal

    # Restore CFR (recompute and save back)
    python scripts/strip_cfr.py --restore assets/data/channels_elaa_m_1k_28g_temporal

    # Dry run (show what would happen)
    python scripts/strip_cfr.py --dry-run assets/data/channels_elaa_m_1k_28g_temporal
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_snapshot_dirs(data_dir: Path):
    return sorted(data_dir.glob("snapshot_*"))


def strip_cfr(data_dir: Path, dry_run: bool = False):
    """Remove CFR from channels.npz, keeping only CIR."""
    snap_dirs = get_snapshot_dirs(data_dir)
    print(f"Found {len(snap_dirs)} snapshots in {data_dir}")

    saved_total = 0
    for i, snap_dir in enumerate(snap_dirs):
        npz_path = snap_dir / "channels.npz"
        if not npz_path.exists():
            continue

        old_size = npz_path.stat().st_size
        data = np.load(npz_path)

        if "cfr" not in data.files:
            print(f"  [{i+1}/{len(snap_dirs)}] {snap_dir.name}: no CFR, skipping")
            continue

        if "cir_a" not in data.files or "cir_tau" not in data.files:
            print(f"  [{i+1}/{len(snap_dirs)}] {snap_dir.name}: no CIR data, skipping (cannot reconstruct)")
            continue

        if dry_run:
            print(f"  [{i+1}/{len(snap_dirs)}] {snap_dir.name}: would strip CFR ({old_size/1024**2:.0f} MB)")
            saved_total += old_size  # approximate
            continue

        # Save without CFR
        arrays = {k: data[k] for k in data.files if k != "cfr"}
        np.savez_compressed(npz_path, **arrays)

        new_size = npz_path.stat().st_size
        saved = old_size - new_size
        saved_total += saved

        if (i + 1) % 50 == 0 or i == 0:
            print(
                f"  [{i+1}/{len(snap_dirs)}] {snap_dir.name}: "
                f"{old_size/1024**2:.0f} MB → {new_size/1024**2:.0f} MB "
                f"(saved {saved/1024**2:.0f} MB)"
            )

    print(f"\nTotal saved: {saved_total/1024**3:.1f} GB")


def restore_cfr(data_dir: Path, dry_run: bool = False, preset_override: str = None):
    """Recompute CFR from CIR and save back into channels.npz."""
    from src.config import SceneConfig

    # Get preset from arg, progress.json, or trajectory_info.json
    preset = preset_override
    if not preset:
        for fname in ["progress.json", "trajectory_info.json"]:
            fpath = data_dir / fname
            if fpath.exists():
                with open(fpath) as f:
                    preset = json.load(f).get("preset")
                if preset:
                    break
        # Also check shared_trajectories
        if not preset:
            traj_info = Path("assets/data/shared_trajectories/trajectory_info.json")
            if traj_info.exists():
                with open(traj_info) as f:
                    preset = json.load(f).get("preset")

    if not preset:
        # Infer from directory name: channels_elaa_m_1k_28g_temporal → munich_elaa_m_1k_28g
        dirname = data_dir.name.replace("channels_", "").replace("_temporal", "")
        preset = f"munich_{dirname}"
        print(f"  Inferred preset: {preset}")
    cfg = SceneConfig.from_preset(preset)
    print(f"Preset: {preset}, {cfg.num_subcarriers} subcarriers, "
          f"{cfg.subcarrier_spacing:.0f} Hz spacing")

    from src.dataset_operation.utils import cir_to_cfr

    snap_dirs = get_snapshot_dirs(data_dir)
    print(f"Found {len(snap_dirs)} snapshots")

    for i, snap_dir in enumerate(snap_dirs):
        npz_path = snap_dir / "channels.npz"
        if not npz_path.exists():
            continue

        data = np.load(npz_path)
        if "cfr" in data.files:
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(snap_dirs)}] {snap_dir.name}: already has CFR")
            continue

        if dry_run:
            print(f"  [{i+1}/{len(snap_dirs)}] {snap_dir.name}: would restore CFR")
            continue

        cfr = cir_to_cfr(
            data["cir_a"], data["cir_tau"],
            cfg.num_subcarriers, cfg.subcarrier_spacing,
        )
        arrays = {k: data[k] for k in data.files}
        arrays["cfr"] = cfr
        np.savez_compressed(npz_path, **arrays)

        if (i + 1) % 50 == 0 or i == 0:
            new_size = npz_path.stat().st_size
            print(f"  [{i+1}/{len(snap_dirs)}] {snap_dir.name}: restored ({new_size/1024**2:.0f} MB)")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Strip/restore CFR from temporal channel data")
    parser.add_argument("data_dir", type=Path, help="Path to temporal channel data directory")
    parser.add_argument("--restore", action="store_true", help="Restore CFR from CIR")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("--preset", type=str, default=None, help="Override preset name")
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: {args.data_dir} does not exist")
        sys.exit(1)

    if args.restore:
        restore_cfr(args.data_dir, dry_run=args.dry_run, preset_override=args.preset)
    else:
        strip_cfr(args.data_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
