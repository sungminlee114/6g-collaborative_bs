#!/usr/bin/env python3
"""Convert temporal snapshot directories (npz per snapshot) to single HDF5 file.

Usage:
    python scripts/npz_to_hdf5.py assets/data/channels_elaa_m_1k_28g_temporal
    python scripts/npz_to_hdf5.py --all          # convert all *_temporal dirs
    python scripts/npz_to_hdf5.py --delete-npz   # remove snapshot dirs after conversion

Output: {data_dir}/channels.h5 with datasets:
    cir_a:   (N_snapshots, N_UE, n_rx, n_tx, max_paths) complex64
    cir_tau: (N_snapshots, N_UE, max_paths) float32
"""
import argparse
import os
import shutil
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from tqdm.auto import tqdm


def _load_snap(args):
    """Load one snapshot npz (module-level for pickling)."""
    snap_dir, snap_idx = args
    path = os.path.join(snap_dir, f"snapshot_{snap_idx:04d}", "channels.npz")
    if not os.path.exists(path):
        return snap_idx, None, None
    d = np.load(path)
    return snap_idx, d["cir_a"], d["cir_tau"]


def convert_dir(data_dir: Path, delete_npz: bool = False):
    """Convert one temporal data directory to HDF5."""
    data_dir = Path(data_dir)
    snap_dirs = sorted(data_dir.glob("snapshot_*"))
    n_snaps = len(snap_dirs)
    if n_snaps == 0:
        print(f"  No snapshots in {data_dir}")
        return

    h5_path = data_dir / "channels.h5"
    if h5_path.exists():
        print(f"  {h5_path} already exists, skipping (delete to reconvert)")
        return

    # Peek at first snapshot to get shapes
    first = np.load(snap_dirs[0] / "channels.npz")
    cir_a_shape = first["cir_a"].shape  # (N_UE, n_rx, n_tx, n_paths)
    cir_tau_shape = first["cir_tau"].shape  # (N_UE, n_paths)
    n_ue, n_rx, n_tx, _ = cir_a_shape

    print(f"  {data_dir.name}: {n_snaps} snapshots, {n_ue} UEs, "
          f"({n_rx}×{n_tx}) ant")

    # Find max paths across all snapshots (they vary slightly)
    print("  Scanning path counts...")
    max_paths = 0
    snap_indices = [int(d.name.split("_")[1]) for d in snap_dirs]

    # Quick scan with multiprocess
    def _get_n_paths(args):
        snap_dir_str, idx = args
        path = os.path.join(snap_dir_str, f"snapshot_{idx:04d}", "channels.npz")
        d = np.load(path)
        return d["cir_a"].shape[-1]

    data_dir_str = str(data_dir)
    with ProcessPoolExecutor(max_workers=32) as pool:
        path_counts = list(pool.map(
            _get_n_paths,
            [(data_dir_str, i) for i in snap_indices],
        ))
    max_paths = max(path_counts)
    print(f"  Max paths: {max_paths} (range {min(path_counts)}-{max_paths})")

    # Create HDF5
    print(f"  Creating {h5_path}...")
    with h5py.File(h5_path, "w") as f:
        ds_a = f.create_dataset(
            "cir_a",
            shape=(n_snaps, n_ue, n_rx, n_tx, max_paths),
            dtype="complex64",
            chunks=(min(100, n_snaps), 1, n_rx, n_tx, max_paths),
            compression="gzip",
            compression_opts=1,  # fast compression
        )
        ds_tau = f.create_dataset(
            "cir_tau",
            shape=(n_snaps, n_ue, max_paths),
            dtype="float32",
            chunks=(min(100, n_snaps), 1, max_paths),
            compression="gzip",
            compression_opts=1,
        )

        # Store metadata
        f.attrs["n_snapshots"] = n_snaps
        f.attrs["n_ue"] = n_ue
        f.attrs["n_rx"] = n_rx
        f.attrs["n_tx"] = n_tx
        f.attrs["max_paths"] = max_paths

        # Load in batches with multiprocess, write to HDF5
        batch_size = 200
        for batch_start in tqdm(range(0, n_snaps, batch_size), desc="Converting", unit="batch"):
            batch_end = min(batch_start + batch_size, n_snaps)
            batch_indices = snap_indices[batch_start:batch_end]

            with ProcessPoolExecutor(max_workers=32) as pool:
                results = list(pool.map(
                    _load_snap,
                    [(data_dir_str, i) for i in batch_indices],
                ))

            for local_idx, (snap_idx, a, tau) in enumerate(results):
                if a is None:
                    continue
                global_idx = batch_start + local_idx
                n_p = a.shape[-1]
                ds_a[global_idx, :, :, :, :n_p] = a
                ds_tau[global_idx, :, :n_p] = tau

    file_size = h5_path.stat().st_size / 1024**3
    print(f"  Done: {h5_path} ({file_size:.1f} GB)")

    if delete_npz:
        print(f"  Deleting {n_snaps} snapshot directories...")
        for d in tqdm(snap_dirs, desc="Deleting", unit="dir"):
            shutil.rmtree(d)
        print(f"  Freed ~{n_snaps * 12 / 1024:.0f} GB")


def main():
    parser = argparse.ArgumentParser(description="Convert npz snapshots to HDF5")
    parser.add_argument("data_dir", nargs="?", type=Path, help="Temporal data directory")
    parser.add_argument("--all", action="store_true", help="Convert all *_temporal dirs")
    parser.add_argument("--delete-npz", action="store_true", help="Delete npz after conversion")
    args = parser.parse_args()

    if args.all:
        dirs = sorted(Path("assets/data").glob("channels_*_temporal"))
    elif args.data_dir:
        dirs = [args.data_dir]
    else:
        parser.print_help()
        sys.exit(1)

    for d in dirs:
        print(f"\n{'═'*60}")
        print(f"  {d}")
        print(f"{'═'*60}")
        convert_dir(d, delete_npz=args.delete_npz)


if __name__ == "__main__":
    main()
