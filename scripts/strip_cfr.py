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


def _load_cir_for_restore(snap_dir_str):
    """Load CIR from snapshot (module-level for pickling)."""
    d = np.load(os.path.join(snap_dir_str, "channels.npz"))
    return d["cir_a"], d["cir_tau"]


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


def restore_cfr(data_dir: Path, dry_run: bool = False, preset_override: str = None,
                 max_snapshots: int = None):
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

    import torch
    from concurrent.futures import ProcessPoolExecutor
    from tqdm.auto import tqdm

    snap_dirs = get_snapshot_dirs(data_dir)
    n_total = len(snap_dirs)
    print(f"Found {n_total} snapshots")

    # Filter: only those missing CFR
    to_restore = []
    for snap_dir in snap_dirs:
        npz_path = snap_dir / "channels.npz"
        if npz_path.exists():
            try:
                d = np.load(npz_path)
                if "cfr" not in d.files:
                    to_restore.append(snap_dir)
            except:
                pass
    if max_snapshots and len(to_restore) > max_snapshots:
        to_restore = to_restore[:max_snapshots]
    print(f"  Need CFR restore: {len(to_restore)}/{n_total}")

    if dry_run or not to_restore:
        return

    # Batch restore: load CIR in batches with ProcessPool, compute CFR with GPU bmm
    n_sc = cfg.num_subcarriers
    sc_spacing = cfg.subcarrier_spacing
    if n_sc % 2 == 0:
        start = -n_sc // 2
    else:
        start = -(n_sc - 1) // 2
    freqs = np.arange(start, start + n_sc, dtype=np.float32) * sc_spacing

    n_gpus = torch.cuda.device_count()
    devices = [torch.device(f"cuda:{i}") for i in range(n_gpus)] if n_gpus > 0 else [torch.device("cpu")]
    print(f"  Using {len(devices)} GPU(s)")

    batch_size = 10000
    n_batches = (len(to_restore) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        batch_start = bi * batch_size
        batch_dirs = to_restore[batch_start:batch_start + batch_size]
        print(f"\n  Batch {bi+1}/{n_batches} ({len(batch_dirs)} snaps)")

        # 1. Disk read
        with ProcessPoolExecutor(max_workers=64) as pool:
            cirs = list(tqdm(
                pool.map(_load_cir_for_restore, [str(d) for d in batch_dirs]),
                total=len(batch_dirs), desc="    Disk read", unit="snap",
            ))

        # Pad to max paths
        max_p = max(a.shape[-1] for a, _ in cirs)
        for idx in range(len(cirs)):
            a, tau = cirs[idx]
            if a.shape[-1] < max_p:
                a = np.pad(a, [(0,0)]*(a.ndim-1) + [(0, max_p - a.shape[-1])])
                tau = np.pad(tau, [(0,0), (0, max_p - tau.shape[-1])])
                cirs[idx] = (a, tau)

        all_a = np.stack([a for a, _ in cirs])
        all_tau = np.stack([t for _, t in cirs])
        B, n_ue, n_rx, n_tx, n_paths = all_a.shape

        # 2. GPU bmm
        flat_a = all_a.reshape(B * n_ue, n_rx, n_tx, n_paths)
        flat_tau = all_tau.reshape(B * n_ue, n_paths)
        total_samples = B * n_ue

        cfr_out = np.zeros((total_samples, n_rx, n_tx, n_sc), dtype=np.complex64)
        chunk = 1500
        gpu_pbar = tqdm(total=total_samples, desc="    GPU bmm", unit="sample")
        for c_start in range(0, total_samples, chunk * len(devices)):
            for di, dev in enumerate(devices):
                d_start = c_start + di * chunk
                d_end = min(d_start + chunk, total_samples)
                if d_start >= d_end:
                    break
                freqs_t = torch.tensor(freqs, dtype=torch.float32, device=dev)
                a_ch = torch.tensor(flat_a[d_start:d_end], dtype=torch.complex64, device=dev)
                tau_ch = torch.tensor(flat_tau[d_start:d_end], dtype=torch.float32, device=dev)
                exp_phase = torch.exp(-2j * torch.pi * tau_ch[:, :, None] * freqs_t[None, None, :])
                n_ch = a_ch.shape[0]
                cfr_flat = torch.bmm(a_ch.reshape(n_ch, n_rx * n_tx, n_paths), exp_phase)
                cfr_out[d_start:d_end] = cfr_flat.reshape(n_ch, n_rx, n_tx, n_sc).cpu().numpy()
                gpu_pbar.update(d_end - d_start)
                del a_ch, tau_ch, exp_phase, cfr_flat
        gpu_pbar.close()

        cfr_all = cfr_out.reshape(B, n_ue, n_rx, n_tx, n_sc)
        del all_a, all_tau, cfr_out

        # 3. Save back
        for idx, snap_dir in enumerate(tqdm(batch_dirs, desc="    Saving", unit="snap")):
            npz_path = snap_dir / "channels.npz"
            d = np.load(npz_path)
            arrays = {k: d[k] for k in d.files}
            arrays["cfr"] = cfr_all[idx]
            np.savez_compressed(npz_path, **arrays)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Strip/restore CFR from temporal channel data")
    parser.add_argument("data_dir", type=Path, help="Path to temporal channel data directory")
    parser.add_argument("--restore", action="store_true", help="Restore CFR from CIR")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    parser.add_argument("--preset", type=str, default=None, help="Override preset name")
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU IDs to use (default: all available)")
    parser.add_argument("--max-snapshots", type=int, default=None,
                        help="Max snapshots to restore (default: all)")
    args = parser.parse_args()

    if args.gpus is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in args.gpus)

    if not args.data_dir.exists():
        print(f"ERROR: {args.data_dir} does not exist")
        sys.exit(1)

    if args.restore:
        restore_cfr(args.data_dir, dry_run=args.dry_run, preset_override=args.preset,
                     max_snapshots=args.max_snapshots)
    else:
        strip_cfr(args.data_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
