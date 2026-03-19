"""Multi-GPU parallel dataset generation.

Splits snapshots across GPUs, each running a separate process.
Auto-manages tasks in assets/backlog.json.

Usage:
    # ELAA (7 GPUs, skip GPU6)
    python -m src.dataset_operation.generate_parallel \
        --preset munich_elaa_s_1k_15g --num_snapshots 100 --num_ue 100 \
        --gpus 0 1 2 3 4 5 7

    # 5G baseline (single GPU)
    python -m src.dataset_operation.generate_parallel \
        --preset munich_5g_mimo_3g5 --num_snapshots 100 --num_ue 100 \
        --gpus 1

    # Both in parallel (ELAA on 0,2,3,4,5,7 + 5G on 1)
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import math
import uuid
import fcntl
import time
from pathlib import Path
from datetime import datetime


# ── Backlog integration ────────────────────────────────────────────

BACKLOG = Path("assets/backlog.json")


def _read_backlog():
    if not BACKLOG.exists():
        return []
    try:
        text = BACKLOG.read_text().strip()
        if not text:
            return []
        items = json.loads(text)
        return items if isinstance(items, list) else []
    except (json.JSONDecodeError, ValueError):
        return []


def _locked_update(modify_fn):
    BACKLOG.parent.mkdir(parents=True, exist_ok=True)
    lock_path = BACKLOG.with_suffix(".lock")
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        items = _read_backlog()
        result = modify_fn(items)
        tmp = BACKLOG.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, BACKLOG)
        return result
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def task_start(preset, num_snapshots, num_ue, gpus):
    task_id = uuid.uuid4().hex[:8]
    name = f"experiment/datagen-{preset}"
    desc = (f"{preset}: {num_snapshots} snapshots × {num_ue} UE, "
            f"GPUs={gpus}")

    def modify(items):
        item = {
            "id": task_id,
            "name": name,
            "type": "experiment",
            "status": "in_progress",
            "description": desc,
            "session": "datagen",
            "active_in": ["datagen"],
            "started_at": datetime.now().isoformat(),
            "script": "src.dataset_operation.generate_parallel",
            "config": {
                "preset": preset,
                "num_snapshots": num_snapshots,
                "num_ue": num_ue,
                "gpus": gpus,
            },
            "args": [],
        }
        items.append(item)
        return item

    _locked_update(modify)
    print(f"📋 Task started [{task_id}] {name}")
    return task_id


def task_done(task_id, summary):
    def modify(items):
        for item in items:
            if item["id"] == task_id:
                item["status"] = "done"
                item["summary"] = summary
                item["finished_at"] = datetime.now().isoformat()
                return item
        return None

    _locked_update(modify)
    print(f"📋 Task done [{task_id}]: {summary}")


def task_fail(task_id, reason):
    def modify(items):
        for item in items:
            if item["id"] == task_id:
                item["status"] = "failed"
                item["reason"] = reason
                item["finished_at"] = datetime.now().isoformat()
                return item
        return None

    _locked_update(modify)
    print(f"📋 Task failed [{task_id}]: {reason}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, required=True)
    parser.add_argument("--num_snapshots", type=int, default=100)
    parser.add_argument("--num_ue", type=int, default=100)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--start_snapshot", type=int, default=0,
                        help="Resume from this snapshot index")
    parser.add_argument("--trajectories", type=str, default=None,
                        help="Path to trajectories.npz for temporal mode")
    args = parser.parse_args()

    num_gpus = len(args.gpus)
    total = args.num_snapshots - args.start_snapshot
    per_gpu = math.ceil(total / num_gpus)

    # Register task
    task_id = task_start(args.preset, args.num_snapshots, args.num_ue, args.gpus)
    t_start = time.time()

    # Signal handler: kill workers + mark task failed on SIGINT/SIGTERM
    procs = []

    def _cleanup(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n⚠ Received {sig_name}, killing workers...")
        for _, p, fd, _, _ in procs:
            p.kill()
            fd.close()
        elapsed = time.time() - t_start
        task_fail(task_id, f"Killed ({sig_name}) after {elapsed/60:.1f}min")
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    # Count workers per GPU
    from collections import Counter
    gpu_counts = Counter(args.gpus)

    print(f"\nPreset: {args.preset}")
    print(f"Snapshots: {args.start_snapshot}..{args.num_snapshots} ({total} total)")
    print(f"Workers: {num_gpus} total, ~{per_gpu} snapshots each")
    print(f"GPU allocation: {dict(gpu_counts)} (workers per GPU)")
    for gpu_id, count in sorted(gpu_counts.items()):
        print(f"  GPU {gpu_id}: {count} workers × ~{per_gpu} snapshots = ~{count * per_gpu} snapshots")
    print()

    # Build worker commands
    for i, gpu_id in enumerate(args.gpus):
        start = args.start_snapshot + i * per_gpu
        end = min(start + per_gpu, args.num_snapshots)
        if start >= end:
            break

        cmd = [
            sys.executable, "-m", "src.dataset_operation.generate_worker",
            "--preset", args.preset,
            "--snapshot_start", str(start),
            "--snapshot_end", str(end),
            "--num_ue", str(args.num_ue),
        ]
        if args.data_dir:
            cmd += ["--data_dir", args.data_dir]
        if args.trajectories:
            cmd += ["--trajectories", args.trajectories]

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        log_file = f"/tmp/datagen_{args.preset}_gpu{gpu_id}.log"
        print(f"  GPU {gpu_id}: snapshots [{start}, {end}) → {log_file}")

        log_fd = open(log_file, "w")
        p = subprocess.Popen(cmd, env=env, stdout=log_fd, stderr=subprocess.STDOUT)
        procs.append((gpu_id, p, log_fd, start, end))

    print(f"\nLaunched {len(procs)} workers. Monitor with:")
    print(f"  tail -f /tmp/datagen_{args.preset}_gpu*.log")
    print(f"  # or check progress:")
    print(f"  ls assets/data/channels_*/snapshot_*/channels.npz | wc -l")

    # Wait for all
    failed = []
    for gpu_id, p, log_fd, start, end in procs:
        rc = p.wait()
        log_fd.close()
        if rc != 0:
            failed.append((gpu_id, rc, start, end))
            print(f"  ✗ GPU {gpu_id} failed (rc={rc}), snapshots [{start}, {end})")
        else:
            print(f"  ✓ GPU {gpu_id} done, snapshots [{start}, {end})")

    elapsed = time.time() - t_start
    elapsed_str = f"{elapsed/60:.1f}min"

    if failed:
        fail_detail = ", ".join(f"GPU{g}(rc={rc})" for g, rc, _, _ in failed)
        task_fail(task_id, f"{len(failed)}/{len(procs)} workers failed: {fail_detail} ({elapsed_str})")
        sys.exit(1)
    else:
        # Merge metadata from all workers
        print(f"\nAll workers done. Merging metadata...")
        merge_cmd = [
            sys.executable, "-m", "src.dataset_operation.generate_worker",
            "--preset", args.preset,
            "--merge_only",
        ]
        if args.data_dir:
            merge_cmd += ["--data_dir", args.data_dir]
        subprocess.run(merge_cmd, check=True)

        task_done(task_id, f"{args.num_snapshots} snapshots × {args.num_ue} UE, "
                           f"{num_gpus} GPUs, {elapsed_str}")
        print("Done!")


if __name__ == "__main__":
    main()
