#!/usr/bin/env python3
"""
Web dashboard for experiment tracking — your own tensorboard.

Zero dependencies beyond stdlib. Serves a single-page app with:
  - Real-time training curves (loss, nmse_db, etc.)
  - Active process monitoring with PID liveness
  - Experiment grouping with tabs
  - Run comparison charts

Usage:
    python web/dashboard.py [port]     # default 8765
    # Then open http://localhost:8765
"""

import json
import os
import re as _re
import subprocess
import sys
import uuid as _uuid
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracker import ASSETS_DIR, RUNS_DIR, Tracker, pid_alive

RESULTS_DIR = ASSETS_DIR / "results"
CHECKPOINTS_DIR = ASSETS_DIR / "checkpoints"
BACKLOG_FILE = ASSETS_DIR / "backlog.json"


# ── Backlog helpers ──────────────────────────────────────────────────


def _load_backlog():
    """Load backlog items from JSON file."""
    if not BACKLOG_FILE.exists():
        return []
    try:
        with open(BACKLOG_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_backlog(items):
    """Atomic write backlog to JSON file."""
    BACKLOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = BACKLOG_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    os.replace(tmp, BACKLOG_FILE)


def _add_backlog_item(item):
    """Add item to backlog. Returns the created item."""
    items = _load_backlog()
    entry = {
        "id": _uuid.uuid4().hex[:8],
        "name": item.get("name", "Untitled"),
        "type": item.get("type", "experiment"),
        "description": item.get("description", ""),
        "script": item.get("script", ""),
        "args": item.get("args", []),
        "config": item.get("config", {}),
        "priority": item.get("priority", "normal"),
        "status": item.get("status", "queued"),
        "summary": item.get("summary", None),
        "session": item.get("session", None),
        "created_at": datetime.now().isoformat(),
        "run_id": None,
    }
    items.append(entry)
    _save_backlog(items)
    return entry


def _update_backlog_item(item_id, updates):
    """Update a backlog item by ID."""
    items = _load_backlog()
    for item in items:
        if item["id"] == item_id:
            item.update(updates)
            _save_backlog(items)
            return item
    return None


def _delete_backlog_item(item_id):
    """Delete a backlog item by ID."""
    items = _load_backlog()
    before = len(items)
    items = [i for i in items if i["id"] != item_id]
    if len(items) < before:
        _save_backlog(items)
        return True
    return False


# ── Data helpers ─────────────────────────────────────────────────────


def _scan_results():
    """Scan results/ and checkpoints/ for JSON evaluation/result files."""
    results = []
    # Scan assets/results/
    for base, prefix in [(RESULTS_DIR, "results"), (CHECKPOINTS_DIR, "checkpoints")]:
        if not base.exists():
            continue
        for p in sorted(base.rglob("*_result.json" if prefix == "checkpoints" else "*.json")):
            try:
                with open(p) as f:
                    data = json.load(f)
                rel = f"{prefix}/{p.relative_to(base)}"
                results.append({"path": rel, "data": data, "mtime": p.stat().st_mtime})
            except (json.JSONDecodeError, OSError):
                continue
    return results


def _scan_runs():
    """Scan all runs, return JSON-serializable list."""
    runs = []
    if not RUNS_DIR.exists():
        return runs

    for d in RUNS_DIR.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        meta_path = d / "run.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                info = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        # Latest metrics
        latest = None
        metrics_path = d / "metrics.jsonl"
        metrics_count = 0
        if metrics_path.exists():
            try:
                with open(metrics_path, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    if size > 0:
                        # Count lines (approx)
                        metrics_count = max(1, size // 80)
                        chunk = min(size, 4096)
                        f.seek(max(0, size - chunk))
                        data = f.read().decode("utf-8", errors="replace")
                        lines = [l for l in data.strip().split("\n") if l.strip()]
                        metrics_count = len(lines) if size <= 4096 else metrics_count
                        if lines:
                            try:
                                latest = json.loads(lines[-1])
                            except json.JSONDecodeError:
                                pass
            except OSError:
                pass

        # Mtime
        try:
            mtime = meta_path.stat().st_mtime
            if metrics_path.exists():
                mtime = max(mtime, metrics_path.stat().st_mtime)
        except OSError:
            mtime = 0

        # PID check
        alive = True
        if info.get("status") == "running":
            alive = pid_alive(info.get("pid"))

        runs.append({
            "info": info,
            "latest": latest,
            "mtime": mtime,
            "alive": alive,
            "metrics_count": metrics_count,
        })

    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def _read_metrics(run_id):
    """Read all metrics from a run's JSONL file."""
    path = RUNS_DIR / run_id / "metrics.jsonl"
    if not path.exists():
        return []
    entries = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        pass
    return entries


# ── System stats ─────────────────────────────────────────────────────


def _get_system_stats():
    """Gather CPU/memory (from /proc) and GPU stats (nvidia-smi)."""
    stats = {"cpu": {}, "memory": {}, "gpus": []}

    # CPU usage from /proc/stat (instantaneous — shows avg since boot)
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
            stats["cpu"]["load_1m"] = float(parts[0])
            stats["cpu"]["load_5m"] = float(parts[1])
        stats["cpu"]["cores"] = os.cpu_count() or 1
    except OSError:
        pass

    # Memory from /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                k, v = line.split(":")
                mem[k.strip()] = int(v.strip().split()[0])  # kB
            total = mem.get("MemTotal", 1)
            avail = mem.get("MemAvailable", 0)
            used = total - avail
            stats["memory"] = {
                "total_gb": round(total / 1048576, 1),
                "used_gb": round(used / 1048576, 1),
                "pct": round(100 * used / total, 1),
            }
    except (OSError, ValueError):
        pass

    # GPU from nvidia-smi (single call with all info)
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,uuid",
             "--format=csv,noheader,nounits"],
            timeout=3, text=True,
        )
        # Get PIDs using GPUs (single call)
        pids_out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=gpu_uuid,pid",
             "--format=csv,noheader"],
            timeout=3, text=True,
        )

        # Build uuid → pids map
        uuid_pids = {}
        for line in pids_out.strip().split("\n"):
            if "," in line:
                uuid, pid = line.split(",", 1)
                uuid_pids.setdefault(uuid.strip(), []).append(int(pid.strip()))

        # Get our tracker PIDs (lightweight: just read PIDs from run.json, skip full scan)
        our_pids = set()
        if RUNS_DIR.exists():
            for d in RUNS_DIR.iterdir():
                rj = d / "run.json"
                if rj.exists():
                    try:
                        with open(rj) as f:
                            info = json.load(f)
                        if info.get("status") == "running" and pid_alive(info.get("pid")):
                            our_pids.add(info["pid"])
                    except (json.JSONDecodeError, OSError):
                        pass

        for line in out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            idx, name, mem_used, mem_total, util, temp, uuid = parts
            mem_total_i = int(mem_total)
            mem_used_i = int(mem_used)
            gpu_pids = uuid_pids.get(uuid, [])
            is_ours = bool(our_pids & set(gpu_pids))
            stats["gpus"].append({
                "index": int(idx),
                "name": name,
                "mem_used_mb": mem_used_i,
                "mem_total_mb": mem_total_i,
                "mem_pct": round(100 * mem_used_i / max(mem_total_i, 1), 1),
                "util_pct": int(util),
                "temp_c": int(temp),
                "ours": is_ours,
            })
    except (FileNotFoundError, subprocess.SubprocessError):
        pass  # No nvidia-smi

    return stats


# ── HTTP Handler ─────────────────────────────────────────────────────


MIME_TYPES = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
    ".json": "application/json",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}


def _read_logs(run_id, tail=None):
    """Read output.log for a run. Optional tail=N for last N lines."""
    path = RUNS_DIR / run_id / "output.log"
    if not path.exists():
        return ""
    try:
        with open(path, "r", errors="replace") as f:
            if tail:
                # Read last N lines efficiently
                lines = f.readlines()
                return "".join(lines[-tail:])
            return f.read()
    except OSError:
        return ""


def _delete_run(run_id):
    """Delete a run directory."""
    import shutil
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        return False
    # Safety: don't delete running processes
    meta_path = run_dir / "run.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                info = json.load(f)
            if info.get("status") == "running" and pid_alive(info.get("pid")):
                return False  # refuse to delete live process
        except (json.JSONDecodeError, OSError):
            pass
    shutil.rmtree(run_dir)
    return True


def _delete_experiment(exp_name):
    """Delete all runs of an experiment."""
    deleted = 0
    if not RUNS_DIR.exists():
        return 0
    for d in list(RUNS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        meta_path = d / "run.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                info = json.load(f)
            if info.get("experiment") == exp_name:
                if _delete_run(d.name):
                    deleted += 1
        except (json.JSONDecodeError, OSError):
            continue
    return deleted


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress access logs

    def do_GET(self):
        path = urlparse(self.path).path

        # API endpoints
        if path == "/api/runs":
            self._respond_json(_scan_runs())
        elif path == "/api/system":
            self._respond_json(_get_system_stats())
        elif path.startswith("/api/logs/"):
            run_id = path[len("/api/logs/"):]
            qs = urlparse(self.path).query
            tail = None
            if "tail=" in qs:
                try:
                    tail = int(qs.split("tail=")[1].split("&")[0])
                except ValueError:
                    pass
            self._respond(200, "text/plain", _read_logs(run_id, tail))
        elif path.startswith("/api/metrics/"):
            run_id = path[len("/api/metrics/"):]
            self._respond_json(_read_metrics(run_id))
        elif path == "/api/results":
            self._respond_json(_scan_results())
        elif path == "/api/backlog":
            self._respond_json(_load_backlog())
        elif path == "/api/experiments":
            self._respond_json(Tracker.list_experiments())
        elif path.startswith("/api/experiment-meta/"):
            exp = path[len("/api/experiment-meta/"):]
            meta = Tracker.get_experiment_meta(exp)
            self._respond_json(meta or {})
        elif path == "/api/cleanup-stale":
            n = Tracker.cleanup_stale()
            self._respond_json({"ok": True, "cleaned": n})
        # Static files
        elif path == "/" or path == "/index.html":
            self._serve_static("index.html")
        elif path.startswith("/static/"):
            rel = path[len("/static/"):]
            self._serve_static(rel)
        else:
            self._respond(404, "text/plain", "Not found")

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._read_body()

        if path == "/api/backlog":
            try:
                item = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self._respond_json({"ok": False, "error": "Invalid JSON"})
                return
            entry = _add_backlog_item(item)
            self._respond_json({"ok": True, "item": entry})
        elif path.startswith("/api/backlog/") and path.endswith("/update"):
            item_id = path.split("/")[3]
            try:
                updates = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self._respond_json({"ok": False, "error": "Invalid JSON"})
                return
            result = _update_backlog_item(item_id, updates)
            self._respond_json({"ok": result is not None, "item": result})
        else:
            self._respond(404, "text/plain", "Not found")

    def do_DELETE(self):
        path = urlparse(self.path).path

        if path.startswith("/api/runs/"):
            run_id = path[len("/api/runs/"):]
            ok = _delete_run(run_id)
            self._respond_json({"ok": ok, "run_id": run_id})
        elif path.startswith("/api/backlog/"):
            item_id = path.split("/")[3]
            ok = _delete_backlog_item(item_id)
            self._respond_json({"ok": ok, "id": item_id})
        elif path.startswith("/api/experiments/"):
            exp = path[len("/api/experiments/"):]
            n = _delete_experiment(exp)
            self._respond_json({"ok": True, "deleted": n, "experiment": exp})
        else:
            self._respond(404, "text/plain", "Not found")

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            return self.rfile.read(length).decode("utf-8", errors="replace")
        return ""

    def _serve_static(self, rel_path):
        """Serve a file from the static directory."""
        # Prevent path traversal
        safe = Path(rel_path).name if "/" not in rel_path else rel_path
        file_path = STATIC_DIR / safe
        if not file_path.exists() or not file_path.is_file():
            self._respond(404, "text/plain", "Not found")
            return
        # Ensure it's within STATIC_DIR
        try:
            file_path.resolve().relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._respond(403, "text/plain", "Forbidden")
            return
        suffix = file_path.suffix
        content_type = MIME_TYPES.get(suffix, "application/octet-stream")
        with open(file_path, "rb") as f:
            body = f.read()
        self._respond(200, content_type, body)

    def _respond(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.end_headers()
        if isinstance(body, str):
            body = body.encode()
        self.wfile.write(body)

    def _respond_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, default=str).encode())


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\n  Dashboard: http://localhost:{port}")
    print(f"  Backlog: {BACKLOG_FILE}")
    print(f"  Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
