"""
Process-level experiment tracker — zero coordination, auto-registration.

Each process creates its own run directory. Registration happens on start,
deregistration on exit (normal, exception, SIGTERM, SIGKILL detected by PID).
No external actor (Claude, CLI) needs to update anything.

Usage:
    from src.tracker import Tracker

    # Context manager (recommended) — auto start/finish, catches crashes
    with Tracker("phase1/maml", config={"lr": 1e-3, "epochs": 100}) as run:
        for epoch in range(100):
            loss = train_epoch(...)
            run.log(epoch=epoch, loss=loss)
        run.set_result(nmse_db=-12.5)

    # Manual
    run = Tracker("exp_name").start()
    run.log(epoch=1, loss=0.5)
    run.finish(result={"nmse_db": -12.5})

    # Decorator
    @Tracker.track("exp_name")
    def train(run, model, data):
        run.log(loss=0.5)

    # Integration with train_local:
    with Tracker("phase1/maml") as run:
        result = train_local(model, loader, ..., tracker=run)
"""

import atexit
import io
import json
import os
import signal
import sys
import tempfile
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
RUNS_DIR = ASSETS_DIR / "runs"
EXPERIMENTS_DIR = RUNS_DIR / ".experiments"
NOTIFY_PATH = RUNS_DIR / ".notify"


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _atomic_json(path: Path, data: dict):
    """Write JSON atomically (tempfile + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", delete=False
    )
    try:
        json.dump(data, fd, indent=2, ensure_ascii=False, default=str)
        fd.write("\n")
        fd.close()
        os.replace(fd.name, path)
    except Exception:
        fd.close()
        try:
            os.unlink(fd.name)
        except OSError:
            pass
        raise


def _notify():
    """Touch sentinel for efficient dashboard watching."""
    try:
        NOTIFY_PATH.parent.mkdir(parents=True, exist_ok=True)
        NOTIFY_PATH.touch()
    except OSError:
        pass


def pid_alive(pid: int) -> bool:
    """Check if a process is still running."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just can't signal it


def _git_commit() -> Optional[str]:
    """Get current git commit hash (short), or None."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=2, text=True,
        ).strip()
    except Exception:
        return None


@dataclass
class ExperimentInfo:
    """Experiment-level metadata shared across all runs."""
    name: str
    purpose: Optional[str] = None        # 실험 목적 (왜 이 실험을 하는가)
    variables: Optional[dict] = None     # {"independent": [...], "dependent": [...], "controlled": [...]}
    hypothesis: Optional[str] = None     # 가설 (예: "LoRA r=64 > r=8 on test BS")
    eval_criteria: Optional[str] = None  # 가설 평가 방법 (예: "test BS avg NMSE < -17 dB")
    git_commit: Optional[str] = None     # 자동 기록
    updated_at: Optional[str] = None


@dataclass
class RunInfo:
    """Serializable metadata for a single run."""
    id: str
    name: str
    experiment: str
    status: str  # running | done | failed
    started_at: str
    ended_at: Optional[str] = None
    tags: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    result: Optional[dict] = None
    error: Optional[str] = None
    pid: Optional[int] = None
    version: int = 0


# ── Active tracker registry (for signal handler cleanup) ─────────────
_active_trackers: list["Tracker"] = []


def _cleanup_all():
    """atexit handler — finish all active trackers."""
    for t in list(_active_trackers):
        t.finish("failed", error="process exited without finishing")


_prev_handlers = {}


def _signal_handler(signum, frame):
    """Signal handler — finish trackers then re-raise."""
    sig_name = signal.Signals(signum).name
    for t in list(_active_trackers):
        t.finish("failed", error=f"killed by {sig_name}")
    # Call previous handler if it was a callable (not SIG_DFL/SIG_IGN)
    prev = _prev_handlers.get(signum)
    if callable(prev) and prev not in (signal.SIG_DFL, signal.SIG_IGN):
        prev(signum, frame)
    # Re-raise with default handler
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


_TRACKED_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)


def _install_signal_handlers():
    """Install signal handlers, chaining with any existing ones.
    Safe to call multiple times — re-registers if overridden by other libs."""
    for sig in _TRACKED_SIGNALS:
        try:
            current = signal.getsignal(sig)
            if current is _signal_handler:
                continue  # already installed
            _prev_handlers[sig] = current
            signal.signal(sig, _signal_handler)
        except (OSError, ValueError):
            pass  # can't set signal handler in non-main thread


# Register at import time (early)
atexit.register(_cleanup_all)
_install_signal_handlers()


class _Tee:
    """Duplicate writes to both the original stream and a log file."""

    def __init__(self, original, log_file):
        self._original = original
        self._log = log_file

    def write(self, data):
        self._original.write(data)
        try:
            self._log.write(data)
            self._log.flush()
        except (OSError, ValueError):
            pass

    def flush(self):
        self._original.flush()
        try:
            self._log.flush()
        except (OSError, ValueError):
            pass

    def __getattr__(self, name):
        return getattr(self._original, name)


class Tracker:
    """Process-level experiment tracker. Each instance = one run."""

    def __init__(self, name: str, *, experiment: str = None,
                 tags: dict = None, config: dict = None,
                 purpose: str = None, variables: dict = None,
                 hypothesis: str = None, eval_criteria: str = None,
                 capture_output: bool = False):
        self.name = name

        # Derive experiment from name: "phase1/maml" → "phase1"
        if experiment is not None:
            self.experiment = experiment
        elif "/" in name:
            self.experiment = name.split("/")[0]
        else:
            self.experiment = name.split("_")[0]

        self.tags = dict(tags) if tags else {}
        self.config = dict(config) if config else {}
        self.purpose = purpose
        self.variables = variables
        self.hypothesis = hypothesis
        self.eval_criteria = eval_criteria
        self._capture = capture_output

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{ts}_{name.replace('/', '_')}"
        self.run_dir = RUNS_DIR / self.run_id
        self._started = False
        self._finished = False
        self._result = None
        self._git_commit_cached = None
        self._version = 0
        self._metrics_path = None
        self._log_path = None
        self._log_file = None
        self._orig_stdout = None
        self._orig_stderr = None
        self._started_at = None

    def start(self):
        """Begin tracking. Creates run dir, registers PID, sets up cleanup."""
        if self._started:
            return self
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._started = True
        self._started_at = _now()
        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._log_path = self.run_dir / "output.log"
        self._save("running")
        _active_trackers.append(self)
        # Re-register signal handlers (PyTorch/etc. may have overridden them)
        _install_signal_handlers()

        # Capture stdout/stderr to output.log
        if self._capture:
            self._log_file = open(self._log_path, "a")
            self._orig_stdout = sys.stdout
            self._orig_stderr = sys.stderr
            sys.stdout = _Tee(sys.stdout, self._log_file)
            sys.stderr = _Tee(sys.stderr, self._log_file)

        return self

    def log(self, step: int = None, **metrics):
        """Append metrics to JSONL log. Auto-starts if needed.

        The metrics.jsonl file's mtime serves as a heartbeat —
        the dashboard uses it to detect stale processes.
        """
        if not self._started:
            self.start()
        entry = {"t": _now()}
        if step is not None:
            entry["step"] = step
        entry.update(metrics)
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        _notify()

    def log_message(self, msg: str):
        """Write a timestamped message to output.log (always works)."""
        if not self._started:
            self.start()
        with open(self._log_path, "a") as f:
            f.write(f"[{_now()}] {msg}\n")

    def set_result(self, result: dict = None, **kwargs):
        """Set final result summary (shown in dashboard)."""
        if result:
            self._result = result
        if kwargs:
            self._result = {**(self._result or {}), **kwargs}

    def finish(self, status: str = "done", *, result: dict = None,
               error: str = None):
        """Mark run as completed/failed. Idempotent."""
        if self._finished:
            return
        self._finished = True
        if result:
            self._result = result

        # Write error traceback to log
        if error and self._log_path:
            try:
                with open(self._log_path, "a") as f:
                    f.write(f"\n[{_now()}] FINISHED status={status}\n")
                    if error:
                        f.write(f"[{_now()}] ERROR: {error}\n")
            except OSError:
                pass

        # Restore stdout/stderr
        if self._capture and self._orig_stdout:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
        if self._log_file:
            try:
                self._log_file.close()
            except OSError:
                pass

        self._save(status, error=error)
        # Remove from active registry
        if self in _active_trackers:
            _active_trackers.remove(self)

    def _save(self, status: str, error: str = None):
        self._version += 1
        info = RunInfo(
            id=self.run_id, name=self.name, experiment=self.experiment,
            status=status, started_at=self._started_at,
            ended_at=_now() if status != "running" else None,
            tags=self.tags, config=self.config,
            result=self._result,
            error=error, pid=os.getpid(), version=self._version,
        )
        _atomic_json(self.run_dir / "run.json", asdict(info))

        # Write experiment-level metadata (once per start, or if fields provided)
        if self._version == 1:
            self._git_commit_cached = _git_commit()
            self._save_experiment_meta()

        _notify()

    def _save_experiment_meta(self):
        """Write/update shared experiment metadata."""
        has_meta = any(x is not None for x in
                       [self.purpose, self.variables, self.hypothesis,
                        self.eval_criteria])
        if not has_meta and not self._git_commit_cached:
            return
        exp_path = EXPERIMENTS_DIR / f"{self.experiment}.json"
        # Merge with existing (don't overwrite fields from earlier runs)
        existing = {}
        if exp_path.exists():
            try:
                with open(exp_path) as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        meta = ExperimentInfo(
            name=self.experiment,
            purpose=self.purpose or existing.get("purpose"),
            variables=self.variables or existing.get("variables"),
            hypothesis=self.hypothesis or existing.get("hypothesis"),
            eval_criteria=self.eval_criteria or existing.get("eval_criteria"),
            git_commit=self._git_commit_cached or existing.get("git_commit"),
            updated_at=_now(),
        )
        _atomic_json(exp_path, asdict(meta))

    # ── Context manager ──────────────────────────────────────────────

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.finish("failed", error=f"{exc_type.__name__}: {exc_val}")
        else:
            self.finish("done")
        return False

    # ── Decorator ────────────────────────────────────────────────────

    @staticmethod
    def track(name: str, **tracker_kwargs):
        """Decorator: wraps function in a tracked run.

        @Tracker.track("phase1/maml")
        def train(run, model, data):
            run.log(loss=0.5)
        """
        def decorator(fn):
            def wrapper(*args, **kwargs):
                with Tracker(name, **tracker_kwargs) as run:
                    return fn(run, *args, **kwargs)
            wrapper.__name__ = fn.__name__
            wrapper.__doc__ = fn.__doc__
            return wrapper
        return decorator

    # ── Static helpers (used by dashboard) ───────────────────────────

    @staticmethod
    def scan_runs(experiment: str = None) -> list[dict]:
        """Scan runs directory. Returns list of run dicts with liveness info.

        Each dict has: info (RunInfo fields), latest_metrics, last_activity,
        alive (bool — PID check for running processes).
        """
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

            if experiment and info.get("experiment") != experiment:
                continue

            # Read latest metrics (last line of JSONL)
            latest = None
            metrics_path = d / "metrics.jsonl"
            if metrics_path.exists():
                try:
                    with open(metrics_path, "rb") as f:
                        f.seek(0, 2)
                        size = f.tell()
                        if size > 0:
                            chunk = min(size, 4096)
                            f.seek(max(0, size - chunk))
                            lines = f.read().decode("utf-8", errors="replace")
                            lines = [l for l in lines.strip().split("\n") if l]
                            if lines:
                                try:
                                    latest = json.loads(lines[-1])
                                except json.JSONDecodeError:
                                    pass
                except OSError:
                    pass

            # Last filesystem activity
            mtime = meta_path.stat().st_mtime
            if metrics_path.exists():
                mtime = max(mtime, metrics_path.stat().st_mtime)

            # PID liveness check (only for "running" status)
            alive = True
            if info.get("status") == "running":
                alive = pid_alive(info.get("pid"))

            runs.append({
                "info": info,
                "latest_metrics": latest,
                "last_activity": mtime,
                "alive": alive,
                "dir": str(d),
            })

        runs.sort(key=lambda r: r["last_activity"], reverse=True)
        return runs

    @staticmethod
    def read_all_metrics(run_id: str) -> list[dict]:
        """Read all metric entries from a run's JSONL."""
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

    @staticmethod
    def mark_dead(run_id: str):
        """Mark a stale run as failed (PID dead but status=running)."""
        meta_path = RUNS_DIR / run_id / "run.json"
        if not meta_path.exists():
            return False
        try:
            with open(meta_path) as f:
                info = json.load(f)
            if info.get("status") == "running" and not pid_alive(info.get("pid")):
                info["status"] = "failed"
                info["ended_at"] = _now()
                info["error"] = f"process died (PID {info.get('pid')} not found)"
                info["version"] = info.get("version", 0) + 1
                _atomic_json(meta_path, info)
                _notify()
                return True
        except (json.JSONDecodeError, OSError):
            pass
        return False

    @staticmethod
    def cleanup_stale() -> int:
        """Find all stale runs (running but PID dead) and mark as failed.
        Returns count of cleaned runs."""
        count = 0
        if not RUNS_DIR.exists():
            return 0
        for d in RUNS_DIR.iterdir():
            if not d.is_dir() or d.name.startswith("."):
                continue
            if Tracker.mark_dead(d.name):
                count += 1
        return count

    @staticmethod
    def get_experiment_meta(experiment: str) -> Optional[dict]:
        """Read experiment-level metadata."""
        path = EXPERIMENTS_DIR / f"{experiment}.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def list_experiments() -> dict[str, dict]:
        """Return all experiment metadata as {name: meta_dict}."""
        result = {}
        if not EXPERIMENTS_DIR.exists():
            return result
        for p in EXPERIMENTS_DIR.glob("*.json"):
            try:
                with open(p) as f:
                    data = json.load(f)
                result[data.get("name", p.stem)] = data
            except (json.JSONDecodeError, OSError):
                continue
        return result
