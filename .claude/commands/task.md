---
description: "Quick task logging. Usage: /task start|done|log|fail [name] [desc/summary]"
allowed-tools: Read, Write, Edit, Bash
---

# Quick Task Logger

Lightweight wrapper around `assets/backlog.json` for Claude to quickly log work.
Each Claude Code panel gets a unique session ID.

## Session ID

Generate or reuse a per-panel session ID using a simple, crash-safe approach:

- Use `os.getpid()` of the current Python process as a seed to find a stable session file.
- Walk `$PPID` once (no `/proc` traversal) — if `CLAUDE_SESSION_ID` env var exists, use it directly.
- Fallback: hash the TTY or use a shared session file.
- Session files expire after 24h.

## User argument: $ARGUMENTS

### `start <name> [description]`
- Add item to `assets/backlog.json` with `status: "in_progress"`, include `session` ID.
- Infer `type` from name prefix: `implement/` → implement, `fix/` → fix, `research/` → research, `phase.../` or `E.../` → experiment, `config/` → config, `review/` → review.
- Example: `/task start implement/right-sidebar "Right sidebar for backlog"`

### `done [summary]`
- Find the most recent `in_progress` item with matching session ID.
- Update `status` to `"done"`, set `summary`.
- If no summary given, generate a one-line summary from context.

### `log <name> [summary]`
- One-shot: add item as `done` immediately with summary.
- For logging completed work retroactively.
- Example: `/task log fix/chart-flicker "Fixed chart flicker on poll by splitting render"`

### `fail [reason]`
- Mark current `in_progress` item (matching session) as `failed` with reason.

## Implementation

Use Python via Bash tool. The full script below handles all subcommands.

**Key safety measures:**
1. File locking (`fcntl.flock`) around read-modify-write to prevent corruption from concurrent access (e.g., dashboard).
2. Graceful JSON recovery — if `backlog.json` is corrupt, back it up and start fresh.
3. No `/proc` filesystem access — session ID derived from env vars or TTY.
4. Atomic writes via temp file + `os.replace`.

```python
#!/usr/bin/env python3
"""
Usage (called via Bash tool):
  python3 -c '<this script>' start 'implement/foo' 'description here'
  python3 -c '<this script>' done 'summary here'
  python3 -c '<this script>' log 'fix/bar' 'summary here'
  python3 -c '<this script>' fail 'reason here'
"""
import json, uuid, os, sys, time, fcntl, hashlib, shutil
from pathlib import Path
from datetime import datetime

BACKLOG = Path("assets/backlog.json")

# ── Session ID (crash-safe, no /proc walking) ──────────────────────
def _get_session_id():
    # 1) Env var override (if Claude Code ever exposes one)
    env_id = os.environ.get("CLAUDE_SESSION_ID")
    if env_id:
        return env_id

    # 2) Derive a stable key from the TTY (unique per terminal/panel)
    try:
        tty = os.ttyname(sys.stdin.fileno())
        key = hashlib.md5(tty.encode()).hexdigest()[:8]
    except Exception:
        # No TTY (piped execution) — fall back to parent PID
        key = str(os.getppid())

    session_file = Path(f"/tmp/claude_task_session_{key}")

    # Reuse if fresh (<24h)
    try:
        if session_file.exists() and (time.time() - session_file.stat().st_mtime) < 86400:
            content = session_file.read_text().strip()
            if content:
                return content
    except OSError:
        pass

    # Generate new
    sid = uuid.uuid4().hex[:6]
    try:
        session_file.write_text(sid)
    except OSError:
        pass  # non-fatal — we still have the ID in memory
    return sid

SESSION = _get_session_id()

# ── Safe JSON read with recovery ───────────────────────────────────
def read_backlog():
    if not BACKLOG.exists():
        return []
    try:
        text = BACKLOG.read_text().strip()
        if not text:
            return []
        items = json.loads(text)
        if not isinstance(items, list):
            raise ValueError("backlog root is not a list")
        return items
    except (json.JSONDecodeError, ValueError) as e:
        # Back up corrupt file and start fresh
        backup = BACKLOG.with_suffix(f".corrupt.{int(time.time())}.json")
        shutil.copy2(BACKLOG, backup)
        print(f"⚠ backlog.json was corrupt ({e}), backed up to {backup.name}")
        return []

# ── Atomic write with file lock ────────────────────────────────────
def write_backlog(items):
    BACKLOG.parent.mkdir(parents=True, exist_ok=True)
    lock_path = BACKLOG.with_suffix(".lock")
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        tmp = BACKLOG.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, BACKLOG)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

# ── Locked read-modify-write cycle ─────────────────────────────────
def locked_update(modify_fn):
    """Read backlog under lock, apply modify_fn, write back."""
    BACKLOG.parent.mkdir(parents=True, exist_ok=True)
    lock_path = BACKLOG.with_suffix(".lock")
    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        items = read_backlog()
        result = modify_fn(items)
        tmp = BACKLOG.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")
        os.replace(tmp, BACKLOG)
        return result
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

# ── Type inference ─────────────────────────────────────────────────
def infer_type(name):
    prefixes = {
        "implement": "implement", "fix": "fix", "research": "research",
        "config": "config", "review": "review",
    }
    first = name.split("/")[0] if "/" in name else ""
    if first in prefixes:
        return prefixes[first]
    if first.startswith("phase") or first.startswith("E"):
        return "experiment"
    return "task"

# ── Subcommands ────────────────────────────────────────────────────
def cmd_start(args):
    if not args:
        print("✗ Usage: /task start <name> [description]")
        return
    name = args[0]
    desc = " ".join(args[1:]) if len(args) > 1 else ""

    def modify(items):
        item = {
            "id": uuid.uuid4().hex[:8],
            "name": name,
            "type": infer_type(name),
            "status": "in_progress",
            "description": desc,
            "session": SESSION,
            "started_at": datetime.now().isoformat(),
            "script": None, "config": {}, "args": [],
        }
        items.append(item)
        return item

    item = locked_update(modify)
    print(f"✓ Started [{item['id']}] {name} (session: {SESSION})")

def cmd_done(args):
    summary = " ".join(args) if args else ""

    def modify(items):
        # Find most recent in_progress for this session
        for item in reversed(items):
            if item.get("status") == "in_progress" and item.get("session") == SESSION:
                item["status"] = "done"
                item["summary"] = summary or "(no summary)"
                item["finished_at"] = datetime.now().isoformat()
                return item
        return None

    item = locked_update(modify)
    if item:
        print(f"✓ Done [{item['id']}] {item.get('name', '?')}: {item.get('summary')}")
    else:
        print(f"✗ No in_progress task found for session {SESSION}")

def cmd_log(args):
    if not args:
        print("✗ Usage: /task log <name> [summary]")
        return
    name = args[0]
    summary = " ".join(args[1:]) if len(args) > 1 else ""

    def modify(items):
        item = {
            "id": uuid.uuid4().hex[:8],
            "name": name,
            "type": infer_type(name),
            "status": "done",
            "description": "",
            "summary": summary or "(no summary)",
            "session": SESSION,
            "started_at": datetime.now().isoformat(),
            "finished_at": datetime.now().isoformat(),
            "script": None, "config": {}, "args": [],
        }
        items.append(item)
        return item

    item = locked_update(modify)
    print(f"✓ Logged [{item['id']}] {name} (session: {SESSION})")

def cmd_fail(args):
    reason = " ".join(args) if args else ""

    def modify(items):
        for item in reversed(items):
            if item.get("status") == "in_progress" and item.get("session") == SESSION:
                item["status"] = "failed"
                item["reason"] = reason or "(no reason)"
                item["finished_at"] = datetime.now().isoformat()
                return item
        return None

    item = locked_update(modify)
    if item:
        print(f"✓ Failed [{item['id']}] {item.get('name', '?')}: {item.get('reason')}")
    else:
        print(f"✗ No in_progress task found for session {SESSION}")

# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__" or True:
    argv = sys.argv[1:] if len(sys.argv) > 1 else "$ARGUMENTS".split()
    if not argv:
        print("Usage: /task start|done|log|fail [args...]")
        sys.exit(1)

    cmd = argv[0].lower()
    rest = argv[1:]

    {"start": cmd_start, "done": cmd_done, "log": cmd_log, "fail": cmd_fail}.get(
        cmd, lambda _: print(f"✗ Unknown command: {cmd}")
    )(rest)
```

## Rules
- The `locked_update` function handles the entire read→modify→write cycle under a single lock — always use it, never read/write separately.
- Keep JSON pretty-printed (indent=2).
- Non-experiment items: `script=null`, `config={}`, `args=[]`.
- Print confirmation with item ID and session.