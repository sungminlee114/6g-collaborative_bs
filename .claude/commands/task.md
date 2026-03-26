---
description: "Quick task logging. Usage: /task start|done|log|fail [name] [desc/summary]"
allowed-tools: Read, Write, Edit, Bash
---

# Quick Task Logger

Lightweight wrapper around `assets/backlog.json` for Claude to quickly log work.
Each Claude Code panel gets a unique session ID.

## Session ID

Hook stdin의 `session_id` (앞 8자리)를 사용. Claude Code가 대화마다 부여하는 고유 UUID.
표시/추적 전용 ("누가 시작/완료했는지"). Task 매칭에는 사용하지 않음.

## Active Task Tracking

No `/tmp` files for task ID. Instead, each backlog item has an `active_in` array of session IDs.
- `start` → adds current session to `active_in`
- `done`/`fail` → finds item where current session is in `active_in`, completes it
- Any session can also close any task (cross-session recovery)
- `active_in` is the source of truth for "which panel owns this task"

## User argument: $ARGUMENTS

### `start <name> [description]`
- Add item to `assets/backlog.json` with `status: "in_progress"`, include `session` ID.
- Infer `type` from name prefix: `implement/` → implement, `fix/` → fix, `research/` → research, `phase.../` or `E.../` → experiment, `config/` → config, `review/` → review.
- Example: `/task start implement/right-sidebar "Right sidebar for backlog"`

### `done [summary]`
- Find `in_progress` item where current session is in `active_in`.
- Update `status` to `"done"`, set `summary`, set `finished_by` to current session.
- If no summary given, generate a one-line summary from context.

### `done <task_id> [summary]`
- Explicit task ID: close that specific task regardless of session (cross-session recovery).

### `log <name> [summary]`
- One-shot: add item as `done` immediately with summary.
- For logging completed work retroactively.
- Example: `/task log fix/chart-flicker "Fixed chart flicker on poll by splitting render"`

### `fail [reason]`
- Find `in_progress` item where current session is in `active_in`.
- Mark as `failed` with reason, set `finished_by` to current session.

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
import json, uuid, os, sys, fcntl, shutil
from pathlib import Path
from datetime import datetime

BACKLOG = Path("assets/backlog.json")

# ── Session ID (display only, set by SessionStart hook) ────────────
def _get_session_id():
    # SessionStart hook writes CLAUDE_SESSION_ID via CLAUDE_ENV_FILE
    env_id = os.environ.get("CLAUDE_SESSION_ID")
    if env_id:
        return env_id[:8]
    # Fallback: generate ephemeral ID (won't persist across sessions)
    return uuid.uuid4().hex[:6]

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
            "active_in": [SESSION],
            "started_at": datetime.now().isoformat(),
            "script": None, "config": {}, "args": [],
        }
        items.append(item)
        return item

    item = locked_update(modify)
    print(f"✓ Started [{item['id']}] {name} (session: {SESSION})")

def cmd_done(args):
    # Check if first arg looks like a task ID (hex, 8 chars)
    explicit_id = None
    summary_args = args
    if args and len(args[0]) == 8 and all(c in "0123456789abcdef" for c in args[0]):
        explicit_id = args[0]
        summary_args = args[1:]
    summary = " ".join(summary_args) if summary_args else ""

    def modify(items):
        # 1) Explicit task ID
        if explicit_id:
            for item in items:
                if item["id"] == explicit_id and item.get("status") == "in_progress":
                    item["status"] = "done"
                    item["summary"] = summary or "(no summary)"
                    item["finished_by"] = SESSION
                    item["finished_at"] = datetime.now().isoformat()
                    return item
            return None
        # 2) Find by active_in (current session in active_in list)
        for item in reversed(items):
            if item.get("status") == "in_progress" and SESSION in item.get("active_in", []):
                item["status"] = "done"
                item["summary"] = summary or "(no summary)"
                item["finished_by"] = SESSION
                item["finished_at"] = datetime.now().isoformat()
                return item
        # 3) Fallback: session field match (backward compat)
        for item in reversed(items):
            if item.get("status") == "in_progress" and item.get("session") == SESSION:
                item["status"] = "done"
                item["summary"] = summary or "(no summary)"
                item["finished_by"] = SESSION
                item["finished_at"] = datetime.now().isoformat()
                return item
        return None

    item = locked_update(modify)
    if item:
        print(f"✓ Done [{item['id']}] {item.get('name', '?')}: {item.get('summary')}")
    else:
        # Show open tasks for manual selection
        open_tasks = [i for i in read_backlog() if i.get("status") == "in_progress"]
        if open_tasks:
            print(f"✗ No active task for session {SESSION}. Open tasks:")
            for t in open_tasks:
                sessions = ", ".join(t.get("active_in", [t.get("session", "?")]))
                print(f"  [{t['id']}] {t['name']} (sessions: {sessions})")
            print(f"  → /task done <task_id> \"summary\"")
        else:
            print(f"✗ No in_progress tasks found")

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
    # Check if first arg looks like a task ID (hex, 8 chars)
    explicit_id = None
    reason_args = args
    if args and len(args[0]) == 8 and all(c in "0123456789abcdef" for c in args[0]):
        explicit_id = args[0]
        reason_args = args[1:]
    reason = " ".join(reason_args) if reason_args else ""

    def modify(items):
        # 1) Explicit task ID
        if explicit_id:
            for item in items:
                if item["id"] == explicit_id and item.get("status") == "in_progress":
                    item["status"] = "failed"
                    item["reason"] = reason or "(no reason)"
                    item["finished_by"] = SESSION
                    item["finished_at"] = datetime.now().isoformat()
                    return item
            return None
        # 2) Find by active_in
        for item in reversed(items):
            if item.get("status") == "in_progress" and SESSION in item.get("active_in", []):
                item["status"] = "failed"
                item["reason"] = reason or "(no reason)"
                item["finished_by"] = SESSION
                item["finished_at"] = datetime.now().isoformat()
                return item
        # 3) Fallback: session field match
        for item in reversed(items):
            if item.get("status") == "in_progress" and item.get("session") == SESSION:
                item["status"] = "failed"
                item["reason"] = reason or "(no reason)"
                item["finished_by"] = SESSION
                item["finished_at"] = datetime.now().isoformat()
                return item
        return None

    item = locked_update(modify)
    if item:
        print(f"✓ Failed [{item['id']}] {item.get('name', '?')}: {item.get('reason')}")
    else:
        open_tasks = [i for i in read_backlog() if i.get("status") == "in_progress"]
        if open_tasks:
            print(f"✗ No active task for session {SESSION}. Open tasks:")
            for t in open_tasks:
                print(f"  [{t['id']}] {t['name']}")
            print(f"  → /task fail <task_id> \"reason\"")
        else:
            print(f"✗ No in_progress tasks found")

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
- **`done`/`log` 실행 전에 반드시 해당 작업 관련 변경사항을 git commit한다.** 커밋 없이 task를 완료 처리하지 말 것.
- The `locked_update` function handles the entire read→modify→write cycle under a single lock — always use it, never read/write separately.
- Keep JSON pretty-printed (indent=2).
- Non-experiment items: `script=null`, `config={}`, `args=[]`.
- Print confirmation with item ID and session.