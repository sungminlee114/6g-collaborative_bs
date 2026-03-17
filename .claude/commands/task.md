---
description: "Quick task logging. Usage: /task start|done|log [name] [desc/summary]"
allowed-tools: Read, Write, Edit, Bash
---

# Quick Task Logger

Lightweight wrapper around `assets/backlog.json` for Claude to quickly log work.
Each Claude session gets a session ID (first 6 chars of a UUID, stored in `/tmp/claude_task_session`).

## Session ID

Generate or reuse session ID:
1. Check if `/tmp/claude_task_session` exists and was created less than 24h ago
2. If yes, read it. If no, generate `uuid.uuid4().hex[:6]` and write it there.
3. Include this as the `session` field in backlog items.

## User argument: $ARGUMENTS

### `start <name> [description]`
- Add item to `assets/backlog.json` with `status: "in_progress"`, include `session` ID
- Infer `type` from name: `implement/...` → implement, `fix/...` → fix, `research/...` → research, `phase.../...` or `E.../...` → experiment, `config/...` → config, `review/...` → review
- Example: `/task start implement/right-sidebar "Right sidebar for backlog"`

### `done [summary]`
- Find the most recent `in_progress` item with matching session ID
- Update `status` to `"done"`, set `summary`
- If no summary given, generate a one-line summary from context

### `log <name> [summary]`
- One-shot: add item as `done` immediately with summary
- For logging completed work retroactively
- Example: `/task log fix/chart-flicker "Fixed chart flicker on poll by splitting render"`

### `fail [reason]`
- Mark current in_progress item as `failed` with reason

## Implementation

Read the file, modify in-memory, write atomically (write to `.tmp` then `os.replace`).
Use Python via Bash tool:

```python
import json, uuid, os, time
from pathlib import Path
from datetime import datetime

BACKLOG = Path("assets/backlog.json")
SESSION_FILE = Path("/tmp/claude_task_session")

# Session ID
if SESSION_FILE.exists() and (time.time() - SESSION_FILE.stat().st_mtime) < 86400:
    session = SESSION_FILE.read_text().strip()
else:
    session = uuid.uuid4().hex[:6]
    SESSION_FILE.write_text(session)

items = json.loads(BACKLOG.read_text()) if BACKLOG.exists() else []
# ... modify items ...
tmp = BACKLOG.with_suffix(".tmp")
tmp.write_text(json.dumps(items, indent=2, ensure_ascii=False))
os.replace(tmp, BACKLOG)
```

## Rules
- Always read before write (avoid race with dashboard)
- Keep JSON pretty-printed (indent=2)
- Non-experiment items: script=null, config={}, args=[]
- Print confirmation with item ID and session
