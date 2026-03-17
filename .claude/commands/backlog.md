---
description: "Manage task backlog (experiments, research, implementation, etc.). Single source of truth: assets/backlog.json"
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Task Backlog Manager

You manage the unified task backlog stored in `assets/backlog.json`.
This file is the single source of truth — both the web dashboard and this command read/write it.
Tracks ALL work: experiments, research, implementation, bug fixes, etc.

## Item schema

```json
{
  "id": "8-char hex",
  "name": "implement/sidebar-toggle",
  "type": "implement",
  "description": "사이드바 접기/펴기 토글 구현",
  "script": null,
  "args": [],
  "config": {},
  "priority": "normal",
  "status": "queued",
  "summary": null,
  "created_at": "ISO timestamp",
  "run_id": null
}
```

### Type values
- `experiment`: ML 학습/평가 (has script, config)
- `research`: 조사/분석 (논문 리뷰, 데이터셋 분석)
- `implement`: 기능 구현 (새 모델, 대시보드)
- `fix`: 버그 수정
- `review`: 코드/결과 리뷰
- `config`: 설정/인프라 변경

## User argument: $ARGUMENTS

Parse the user's argument to determine the action:

### `list` (or no argument)
- Read `assets/backlog.json`
- Display all items grouped by status (in_progress → queued → done → failed)
- Show: type icon, name, priority, description, status

### `add <name> [options]`
- Add a new item to the backlog
- Generate 8-char hex id via Python: `uuid.uuid4().hex[:8]`
- Infer `type` from name prefix (e.g., "implement/..." → implement, "phase1/..." → experiment)
- For experiments: look at existing scripts in `src/experiments/` to match
- For non-experiments: script/config/args can be null/empty

### `remove <id or name>`
- Remove item by id or name match
- Confirm before removing

### `run <id or name or "next">`
- For experiments: execute the training script with Tracker
- For non-experiments: update status to "in_progress"
- On completion: update status to "done", set `summary`

### `done <id or name> [summary]`
- Mark item as done with optional summary text

### `clear`
- Remove all "done" and "failed" items from backlog

### `edit <id or name> <field>=<value>`
- Update a specific field of an item

## Important rules
- Always read the current file before modifying (avoid race conditions with dashboard)
- Use atomic writes: write to `.tmp` then rename
- Keep JSON pretty-printed with indent=2
- When running experiments, follow the patterns in existing `src/experiments/` scripts
- Reference `CLAUDE.md` for Tracker usage and task logging patterns
