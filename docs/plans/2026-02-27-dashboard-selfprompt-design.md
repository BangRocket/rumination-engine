# Web Dashboard + Self-Prompting Refactor

**Date:** 2026-02-27
**Status:** Approved

## Overview

Two changes: (1) replace `## Next` parsing with an explicit `set_next_prompt` tool and prompt archiving, (2) add a live web dashboard with SSE streaming.

## 1. Self-Prompting Refactor

### Current System
Engine parses `## Next` from thought text, saves to `next_prompt.txt`. Fragile — depends on Clara following a text convention.

### New System
New built-in tool `set_next_prompt(prompt)`. The engine:

1. Saves prompt to `prompts/YYYY-MM-DD_HH-MM-SS.md` (archived)
2. Also saves to `prompts/current.md` (what the engine reads next cycle)
3. `build_context()` reads `prompts/current.md` for thought 2+, falls back to `prompt.txt` for thought 1
4. `## Next` parsing removed entirely

**Prompt archive:**
```
prompts/
  current.md
  2026-02-27_14-30-22.md
  2026-02-27_17-45-10.md
```

### Why a tool instead of file convention
- Discoverable in tool list
- Engine can log when prompt is set
- Clara doesn't need to know file paths
- No parsing ambiguity

## 2. Web Dashboard

### Architecture
FastAPI + SSE embedded in the ruminate process. Shares asyncio event loop with thought loop.

```
ruminate.py
  ├── main()           — thought loop (existing)
  ├── FastAPI app       — dashboard + SSE
  │     ├── GET /                — HTML dashboard
  │     ├── GET /api/stream      — SSE (live thoughts)
  │     ├── GET /api/thoughts    — JSON archived thoughts
  │     ├── GET /api/prompts     — JSON archived prompts
  │     └── GET /api/thought/:id — single thought
  └── Uvicorn           — runs both concurrently
```

### SSE Events
- `thought_start` — thought number, timestamp
- `thought_chunk` — streamed text
- `tool_call` — tool name, args
- `thought_end` — complete thought, prompt set
- `prompt_set` — new self-prompt written

### Dashboard UI
Single `dashboard.html`, vanilla JS + CSS, no build step:
- Live streaming thought (top, auto-scroll)
- Recent thoughts list (sidebar/below)
- Current prompt (top)
- Color coding: green thoughts, blue tools, magenta memory

## 3. Dependencies

Add to pyproject.toml:
- `fastapi`
- `uvicorn`
- `sse-starlette`

## 4. File Changes

- `ruminate.py` — remove `## Next` parsing, add `set_next_prompt` tool, SSE broadcasting, Uvicorn startup
- `dashboard.py` — new: FastAPI app, routes, SSE
- `static/dashboard.html` — new: single-page dashboard
- `prompts/` — new dir (gitignored)
- `pyproject.toml` — new deps
- `prompt.txt` — update continuity section for new tool

## 5. Startup

Uvicorn runs the FastAPI app; thought loop starts as background asyncio task. Dashboard on port 8080 (configurable via `DASHBOARD_PORT`).

## 6. Thought Archiving

Unchanged from current system — thoughts still saved to `thoughts/YYYY-MM-DD_HH-MM-SS.md`.
