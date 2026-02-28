"""
Web dashboard for the Rumination Engine.

Provides a live SSE stream of thoughts, tool calls, and prompts,
plus REST endpoints for archived thoughts and prompts.
"""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse


app = FastAPI(title="Rumination Engine")

# SSE subscriber management
_subscribers: list[asyncio.Queue] = []


def broadcast(event: str, data: dict) -> None:
    """Push an event to all connected SSE subscribers."""
    msg = {"event": event, "data": json.dumps(data)}
    dead = []
    for q in _subscribers:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _subscribers.remove(q)


@app.get("/api/stream")
async def stream(request: Request):
    """Server-Sent Events stream of live thought activity."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    _subscribers.append(queue)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=30)
                    yield msg
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
        finally:
            if queue in _subscribers:
                _subscribers.remove(queue)

    return EventSourceResponse(event_generator())


@app.get("/api/thoughts")
async def list_thoughts():
    """List archived thoughts, newest first."""
    from ruminate import THOUGHTS_DIR
    thoughts = []
    for f in sorted(THOUGHTS_DIR.glob("*.md"), reverse=True):
        content = f.read_text(encoding="utf-8")
        lines = content.strip().split("\n", 1)
        title = lines[0].lstrip("# ").strip() if lines else f.stem
        preview = lines[1].strip()[:200] if len(lines) > 1 else ""
        thoughts.append({
            "id": f.stem,
            "title": title,
            "preview": preview,
            "timestamp": f.stem,
        })
    return {"thoughts": thoughts}


@app.get("/api/thought/{thought_id}")
async def get_thought(thought_id: str):
    """Get a single thought by ID (filename stem)."""
    from ruminate import THOUGHTS_DIR
    path = THOUGHTS_DIR / f"{thought_id}.md"
    if not path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return {"id": thought_id, "content": path.read_text(encoding="utf-8")}


@app.get("/api/prompts")
async def list_prompts():
    """List archived prompts, newest first."""
    from ruminate import PROMPTS_DIR
    prompts = []
    for f in sorted(PROMPTS_DIR.glob("*.md"), reverse=True):
        if f.name == "current.md":
            continue
        content = f.read_text(encoding="utf-8")
        prompts.append({
            "id": f.stem,
            "content": content,
            "timestamp": f.stem,
        })
    return {"prompts": prompts}


@app.get("/api/current-prompt")
async def get_current_prompt():
    """Get the current self-prompt."""
    from ruminate import CURRENT_PROMPT_FILE, INITIAL_PROMPT
    if CURRENT_PROMPT_FILE.exists():
        return {"prompt": CURRENT_PROMPT_FILE.read_text(encoding="utf-8"), "source": "self"}
    return {"prompt": INITIAL_PROMPT, "source": "initial"}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML."""
    html_path = Path(__file__).parent / "static" / "dashboard.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))
