# Web Dashboard + Self-Prompting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a live web dashboard with SSE streaming and refactor self-prompting to use an explicit tool with prompt archiving.

**Architecture:** FastAPI embedded in ruminate.py's asyncio loop serves a single-page dashboard. SSE broadcasts thought events to connected clients. Self-prompting switches from `## Next` text parsing to a `set_next_prompt` built-in tool that archives each prompt to `prompts/`.

**Tech Stack:** FastAPI, Uvicorn, sse-starlette, vanilla HTML/JS/CSS

---

### Task 1: Add dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add new dependencies to pyproject.toml**

Add to `[tool.poetry.dependencies]`:
```toml
fastapi = "*"
uvicorn = "*"
sse-starlette = "*"
```

**Step 2: Lock and install**

Run: `poetry lock && poetry install`
Expected: All deps resolve and install successfully

**Step 3: Commit**

```bash
git add pyproject.toml poetry.lock
git commit -m "feat: add fastapi, uvicorn, sse-starlette dependencies"
```

---

### Task 2: Refactor self-prompting system

**Files:**
- Modify: `ruminate.py` (self-prompting section ~lines 815-860, BUILTIN_TOOLS ~line 398, handle_builtin_tool ~line 711, main loop ~lines 1115-1119)
- Modify: `.gitignore`

**Step 1: Update .gitignore**

Replace:
```
# Thought archive & self-prompt
thoughts/
next_prompt.txt
```

With:
```
# Thought archive & prompts
thoughts/
prompts/
```

**Step 2: Replace self-prompting functions in ruminate.py**

Replace the entire `# === Self-Prompting & Thought Archiving ===` section (lines 815-858) with:

```python
# === Self-Prompting & Thought Archiving ===

PROMPTS_DIR = APP_DIR / "prompts"
PROMPTS_DIR.mkdir(exist_ok=True)
THOUGHTS_DIR = APP_DIR / "thoughts"
THOUGHTS_DIR.mkdir(exist_ok=True)
CURRENT_PROMPT_FILE = PROMPTS_DIR / "current.md"


def save_thought_to_file(thought: str, thought_count: int) -> Path:
    """Save a thought as a markdown file in the thoughts/ directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}.md"
    path = THOUGHTS_DIR / filename
    header = f"# Thought {thought_count} — {timestamp.replace('_', ' ')}\n\n"
    path.write_text(header + thought, encoding="utf-8")
    return path


def load_current_prompt() -> str | None:
    """Load the current self-written prompt, if it exists."""
    if CURRENT_PROMPT_FILE.exists():
        content = CURRENT_PROMPT_FILE.read_text(encoding="utf-8").strip()
        return content if content else None
    return None


def save_prompt(prompt: str) -> Path:
    """Save a prompt to the archive and update current.md."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_path = PROMPTS_DIR / f"{timestamp}.md"
    archive_path.write_text(prompt, encoding="utf-8")
    CURRENT_PROMPT_FILE.write_text(prompt, encoding="utf-8")
    log_file_only(f"Prompt saved ({len(prompt)} chars) -> {archive_path.name}")
    return archive_path
```

**Step 3: Add `set_next_prompt` to BUILTIN_TOOLS list**

After the `restart_self` tool definition (around line 524), add:

```python
    {
        "name": "set_next_prompt",
        "description": (
            "Set the prompt for your next thought cycle. Write what you want to "
            "think about or work on next. This becomes the direction you give yourself "
            "when the next thought begins. Each prompt is archived with a timestamp."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt/direction for your next thought cycle",
                },
            },
            "required": ["prompt"],
        },
    },
```

**Step 4: Add handler in handle_builtin_tool**

After the `restart_self` handler (around line 722), add:

```python
    if tool_name == "set_next_prompt":
        prompt_text = tool_input.get("prompt", "")
        if not prompt_text.strip():
            return json.dumps({"error": "Prompt text is required"})
        archive_path = save_prompt(prompt_text)
        return json.dumps({
            "ok": True,
            "message": f"Next prompt set ({len(prompt_text)} chars). Archived to {archive_path.name}.",
        })
```

**Step 5: Update `build_context` to use `load_current_prompt`**

In `build_context()`, replace:
```python
    next_prompt = load_next_prompt()
    if next_prompt and thought_count > 1:
```
with:
```python
    next_prompt = load_current_prompt()
    if next_prompt and thought_count > 1:
```

**Step 6: Update main loop — remove `## Next` parsing**

Replace the post-thought section (around lines 1115-1121):
```python
            # Extract self-prompt and archive thought
            clean_thought, next_prompt = extract_next_prompt(thought)
            if next_prompt:
                save_next_prompt(next_prompt)
                console.print(f"[dim]next prompt: {next_prompt[:100]}{'…' if len(next_prompt) > 100 else ''}[/dim]")

            save_thought_to_file(clean_thought, thought_count)

            thought_history.append(clean_thought)
```

With:
```python
            # Archive thought
            save_thought_to_file(thought, thought_count)

            thought_history.append(thought)
```

**Step 7: Update context_info — replace ## Next instruction with set_next_prompt**

In `build_context()`, replace:
```python
**Continuity** — End each thought with a `## Next` section describing what you want to think
about or work on next time. This becomes your prompt for the next thought cycle.
```
with:
```python
**Continuity** — Use `set_next_prompt` to write the direction for your next thought cycle.
This is how you maintain intentional focus across thoughts.
```

**Step 8: Delete `next_prompt.txt` if it exists, verify syntax**

Run: `rm -f next_prompt.txt && python3 -c "compile(open('ruminate.py').read(), 'ruminate.py', 'exec'); print('OK')"`
Expected: `OK`

**Step 9: Commit**

```bash
git add ruminate.py .gitignore
git commit -m "feat: refactor self-prompting to use set_next_prompt tool with prompt archiving"
```

---

### Task 3: Create SSE event broadcasting system

**Files:**
- Create: `dashboard.py`

**Step 1: Create `dashboard.py` with SSE broadcaster and FastAPI app**

```python
"""
Web dashboard for the Rumination Engine.

Provides a live SSE stream of thoughts, tool calls, and prompts,
plus REST endpoints for archived thoughts and prompts.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
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


# --- SSE endpoint ---

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
            _subscribers.remove(queue)

    return EventSourceResponse(event_generator())


# --- REST endpoints ---

@app.get("/api/thoughts")
async def list_thoughts():
    """List archived thoughts, newest first."""
    from ruminate import THOUGHTS_DIR
    thoughts = []
    for f in sorted(THOUGHTS_DIR.glob("*.md"), reverse=True):
        content = f.read_text(encoding="utf-8")
        # Extract first line as title, rest as preview
        lines = content.strip().split("\n", 1)
        title = lines[0].lstrip("# ").strip() if lines else f.stem
        preview = lines[1].strip()[:200] if len(lines) > 1 else ""
        thoughts.append({
            "id": f.stem,
            "title": title,
            "preview": preview,
            "timestamp": f.stem,  # filename is the timestamp
        })
    return {"thoughts": thoughts}


@app.get("/api/thought/{thought_id}")
async def get_thought(thought_id: str):
    """Get a single thought by ID (filename stem)."""
    from ruminate import THOUGHTS_DIR
    path = THOUGHTS_DIR / f"{thought_id}.md"
    if not path.exists():
        return {"error": "Not found"}, 404
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
```

**Step 2: Verify syntax**

Run: `python3 -c "compile(open('dashboard.py').read(), 'dashboard.py', 'exec'); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add dashboard.py
git commit -m "feat: add dashboard FastAPI app with SSE broadcasting and REST endpoints"
```

---

### Task 4: Create dashboard HTML

**Files:**
- Create: `static/dashboard.html`

**Step 1: Create `static/` directory and `dashboard.html`**

Create a single-page dashboard with:
- Live thought stream area (top, auto-scrolling, green text)
- Current prompt display (top bar)
- Tool call indicators (blue for tools, magenta for memory)
- Recent thoughts list (sidebar)
- Dark theme, monospace, minimal
- Vanilla JS EventSource for SSE
- No build step, no framework

The HTML file should be self-contained (~200-300 lines) with embedded CSS and JS. Key behaviors:
- Connect to `/api/stream` via EventSource
- Handle events: `thought_start`, `thought_chunk`, `tool_call`, `thought_end`, `prompt_set`
- Load initial data from `/api/thoughts` and `/api/current-prompt` on page load
- Auto-reconnect on SSE disconnect
- Mobile-responsive layout

**Step 2: Commit**

```bash
git add static/dashboard.html
git commit -m "feat: add dashboard single-page HTML with SSE live streaming"
```

---

### Task 5: Integrate SSE broadcasting into ruminate.py

**Files:**
- Modify: `ruminate.py` (main loop, thought events)
- Modify: `providers.py` (stream text + tool call events)

**Step 1: Add broadcast calls to ruminate.py main loop**

At the top of ruminate.py, add import:
```python
from dashboard import broadcast
```

In the main loop, add broadcast calls:

Before `provider.think()` call:
```python
            broadcast("thought_start", {
                "thought_number": thought_count,
                "timestamp": timestamp,
            })
```

After thought completes (after `console.print()`):
```python
            broadcast("thought_end", {
                "thought_number": thought_count,
                "thought": thought,
                "timestamp": timestamp,
            })
```

**Step 2: Add broadcast calls to providers.py**

In both AnthropicProvider and OpenAIProvider `think()` methods:

After each streamed text chunk, broadcast:
```python
from dashboard import broadcast
# ... inside the streaming loop:
broadcast("thought_chunk", {"text": text_or_delta})
```

After each tool call display, broadcast:
```python
broadcast("tool_call", {
    "name": tool_name,
    "args": input_brief,
    "is_memory": is_memory,
})
```

**Step 3: Add broadcast to set_next_prompt handler**

In the `set_next_prompt` handler in `handle_builtin_tool`, after saving:
```python
broadcast("prompt_set", {"prompt": prompt_text})
```

**Step 4: Verify syntax**

Run: `python3 -c "compile(open('ruminate.py').read(), 'ruminate.py', 'exec'); compile(open('providers.py').read(), 'providers.py', 'exec'); print('OK')"`

**Step 5: Commit**

```bash
git add ruminate.py providers.py
git commit -m "feat: integrate SSE broadcasting into thought loop and providers"
```

---

### Task 6: Embed Uvicorn in the startup

**Files:**
- Modify: `ruminate.py` (replace `asyncio.run(main())` with Uvicorn startup)

**Step 1: Refactor main() entry point**

Replace the `if __name__ == "__main__":` block at the bottom of ruminate.py:

```python
if __name__ == "__main__":
    asyncio.run(main())
```

With:

```python
DASHBOARD_PORT = int(os.environ.get("DASHBOARD_PORT", "8080"))


async def start():
    """Start the thought loop and web dashboard concurrently."""
    import uvicorn
    from dashboard import app

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=DASHBOARD_PORT,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    # Run both concurrently — thought loop + web server
    await asyncio.gather(
        server.serve(),
        main(),
    )


if __name__ == "__main__":
    asyncio.run(start())
```

**Step 2: Add DASHBOARD_PORT to docker-compose.yml**

Add to environment section:
```yaml
      DASHBOARD_PORT: ${DASHBOARD_PORT:-8080}
```

Add to ports (if using Docker):
```yaml
    ports:
      - "${DASHBOARD_PORT:-8080}:8080"
```

**Step 3: Add DASHBOARD_PORT to ruminate.service**

The systemd service doesn't need changes for the port itself, but if the VPS has a firewall, port 8080 needs to be open.

**Step 4: Verify syntax**

Run: `python3 -c "compile(open('ruminate.py').read(), 'ruminate.py', 'exec'); print('OK')"`

**Step 5: Commit**

```bash
git add ruminate.py docker-compose.yml
git commit -m "feat: embed Uvicorn alongside thought loop, serve dashboard on port 8080"
```

---

### Task 7: Update prompt.txt and Dockerfile

**Files:**
- Modify: `prompt.txt`
- Modify: `Dockerfile`

**Step 1: Update prompt.txt continuity section**

Replace the `## Continuity` section with:

```markdown
## Continuity

Use `set_next_prompt` to write the direction for your next thought cycle. This is your
self-authored thread of intention — each thought can set the agenda for the next.

Example: at the end of a thought about building a router, call:
  set_next_prompt("Continue implementing the message router. Left off at the middleware
  layer — need to add the context injection before testing the full pipeline.")

This is how you maintain focus across thoughts. Without it, you start fresh each time.
With it, you pick up where you left off — by your own choice.
```

**Step 2: Update Dockerfile**

Change the COPY line to include new files:
```dockerfile
COPY ruminate.py providers.py dashboard.py ./
COPY static/ ./static/
```

**Step 3: Commit**

```bash
git add prompt.txt Dockerfile
git commit -m "feat: update prompt for set_next_prompt tool, update Dockerfile for dashboard"
```

---

### Task 8: Smoke test

**Step 1: Verify all files parse cleanly**

Run: `python3 -c "compile(open('ruminate.py').read(), 'ruminate.py', 'exec'); compile(open('providers.py').read(), 'providers.py', 'exec'); compile(open('dashboard.py').read(), 'dashboard.py', 'exec'); print('ALL OK')"`

**Step 2: Verify imports work**

Run: `python3 -c "from dashboard import app, broadcast; print('imports OK')"`

**Step 3: Push**

```bash
git push origin master
```
