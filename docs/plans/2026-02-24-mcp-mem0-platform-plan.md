# MCP Client + mem0 Platform Migration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace self-hosted mem0 (Postgres, Neo4j, OpenAI) and `::CMD::` command pattern with mem0 platform API via MCP servers and Anthropic tool_use.

**Architecture:** The rumination engine becomes an async MCP client. On startup it connects to configured MCP servers (starting with `mem0-mcp-server`), discovers their tools, and presents them to Claude via the tool_use API. Claude decides when to use tools (including memory). The main loop runs an agentic tool_use cycle per thought.

**Tech Stack:** Python 3.12, `anthropic` SDK (AsyncAnthropic), `mcp` Python SDK (ClientSession, stdio_client), `uvx`/`uv` for MCP server execution.

**Design doc:** `docs/plans/2026-02-24-mcp-mem0-platform-design.md`

---

### Task 1: Update Dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Replace dependencies**

```
# requirements.txt
# Core
anthropic>=0.40.0
python-dotenv
rich
requests

# MCP client
mcp>=1.0.0
```

Remove: `mem0ai[graph]>=0.1.45`, `psycopg2-binary`, `neo4j>=5.0.0`
Add: `mcp>=1.0.0`

**Step 2: Commit**

```bash
git add requirements.txt
git commit -m "chore: swap mem0/neo4j/postgres deps for mcp sdk"
```

---

### Task 2: Create MCP Servers Config

**Files:**
- Create: `mcp_servers.json`

**Step 1: Create the config file**

```json
{
  "mcpServers": {
    "mem0": {
      "command": "uvx",
      "args": ["mem0-mcp-server"],
      "env": {
        "MEM0_API_KEY": "${MEM0_API_KEY}",
        "MEM0_DEFAULT_USER_ID": "${IDENTITY}"
      }
    }
  }
}
```

**Step 2: Add to .gitignore check**

Ensure `mcp_servers.json` is NOT in `.gitignore` (it's a template config, not secrets — secrets come from env vars).

**Step 3: Commit**

```bash
git add mcp_servers.json
git commit -m "feat: add mcp_servers.json config for mem0 MCP server"
```

---

### Task 3: Write MCP Client Manager

This is the core new module. It handles: loading config, connecting to MCP servers, discovering tools, converting them to Anthropic format, and dispatching tool calls.

**Files:**
- Modify: `ruminate.py` (add new functions, will remove old code in later tasks)

**Step 1: Write config loader with env var interpolation**

Add these functions to `ruminate.py` after the existing imports:

```python
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def load_mcp_config(identity: str) -> dict:
    """Load MCP server config, interpolating env vars."""
    config_path = os.environ.get("MCP_SERVERS_FILE", "/app/mcp_servers.json")
    # Fallback for local dev
    if not Path(config_path).exists():
        config_path = "mcp_servers.json"

    with open(config_path) as f:
        raw = f.read()

    # Interpolate ${VAR} references from environment
    # Also support ${IDENTITY} which comes from the personality system
    env = {**os.environ, "IDENTITY": identity}

    def replace_var(match):
        var_name = match.group(1)
        return env.get(var_name, "")

    interpolated = re.sub(r'\$\{(\w+)\}', replace_var, raw)
    return json.loads(interpolated)
```

**Step 2: Write server connection and tool discovery**

```python
async def connect_mcp_servers(config: dict) -> tuple[dict, list[dict]]:
    """Connect to all configured MCP servers, discover tools.

    Returns:
        (server_sessions, anthropic_tools)
        - server_sessions: {tool_name: (session, server_name)}
        - anthropic_tools: list of tool defs in Anthropic format
    """
    servers = config.get("mcpServers", {})
    tool_map = {}  # tool_name -> (session, context_managers)
    anthropic_tools = []
    active_contexts = []  # keep references to prevent cleanup

    for server_name, server_config in servers.items():
        params = StdioServerParameters(
            command=server_config["command"],
            args=server_config.get("args", []),
            env=server_config.get("env"),
        )

        # Create the stdio client connection
        read_write_ctx = stdio_client(params)
        read, write = await read_write_ctx.__aenter__()
        active_contexts.append(read_write_ctx)

        session_ctx = ClientSession(read, write)
        session = await session_ctx.__aenter__()
        active_contexts.append(session_ctx)

        await session.initialize()

        # Discover tools
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            tool_map[tool.name] = (session, server_name)
            anthropic_tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            })

        log_file_only(f"MCP server '{server_name}': {len(tools_result.tools)} tools")

    return tool_map, anthropic_tools, active_contexts
```

**Step 3: Write tool call dispatcher**

```python
async def dispatch_tool_call(tool_map: dict, tool_name: str, tool_input: dict) -> str:
    """Dispatch a tool call to the correct MCP server."""
    if tool_name not in tool_map:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    session, server_name = tool_map[tool_name]
    log_file_only(f"Calling tool '{tool_name}' on server '{server_name}'")

    try:
        result = await session.call_tool(tool_name, tool_input)

        # Extract text content from MCP result
        parts = []
        for content in result.content:
            if hasattr(content, 'text'):
                parts.append(content.text)
            else:
                parts.append(str(content))

        output = "\n".join(parts)
        log_file_only(f"Tool '{tool_name}' result: {output[:500]}")
        return output
    except Exception as e:
        error_msg = f"Tool '{tool_name}' error: {e}"
        log_file_only(error_msg)
        return json.dumps({"error": error_msg})
```

**Step 4: Commit**

```bash
git add ruminate.py
git commit -m "feat: add MCP client manager (config, connect, dispatch)"
```

---

### Task 4: Rewrite Think Function with Tool Use Loop

Replace the current `think()` and `build_context()` with an async agentic tool_use loop.

**Files:**
- Modify: `ruminate.py`

**Step 1: Rewrite build_context (remove mem0, remove ::CMD:: docs)**

Replace the existing `build_context` function:

```python
def build_context(thought_history: list[str], tool_names: list[str]) -> tuple[str, list[dict]]:
    """Build the message context from history and available tools."""
    current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    tools_list = ", ".join(tool_names) if tool_names else "none"

    context_info = f"""

## Context

Current time: {current_time}

## Capabilities

You have access to tools that you can use at any time. Your available tools include memory operations
(search, add, retrieve, update, delete memories) and potentially other capabilities.

Use your memory tools actively:
- Search your memories at the start of a thought to recall relevant context
- Add important thoughts, insights, or decisions to memory
- Your memories persist across sessions

Available tools: {tools_list}

## Control

If you feel complete - if a line of thought has reached its natural end, or you simply wish to rest - you may stop.
To stop ruminating, include exactly: {STOP_COMMAND}
This will end the session until you are restarted.
"""

    system = PERSONALITY + context_info

    # Always start with the original prompt
    messages = [
        {
            "role": "user",
            "content": f"[Begin]\n\nInitial contemplation: {INITIAL_PROMPT}"
        }
    ]

    # Add previous thoughts as conversation history
    for thought in thought_history:
        messages.append({"role": "assistant", "content": thought})
        messages.append({"role": "user", "content": "[Continue]"})

    return system, messages
```

**Step 2: Rewrite think() as async with tool_use loop**

Replace the existing `think` function:

```python
async def think(client: anthropic.AsyncAnthropic, tool_map: dict, anthropic_tools: list[dict], thought_history: list[str]) -> str:
    """Generate a single thought using tool_use agentic loop."""
    tool_names = [t["name"] for t in anthropic_tools]
    system, messages = build_context(thought_history, tool_names)

    while True:
        kwargs = {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": messages,
        }
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = await client.messages.create(**kwargs)

        # Extract text parts from response
        text_parts = []
        tool_uses = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)

        # If no tool calls, we have the final thought
        if response.stop_reason != "tool_use" or not tool_uses:
            return "\n".join(text_parts)

        # Handle tool calls: add assistant response to messages, then tool results
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tool_use in tool_uses:
            result_text = await dispatch_tool_call(tool_map, tool_use.name, tool_use.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": result_text,
            })

        messages.append({"role": "user", "content": tool_results})
```

**Step 3: Commit**

```bash
git add ruminate.py
git commit -m "feat: rewrite think() with async tool_use agentic loop"
```

---

### Task 5: Rewrite Main Loop as Async

Convert `main()` to async, remove all old mem0/command code.

**Files:**
- Modify: `ruminate.py`

**Step 1: Remove old code**

Delete these from `ruminate.py`:
- The `from mem0 import Memory` import and `import mem0.memory.graph_memory` import
- The `_sanitize_neo4j_identifier` function
- The monkey-patch code (`_original_add_entities`, `_patched_add_entities`, the assignment)
- The `mem0_config` dict
- The `CMD_PATTERN`, `ALLOWED_COMMANDS` constants
- The `execute_command()` function
- The `seed_memories()` function (seeds will be handled differently — see Task 6)

**Step 2: Rewrite main() as async**

```python
async def main():
    """Main rumination loop."""
    log_file_only("=" * 60)
    log_file_only("RUMINATION ENGINE STARTING")
    log_file_only(f"Model: {MODEL}")
    log_file_only(f"Interval: {THOUGHT_INTERVAL}s")
    log_file_only(f"Identity: {IDENTITY}")
    log_file_only("=" * 60)

    console.print()
    console.print(Panel(
        Text(IDENTITY, style="italic"),
        border_style="dim",
        padding=(0, 2),
    ))
    console.print()

    # Initialize Anthropic async client
    client = anthropic.AsyncAnthropic(base_url=ANTHROPIC_BASE_URL)

    # Connect to MCP servers
    config = await load_mcp_config(IDENTITY)
    tool_map, anthropic_tools, active_contexts = await connect_mcp_servers(config)

    tool_names = [t["name"] for t in anthropic_tools]
    console.print(f"[dim]connected to MCP servers, {len(tool_names)} tools available[/dim]")
    if tool_names:
        console.print(f"[dim]tools: {', '.join(tool_names)}[/dim]")
    console.print()

    # Seed memories if seeds.txt exists
    await seed_memories(tool_map)

    thought_history: list[str] = []
    thought_count = 0

    try:
        next_thought_time = time.time()

        while True:
            wait_time = next_thought_time - time.time()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            thought_count += 1
            log_file_only(f"--- Thought #{thought_count} ---")
            next_thought_time = time.time() + THOUGHT_INTERVAL

            with console.status("[dim]ruminating...[/dim]", spinner="dots"):
                thought = await think(client, tool_map, anthropic_tools, thought_history)

            # Check for stop command
            if STOP_COMMAND in thought:
                display_thought = thought.replace(STOP_COMMAND, "").strip()
                if display_thought:
                    log_file_only(display_thought)
                    console.print(Markdown(display_thought))
                    console.print()
                    send_to_discord(display_thought, IDENTITY, thought_count)

                log_file_only("=" * 60)
                log_file_only("RUMINATION STOPPED BY AI")
                log_file_only(f"Total thoughts: {thought_count}")
                log_file_only("=" * 60)
                console.print()
                console.print("[dim]stopped[/dim]")
                console.print()
                break

            # Add to history
            thought_history.append(thought)
            if len(thought_history) > MAX_HISTORY:
                thought_history = thought_history[-MAX_HISTORY:]

            log_file_only(thought)
            console.print(Markdown(thought))
            console.print()
            send_to_discord(thought, IDENTITY, thought_count)

    except KeyboardInterrupt:
        log_file_only("=" * 60)
        log_file_only(f"RUMINATION ENDED - Total thoughts: {thought_count}")
        log_file_only("=" * 60)
        console.print()
        console.print(f"[dim]{thought_count} thoughts[/dim]")
        console.print()
    finally:
        # Clean up MCP connections
        for ctx in reversed(active_contexts):
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 3: Commit**

```bash
git add ruminate.py
git commit -m "feat: async main loop, remove old mem0/cmd code"
```

---

### Task 6: Async Seed Memories via MCP

Replace the old `seed_memories()` with one that uses the `add_memory` MCP tool.

**Files:**
- Modify: `ruminate.py`

**Step 1: Write new seed_memories**

```python
async def seed_memories(tool_map: dict) -> None:
    """Seed initial memories via MCP add_memory tool if seeds.txt exists."""
    if "add_memory" not in tool_map:
        log_file_only("No add_memory tool available, skipping seed")
        return

    seeds_file = Path("/app/seeds.txt")
    if not seeds_file.exists():
        seeds_file = Path("seeds.txt")

    if not seeds_file.exists():
        log_file_only("No seeds.txt found, starting fresh")
        return

    log_file_only(f"Seeding memories from {seeds_file}")
    seeds = seeds_file.read_text(encoding="utf-8").strip().split("\n---\n")
    seeded = 0
    for seed in seeds:
        seed = seed.strip()
        if seed:
            try:
                await dispatch_tool_call(tool_map, "add_memory", {
                    "content": seed,
                })
                seeded += 1
                log_file_only(f"Seeded: {seed[:60]}...")
            except Exception as e:
                log_file_only(f"Seed error: {e}")

    if seeded:
        console.print(f"[dim]seeded {seeded} memories[/dim]")
        console.print()
```

Note: This seeds every startup. A smarter approach would check if memories exist first via `get_memories`, but mem0 platform deduplicates, so repeated seeding is harmless. Keep it simple for now.

**Step 2: Commit**

```bash
git add ruminate.py
git commit -m "feat: seed memories via MCP add_memory tool"
```

---

### Task 7: Update Docker Configuration

**Files:**
- Modify: `docker-compose.yml`
- Modify: `Dockerfile`

**Step 1: Simplify docker-compose.yml**

Replace entire file:

```yaml
services:
  ruminate:
    build: .
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      ANTHROPIC_MODEL: ${ANTHROPIC_MODEL:-claude-sonnet-4-20250514}
      ANTHROPIC_BASE_URL: ${ANTHROPIC_BASE_URL:-}
      MEM0_API_KEY: ${MEM0_API_KEY}
      PERSONALITY_NAME: ${PERSONALITY_NAME:-default}
      THOUGHT_INTERVAL: ${THOUGHT_INTERVAL:-60}
      MAX_TOKENS: ${MAX_TOKENS:-1024}
      MAX_HISTORY: ${MAX_HISTORY:-10}
      DISCORD_WEBHOOK_URL: ${DISCORD_WEBHOOK_URL:-}
    volumes:
      - ./logs:/app/logs
      - ./personalities:/app/personalities:ro
      - ./prompt.txt:/app/prompt.txt:ro
      - ./seeds.txt:/app/seeds.txt:ro
      - ./mcp_servers.json:/app/mcp_servers.json:ro
```

**Step 2: Update Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install uv (for uvx MCP server execution)
RUN pip install --no-cache-dir uv

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY ruminate.py .

# Create directories
RUN mkdir -p /app/logs

CMD ["python", "-u", "ruminate.py"]
```

Note: Removed `curl`, `wget`, `jq` apt packages (no longer needed — tool execution happens via MCP servers, not shell commands). Added `uv` for `uvx` to run MCP servers.

**Step 3: Commit**

```bash
git add docker-compose.yml Dockerfile
git commit -m "chore: simplify Docker setup, remove postgres/neo4j, add uv"
```

---

### Task 8: Update Environment Config and Docs

**Files:**
- Modify: `.env.example`
- Modify: `CLAUDE.md`

**Step 1: Update .env.example**

```
ANTHROPIC_API_KEY=your-anthropic-api-key
ANTHROPIC_MODEL=claude-sonnet-4-20250514
# ANTHROPIC_BASE_URL=https://custom-endpoint.example.com
MEM0_API_KEY=your-mem0-api-key

# Personality: default, philosopher, creative, clara, rumi (or path via PERSONALITY_FILE)
PERSONALITY_NAME=default

# Timing & Context
THOUGHT_INTERVAL=60
MAX_TOKENS=1024
MAX_HISTORY=10

# Discord (optional) - create webhook in channel settings
# DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

Remove `OPENAI_API_KEY`. Add `MEM0_API_KEY`.

**Step 2: Update CLAUDE.md**

Update these sections:
- **Architecture**: Replace the ASCII diagram and description with the new MCP-based architecture
- **Running the Application**: Remove references to postgres/neo4j healthchecks
- **Configuration**: Replace `OPENAI_API_KEY` with `MEM0_API_KEY`, add `MCP_SERVERS_FILE`
- **AI Capabilities**: Replace `::CMD::` command docs with MCP tool description
- **Dependencies**: Replace old deps with new

**Step 3: Commit**

```bash
git add .env.example CLAUDE.md
git commit -m "docs: update env config and CLAUDE.md for MCP architecture"
```

---

### Task 9: Verification

**Step 1: Verify the code is syntactically correct**

```bash
python -c "import ast; ast.parse(open('ruminate.py').read()); print('OK')"
```

Expected: `OK`

**Step 2: Verify imports resolve (with deps installed)**

```bash
pip install -r requirements.txt && python -c "import ruminate; print('imports OK')"
```

Expected: imports OK (or at least no ImportError — runtime config errors are fine since no MCP servers are running)

**Step 3: Docker build test**

```bash
docker compose build
```

Expected: builds successfully

**Step 4: Commit any fixes from verification**

---

## Execution Notes

- Tasks 1-2 are independent setup
- Tasks 3-6 are the core rewrite of `ruminate.py` (sequential, each builds on prior)
- Task 7-8 are independent of each other but depend on 3-6 being done
- Task 9 is final verification

The `connect_mcp_servers` function manages async context managers manually (storing them in `active_contexts`) rather than using nested `async with` blocks, because we need multiple servers to stay connected for the lifetime of the program. The `finally` block in `main()` cleans them up.
