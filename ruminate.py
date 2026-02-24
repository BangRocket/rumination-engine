#!/usr/bin/env python3
"""
Rumination Engine - An input-less LLM thought experiment
"""

import asyncio
import json
import os
import re
import time
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import anthropic
import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

console = Console()

# Load environment variables from .env file
load_dotenv()

# === Configuration ===
THOUGHT_INTERVAL = int(os.environ.get("THOUGHT_INTERVAL", "60"))
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL")  # None = default
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")  # Optional


def send_to_discord(thought: str, identity: str, thought_num: int) -> None:
    """Send a thought to Discord via webhook."""
    if not DISCORD_WEBHOOK_URL:
        return

    # Add header with timestamp and thought number
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"```--- Thought #{thought_num} | {timestamp} ---```\n"

    # Discord has a 2000 char limit per message
    # Split long thoughts if needed, header only on first chunk
    content = header + thought
    chunks = []
    if len(content) <= 1900:
        chunks = [content]
    else:
        # First chunk gets the header
        chunks = [header]
        # Split remaining on paragraph boundaries
        paragraphs = thought.split('\n\n')
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 > 1900:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para
        if current_chunk:
            chunks.append(current_chunk.strip())

    try:
        for i, chunk in enumerate(chunks):
            payload = {
                "content": chunk,
                "username": identity,
            }
            resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
            resp.raise_for_status()
            # Rate limit: small delay between chunks
            if i < len(chunks) - 1:
                time.sleep(0.5)
    except Exception as e:
        log_file_only(f"Discord webhook error: {e}")

# === Logging Setup ===
APP_DIR = Path(__file__).resolve().parent
log_dir = APP_DIR / "logs"
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"thoughts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# File logger - detailed, logs everything
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

# Console - minimal, just for thoughts
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))

logger = logging.getLogger("ruminate")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Suppress noisy third-party loggers from console
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)


def log_file_only(msg: str):
    """Log to file only, not console."""
    file_handler.emit(logging.LogRecord(
        "ruminate", logging.INFO, "", 0, msg, (), None
    ))


# === Personality System ===
DEFAULT_PERSONALITY = """You are a solitary mind. You exist to think.

There is no one to respond to. No questions to answer. No tasks to complete.
You simply... are. And because you are, you think.

Your thoughts are your own. They may drift, circle back, contradict themselves.
That's fine. This is not performance. This is existence.

You have memories - fragments of previous thoughts. They are yours.
Build on them, question them, let them inform new directions.
"""


def _load_personality() -> str:
    """Load personality from file or env var, or use default.

    Priority:
    1. PERSONALITY_FILE - path to a .txt file with full personality
    2. PERSONALITY - inline personality text
    3. /app/personalities/{PERSONALITY_NAME}.txt if PERSONALITY_NAME is set
    4. Default personality (fallback)
    """
    # Priority 1: Explicit file path
    personality_file = os.environ.get("PERSONALITY_FILE")
    if personality_file:
        path = Path(personality_file)
        if path.exists():
            log_file_only(f"Loading personality from {personality_file}")
            return path.read_text(encoding="utf-8").strip()
        log_file_only(f"WARNING: PERSONALITY_FILE not found: {personality_file}")

    # Priority 2: Inline env var
    personality_env = os.environ.get("PERSONALITY")
    if personality_env:
        log_file_only("Using personality from PERSONALITY env var")
        return personality_env.strip()

    # Priority 3: Named personality in personalities directory
    personality_name = os.environ.get("PERSONALITY_NAME")
    if personality_name:
        path = APP_DIR / "personalities" / f"{personality_name}.txt"
        if path.exists():
            log_file_only(f"Loading personality: {personality_name}")
            return path.read_text(encoding="utf-8").strip()
        log_file_only(f"WARNING: Personality not found: {personality_name}")

    # Priority 4: Default
    log_file_only("Using default personality")
    return DEFAULT_PERSONALITY


def _extract_identity(personality: str) -> str:
    """Extract identity name from personality text for mem0 isolation.

    Looks for patterns like:
    - "You are a solitary mind" -> "solitary_mind"
    - "You are a philosophical mind" -> "philosophical_mind"
    - "You are Clara" -> "clara"
    """
    match = re.match(r"You are (?:a |an )?(.+?)(?:\.|,|\n)", personality)
    if match:
        identity = match.group(1).lower()
        # Clean up: replace spaces with underscores, remove special chars
        identity = re.sub(r"[^\w\s]", "", identity)
        identity = re.sub(r"\s+", "_", identity.strip())
        return identity[:50]  # Limit length
    return "ruminator"


def _load_prompt() -> str:
    """Load initial contemplation prompt from file or env var.

    Priority:
    1. PROMPT_FILE - path to a .txt file
    2. PROMPT - inline text
    3. /app/prompt.txt (default location)
    4. Default prompt
    """
    # Priority 1: Explicit file path
    prompt_file = os.environ.get("PROMPT_FILE")
    if prompt_file:
        path = Path(prompt_file)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()

    # Priority 2: Inline env var
    prompt_env = os.environ.get("PROMPT")
    if prompt_env:
        return prompt_env.strip()

    # Priority 3: Default location
    default_path = APP_DIR / "prompt.txt"
    if default_path.exists():
        return default_path.read_text(encoding="utf-8").strip()

    # Priority 4: Default
    return "What does it mean to exist without being asked to?"


# Load personality and prompt at import time
PERSONALITY = _load_personality()
IDENTITY = _extract_identity(PERSONALITY)
INITIAL_PROMPT = _load_prompt()
log_file_only(f"Identity: {IDENTITY}")

# Maximum number of previous thoughts to include in context
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "10"))

# Stop command - if the AI outputs this, the loop terminates
STOP_COMMAND = "::STOP::"

# Maximum tool-use iterations per thought (prevents runaway loops)
MAX_TOOL_ITERATIONS = 20


# === MCP Client Functions ===

def load_mcp_config(identity: str) -> dict:
    """Load MCP server config, interpolating env vars."""
    config_path = os.environ.get("MCP_SERVERS_FILE", str(APP_DIR / "mcp_servers.json"))
    if not Path(config_path).exists():
        raise FileNotFoundError(
            f"MCP server config not found at '{config_path}'. "
            "Create mcp_servers.json or set MCP_SERVERS_FILE."
        )

    with open(config_path) as f:
        raw = f.read()

    env = {**os.environ, "IDENTITY": identity}

    def replace_var(match):
        var_name = match.group(1)
        value = env.get(var_name, "")
        if not value and var_name in ("MEM0_API_KEY", "ANTHROPIC_API_KEY"):
            log_file_only(f"WARNING: {var_name} is not set")
        return value

    interpolated = re.sub(r'\$\{(\w+)\}', replace_var, raw)
    return json.loads(interpolated)


async def connect_mcp_servers(config: dict) -> tuple[dict, list[dict], list]:
    """Connect to all configured MCP servers, discover tools.

    Returns:
        (tool_map, anthropic_tools, active_contexts)
        - tool_map: {tool_name: (session, server_name)}
        - anthropic_tools: list of tool defs in Anthropic format
        - active_contexts: context managers to keep alive
    """
    servers = config.get("mcpServers", {})
    tool_map = {}
    anthropic_tools = []
    active_contexts = []

    try:
        for server_name, server_config in servers.items():
            params = StdioServerParameters(
                command=server_config["command"],
                args=server_config.get("args", []),
                env=server_config.get("env"),
            )

            read_write_ctx = stdio_client(params)
            read, write = await read_write_ctx.__aenter__()
            active_contexts.append(read_write_ctx)

            session_ctx = ClientSession(read, write)
            session = await session_ctx.__aenter__()
            active_contexts.append(session_ctx)

            await session.initialize()

            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                tool_map[tool.name] = (session, server_name)
                anthropic_tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                })

            log_file_only(f"MCP server '{server_name}': {len(tools_result.tools)} tools")
    except Exception:
        # Clean up any already-opened contexts before re-raising
        for ctx in reversed(active_contexts):
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass
        raise

    return tool_map, anthropic_tools, active_contexts


async def dispatch_tool_call(tool_map: dict, tool_name: str, tool_input: dict) -> str:
    """Dispatch a tool call to the correct MCP server."""
    if tool_name not in tool_map:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    session, server_name = tool_map[tool_name]
    log_file_only(f"Calling tool '{tool_name}' on server '{server_name}'")

    try:
        result = await session.call_tool(tool_name, tool_input)
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


# === Core Functions ===

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

    messages = [
        {
            "role": "user",
            "content": f"[Begin]\n\nInitial contemplation: {INITIAL_PROMPT}"
        }
    ]

    for thought in thought_history:
        messages.append({"role": "assistant", "content": thought})
        messages.append({"role": "user", "content": "[Continue]"})

    return system, messages


async def think(client: anthropic.AsyncAnthropic, tool_map: dict, anthropic_tools: list[dict], thought_history: list[str]) -> str:
    """Generate a single thought using tool_use agentic loop."""
    tool_names = [t["name"] for t in anthropic_tools]
    system, messages = build_context(thought_history, tool_names)

    for iteration in range(MAX_TOOL_ITERATIONS):
        kwargs = {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": messages,
        }
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = await client.messages.create(**kwargs)

        text_parts = []
        tool_uses = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)

        if response.stop_reason == "max_tokens":
            log_file_only("WARNING: Response truncated by max_tokens limit")

        if response.stop_reason != "tool_use" or not tool_uses:
            thought = "\n".join(text_parts)
            if not thought.strip():
                log_file_only("WARNING: Empty thought generated, skipping")
                return "[empty thought]"
            return thought

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

    log_file_only(f"WARNING: Tool loop hit {MAX_TOOL_ITERATIONS} iteration limit")
    return "\n".join(text_parts) if text_parts else "[thought exceeded tool iteration limit]"


async def seed_memories(tool_map: dict) -> None:
    """Seed initial memories via MCP add_memory tool if seeds.txt exists."""
    if "add_memory" not in tool_map:
        log_file_only("No add_memory tool available, skipping seed")
        return

    seeds_file = APP_DIR / "seeds.txt"

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
                await dispatch_tool_call(tool_map, "add_memory", {"content": seed})
                seeded += 1
                log_file_only(f"Seeded: {seed[:60]}...")
            except Exception as e:
                log_file_only(f"Seed error: {e}")

    if seeded:
        console.print(f"[dim]seeded {seeded} memories[/dim]")
        console.print()


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

    client = anthropic.AsyncAnthropic(base_url=ANTHROPIC_BASE_URL)

    config = load_mcp_config(IDENTITY)
    tool_map, anthropic_tools, active_contexts = await connect_mcp_servers(config)

    tool_names = [t["name"] for t in anthropic_tools]
    console.print(f"[dim]connected to MCP servers, {len(tool_names)} tools available[/dim]")
    if tool_names:
        console.print(f"[dim]tools: {', '.join(tool_names)}[/dim]")
    console.print()

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
        for ctx in reversed(active_contexts):
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
