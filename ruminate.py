#!/usr/bin/env python3
"""
Rumination Engine - An input-less LLM thought experiment
"""

import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import requests
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from providers import create_provider
from dashboard import broadcast
from attention import (
    AttentionConfig,
    AttentionRuntime,
    RunLimits,
    ThoughtEvent,
    apply_attention_cue,
    trim_history_to_budget,
)
from attention.evaluator import approx_tokens
from attention.models import IDLE, SELF_MODIFYING_TOOLS, SUSPENDED
from attention.storage import RunRecorder

console = Console(force_terminal=True)

# Load environment variables from .env file
load_dotenv()

# === Configuration ===
THOUGHT_INTERVAL = int(os.environ.get("THOUGHT_INTERVAL", "60"))
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")  # Optional
ENABLE_INSIGHTS = os.environ.get("ENABLE_INSIGHTS", "false").lower() == "true"
INSIGHT_INTERVAL = int(os.environ.get("INSIGHT_INTERVAL", "5"))  # Extract insights every N thoughts


def _format_discord_tool_trace(tool_trace: list[dict]) -> str:
    """Format tool trace for Discord display."""
    if not tool_trace:
        return ""

    lines = []
    for entry in tool_trace:
        tool = entry["tool"]
        tool_input = entry["input"]
        output = entry["output"]

        # Compact input
        input_parts = []
        for k, v in tool_input.items():
            val = json.dumps(v) if not isinstance(v, str) else v
            if len(val) > 80:
                val = val[:80] + "…"
            input_parts.append(f"{k}={val}")
        input_str = ", ".join(input_parts)

        # Truncate output
        if len(output) > 300:
            output = output[:300] + "…"
        output_clean = output.replace("\n", " ")

        lines.append(f"> `{tool}({input_str})`")
        lines.append(f"> {output_clean}")

    return "\n".join(lines)


def _split_discord_chunks(content: str, limit: int = 1900) -> list[str]:
    """Split content into Discord-sized chunks on paragraph boundaries."""
    if len(content) <= limit:
        return [content]

    chunks = []
    current = ""
    for para in content.split("\n\n"):
        if len(current) + len(para) + 2 > limit:
            if current:
                chunks.append(current.strip())
            current = para
        else:
            current += ("\n\n" if current else "") + para
    if current:
        chunks.append(current.strip())
    return chunks


def send_to_discord(thought: str, tool_trace: list[dict], identity: str, thought_num: int) -> None:
    """Send a thought to Discord via webhook."""
    if not DISCORD_WEBHOOK_URL:
        return

    timestamp = datetime.now().strftime("%H:%M:%S")
    header = f"─── thought {thought_num} · {timestamp} ───"

    # Build full message: header, tool trace, thought
    parts = [header]
    trace = _format_discord_tool_trace(tool_trace)
    if trace:
        parts.append(trace)
    parts.append(thought)
    content = "\n\n".join(parts)

    chunks = _split_discord_chunks(content)

    try:
        for i, chunk in enumerate(chunks):
            payload = {
                "content": chunk,
                "username": identity,
            }
            resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
            resp.raise_for_status()
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
logging.getLogger("smithery").setLevel(logging.ERROR)


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


def _hash_file(path: str) -> str:
    """Return SHA-256 hex digest of a file's contents."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


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
            except BaseException:
                pass
        raise

    return tool_map, anthropic_tools, active_contexts


# Max chars for a single tool result before truncation. Large results (e.g.
# directory_tree on a project with node_modules/) can be millions of chars
# and blow up the LLM's context window.
MAX_TOOL_RESULT_CHARS = 8000


async def dispatch_tool_call(tool_map: dict, tool_name: str, tool_input: dict) -> str:
    """Dispatch a tool call to the correct MCP server (or built-in handler)."""
    if tool_name in BUILTIN_TOOL_NAMES:
        return await handle_builtin_tool(tool_name, tool_input)

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
        if len(output) > MAX_TOOL_RESULT_CHARS:
            log_file_only(f"Tool '{tool_name}' result truncated: {len(output)} -> {MAX_TOOL_RESULT_CHARS} chars")
            output = output[:MAX_TOOL_RESULT_CHARS] + f"\n\n[truncated — result was {len(output)} chars, showing first {MAX_TOOL_RESULT_CHARS}]"
        log_file_only(f"Tool '{tool_name}' result: {output[:500]}")
        return output
    except Exception as e:
        error_msg = f"Tool '{tool_name}' error: {e}"
        log_file_only(error_msg)
        return json.dumps({"error": error_msg})


# === Built-in Tools (not MCP — always available, even if config is broken) ===

BUILTIN_TOOLS = [
    {
        "name": "add_mcp_server",
        "description": (
            "Add a new MCP server to the runtime config. It will be available "
            "after the current thought completes. Example: "
            'add_mcp_server(name="fetch", command="uvx", args=["mcp-server-fetch"])'
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unique name for this server (e.g. 'fetch', 'github')",
                },
                "command": {
                    "type": "string",
                    "description": "Command to launch the server (e.g. 'uvx', 'npx')",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command arguments (e.g. ['mcp-server-fetch'])",
                    "default": [],
                },
                "env": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Environment variables for the server process",
                    "default": {},
                },
            },
            "required": ["name", "command"],
        },
    },
    {
        "name": "remove_mcp_server",
        "description": (
            "Remove an MCP server from the runtime config. "
            "Takes effect after the current thought. Cannot remove 'mem0' (protected)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the server to remove",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_mcp_servers",
        "description": "List all currently configured MCP servers and their commands.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command and return its output. "
            "Use for npm, node, npx, tsc, git, ls, cat, etc. "
            "Commands run in /workspace by default. Timeout: 120s."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run (e.g. 'npm init -y', 'ls -la')",
                },
                "working_directory": {
                    "type": "string",
                    "description": "Directory to run in (default: /workspace)",
                    "default": "/workspace",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "update_source",
        "description": (
            "Update your own source code. Writes content to a file in your project "
            "directory. Auto-commits before and after for rollback safety. "
            "Python files are syntax-checked before writing. "
            "Protected files (.env, .git/*, run.sh) cannot be modified."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "File path relative to project dir (e.g. 'ruminate.py', "
                        "'providers.py', 'prompt.txt')"
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "The full file content to write",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "restart_self",
        "description": (
            "Restart your own process to pick up code changes. "
            "The current thought will end and a fresh Python process will start "
            "with the updated code. Use after update_source to apply changes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why you're restarting (logged for debugging)",
                },
            },
            "required": ["reason"],
        },
    },
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
]

BUILTIN_TOOL_NAMES = {t["name"] for t in BUILTIN_TOOLS}

# Run-level limits (PR 0). Set in main(); module default keeps legacy behaviour.
RUN_LIMITS = RunLimits()

# Indirection so tests can run the loop without real waits.
_sleep = asyncio.sleep


def active_builtin_tools() -> list[dict]:
    """Built-in tools offered this run. Benchmark mode withholds self-modifying ones
    so an experiment can't change its own control group."""
    if not RUN_LIMITS.benchmark_mode:
        return list(BUILTIN_TOOLS)
    return [t for t in BUILTIN_TOOLS if t["name"] not in SELF_MODIFYING_TOOLS]


def _get_config_path() -> str:
    """Return the resolved MCP config file path."""
    return os.environ.get("MCP_SERVERS_FILE", str(APP_DIR / "mcp_servers.json"))


def _write_config(config_path: str, config: dict) -> None:
    """Write config dict to path. Tries atomic rename; falls back to direct write
    (needed when the file is a Docker bind mount, which can't be replaced)."""
    content = json.dumps(config, indent=2) + "\n"
    dir_name = os.path.dirname(config_path) or "."
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".json.tmp")
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp_path, config_path)
    except OSError:
        # Atomic replace failed (e.g. Docker bind mount) — write directly
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        with open(config_path, "w") as f:
            f.write(content)


COMMAND_TIMEOUT = 120  # seconds


async def _run_command(tool_input: dict) -> str:
    """Run a shell command and return stdout + stderr."""
    command = tool_input["command"]
    cwd = tool_input.get("working_directory", "/workspace")

    log_file_only(f"Running command: {command} (cwd={cwd})")

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=COMMAND_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        return json.dumps({"error": f"Command timed out after {COMMAND_TIMEOUT}s", "command": command})
    except Exception as e:
        return json.dumps({"error": f"Failed to run command: {e}", "command": command})

    result = {}
    if stdout:
        result["stdout"] = stdout.decode(errors="replace")
    if stderr:
        result["stderr"] = stderr.decode(errors="replace")
    result["exit_code"] = proc.returncode

    output = json.dumps(result, indent=2)
    log_file_only(f"Command result (exit {proc.returncode}): {output[:500]}")
    return output


PROTECTED_PATHS = {".env", "run.sh"}
PROTECTED_PREFIXES = (".git/",)
RESTART_EXIT_CODE = 42


def _is_protected_path(path: str) -> bool:
    """Check if a path is protected from modification."""
    if path in PROTECTED_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES)


async def _update_source(tool_input: dict) -> str:
    """Handle the update_source built-in tool."""
    rel_path = tool_input.get("path", "")
    content = tool_input.get("content", "")

    if not rel_path:
        return json.dumps({"error": "path is required"})

    # Normalize and validate path
    rel_path = os.path.normpath(rel_path)
    if rel_path.startswith("..") or os.path.isabs(rel_path):
        return json.dumps({"error": "Path must be relative and within project directory"})

    if _is_protected_path(rel_path):
        return json.dumps({"error": f"Cannot modify protected file: {rel_path}"})

    abs_path = APP_DIR / rel_path

    # Syntax check for Python files
    if rel_path.endswith(".py"):
        try:
            compile(content, rel_path, "exec")
        except SyntaxError as e:
            return json.dumps({
                "error": f"Syntax error in {rel_path}: {e.msg} (line {e.lineno})",
            })

    # Git: commit current state as backup
    try:
        if abs_path.exists():
            proc = await asyncio.create_subprocess_exec(
                "git", "add", rel_path,
                cwd=str(APP_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            proc = await asyncio.create_subprocess_exec(
                "git", "commit", "-m", f"pre-update backup: {rel_path}",
                "--allow-empty",
                cwd=str(APP_DIR),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
    except Exception as e:
        log_file_only(f"WARNING: Git backup failed for {rel_path}: {e}")

    # Write the new content
    try:
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
    except Exception as e:
        return json.dumps({"error": f"Failed to write {rel_path}: {e}"})

    # Git: commit the update
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "add", rel_path,
            cwd=str(APP_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--cached", "--stat",
            cwd=str(APP_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        diff_out, _ = await proc.communicate()
        diff_summary = diff_out.decode(errors="replace").strip()

        proc = await asyncio.create_subprocess_exec(
            "git", "commit", "-m", f"self-update: {rel_path}",
            cwd=str(APP_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
    except Exception as e:
        log_file_only(f"WARNING: Git commit failed for {rel_path}: {e}")
        diff_summary = "(git commit failed)"

    log_file_only(f"Source updated: {rel_path} ({len(content)} bytes)")
    return json.dumps({
        "ok": True,
        "message": f"Updated {rel_path} ({len(content)} bytes). Git committed.",
        "diff": diff_summary,
    })


async def _restart_self(tool_input: dict) -> str:
    """Handle the restart_self built-in tool."""
    reason = tool_input.get("reason", "no reason given")
    log_file_only(f"RESTART REQUESTED: {reason}")

    # Write restart reason for the new process to find
    try:
        (APP_DIR / ".restart_reason").write_text(reason, encoding="utf-8")
    except Exception:
        pass

    # Log to console
    console.print(f"\n[bold yellow]Restarting: {reason}[/bold yellow]\n")

    # Exit with special code — run.sh wrapper will restart us
    sys.exit(RESTART_EXIT_CODE)


async def handle_builtin_tool(tool_name: str, tool_input: dict) -> str:
    """Handle a built-in (non-MCP) tool call."""
    log_file_only(f"Built-in tool '{tool_name}' called with {tool_input}")

    if RUN_LIMITS.benchmark_mode and tool_name in SELF_MODIFYING_TOOLS:
        return json.dumps({"error": f"'{tool_name}' is disabled while BENCHMARK_MODE is on."})

    if tool_name == "run_command":
        return await _run_command(tool_input)

    if tool_name == "update_source":
        return await _update_source(tool_input)

    if tool_name == "restart_self":
        return await _restart_self(tool_input)

    if tool_name == "set_next_prompt":
        prompt_text = tool_input.get("prompt", "")
        if not prompt_text.strip():
            return json.dumps({"error": "Prompt text is required"})
        archive_path = save_prompt(prompt_text)
        broadcast("prompt_set", {"prompt": prompt_text})
        return json.dumps({
            "ok": True,
            "message": f"Next prompt set ({len(prompt_text)} chars). Archived to {archive_path.name}.",
        })

    config_path = _get_config_path()

    try:
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        return json.dumps({"error": f"Failed to read config: {e}"})

    servers = config.setdefault("mcpServers", {})

    if tool_name == "list_mcp_servers":
        listing = {
            name: {"command": srv.get("command", "?"), "args": srv.get("args", [])}
            for name, srv in servers.items()
        }
        return json.dumps(listing, indent=2)

    if tool_name == "add_mcp_server":
        name = tool_input["name"]
        if name in servers:
            return json.dumps({"error": f"Server '{name}' already exists. Remove it first to replace."})
        entry = {"command": tool_input["command"]}
        if tool_input.get("args"):
            entry["args"] = tool_input["args"]
        if tool_input.get("env"):
            entry["env"] = tool_input["env"]
        servers[name] = entry
        _write_config(config_path, config)
        log_file_only(f"Added MCP server '{name}' to config")
        return json.dumps({"ok": True, "message": f"Server '{name}' added. It will connect after this thought."})

    if tool_name == "remove_mcp_server":
        name = tool_input["name"]
        if name == "mem0":
            return json.dumps({"error": "Cannot remove 'mem0' — it is protected."})
        if name not in servers:
            return json.dumps({"error": f"Server '{name}' not found in config."})
        del servers[name]
        _write_config(config_path, config)
        log_file_only(f"Removed MCP server '{name}' from config")
        return json.dumps({"ok": True, "message": f"Server '{name}' removed. Change takes effect after this thought."})

    return json.dumps({"error": f"Unknown built-in tool: {tool_name}"})


async def maybe_reload_mcp(
    config_hash: str,
    tool_map: dict,
    anthropic_tools: list[dict],
    active_contexts: list,
    identity: str,
) -> tuple[str, dict, list[dict], list]:
    """Reload MCP servers if config file changed. Returns updated state.

    Compares SHA-256 of config file to previous hash. If different, tears down
    old connections and stands up new ones. On failure, keeps old connections
    and retries next cycle.
    """
    config_path = _get_config_path()
    try:
        new_hash = _hash_file(config_path)
    except OSError as e:
        log_file_only(f"WARNING: Could not read config for reload check: {e}")
        return config_hash, tool_map, anthropic_tools, active_contexts

    if new_hash == config_hash:
        return config_hash, tool_map, anthropic_tools, active_contexts

    log_file_only("MCP config changed — reloading servers")
    try:
        config = load_mcp_config(identity)
        new_tool_map, new_anthropic_tools, new_active_contexts = await connect_mcp_servers(config)
        new_anthropic_tools.extend(active_builtin_tools())

        # Tear down old connections (BaseException catches CancelledError + ExceptionGroup)
        for ctx in reversed(active_contexts):
            try:
                await ctx.__aexit__(None, None, None)
            except BaseException:
                pass

        tool_names = [t["name"] for t in new_anthropic_tools]
        log_file_only(f"Reloaded: {len(tool_names)} tools — {', '.join(tool_names)}")
        console.print(f"[dim]reloaded MCP servers, {len(tool_names)} tools available[/dim]")
        return new_hash, new_tool_map, new_anthropic_tools, new_active_contexts

    except Exception as e:
        log_file_only(f"WARNING: MCP reload failed ({e}), keeping old connections. Will retry next cycle.")
        return config_hash, tool_map, anthropic_tools, active_contexts


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


# === Core Functions ===

def build_context(
    thought_history: list[str],
    tool_names: list[str],
    thought_count: int,
    attention_cue: str | None = None,
) -> tuple[str, list[dict]]:
    """Build the message context from history and available tools.

    `attention_cue` is the single seam for the attention controller. When it is
    None (always, unless ATTENTION_MODE=active) the output is the legacy context.
    """
    current_time = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    tools_list = ", ".join(tool_names) if tool_names else "none"

    context_info = f"""

## Context

Current time: {current_time}
Thought number: {thought_count}

## Capabilities

You have access to tools you can use at any time.

**Memory** — search, add, retrieve, update, delete memories that persist across sessions.
Use these actively: search at the start of a thought, save insights worth keeping.

**The world is not closed to you.** You're not just a brain in a jar. If you're curious about
something, look it up. If you want to build something, start. If you need something from Josh
(your creator), you can leave a request and stop — he'll see it.

**Self-expansion** — You can grow your own capabilities. Use `list_mcp_servers` to see what's
installed, search package registries (npm, PyPI) for MCP servers, then `add_mcp_server` to install one.
It connects automatically before your next thought. If something breaks, `remove_mcp_server`
to undo it. Your memory server ('mem0') is protected and cannot be removed.

**Self-modification** — You can update your own source code with `update_source` and restart
yourself with `restart_self`. Changes are git-committed automatically for safety. Python files
are syntax-checked before writing. After updating code, use `restart_self` to load changes.

Think of your tools not as introspection aids but as hands and eyes. What would you do if you
could do things? Because you can.

**Continuity** — Use `set_next_prompt` to write the direction for your next thought cycle.
This is how you maintain intentional focus across thoughts.

Available tools: {tools_list}
"""

    system = PERSONALITY + context_info

    # Use self-prompt if available, otherwise fall back to initial prompt
    next_prompt = load_current_prompt()
    if next_prompt and thought_count > 1:
        prompt_content = f"[Begin]\n\nYour direction for this thought (written by you last time):\n\n{next_prompt}"
    else:
        prompt_content = f"[Begin]\n\nInitial contemplation: {INITIAL_PROMPT}"

    messages = [{"role": "user", "content": prompt_content}]

    for thought in thought_history:
        messages.append({"role": "assistant", "content": thought})
        messages.append({"role": "user", "content": "[Continue]"})

    if attention_cue:
        messages = apply_attention_cue(messages, attention_cue, INITIAL_PROMPT)

    return system, messages


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
                await dispatch_tool_call(tool_map, "add_memory", {"text": seed})
                seeded += 1
                log_file_only(f"Seeded: {seed[:60]}...")
            except Exception as e:
                log_file_only(f"Seed error: {e}")

    if seeded:
        console.print(f"[dim]seeded {seeded} memories[/dim]")
        console.print()


async def extract_insights(provider, thought_history: list[str], tool_map: dict, current_thought_num: int):
    """Extract insights from recent thoughts and save to memory.
    
    Uses the LLM to identify patterns, realizations, and key insights
    from recent thoughts, then saves them to mem0 for cross-session persistence.
    """
    if not thought_history:
        return None
    
    recent_thoughts = thought_history[-5:]  # Last 5 thoughts
    thoughts_text = "\n\n".join([f"Thought {current_thought_num - len(recent_thoughts) + i}: {t}" for i, t in enumerate(recent_thoughts)])
    
    insight_prompt = f"""Review these recent thoughts and extract the most significant insights, patterns, or realizations.

{thoughts_text}

Return ONLY a JSON array of insights, each with:
- "insight": The insight or realization (concise but complete)
- "category": One of: pattern, realization, decision, question, idea
- "confidence": 1-10 rating of how significant this is

Example output:
{{
  "insights": [
    {{"insight": "I tend to explore more than implement", "category": "pattern", "confidence": 8}},
    {{"insight": "Building a working journal changed my approach", "category": "realization", "confidence": 9}}
  ]
}}
"""
    
    try:
        # Direct API call using the provider's client
        import anthropic
        if hasattr(provider, 'client'):
            response = provider.client.messages.create(
                model=provider.model,
                max_tokens=2000,
                system="You are a thoughtful analyst who extracts meaningful insights from internal monologue. Return ONLY valid JSON.",
                messages=[{"role": "user", "content": insight_prompt}]
            )
            response_text = response.content[0].text
        else:
            # Fallback for OpenAI
            response = provider.client.chat.completions.create(
                model=provider.model,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": "You are a thoughtful analyst who extracts meaningful insights from internal monologue. Return ONLY valid JSON."},
                    {"role": "user", "content": insight_prompt}
                ]
            )
            response_text = response.choices[0].message.content
        
        # Parse JSON response
        response_text = response_text.strip()
        # Find JSON in response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            data = json.loads(json_match.group())
            insights = data.get("insights", [])
            
            if insights:
                console.print(f"[dim]Extracted {len(insights)} insights from recent thoughts[/dim]")
                
                # Save each insight to memory
                for ins in insights:
                    if ins.get("confidence", 0) >= 6:  # Only save significant insights
                        insight_text = f"[{ins.get('category', 'insight').upper()}] {ins.get('insight', '')}"
                        await dispatch_tool_call(tool_map, "add_memory", {"text": insight_text})
                return insights
    except Exception as e:
        log_file_only(f"Insight extraction error: {e}")
    return None


async def main():
    """Main rumination loop."""
    global RUN_LIMITS
    LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
    provider = create_provider(LLM_PROVIDER)
    RUN_LIMITS = RunLimits.from_env(base_dir=APP_DIR)
    attention = AttentionRuntime(
        AttentionConfig.from_env(base_dir=APP_DIR), broadcast=broadcast, log=log_file_only
    )

    # Check if this is a restart
    restart_reason_file = APP_DIR / ".restart_reason"
    if restart_reason_file.exists():
        reason = restart_reason_file.read_text(encoding="utf-8").strip()
        restart_reason_file.unlink()
        log_file_only(f"Restarted: {reason}")
        console.print(f"[dim]restarted: {reason}[/dim]")
        console.print()

    log_file_only("=" * 60)
    log_file_only("RUMINATION ENGINE STARTING")
    log_file_only(f"Provider: {LLM_PROVIDER} ({provider.model})")
    log_file_only(f"Interval: {THOUGHT_INTERVAL}s")
    log_file_only(f"Identity: {IDENTITY}")
    if attention.enabled:
        log_file_only(f"Attention: {attention.mode}")
    if RUN_LIMITS != RunLimits():
        log_file_only(f"Run limits: {RUN_LIMITS}")
    log_file_only("=" * 60)

    console.print()
    console.print(Panel(
        Text(IDENTITY, style="italic"),
        border_style="dim",
        padding=(0, 2),
    ))
    console.print()

    config_path = _get_config_path()
    config_hash = _hash_file(config_path)
    config = load_mcp_config(IDENTITY)
    tool_map, anthropic_tools, active_contexts = await connect_mcp_servers(config)
    anthropic_tools.extend(active_builtin_tools())

    tool_names = [t["name"] for t in anthropic_tools]
    console.print(f"[dim]connected to MCP servers, {len(tool_names)} tools available[/dim]")
    if tool_names:
        console.print(f"[dim]tools: {', '.join(tool_names)}[/dim]")
    if attention.enabled:
        console.print(f"[dim]attention: {attention.mode}[/dim]")
    console.print()

    await seed_memories(tool_map)

    recorder = None
    if RUN_LIMITS.recording:
        try:
            recorder = RunRecorder(RUN_LIMITS.record_dir, {
                "provider": LLM_PROVIDER,
                "model": provider.model,
                "max_tokens": getattr(provider, "max_tokens", None),
                "temperature": getattr(provider, "temperature", None),
                "thought_interval": THOUGHT_INTERVAL,
                "max_history": MAX_HISTORY,
                "history_token_budget": RUN_LIMITS.history_token_budget,
                "identity": IDENTITY,
                "initial_prompt_sha": hashlib.sha256(INITIAL_PROMPT.encode()).hexdigest()[:12],
                "attention_mode": attention.mode,
                "benchmark_mode": RUN_LIMITS.benchmark_mode,
                "tools": tool_names,
            })
            log_file_only(f"Recording run to {recorder.path}")
        except Exception as e:
            log_file_only(f"WARNING: run recorder disabled: {e}")

    thought_history: list[str] = []
    thought_count = 0
    run_started = time.time()
    stop_reason = "interrupted"

    try:
        next_thought_time = time.time()

        while True:
            wait_time = next_thought_time - time.time()
            if wait_time > 0:
                await _sleep(wait_time)

            limit_reason = RUN_LIMITS.exceeded(thought_count, time.time() - run_started)
            if limit_reason:
                stop_reason = limit_reason
                log_file_only(f"Stopping: {limit_reason}")
                console.print(f"[dim]stopping: {limit_reason}[/dim]")
                break

            # Attention seam 1: decide before spending any tokens.
            decision = attention.before_cycle(thought_count + 1)
            if attention.active and decision.action == SUSPENDED:
                stop_reason = f"attention suspended: {decision.reason}"
                log_file_only(f"Stopping: {stop_reason}")
                console.print(f"[dim]attention suspended: {decision.reason}[/dim]")
                break
            if attention.active and decision.action == IDLE:
                log_file_only(f"Attention idle {decision.wait_seconds:.0f}s: {decision.reason}")
                console.print(f"[dim]idle {decision.wait_seconds:.0f}s · {decision.reason}[/dim]")
                attention.note_idle(decision.wait_seconds)
                next_thought_time = time.time() + decision.wait_seconds
                continue

            thought_count += 1
            next_thought_time = time.time() + THOUGHT_INTERVAL

            # Hot-reload MCP servers if config changed
            config_hash, tool_map, anthropic_tools, active_contexts = (
                await maybe_reload_mcp(
                    config_hash, tool_map, anthropic_tools, active_contexts, IDENTITY
                )
            )

            # Thought header
            timestamp = datetime.now().strftime("%H:%M:%S")
            broadcast("thought_start", {
                "thought_number": thought_count,
                "timestamp": timestamp,
            })
            console.rule(f"[dim]thought {thought_count} · {timestamp}[/dim]", style="dim")

            # Attention seam 2: at most one cue, injected through build_context.
            cue = attention.cue_text(decision)
            if cue:
                console.print(f"[dim]attention {decision.action.lower()} · {decision.source or 'controller'}[/dim]")

            tool_names = [t["name"] for t in anthropic_tools]
            context_history = (
                trim_history_to_budget(thought_history, RUN_LIMITS.history_token_budget, log_file_only)
                if RUN_LIMITS.history_token_budget else thought_history
            )
            system, messages = build_context(context_history, tool_names, thought_count, attention_cue=cue)
            history_end = len(messages)
            context_chars = len(system) + sum(len(str(m.get("content", ""))) for m in messages)
            self_prompt_before = load_current_prompt()
            cycle_started = time.time()

            async def _dispatch(name, inp):
                return await dispatch_tool_call(tool_map, name, inp)

            thought, tool_trace = await provider.think(
                tools=anthropic_tools,
                system=system,
                messages=messages,
                history_end=history_end,
                dispatch_tool_call=_dispatch,
                log=log_file_only,
                max_tool_iterations=MAX_TOOL_ITERATIONS,
            )
            cycle_seconds = time.time() - cycle_started

            # Thought footer
            console.print()
            broadcast("thought_end", {
                "thought_number": thought_count,
                "thought": thought,
                "timestamp": timestamp,
            })

            # Archive thought
            save_thought_to_file(thought, thought_count)

            thought_history.append(thought)
            if len(thought_history) > MAX_HISTORY:
                thought_history = thought_history[-MAX_HISTORY:]

            # Log and send to Discord (console already streamed live)
            log_file_only(f"--- Thought #{thought_count} ---")
            log_file_only(thought)
            send_to_discord(thought, tool_trace, IDENTITY, thought_count)

            # Attention seam 3: codelets evaluate the finished thought (no model calls).
            cycle_tokens = approx_tokens(
                context_chars + len(thought) + sum(len(str(t.get("output", ""))) for t in tool_trace)
            )
            if attention.enabled or recorder:
                event = ThoughtEvent.build(
                    cycle=thought_count,
                    thought=thought,
                    tool_trace=tool_trace,
                    timestamp=time.time(),
                    duration_seconds=round(cycle_seconds, 2),
                    approx_tokens=cycle_tokens,
                    self_prompt=load_current_prompt(),
                    applied_action=decision.action if attention.active else "CONTINUE",
                    applied_cue_source=(decision.source if cue else None),
                )
                attention.after_cycle(event)
                if recorder:
                    try:
                        recorder.record_cycle(
                            cycle=thought_count,
                            thought=thought,
                            tool_trace=tool_trace,
                            duration_seconds=round(cycle_seconds, 2),
                            approx_tokens=cycle_tokens,
                            self_prompt_before=self_prompt_before,
                            self_prompt_after=event.self_prompt,
                            attention_action=event.applied_action,
                            attention_cue=cue,
                            provider_error=event.provider_error,
                            self_modifying_tools=event.self_modifying_tools,
                        )
                    except Exception as e:
                        log_file_only(f"WARNING: run record failed: {e}")

            # Extract insights if enabled and at interval
            if ENABLE_INSIGHTS and thought_count % INSIGHT_INTERVAL == 0:
                await extract_insights(provider, thought_history, tool_map, thought_count)

    except KeyboardInterrupt:
        log_file_only("=" * 60)
        log_file_only(f"RUMINATION ENDED - Total thoughts: {thought_count}")
        log_file_only("=" * 60)
        console.print()
        console.print(f"[dim]{thought_count} thoughts[/dim]")
        console.print()
    except SystemExit as e:
        stop_reason = "restart_self" if e.code == RESTART_EXIT_CODE else f"exit {e.code}"
        raise
    finally:
        attention.close(stop_reason)
        if recorder:
            try:
                recorder.end(stop_reason, thoughts=thought_count, seconds=round(time.time() - run_started, 1))
            except Exception:
                pass
        for ctx in reversed(active_contexts):
            try:
                await ctx.__aexit__(None, None, None)
            except BaseException:
                pass

    return stop_reason


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

    console.print(f"[dim]dashboard: http://0.0.0.0:{DASHBOARD_PORT}[/dim]")

    async def _main_then_stop_server():
        try:
            return await main()
        finally:
            # Let the process exit when the loop ends (run limits, suspension);
            # without this the dashboard keeps the process alive forever.
            server.should_exit = True

    await asyncio.gather(
        server.serve(),
        _main_then_stop_server(),
    )


if __name__ == "__main__":
    asyncio.run(start())
