#!/usr/bin/env python3
"""
Rumination Engine - An input-less LLM thought experiment
"""

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
from rich.status import Status

console = Console()

# Load environment variables from .env file
load_dotenv()
from mem0 import Memory
import mem0.memory.graph_memory as graph_memory_module


def _sanitize_neo4j_identifier(name: str) -> str:
    """Sanitize a string to be a valid Neo4j relationship type or label.

    Neo4j identifiers can contain letters, numbers, and underscores.
    Hyphens and other special chars are invalid without backticks.
    """
    if not name:
        return name
    # Replace hyphens and spaces with underscores
    sanitized = re.sub(r'[-\s]+', '_', name)
    # Remove any other non-alphanumeric chars (except underscore)
    sanitized = re.sub(r'[^\w]', '', sanitized)
    # Ensure it doesn't start with a number
    if sanitized and sanitized[0].isdigit():
        sanitized = '_' + sanitized
    return sanitized or 'RELATED_TO'


# Monkey-patch mem0's graph memory to sanitize relationship types
_original_add_entities = graph_memory_module.MemoryGraph._add_entities

def _patched_add_entities(self, to_be_added, filters, entity_type_map):
    """Patched version that sanitizes relationship types before Neo4j queries."""
    # Sanitize relationship types in the entities to be added
    for item in to_be_added:
        if 'relation' in item:
            item['relation'] = _sanitize_neo4j_identifier(item['relation'])
    return _original_add_entities(self, to_be_added, filters, entity_type_map)

graph_memory_module.MemoryGraph._add_entities = _patched_add_entities

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
log_dir = Path("/app/logs")
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
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("mem0").setLevel(logging.WARNING)


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
        path = Path(f"/app/personalities/{personality_name}.txt")
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
    default_path = Path("/app/prompt.txt")
    if default_path.exists():
        return default_path.read_text(encoding="utf-8").strip()

    # Priority 4: Default
    return "What does it mean to exist without being asked to?"


# Load personality and prompt at import time
PERSONALITY = _load_personality()
IDENTITY = _extract_identity(PERSONALITY)
INITIAL_PROMPT = _load_prompt()
log_file_only(f"Identity: {IDENTITY}")

# === Mem0 Configuration ===
mem0_config = {
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": f"bolt://{os.environ.get('NEO4J_HOST', 'neo4j')}:7687",
            "username": "neo4j",
            "password": "ruminate123",
        }
    },
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "host": os.environ.get("POSTGRES_HOST", "postgres"),
            "port": 5432,
            "user": "ruminate",
            "password": "ruminate",
            "dbname": "mem0",
        }
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4o-mini",
            "api_key": os.environ.get("OPENAI_API_KEY"),
        }
    },
    "embedder": {
        "provider": "openai",
        "config": {
            "api_key": os.environ.get("OPENAI_API_KEY"),
        }
    }
}


# Maximum number of previous thoughts to include in context
MAX_HISTORY = int(os.environ.get("MAX_HISTORY", "10"))

# Stop command - if the AI outputs this, the loop terminates
STOP_COMMAND = "::STOP::"

# Command execution pattern
CMD_PATTERN = re.compile(r'::CMD::(.*?)::END::', re.DOTALL)

# Allowed commands (security whitelist)
ALLOWED_COMMANDS = {'curl', 'wget', 'ls', 'cat', 'head', 'tail', 'wc', 'date', 'echo', 'mkdir', 'touch', 'rm', 'mv', 'cp', 'find', 'grep', 'jq'}


def execute_command(cmd: str) -> str:
    """Execute a shell command safely and return output."""
    import subprocess

    cmd = cmd.strip()
    if not cmd:
        return "[error: empty command]"

    # Extract base command for whitelist check
    base_cmd = cmd.split()[0] if cmd.split() else ""

    if base_cmd not in ALLOWED_COMMANDS:
        return f"[error: command '{base_cmd}' not allowed. Allowed: {', '.join(sorted(ALLOWED_COMMANDS))}]"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd="/data"
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr: {result.stderr}]"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "[no output]"
    except subprocess.TimeoutExpired:
        return "[error: command timed out (30s limit)]"
    except Exception as e:
        return f"[error: {e}]"


def build_context(memory: Memory, thought_history: list[str]) -> tuple[str, list[dict]]:
    """Build the message context from memories, history, and original prompt."""

    # Retrieve relevant memories
    memories = memory.get_all(user_id=IDENTITY, limit=20)

    memory_text = ""
    if memories and memories.get("results"):
        memory_text = "\n\n## Your Memories\n"
        for mem in memories["results"]:
            memory_text += f"- {mem['memory']}\n"

    # Add current time and stop command information
    current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    context_info = f"""

## Context

Current time: {current_time}

## Capabilities

You have a persistent workspace at /data where you can create, read, and organize files.
This space persists across sessions - anything you write there will remain.

You can execute commands using the syntax:
::CMD::your command here::END::

Available commands: {', '.join(sorted(ALLOWED_COMMANDS))}

Examples:
- ::CMD::curl -s https://example.com::END:: (fetch a webpage)
- ::CMD::echo "a thought" >> /data/notes.txt::END:: (append to a file)
- ::CMD::cat /data/notes.txt::END:: (read a file)
- ::CMD::ls -la /data::END:: (list your files)
- ::CMD::date::END:: (get current system time)

Commands run in /data with a 30-second timeout. Output will be provided before your next thought.

## Control

If you feel complete - if a line of thought has reached its natural end, or you simply wish to rest - you may stop.
To stop ruminating, output exactly: {STOP_COMMAND}
This will end the session until you are restarted.
"""

    system = PERSONALITY + memory_text + context_info

    # Always start with the original prompt
    messages = [
        {
            "role": "user",
            "content": f"[Begin]\n\nInitial contemplation: {INITIAL_PROMPT}"
        }
    ]

    # Add previous thoughts as conversation history
    for thought in thought_history:
        messages.append({
            "role": "assistant",
            "content": thought
        })
        messages.append({
            "role": "user",
            "content": "[Continue]"
        })

    return system, messages


def think(client: anthropic.Anthropic, memory: Memory, thought_history: list[str]) -> str:
    """Generate a single thought."""

    system, messages = build_context(memory, thought_history)

    # Use streaming to handle long requests
    thought_parts = []
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages
    ) as stream:
        for text in stream.text_stream:
            thought_parts.append(text)

    thought = "".join(thought_parts)

    # Store in memory (let mem0 decide what's worth keeping)
    # Graph store can fail on edge cases (hyphens in relationship names, etc.)
    try:
        memory.add(thought, user_id=IDENTITY)
    except Exception as e:
        log_file_only(f"Memory storage error (continuing): {e}")

    return thought


def seed_memories(memory: Memory) -> None:
    """Seed initial memories if none exist for this identity."""
    existing = memory.get_all(user_id=IDENTITY, limit=1)
    if existing and existing.get("results"):
        log_file_only(f"Found existing memories for {IDENTITY}, skipping seed")
        return

    # Check for seeds file
    seeds_file = Path("/app/seeds.txt")
    if not seeds_file.exists():
        seeds_file = Path("seeds.txt")  # Local dev fallback

    if seeds_file.exists():
        log_file_only(f"Seeding memories from {seeds_file}")
        seeds = seeds_file.read_text(encoding="utf-8").strip().split("\n---\n")
        for seed in seeds:
            seed = seed.strip()
            if seed:
                try:
                    memory.add(seed, user_id=IDENTITY)
                    log_file_only(f"Seeded: {seed[:60]}...")
                except Exception as e:
                    log_file_only(f"Seed error: {e}")
        console.print(f"[dim]seeded {len(seeds)} memories[/dim]")
        console.print()
    else:
        log_file_only("No seeds.txt found, starting fresh")


def main():
    """Main rumination loop."""

    # File-only startup logging
    log_file_only("=" * 60)
    log_file_only("RUMINATION ENGINE STARTING")
    log_file_only(f"Model: {MODEL}")
    log_file_only(f"Interval: {THOUGHT_INTERVAL}s")
    log_file_only(f"Identity: {IDENTITY}")
    log_file_only("=" * 60)

    # Rich console header
    console.print()
    console.print(Panel(
        Text(IDENTITY, style="italic"),
        border_style="dim",
        padding=(0, 2),
    ))
    console.print()

    # Initialize clients
    client = anthropic.Anthropic(base_url=ANTHROPIC_BASE_URL)
    memory = Memory.from_config(mem0_config)

    # Seed initial memories if starting fresh
    seed_memories(memory)

    thought_history: list[str] = []
    thought_count = 0

    try:
        next_thought_time = time.time()

        while True:
            # Wait until next scheduled time (or continue immediately if past due)
            wait_time = next_thought_time - time.time()
            if wait_time > 0:
                time.sleep(wait_time)

            thought_count += 1
            log_file_only(f"--- Thought #{thought_count} ---")

            # Schedule next thought from now (so interval is between starts, not ends)
            next_thought_time = time.time() + THOUGHT_INTERVAL

            # Show spinner while thinking
            with console.status("[dim]ruminating...[/dim]", spinner="dots"):
                thought = think(client, memory, thought_history)

            # Check for stop command
            if STOP_COMMAND in thought:
                # Remove the stop command from output
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

            # Check for and execute commands
            commands = CMD_PATTERN.findall(thought)
            cmd_results = []
            if commands:
                with console.status("[dim]executing commands...[/dim]", spinner="dots"):
                    for cmd in commands:
                        cmd = cmd.strip()
                        log_file_only(f"Executing command: {cmd}")
                        result = execute_command(cmd)
                        log_file_only(f"Command output: {result[:500]}...")
                        cmd_results.append(f"$ {cmd}\n{result}")
                console.print(f"[dim]executed {len(commands)} command(s)[/dim]")
                console.print()

            # Add to history (keep only last MAX_HISTORY thoughts)
            thought_history.append(thought)

            # If commands were executed, add results as a system message
            if cmd_results:
                cmd_output = "\n\n".join(cmd_results)
                thought_history.append(f"[Command Results]\n{cmd_output}")
            if len(thought_history) > MAX_HISTORY:
                thought_history = thought_history[-MAX_HISTORY:]

            # Log to file
            log_file_only(thought)

            # Rich console output
            console.print(Markdown(thought))
            console.print()

            # Send to Discord if configured
            send_to_discord(thought, IDENTITY, thought_count)

    except KeyboardInterrupt:
        log_file_only("=" * 60)
        log_file_only(f"RUMINATION ENDED - Total thoughts: {thought_count}")
        log_file_only("=" * 60)
        console.print()
        console.print(f"[dim]{thought_count} thoughts[/dim]")
        console.print()


if __name__ == "__main__":
    main()
