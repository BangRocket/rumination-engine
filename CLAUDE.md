# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rumination Engine is an input-less LLM thought experiment. It runs a continuous loop where Claude generates "thoughts" without external prompts, using mem0 for persistent memory across sessions.

## Architecture

**Single-file Python application** (`ruminate.py`) with three external services:

```
ruminate.py
    │
    ├── Anthropic API (Claude) → generates thoughts
    │
    └── mem0 Memory System
            ├── PostgreSQL/pgvector → vector store for semantic memory
            └── Neo4j → graph store for memory relationships
```

The main loop:
1. Retrieves memories from mem0 (up to 20)
2. Builds context with system prompt + memories + thought history + original prompt
3. Calls Claude (streaming) to generate next thought
4. Executes any `::CMD::command::END::` commands found in the output
5. Stores new thought in mem0
6. Waits `THOUGHT_INTERVAL` seconds, repeats

The AI can output `::STOP::` to terminate the session.

## Running the Application

```bash
# Start all services (requires ANTHROPIC_API_KEY and OPENAI_API_KEY env vars)
docker compose up --build

# View logs
tail -f logs/thoughts_*.log

# Stop
docker compose down

# Reset all data
docker compose down -v
```

## Configuration

Environment variables (see `.env.example`):
- `ANTHROPIC_MODEL`: Claude model to use
- `ANTHROPIC_BASE_URL`: Custom endpoint (optional)
- `THOUGHT_INTERVAL`: seconds between thoughts (default: 60)
- `MAX_TOKENS`: max response length (default: 1024)
- `MAX_HISTORY`: number of previous thoughts in context (default: 10)
- `DISCORD_WEBHOOK_URL`: optional webhook for thought output

## AI Capabilities

The ruminating AI has access to:

**Persistent Storage** (`/data` volume):
- Files persist across container restarts
- Docker named volume `ruminate_data`

**Command Execution** (via `::CMD::command::END::` in output):
- Allowed: `cat`, `cp`, `curl`, `date`, `echo`, `find`, `grep`, `head`, `jq`, `ls`, `mkdir`, `mv`, `rm`, `tail`, `touch`, `wc`, `wget`
- 30-second timeout per command
- Commands run in `/data` working directory
- Results appear in the AI's context on next iteration

## Personality System

Personalities and prompts are separate, combined at runtime:

- **Personality** (`personalities/*.txt`): Defines the mind's nature, style, inclinations
- **Prompt** (`prompt.txt`): The initial contemplation seed

Selection priority:
1. `PERSONALITY_FILE` - explicit path
2. `PERSONALITY` - inline env var
3. `PERSONALITY_NAME` - name in `personalities/` dir (default, philosopher, creative)
4. Built-in default

The identity (for mem0 memory isolation) is extracted from the personality's first line (e.g., "You are a philosophical mind" → `philosophical_mind`).

## Dependencies

- `anthropic`: Claude API client
- `mem0ai`: Memory system with vector + graph stores
- `psycopg2-binary`: PostgreSQL driver for pgvector
- `neo4j`: Neo4j driver for graph store
- OpenAI API key required for mem0's default embedder
