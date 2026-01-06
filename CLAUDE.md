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
2. Builds context with system prompt + memories + last thought
3. Calls Claude to generate next thought
4. Stores new thought in mem0
5. Waits `THOUGHT_INTERVAL` seconds, repeats

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

Key constants in `ruminate.py`:
- `THOUGHT_INTERVAL`: seconds between thoughts (default: 60)
- `MODEL`: Claude model to use
- `MAX_TOKENS`: max response length (default: 1024)
- `USER_ID`: mem0 identity namespace (default: "ruminator")

The system prompt in `prompt.txt` defines the "personality" and initial contemplation seed.

## Dependencies

- `anthropic`: Claude API client
- `mem0ai`: Memory system with vector + graph stores
- `psycopg2-binary`: PostgreSQL driver for pgvector
- `neo4j`: Neo4j driver for graph store
- OpenAI API key required for mem0's default embedder
