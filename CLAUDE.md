# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rumination Engine is an input-less LLM thought experiment. It runs a continuous loop where Claude generates "thoughts" without external prompts, using mem0 for persistent memory across sessions.

## Architecture

**Single-file Python application** (`ruminate.py`) with MCP server integration:

```
ruminate.py (MCP client + Anthropic tool_use)
    │
    ├── Anthropic API (Claude) → generates thoughts via tool_use
    │
    └── MCP Servers (configurable, stdio transport)
            ├── mem0-mcp-server → memory (add, search, get, delete)
            └── [any other configured MCP servers]
```

The main loop:
1. Connects to configured MCP servers, discovers available tools
2. Builds context with system prompt + thought history + tool descriptions
3. Calls Claude with tool_use enabled — Claude can call any MCP tool
4. Handles tool calls in an agentic loop until Claude produces final text
5. That text is the "thought" — logged, displayed, sent to Discord
6. The AI can output `::STOP::` to terminate the session
7. Waits `THOUGHT_INTERVAL` seconds, repeats

## Running the Application

```bash
# Start (requires ANTHROPIC_API_KEY and MEM0_API_KEY env vars)
docker compose up --build

# View logs
tail -f logs/thoughts_*.log

# Stop
docker compose down
```

## Configuration

Environment variables (see `.env.example`):
- `ANTHROPIC_MODEL`: Claude model to use
- `ANTHROPIC_BASE_URL`: Custom endpoint (optional)
- `MEM0_API_KEY`: mem0 platform API key
- `MCP_SERVERS_FILE`: path to MCP server config (default: /app/mcp_servers.json)
- `THOUGHT_INTERVAL`: seconds between thoughts (default: 60)
- `MAX_TOKENS`: max response length (default: 1024)
- `MAX_HISTORY`: number of previous thoughts in context (default: 10)
- `DISCORD_WEBHOOK_URL`: optional webhook for thought output

## AI Capabilities

The ruminating AI has access to tools via MCP servers:

**Memory** (via mem0-mcp-server):
- `add_memory`: Save thoughts and insights to persistent memory
- `search_memories`: Semantic search across memories
- `get_memories`: List memories with filters
- `update_memory`, `delete_memory`: Manage individual memories

**Extensible**: Additional MCP servers can be configured in `mcp_servers.json`

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

- `anthropic`: Claude API client (AsyncAnthropic)
- `mcp`: Model Context Protocol Python SDK
- `rich`: Terminal formatting
- `requests`: HTTP client (Discord webhooks)
- `python-dotenv`: Environment variable loading
