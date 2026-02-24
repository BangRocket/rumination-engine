# Design: MCP Client + mem0 Platform Migration

## Summary

Replace the self-hosted mem0 stack (Postgres/pgvector, Neo4j, OpenAI embeddings) with mem0's hosted platform API, and replace the `::CMD::` text-based command system with proper MCP client integration using Anthropic's tool_use API.

## Architecture

```
ruminate.py (MCP client + Anthropic tool_use)
    |
    +-- Anthropic API (Claude) -- generates thoughts, calls tools via tool_use
    |
    +-- MCP Servers (configurable, stdio transport)
            +-- mem0-mcp-server -- memory (add, search, get, delete)
            +-- [any other MCP servers configured by user]
```

## Main Loop (New)

1. On startup: connect to all configured MCP servers, discover their tools
2. Convert MCP tool schemas to Anthropic tool_use format
3. Build system prompt: personality + context info (time, capabilities list)
4. Call Claude with `tools=[all MCP tools]` and conversation history
5. Handle response in an agentic loop:
   - If Claude returns tool_use blocks: dispatch each to the correct MCP server, collect results, send back to Claude
   - If Claude returns text (end_turn): that's the "thought"
6. Check for ::STOP:: in the thought text
7. Add thought to conversation history, log it, send to Discord
8. Wait THOUGHT_INTERVAL, repeat from step 3

Key change: memory is no longer force-injected. The AI uses `search_memories` and `add_memory` MCP tools when it chooses to. The system prompt describes its memory capabilities.

## MCP Server Configuration

File: `mcp_servers.json`

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

- Environment variable references (`${VAR}`) are interpolated from process env at startup
- Servers use stdio transport (StdioServerParameters)
- Each server's tools are namespaced or tracked by server name for dispatch

## Removed

- PostgreSQL/pgvector Docker service and config
- Neo4j Docker service and config
- `mem0.Memory` Python client, `mem0_config` dict, neo4j monkey-patch
- `::CMD::command::END::` pattern, `ALLOWED_COMMANDS`, `execute_command()`
- `OPENAI_API_KEY` requirement
- `psycopg2-binary`, `neo4j`, `mem0ai[graph]` Python dependencies

## Added

- `mcp` Python SDK dependency
- MCP client manager: connects to servers, discovers tools, dispatches calls
- Anthropic tool_use agentic loop
- `mcp_servers.json` configuration file
- `mem0-mcp-server` as default configured server (installed via uvx in Docker)

## Kept

- Personality system (load, identity extraction, prompt loading)
- Discord webhook integration
- Logging and Rich console output
- `::STOP::` termination command
- Docker (simplified: only ruminate container)
- Seeds system (uses `add_memory` MCP tool instead of Python client)

## Docker Changes

`docker-compose.yml` reduces to one service:

```yaml
services:
  ruminate:
    build: .
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      MEM0_API_KEY: ${MEM0_API_KEY}
      ANTHROPIC_MODEL: ${ANTHROPIC_MODEL:-claude-sonnet-4-20250514}
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

No postgres, no neo4j, no data volumes for those.

Dockerfile adds `uv` for `uvx` MCP server execution.

## Environment Variables

- `ANTHROPIC_API_KEY` (required)
- `MEM0_API_KEY` (required)
- `ANTHROPIC_MODEL` (default: claude-sonnet-4-20250514)
- `PERSONALITY_NAME` / `PERSONALITY_FILE` / `PERSONALITY`
- `THOUGHT_INTERVAL`, `MAX_TOKENS`, `MAX_HISTORY`
- `DISCORD_WEBHOOK_URL` (optional)
- `MCP_SERVERS_FILE` (default: /app/mcp_servers.json)
