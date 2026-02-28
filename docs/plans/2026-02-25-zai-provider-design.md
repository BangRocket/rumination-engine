# z.ai / GLM 4.7 Provider Integration

**Date:** 2025-02-25
**Goal:** Allow rumi to use z.ai's GLM 4.7 model via OpenAI-compatible API, switchable with Anthropic via config.

## Configuration

New env vars:

```
LLM_PROVIDER=openai          # "anthropic" (default) or "openai"
OPENAI_API_KEY=your_zai_key
OPENAI_BASE_URL=https://api.z.ai/api/paas/v4
OPENAI_MODEL=glm-4.7
```

Existing Anthropic vars unchanged. `LLM_PROVIDER` selects which client is used at startup.

## Architecture

```
ruminate.py (main loop, MCP, config)
    │
    └── providers.py
            ├── Provider (Protocol)
            ├── AnthropicProvider (existing think() logic)
            └── OpenAIProvider (new, OpenAI SDK for z.ai)
```

### Provider Protocol

```python
class Provider(Protocol):
    async def think(
        self,
        tool_map: dict,
        anthropic_tools: list[dict],
        thought_history: list[str],
        thought_count: int,
        system: str,
        messages: list[dict],
        history_end: int,
    ) -> tuple[str, list[dict]]
```

Both providers implement this. `ruminate.py` calls `build_context()` then hands off to `provider.think()`.

### Tool Format Translation

Anthropic tools → OpenAI tools:

```python
# Anthropic: {"name": "x", "description": "y", "input_schema": {...}}
# OpenAI:    {"type": "function", "function": {"name": "x", "description": "y", "parameters": {...}}}
```

Converter lives in `providers.py`.

### What stays in ruminate.py

- MCP server connection/management
- `build_context()`
- `dispatch_tool_call()`
- Built-in tools
- Discord/logging
- Main loop (calls `provider.think()`)

### What moves to providers.py

- Anthropic SDK import + streaming + tool_use loop → `AnthropicProvider`
- OpenAI SDK import + streaming + function calling loop → `OpenAIProvider`
- `create_provider(provider_name)` factory function

## File Changes

| File | Change |
|------|--------|
| `providers.py` | New file: Provider protocol, AnthropicProvider, OpenAIProvider |
| `ruminate.py` | Import provider, remove inline think(), call provider.think() |
| `requirements.txt` | Add `openai` |
| `docker-compose.yml` | Add LLM_PROVIDER, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL env vars |

## Decisions

- Provider selected at startup, not per-thought (no hot-switching)
- `build_context()` stays in ruminate.py — both providers consume its output
- Prompt caching (Anthropic-specific `cache_control`) only applied in AnthropicProvider
- OpenAI provider strips `cache_control` from tools before sending
