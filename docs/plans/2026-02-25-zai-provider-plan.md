# z.ai Provider Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add switchable LLM provider support so rumi can use z.ai's GLM 4.7 via OpenAI-compatible API alongside existing Anthropic/Claude support.

**Architecture:** Extract `think()` into a `providers.py` module with a `Provider` protocol. `AnthropicProvider` wraps existing logic; `OpenAIProvider` uses `openai.AsyncOpenAI` with streaming + function calling. `ruminate.py` selects provider at startup via `LLM_PROVIDER` env var.

**Tech Stack:** Python 3.12, `anthropic` SDK, `openai` SDK, asyncio

---

### Task 1: Add `openai` to requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Add the dependency**

Add `openai>=1.40.0` to `requirements.txt` after the `anthropic` line.

```
# Core
anthropic>=0.40.0
openai>=1.40.0
python-dotenv
rich
requests

# MCP client
mcp>=1.0.0
```

**Step 2: Install it locally**

Run: `pip install openai>=1.40.0`
Expected: installs successfully

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add openai SDK dependency for z.ai provider support"
```

---

### Task 2: Create `providers.py` with AnthropicProvider

Extract the existing `think()` function from `ruminate.py` into `providers.py` as `AnthropicProvider`.

**Files:**
- Create: `providers.py`
- Modify: `ruminate.py`

**Step 1: Create `providers.py` with the provider protocol and AnthropicProvider**

The provider needs access to:
- `dispatch_tool_call` (passed as callable)
- `log_file_only` (passed as callable)
- Model config (model name, max_tokens, base_url)

```python
"""LLM provider abstraction for the rumination engine."""

import asyncio
import json
import sys
from typing import Callable, Awaitable, Protocol

import anthropic
import httpx
from rich.console import Console

console = Console()


class Provider(Protocol):
    """Interface for LLM providers."""

    async def think(
        self,
        tools: list[dict],
        system: str,
        messages: list[dict],
        history_end: int,
        dispatch_tool_call: Callable[[str, dict], Awaitable[str]],
        log: Callable[[str], None],
        max_tool_iterations: int,
    ) -> tuple[str, list[dict]]:
        """Generate a single thought via streaming + tool-use loop.

        Args:
            tools: Tool definitions in Anthropic format (name, description, input_schema).
            system: System prompt string.
            messages: Conversation messages.
            history_end: Index where history ends (for trimming on overflow).
            dispatch_tool_call: Async callable(tool_name, tool_input) -> result string.
            log: Callable for file-only logging.
            max_tool_iterations: Max tool-use loop iterations.

        Returns:
            (thought_text, tool_trace) where tool_trace is list of
            {"tool": name, "input": {...}, "output": "..."} dicts.
        """
        ...


class AnthropicProvider:
    """Anthropic/Claude provider using the Anthropic SDK."""

    def __init__(self, model: str, max_tokens: int, base_url: str | None = None):
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.AsyncAnthropic(base_url=base_url)

    async def think(
        self,
        tools: list[dict],
        system: str,
        messages: list[dict],
        history_end: int,
        dispatch_tool_call: Callable[[str, dict], Awaitable[str]],
        log: Callable[[str], None],
        max_tool_iterations: int,
    ) -> tuple[str, list[dict]]:
        tool_trace: list[dict] = []

        # Mark tools for prompt caching (cache_control on last tool)
        cached_tools = None
        if tools:
            cached_tools = [*tools]
            cached_tools[-1] = {**cached_tools[-1], "cache_control": {"type": "ephemeral"}}

        # System prompt as cacheable content block
        cached_system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

        text_parts = []
        for iteration in range(max_tool_iterations):
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": cached_system,
                "messages": messages,
            }
            if cached_tools:
                kwargs["tools"] = cached_tools

            response = None
            streamed_text = ""
            for attempt in range(3):
                try:
                    async with self.client.messages.stream(**kwargs) as stream:
                        async for text in stream.text_stream:
                            sys.stdout.write(text)
                            sys.stdout.flush()
                            streamed_text += text
                        response = await stream.get_final_message()
                    break
                except anthropic.BadRequestError as e:
                    if "prompt is too long" in str(e) and history_end > 3:
                        history = messages[:history_end]
                        tool_chain = messages[history_end:]
                        pairs_to_remove = max(1, (history_end - 1) // 4)
                        cut = pairs_to_remove * 2
                        history = [history[0]] + history[1 + cut:]
                        history_end = len(history)
                        messages = history + tool_chain
                        log(f"WARNING: Prompt too long, trimmed {pairs_to_remove} history pairs ({len(messages)} messages remain)")
                        break
                    raise
                except (anthropic.APIConnectionError, anthropic.APITimeoutError, httpx.RemoteProtocolError) as e:
                    log(f"WARNING: Transient API error (attempt {attempt + 1}/3): {e}")
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                        streamed_text = ""
                        continue
                    sys.stdout.write("\n")
                    return f"[thought interrupted: {e}]", tool_trace

            if response is None:
                continue

            text_parts = []
            tool_uses = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            if response.stop_reason == "max_tokens":
                log("WARNING: Response truncated by max_tokens limit")

            if response.stop_reason != "tool_use" or not tool_uses:
                sys.stdout.write("\n\n")
                sys.stdout.flush()
                thought = "\n".join(text_parts)
                if not thought.strip():
                    log("WARNING: Empty thought generated, skipping")
                    return "[empty thought]", tool_trace
                return thought, tool_trace

            # Tool calls
            messages.append({"role": "assistant", "content": response.content})
            if streamed_text:
                sys.stdout.write("\n")

            tool_results = []
            for tool_use in tool_uses:
                input_brief = ", ".join(
                    f"{k}={json.dumps(v)[:60]}" for k, v in tool_use.input.items()
                )
                console.print(f"[dim italic]  ↳ {tool_use.name}({input_brief})[/dim italic]")

                result_text = await dispatch_tool_call(tool_use.name, tool_use.input)

                result_preview = result_text[:300].replace("\n", " ").strip()
                if len(result_text) > 300:
                    result_preview += "…"
                console.print(f"[dim]    {result_preview}[/dim]")
                console.print()

                tool_trace.append({
                    "tool": tool_use.name,
                    "input": tool_use.input,
                    "output": result_text,
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result_text,
                })

            messages.append({"role": "user", "content": tool_results})
            streamed_text = ""

        log(f"WARNING: Tool loop hit {max_tool_iterations} iteration limit")
        sys.stdout.write("\n")
        thought = "\n".join(text_parts) if text_parts else "[thought exceeded tool iteration limit]"
        return thought, tool_trace
```

**Step 2: Update `ruminate.py` to import and use AnthropicProvider**

Remove the existing `think()` function from `ruminate.py`. Replace with:

In the imports section, add:
```python
from providers import AnthropicProvider, create_provider
```

Remove: `import anthropic`, `import httpx` (moved to providers.py).

In `main()`, replace the client creation:
```python
# Old:
client = anthropic.AsyncAnthropic(base_url=ANTHROPIC_BASE_URL)

# New:
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
provider = create_provider(LLM_PROVIDER)
```

Replace the `think()` call in the main loop:
```python
# Old:
thought, tool_trace = await think(client, tool_map, anthropic_tools, thought_history, thought_count)

# New:
tool_names = [t["name"] for t in anthropic_tools]
system, messages = build_context(thought_history, tool_names, thought_count)
history_end = len(messages)

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
```

Add a `create_provider` factory to `providers.py`:
```python
def create_provider(provider_name: str) -> AnthropicProvider:
    """Create an LLM provider from config. Only anthropic for now."""
    import os
    if provider_name == "anthropic":
        return AnthropicProvider(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=int(os.environ.get("MAX_TOKENS", "1024")),
            base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        )
    raise ValueError(f"Unknown LLM provider: {provider_name}")
```

**Step 3: Verify no import errors**

Run: `cd /Users/heidornj/Code/rumination-engine && python -c "from providers import AnthropicProvider, create_provider; print('OK')"`
Expected: `OK`

**Step 4: Verify ruminate.py still imports cleanly**

Run: `python -c "import ruminate; print('OK')"`
Expected: `OK` (may warn about missing env vars, that's fine)

**Step 5: Commit**

```bash
git add providers.py ruminate.py
git commit -m "refactor: extract think() into providers.py with AnthropicProvider"
```

---

### Task 3: Add OpenAIProvider

**Files:**
- Modify: `providers.py`

**Step 1: Add OpenAIProvider to providers.py**

Add after `AnthropicProvider`:

```python
from openai import AsyncOpenAI


def _to_openai_tools(anthropic_tools: list[dict]) -> list[dict]:
    """Convert Anthropic tool format to OpenAI function calling format."""
    openai_tools = []
    for tool in anthropic_tools:
        schema = dict(tool.get("input_schema", {}))
        schema.pop("cache_control", None)
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": schema,
            },
        })
    return openai_tools


class OpenAIProvider:
    """OpenAI-compatible provider (for z.ai, etc.)."""

    def __init__(self, model: str, max_tokens: int, api_key: str, base_url: str | None = None):
        self.model = model
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def think(
        self,
        tools: list[dict],
        system: str,
        messages: list[dict],
        history_end: int,
        dispatch_tool_call: Callable[[str, dict], Awaitable[str]],
        log: Callable[[str], None],
        max_tool_iterations: int,
    ) -> tuple[str, list[dict]]:
        tool_trace: list[dict] = []
        openai_tools = _to_openai_tools(tools) if tools else None

        # Convert messages to OpenAI format
        # Anthropic uses {"role": "user", "content": [{"type": "tool_result", ...}]}
        # OpenAI uses separate {"role": "tool", "content": "...", "tool_call_id": "..."}
        oai_messages = [{"role": "system", "content": system}]
        for msg in messages:
            oai_messages.append(_to_openai_message(msg))

        text_parts = []
        for iteration in range(max_tool_iterations):
            kwargs = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": oai_messages,
            }
            if openai_tools:
                kwargs["tools"] = openai_tools

            try:
                async with self.client.chat.completions.stream(**kwargs) as stream:
                    async for event in stream:
                        if event.type == "content.delta":
                            if event.delta:
                                sys.stdout.write(event.delta)
                                sys.stdout.flush()

                    response = await stream.get_final_completion()
            except Exception as e:
                log(f"WARNING: OpenAI API error: {e}")
                sys.stdout.write("\n")
                return f"[thought interrupted: {e}]", tool_trace

            choice = response.choices[0]
            message = choice.message

            if choice.finish_reason == "length":
                log("WARNING: Response truncated by max_tokens limit")

            # No tool calls — return the text
            if choice.finish_reason != "tool_calls" or not message.tool_calls:
                sys.stdout.write("\n\n")
                sys.stdout.flush()
                thought = message.content or ""
                if not thought.strip():
                    log("WARNING: Empty thought generated, skipping")
                    return "[empty thought]", tool_trace
                return thought, tool_trace

            # Tool calls
            # Append assistant message with tool_calls
            oai_messages.append(message)
            if message.content:
                sys.stdout.write("\n")

            for tool_call in message.tool_calls:
                func = tool_call.function
                try:
                    tool_input = json.loads(func.arguments)
                except json.JSONDecodeError:
                    tool_input = {}

                input_brief = ", ".join(
                    f"{k}={json.dumps(v)[:60]}" for k, v in tool_input.items()
                )
                console.print(f"[dim italic]  ↳ {func.name}({input_brief})[/dim italic]")

                result_text = await dispatch_tool_call(func.name, tool_input)

                result_preview = result_text[:300].replace("\n", " ").strip()
                if len(result_text) > 300:
                    result_preview += "…"
                console.print(f"[dim]    {result_preview}[/dim]")
                console.print()

                tool_trace.append({
                    "tool": func.name,
                    "input": tool_input,
                    "output": result_text,
                })

                oai_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                })

        log(f"WARNING: Tool loop hit {max_tool_iterations} iteration limit")
        sys.stdout.write("\n")
        thought = message.content if message and message.content else "[thought exceeded tool iteration limit]"
        return thought, tool_trace


def _to_openai_message(msg: dict) -> dict:
    """Convert an Anthropic-format message to OpenAI format.

    Handles the special case of tool_result content blocks:
    Anthropic: {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "y"}]}
    OpenAI: {"role": "tool", "tool_call_id": "x", "content": "y"}

    For assistant messages with tool_use content blocks, converts to tool_calls format.
    """
    content = msg.get("content")

    # Simple string content — pass through
    if isinstance(content, str):
        return {"role": msg["role"], "content": content}

    # List content — check for tool_result or tool_use blocks
    if isinstance(content, list):
        # Tool results from user role
        if msg["role"] == "user" and content and isinstance(content[0], dict) and content[0].get("type") == "tool_result":
            # Return multiple tool messages (caller should handle this)
            # For simplicity, return the first one — the caller handles multi-tool
            # Actually, OpenAI expects separate messages, so we return the first.
            # The full conversion happens at a higher level.
            result = content[0]
            return {
                "role": "tool",
                "tool_call_id": result.get("tool_use_id", ""),
                "content": result.get("content", ""),
            }

        # Just join text blocks
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        if text_parts:
            return {"role": msg["role"], "content": "\n".join(text_parts)}

    return {"role": msg["role"], "content": str(content) if content else ""}
```

**Step 2: Update `create_provider` to handle openai**

```python
def create_provider(provider_name: str) -> AnthropicProvider | OpenAIProvider:
    """Create an LLM provider from environment config."""
    import os
    if provider_name == "anthropic":
        return AnthropicProvider(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=int(os.environ.get("MAX_TOKENS", "1024")),
            base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        )
    if provider_name == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        return OpenAIProvider(
            model=os.environ.get("OPENAI_MODEL", "glm-4.7"),
            max_tokens=int(os.environ.get("MAX_TOKENS", "1024")),
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
    raise ValueError(f"Unknown LLM provider: {provider_name}. Use 'anthropic' or 'openai'.")
```

**Step 3: Verify import**

Run: `python -c "from providers import OpenAIProvider, create_provider; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add providers.py
git commit -m "feat: add OpenAIProvider for z.ai/GLM 4.7 support"
```

---

### Task 4: Update `docker-compose.yml` with new env vars

**Files:**
- Modify: `docker-compose.yml`

**Step 1: Add the new environment variables**

Add after the existing env vars:

```yaml
      LLM_PROVIDER: ${LLM_PROVIDER:-anthropic}
      OPENAI_API_KEY: ${OPENAI_API_KEY:-}
      OPENAI_BASE_URL: ${OPENAI_BASE_URL:-}
      OPENAI_MODEL: ${OPENAI_MODEL:-glm-4.7}
```

**Step 2: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: add LLM_PROVIDER and OpenAI env vars to docker-compose"
```

---

### Task 5: Handle message format conversion edge cases

The initial `_to_openai_message` in Task 3 has a known limitation: Anthropic's tool_result messages can contain multiple results in one user message, but OpenAI expects separate `role: tool` messages. Also, Anthropic assistant messages with tool_use blocks need conversion to OpenAI's `tool_calls` format.

In practice, the OpenAI provider only hits these during the tool loop, where it builds its own `oai_messages` natively. The conversion is only needed for the initial history messages from `build_context()`, which are simple string content (user/assistant pairs). So the edge case doesn't actually arise in normal flow.

**Files:**
- Modify: `providers.py`

**Step 1: Simplify `_to_openai_message` and add a docstring clarifying scope**

Replace the function with:

```python
def _to_openai_message(msg: dict) -> dict:
    """Convert a build_context() message to OpenAI format.

    Only handles simple string-content messages from thought history.
    Tool interactions during the think() loop are built natively in OpenAI format.
    """
    content = msg.get("content", "")
    if isinstance(content, str):
        return {"role": msg["role"], "content": content}
    # Fallback: stringify complex content (shouldn't happen for history messages)
    return {"role": msg["role"], "content": str(content)}
```

**Step 2: Commit**

```bash
git add providers.py
git commit -m "refactor: simplify message conversion for OpenAI provider"
```

---

### Task 6: Verify end-to-end with Anthropic provider (regression check)

**Files:** None (verification only)

**Step 1: Run the app briefly with anthropic provider**

Run: `cd /Users/heidornj/Code/rumination-engine && LLM_PROVIDER=anthropic timeout 30 python -u ruminate.py 2>&1 | head -50`

Expected: Starts up, connects to MCP servers, generates at least one thought (or shows connection error if no API key — that's fine, we're checking it doesn't crash on import/init).

**Step 2: Run with openai provider (no key — verify clean error)**

Run: `cd /Users/heidornj/Code/rumination-engine && LLM_PROVIDER=openai python -c "from providers import create_provider; create_provider('openai')" 2>&1`

Expected: `ValueError: OPENAI_API_KEY is required when LLM_PROVIDER=openai`

**Step 3: Verify unknown provider gives clean error**

Run: `python -c "from providers import create_provider; create_provider('gemini')"`

Expected: `ValueError: Unknown LLM provider: gemini. Use 'anthropic' or 'openai'.`

---

### Task 7: Final cleanup and commit

**Files:**
- Modify: `ruminate.py` (remove stale imports if any remain)

**Step 1: Verify no unused imports in ruminate.py**

Check that `import anthropic` and `import httpx` are removed from `ruminate.py` (they live in `providers.py` now).

**Step 2: Final commit**

```bash
git add ruminate.py providers.py
git commit -m "chore: clean up imports after provider extraction"
```
