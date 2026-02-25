"""
LLM provider abstraction for the Rumination Engine.

Defines a Provider protocol and concrete implementations (Anthropic).
"""

import asyncio
import json
import os
import sys
from collections.abc import Awaitable, Callable
from typing import Protocol

import anthropic
import httpx
from rich.console import Console


class Provider(Protocol):
    """Protocol for LLM providers that can generate thoughts."""

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
        """Generate a single thought using an agentic tool-use loop.

        Args:
            tools: Tool definitions in Anthropic format.
            system: System prompt text.
            messages: Conversation messages (history + tool interactions).
            history_end: Index where history ends and tool interactions begin.
            dispatch_tool_call: Callback to dispatch tool calls (name, input) -> result.
            log: Callback for file-only logging.
            max_tool_iterations: Maximum tool-use loop iterations.

        Returns:
            (thought_text, tool_trace) where tool_trace is a list of
            {"tool": name, "input": {...}, "output": "..."} dicts.
        """
        ...


class AnthropicProvider:
    """Anthropic Claude provider for the Rumination Engine."""

    def __init__(self, model: str, max_tokens: int, base_url: str | None = None):
        self.model = model
        self.max_tokens = max_tokens
        self.client = anthropic.AsyncAnthropic(base_url=base_url)
        self.console = Console()

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
        """Generate a single thought using tool_use agentic loop.

        Returns:
            (thought_text, tool_trace) where tool_trace is a list of
            {"tool": name, "input": {...}, "output": "..."} dicts.
        """
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
                        # Stream text to console as it arrives
                        async for text in stream.text_stream:
                            sys.stdout.write(text)
                            sys.stdout.flush()
                            streamed_text += text
                        response = await stream.get_final_message()
                    break  # success
                except anthropic.BadRequestError as e:
                    if "prompt is too long" in str(e) and history_end > 3:
                        # Trim history only (in pairs), preserve tool interactions
                        history = messages[:history_end]
                        tool_chain = messages[history_end:]
                        # Remove oldest pairs from history (keep first user msg + last pair)
                        pairs_to_remove = max(1, (history_end - 1) // 4)  # remove ~quarter
                        cut = pairs_to_remove * 2  # each "thought" is assistant+user
                        history = [history[0]] + history[1 + cut:]
                        history_end = len(history)
                        messages = history + tool_chain
                        log(f"WARNING: Prompt too long, trimmed {pairs_to_remove} history pairs ({len(messages)} messages remain)")
                        break  # will retry via outer loop since response is None
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
                continue  # prompt was trimmed, retry the outer tool iteration loop

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

            # Tool calls — show activity live
            messages.append({"role": "assistant", "content": response.content})
            if streamed_text:
                sys.stdout.write("\n")

            tool_results = []
            for tool_use in tool_uses:
                # Show tool call
                input_brief = ", ".join(
                    f"{k}={json.dumps(v)[:60]}" for k, v in tool_use.input.items()
                )
                self.console.print(f"[dim italic]  ↳ {tool_use.name}({input_brief})[/dim italic]")

                result_text = await dispatch_tool_call(tool_use.name, tool_use.input)

                # Show result (truncated, single line)
                result_preview = result_text[:300].replace("\n", " ").strip()
                if len(result_text) > 300:
                    result_preview += "…"
                self.console.print(f"[dim]    {result_preview}[/dim]")
                self.console.print()

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


def create_provider(provider_name: str) -> AnthropicProvider:
    """Factory function to create an LLM provider by name.

    Args:
        provider_name: Provider identifier (currently only "anthropic").

    Returns:
        A Provider instance.

    Raises:
        ValueError: If provider_name is not recognized.
    """
    if provider_name == "anthropic":
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        max_tokens = int(os.environ.get("MAX_TOKENS", "1024"))
        base_url = os.environ.get("ANTHROPIC_BASE_URL")  # None = default
        return AnthropicProvider(model=model, max_tokens=max_tokens, base_url=base_url)

    raise ValueError(
        f"Unknown LLM provider: {provider_name!r}. Supported: 'anthropic'"
    )
