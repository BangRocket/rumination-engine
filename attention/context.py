"""Translate a controller Decision into one narrow cue for the model, and
trim thought history to a token budget.

Rule: the model sees at most ONE direction per cycle. When a cue is applied,
the stale self-prompt is removed from the opener and the trailing
"[Continue]" is replaced by the cue, so nothing stacks or conflicts.
"""

from __future__ import annotations

from collections.abc import Callable

from .evaluator import approx_tokens
from .models import CONTINUE, EXPLORE, REFRAME, REVISIT, Decision

SELF_PROMPT_TEMPLATE = "Your direction for this thought (written by you last time):\n\n{cue}"


def render_cue(decision: Decision | None) -> str | None:
    """Return the text of the single cue, or None for the legacy prompt."""
    if decision is None or not decision.cue:
        return None
    cue = decision.cue.strip()
    a = decision.action
    if a == CONTINUE:
        if decision.source == "self_prompt":
            return SELF_PROMPT_TEMPLATE.format(cue=cue)
        return cue
    if a == REFRAME:
        return ("[Attention] Your last thoughts have been circling the same ground. "
                f"Come at it from a different angle this time:\n\n{cue}")
    if a == REVISIT:
        return ("[Attention] Earlier you raised something you never came back to. "
                f"Pick it up now:\n\n{cue}")
    if a == EXPLORE:
        return f"[Attention] Set the current thread aside for now.\n\n{cue}"
    return None


def apply_attention_cue(messages: list[dict], cue_text: str | None, initial_prompt: str) -> list[dict]:
    """Inject one cue into messages built by ruminate.build_context().

    No cue -> messages are returned unchanged (legacy behaviour).
    """
    if not cue_text or not messages:
        return messages
    out = [dict(m) for m in messages]
    if len(out) == 1:
        out[0]["content"] = f"[Begin]\n\n{cue_text}"
        return out
    # Neutral opener (drop the stale self-prompt) + cue in place of the final "[Continue]".
    out[0]["content"] = f"[Begin]\n\nInitial contemplation: {initial_prompt}"
    if out[-1].get("role") == "user":
        out[-1]["content"] = cue_text
    else:
        out.append({"role": "user", "content": cue_text})
    return out


TRUNCATION_MARK = "\n\n[…truncated to fit the context budget]"


def trim_history_to_budget(history: list[str], token_budget: int,
                           log: Callable[[str], None] | None = None) -> list[str]:
    """Keep the newest whole thoughts that fit `token_budget` (chars/4 estimate).

    Policy (deterministic, documented):
      * budget <= 0 -> history unchanged.
      * walk newest -> oldest, keep whole thoughts while they fit.
      * if even the newest thought doesn't fit, keep its head, truncated, with a marker.
      * the full text always remains in thoughts/*.md; only the context window is trimmed.
    """
    if token_budget <= 0 or not history:
        return list(history)
    kept: list[str] = []
    used = 0
    for t in reversed(history):
        cost = approx_tokens(t) + 4  # + "[Continue]" turn
        if used + cost > token_budget:
            break
        kept.append(t)
        used += cost
    kept.reverse()
    if not kept:
        max_chars = max(0, token_budget * 4 - len(TRUNCATION_MARK) - 16)
        kept = [history[-1][:max_chars] + TRUNCATION_MARK]
    dropped = len(history) - len(kept)
    if log and (dropped or kept[-1].endswith(TRUNCATION_MARK)):
        log(f"History budget {token_budget} tok: kept {len(kept)}/{len(history)} thoughts (~{used} tok)")
    return kept
