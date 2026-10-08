"""LIDA-inspired attention layer for the Rumination Engine.

Borrows LIDA's attention-codelet / competition / global-workspace ideas; it
is not an implementation of the full LIDA cognitive architecture.

Entry point for ruminate.py is `AttentionRuntime`. See docs/attention.md.
"""

from .config import AttentionConfig, RunLimits
from .context import apply_attention_cue, render_cue, trim_history_to_budget
from .models import (
    CONTINUE, EXPLORE, IDLE, REFRAME, REVISIT, SUSPENDED,
    AttentionProposal, AttentionState, Decision, ThoughtEvent,
)
from .runtime import AttentionRuntime

__all__ = [
    "AttentionConfig", "RunLimits", "AttentionRuntime", "AttentionProposal", "AttentionState",
    "Decision", "ThoughtEvent", "apply_attention_cue", "render_cue", "trim_history_to_budget",
    "CONTINUE", "REFRAME", "REVISIT", "EXPLORE", "IDLE", "SUSPENDED",
]
