"""Typed data passed between codelets, the workspace and the controller.

Everything here is plain data: no I/O, no provider references. All of it
round-trips through JSON so it can be logged and snapshotted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# --- Actions the controller can choose ---------------------------------------
CONTINUE = "CONTINUE"
REFRAME = "REFRAME"
REVISIT = "REVISIT"
EXPLORE = "EXPLORE"
IDLE = "IDLE"
SUSPENDED = "SUSPENDED"   # terminal action: stop the loop

ACTIONS = (CONTINUE, REFRAME, REVISIT, EXPLORE, IDLE)
# Deterministic tie-break order (lower index wins).
ACTION_RANK = {a: i for i, a in enumerate((CONTINUE, REVISIT, REFRAME, EXPLORE, IDLE, SUSPENDED))}

# --- Controller FSM states ----------------------------------------------------
ENGAGED = "ENGAGED"
REFLECTING = "REFLECTING"
STALLED = "STALLED"
IDLING = "IDLE"
EXPLORING = "EXPLORING"
SUSPENDED_STATE = "SUSPENDED"

STATES = (ENGAGED, REFLECTING, STALLED, IDLING, EXPLORING, SUSPENDED_STATE)

# Built-in tools that change the engine's own environment. Used to flag
# cycles whose outcome can't be attributed to attention alone.
SELF_MODIFYING_TOOLS = frozenset({"update_source", "restart_self", "add_mcp_server", "remove_mcp_server"})

# Markers the providers return instead of a real thought.
PROVIDER_ERROR_PREFIXES = ("[thought interrupted", "[empty thought]", "[thought exceeded tool iteration limit]")


def is_provider_error(thought: str) -> bool:
    t = (thought or "").strip()
    return not t or t.startswith(PROVIDER_ERROR_PREFIXES)


@dataclass(frozen=True)
class ToolCall:
    """Compact, provider-independent summary of one tool call."""
    tool: str
    input_brief: str
    output_chars: int
    ok: bool

    @classmethod
    def from_trace(cls, entry: dict) -> "ToolCall":
        name = str(entry.get("tool", "?"))
        inp = entry.get("input") or {}
        try:
            brief = ", ".join(f"{k}={str(v)[:60]}" for k, v in dict(inp).items())
        except Exception:
            brief = str(inp)[:120]
        out = str(entry.get("output", ""))
        return cls(tool=name, input_brief=brief[:200], output_chars=len(out), ok=_tool_output_ok(out))


def _tool_output_ok(output: str) -> bool:
    head = output.lstrip()[:200].lower()
    if head.startswith('{"error"') or head.startswith("{\n  \"error\""):
        return False
    if '"exit_code":' in output:
        # run_command: non-zero exit is not a success
        return '"exit_code": 0' in output
    return bool(output.strip())


@dataclass(frozen=True)
class ThoughtEvent:
    """A completed thought plus everything codelets may look at."""
    cycle: int
    thought: str
    tool_calls: tuple[ToolCall, ...] = ()
    timestamp: float = 0.0
    duration_seconds: float = 0.0
    approx_tokens: int = 0                 # context + output, chars/4 estimate
    self_prompt: str | None = None         # prompts/current.md after the cycle
    self_prompt_set: bool = False          # set_next_prompt called this cycle
    applied_action: str = CONTINUE         # what the controller asked for this cycle
    applied_cue_source: str | None = None

    @property
    def provider_error(self) -> bool:
        return is_provider_error(self.thought)

    @property
    def self_modifying_tools(self) -> list[str]:
        return sorted({c.tool for c in self.tool_calls if c.tool in SELF_MODIFYING_TOOLS})

    @classmethod
    def build(cls, *, cycle: int, thought: str, tool_trace: list[dict] | None = None, **kw: Any) -> "ThoughtEvent":
        calls = tuple(ToolCall.from_trace(e) for e in (tool_trace or []))
        set_prompt = any(c.tool == "set_next_prompt" and c.ok for c in calls)
        kw.setdefault("self_prompt_set", set_prompt)
        return cls(cycle=cycle, thought=thought or "", tool_calls=calls, **kw)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["thought"] = self.thought[:500]
        d["provider_error"] = self.provider_error
        return d


@dataclass(frozen=True)
class AttentionProposal:
    source: str                 # progress / repetition / contradiction / questions / self_prompt
    action: str                 # CONTINUE / REFRAME / REVISIT / EXPLORE / IDLE
    priority: float             # 0.0–1.0
    topic_key: str | None
    cue: str | None
    evidence: str               # observable reason for the recommendation
    confidence: float = 0.5     # separately recorded, not treated as truth
    recency: float = 1.0        # 1.0 = about this cycle; lower = older material

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CodeletOutput:
    """What one codelet returns: proposals plus raw signals for the controller."""
    source: str
    proposals: list[AttentionProposal] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class Assessment:
    """Controller's summary of a cycle, derived from codelet signals."""
    progress_score: float = 0.0
    productive: bool = False
    stalled: bool = False
    stall_reasons: list[str] = field(default_factory=list)
    max_similarity: float = 0.0
    deferral: bool = False
    provider_error: bool = False
    contradiction: bool = False
    voluntary_stop: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    """What the controller wants the *next* cycle to do."""
    action: str = CONTINUE
    state: str = ENGAGED
    cycle: int = 0                      # the cycle this decision is for
    cue: str | None = None              # None = legacy prompt (self-prompt or initial)
    source: str | None = None
    topic_key: str | None = None
    priority: float = 0.0
    evidence: str = ""
    reason: str = ""
    wait_seconds: float = 0.0
    overrode_self_prompt: bool = False
    nominees: list[dict] = field(default_factory=list)
    mode: str = "off"
    applied: bool = False

    @classmethod
    def passthrough(cls, cycle: int = 0, reason: str = "legacy flow", mode: str = "off") -> "Decision":
        return cls(action=CONTINUE, state=ENGAGED, cycle=cycle, cue=None, reason=reason, mode=mode)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "Decision | None":
        if not d:
            return None
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@dataclass
class OpenThread:
    """An unanswered question or unfinished next step the mind raised."""
    key: str
    text: str
    kind: str                # question / next_step / marked
    created_cycle: int
    last_seen_cycle: int
    revisits: int = 0
    last_revisited_cycle: int | None = None
    confidence: float = 0.5

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class AttentionState:
    """Controller-only working state. Never shown to the model wholesale."""
    fsm_state: str = ENGAGED
    cycle: int = 0
    recent_thoughts: list[dict] = field(default_factory=list)   # {"cycle", "text"}
    recent_self_prompts: list[str] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)            # {"cycle", "claim", "polarity"}
    open_threads: list[OpenThread] = field(default_factory=list)
    cooldowns: dict[str, int] = field(default_factory=dict)     # key -> cycle it was last used
    consecutive_stalls: int = 0
    consecutive_unproductive: int = 0
    consecutive_redirects: int = 0
    consecutive_errors: int = 0
    reframe_used: bool = False
    idle_level: int = 0
    longest_stall_streak: int = 0
    approx_tokens_total: int = 0
    idle_seconds_total: float = 0.0
    interventions: int = 0
    last_decision: dict | None = None
    pending_decision: dict | None = None
    wake_decision: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AttentionState":
        d = dict(d or {})
        threads = [OpenThread(**t) for t in d.pop("open_threads", []) if isinstance(t, dict)]
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        st = cls(**known)
        st.open_threads = threads
        return st

    def summary(self) -> dict:
        """Compact view for SSE/dashboard."""
        return {
            "fsm_state": self.fsm_state,
            "cycle": self.cycle,
            "consecutive_stalls": self.consecutive_stalls,
            "consecutive_redirects": self.consecutive_redirects,
            "consecutive_errors": self.consecutive_errors,
            "longest_stall_streak": self.longest_stall_streak,
            "reframe_used": self.reframe_used,
            "idle_level": self.idle_level,
            "open_threads": [
                {"key": t.key, "text": t.text[:160], "kind": t.kind, "created_cycle": t.created_cycle, "revisits": t.revisits}
                for t in self.open_threads[-8:]
            ],
            "open_thread_count": len(self.open_threads),
            "cooldowns": dict(sorted(self.cooldowns.items(), key=lambda kv: -kv[1])[:8]),
            "approx_tokens_total": self.approx_tokens_total,
            "idle_seconds_total": round(self.idle_seconds_total, 1),
            "interventions": self.interventions,
        }
