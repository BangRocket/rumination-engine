"""Regression tests against the real ruminate.py loop, with a fake provider.

No network, no MCP servers, no waiting: connect/reload/seed are stubbed and
`ruminate._sleep` is replaced so scheduler waits are recorded, not slept.
"""

import asyncio
import json

import pytest

import ruminate
from attention.config import AttentionConfig
from attention.models import SELF_MODIFYING_TOOLS
from attention.runtime import AttentionRuntime
from attention.models import ThoughtEvent
from conftest import load_fixture

DEFERRAL = "There is nothing new to think about right now. I'll come back to this later."


class FakeProvider:
    model = "fake-model"
    max_tokens = 64
    temperature = None

    def __init__(self, thoughts, set_prompts=None):
        self.thoughts = list(thoughts)
        self.set_prompts = set_prompts or {}
        self.calls = []

    async def think(self, tools, system, messages, history_end, dispatch_tool_call, log, max_tool_iterations):
        n = len(self.calls)
        self.calls.append({"system": system, "messages": [dict(m) for m in messages],
                           "tools": [t["name"] for t in tools], "history_end": history_end})
        trace = []
        if n in self.set_prompts:
            inp = {"prompt": self.set_prompts[n]}
            trace.append({"tool": "set_next_prompt", "input": inp,
                          "output": await dispatch_tool_call("set_next_prompt", inp)})
        return self.thoughts[n % len(self.thoughts)], trace


@pytest.fixture
def loop_env(monkeypatch, tmp_path):
    """Patch ruminate so main() runs offline and fast; return a runner."""
    for k in list(__import__("os").environ):
        if k.startswith(("ATTENTION_", "RUN_", "BENCHMARK", "HISTORY_TOKEN", "LLM_TEMPERATURE")):
            monkeypatch.delenv(k, raising=False)
    prompts = tmp_path / "prompts"
    thoughts = tmp_path / "thoughts"
    prompts.mkdir()
    thoughts.mkdir()
    monkeypatch.setattr(ruminate, "PROMPTS_DIR", prompts)
    monkeypatch.setattr(ruminate, "CURRENT_PROMPT_FILE", prompts / "current.md")
    monkeypatch.setattr(ruminate, "THOUGHTS_DIR", thoughts)
    monkeypatch.setattr(ruminate, "THOUGHT_INTERVAL", 0)
    monkeypatch.setattr(ruminate, "MAX_HISTORY", 10)
    monkeypatch.setattr(ruminate, "ENABLE_INSIGHTS", False)
    monkeypatch.setattr(ruminate, "DISCORD_WEBHOOK_URL", None)
    monkeypatch.setattr(ruminate, "_get_config_path", lambda: str(tmp_path / "mcp.json"))
    (tmp_path / "mcp.json").write_text("{}")
    monkeypatch.setattr(ruminate, "load_mcp_config", lambda identity: {})

    async def _connect(config):
        return {}, [], []

    async def _reload(h, tm, at, ac, ident):
        return h, tm, at, ac

    async def _seed(tm):
        return None

    monkeypatch.setattr(ruminate, "connect_mcp_servers", _connect)
    monkeypatch.setattr(ruminate, "maybe_reload_mcp", _reload)
    monkeypatch.setattr(ruminate, "seed_memories", _seed)

    waits = []

    async def _fake_sleep(s):
        waits.append(s)

    monkeypatch.setattr(ruminate, "_sleep", _fake_sleep)

    def run(provider, **env):
        env.setdefault("ATTENTION_LOG_PATH", str(tmp_path / "attention.jsonl"))
        env.setdefault("ATTENTION_STATE_PATH", str(tmp_path / "attention_state.json"))
        env.setdefault("RUN_RECORD_DIR", str(tmp_path / "runs"))
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))
        monkeypatch.setattr(ruminate, "create_provider", lambda name: provider)
        reason = asyncio.run(ruminate.main())
        return reason, waits

    run.tmp = tmp_path
    return run


# --- build_context ---------------------------------------------------------------

def test_build_context_legacy_without_cue(monkeypatch, tmp_path):
    monkeypatch.setattr(ruminate, "CURRENT_PROMPT_FILE", tmp_path / "current.md")
    sys1, m1 = ruminate.build_context(["a", "b"], ["t1"], 1)
    assert m1[0]["content"] == f"[Begin]\n\nInitial contemplation: {ruminate.INITIAL_PROMPT}"
    assert [m["content"] for m in m1[1:]] == ["a", "[Continue]", "b", "[Continue]"]
    (tmp_path / "current.md").write_text("my direction")
    _, m2 = ruminate.build_context(["a"], ["t1"], 2)
    assert m2[0]["content"].endswith("written by you last time):\n\nmy direction")
    _, m3 = ruminate.build_context(["a"], ["t1"], 2, attention_cue=None)
    assert m3 == m2


def test_build_context_with_cue_has_single_direction(monkeypatch, tmp_path):
    monkeypatch.setattr(ruminate, "CURRENT_PROMPT_FILE", tmp_path / "current.md")
    (tmp_path / "current.md").write_text("stale direction")
    _, m = ruminate.build_context(["a", "b"], ["t1"], 3, attention_cue="CUE")
    text = json.dumps(m)
    assert "stale direction" not in text and text.count("CUE") == 1
    assert m[-1] == {"role": "user", "content": "CUE"}


# --- end-to-end loop -------------------------------------------------------------

def test_off_mode_is_legacy_and_writes_no_attention_files(loop_env):
    p = FakeProvider([f"Thought number {i} about something new and different." for i in range(5)],
                     set_prompts={1: "look at dates"})
    reason, waits = loop_env(p, RUN_MAX_THOUGHTS=4)
    assert reason.startswith("RUN_MAX_THOUGHTS")
    assert len(p.calls) == 4
    assert not (loop_env.tmp / "attention.jsonl").exists()
    assert not (loop_env.tmp / "runs").exists()
    # Third call uses the self-prompt written in call 2 — the legacy mechanism.
    assert "look at dates" in p.calls[2]["messages"][0]["content"]
    assert all(c["messages"][-1]["content"] in ("[Continue]",) or len(c["messages"]) == 1 for c in p.calls)
    assert "update_source" in p.calls[0]["tools"]


def test_observe_mode_never_changes_what_the_model_sees(loop_env, tmp_path):
    thoughts = [DEFERRAL] * 6
    off = FakeProvider(thoughts, set_prompts={1: "p"})
    loop_env(off, RUN_MAX_THOUGHTS=6)
    (tmp_path / "prompts" / "current.md").unlink()        # fresh start for the second run
    obs = FakeProvider(thoughts, set_prompts={1: "p"})
    _, waits = loop_env(obs, RUN_MAX_THOUGHTS=6, ATTENTION_MODE="observe")
    assert [c["messages"] for c in obs.calls] == [c["messages"] for c in off.calls]
    assert all(w == 0 or w < 1 for w in waits)          # no idle backoff honoured
    recs = [json.loads(l) for l in (tmp_path / "attention.jsonl").read_text().splitlines()]
    kinds = {r["kind"] for r in recs}
    assert {"start", "decision", "cycle", "stop"} <= kinds
    would = {r["action"] for r in recs if r["kind"] == "decision"}
    assert "IDLE" in would                                 # it *would* have idled
    assert not any(r.get("applied") for r in recs if r["kind"] == "decision")


def test_active_mode_redirects_and_idles_without_model_calls(loop_env, tmp_path):
    p = FakeProvider([DEFERRAL] * 10)
    reason, waits = loop_env(p, RUN_MAX_THOUGHTS=5, ATTENTION_MODE="active", ATTENTION_IDLE_BACKOFF_SECONDS=30)
    assert len(p.calls) == 5
    # One redirect cue reached the model.
    cues = [c["messages"][-1]["content"] for c in p.calls if "[Attention]" in c["messages"][-1]["content"]]
    assert cues
    # Idle backoffs were scheduled (30, 60, ...) — and no provider call happened for them.
    assert 30 in [round(w) for w in waits]
    recs = [json.loads(l) for l in (tmp_path / "attention.jsonl").read_text().splitlines()]
    idles = [r for r in recs if r["kind"] == "decision" and r["action"] == "IDLE" and r["applied"]]
    assert idles
    assert sum(1 for r in recs if r["kind"] == "cycle") == len(p.calls)


def test_active_mode_productive_run_continues(loop_env):
    items = load_fixture("productive")
    p = FakeProvider([i["thought"] for i in items])
    _, waits = loop_env(p, RUN_MAX_THOUGHTS=5, ATTENTION_MODE="active")
    assert len(p.calls) == 5
    assert not any("[Attention]" in str(c["messages"]) for c in p.calls)
    assert all(w < 1 for w in waits)


def test_active_mode_suspends_on_budget(loop_env):
    p = FakeProvider(["A thought with several new ideas about memory, time, and tools."])
    reason, _ = loop_env(p, ATTENTION_MODE="active", ATTENTION_MAX_CYCLES=3, RUN_MAX_THOUGHTS=10)
    assert reason.startswith("attention suspended")
    assert len(p.calls) == 3


def test_controller_fault_fails_open(loop_env, monkeypatch):
    def boom(self, *a, **k):
        raise RuntimeError("controller exploded")

    monkeypatch.setattr("attention.controller.AttentionController.after_cycle", boom)
    monkeypatch.setattr("attention.controller.AttentionController.before_cycle", boom)
    p = FakeProvider([DEFERRAL])
    reason, _ = loop_env(p, RUN_MAX_THOUGHTS=3, ATTENTION_MODE="active")
    assert len(p.calls) == 3                              # legacy flow kept running
    recs = [json.loads(l) for l in (loop_env.tmp / "attention.jsonl").read_text().splitlines()]
    assert sum(1 for r in recs if r["kind"] == "fault") >= 6


def test_benchmark_mode_hides_and_blocks_self_modifying_tools(loop_env):
    p = FakeProvider(["One thought, then done."])
    loop_env(p, RUN_MAX_THOUGHTS=1, BENCHMARK_MODE="true")
    assert not (SELF_MODIFYING_TOOLS & set(p.calls[0]["tools"]))
    out = asyncio.run(ruminate.handle_builtin_tool("restart_self", {"reason": "x"}))
    assert "disabled" in json.loads(out)["error"]
    runs = list((loop_env.tmp / "runs").glob("run_*.jsonl"))
    assert len(runs) == 1                                 # benchmark mode records the run
    kinds = [json.loads(l)["kind"] for l in runs[0].read_text().splitlines()]
    assert kinds == ["manifest", "cycle", "end"]


def test_history_token_budget_trims_context(loop_env):
    p = FakeProvider(["x" * 2000])
    loop_env(p, RUN_MAX_THOUGHTS=4, HISTORY_TOKEN_BUDGET=1100)
    hist_lens = [sum(1 for m in c["messages"] if m["role"] == "assistant") for c in p.calls]
    assert hist_lens == [0, 1, 2, 2]


def test_recorded_run_replays_offline(loop_env):
    from scripts.attention_eval import load_source, replay
    p = FakeProvider([DEFERRAL] * 4)
    loop_env(p, RUN_MAX_THOUGHTS=4, RUN_RECORD="true")
    run = next((loop_env.tmp / "runs").glob("run_*.jsonl"))
    items = load_source(run)
    assert [i["thought"] for i in items] == [DEFERRAL] * 4
    rows = replay(items, AttentionConfig(mode="observe", log_path="", state_path=""))
    assert "IDLE" in [r["next"]["action"] for r in rows]


# --- runtime unit checks ---------------------------------------------------------

def test_runtime_off_is_inert(tmp_path):
    rt = AttentionRuntime(AttentionConfig(mode="off", log_path=str(tmp_path / "a.jsonl")))
    assert not rt.enabled and rt.cue_text(rt.before_cycle(1)) is None
    assert rt.after_cycle(ThoughtEvent.build(cycle=1, thought="x")) is None
    assert not (tmp_path / "a.jsonl").exists()


def test_runtime_resumes_state_across_restart(tmp_path):
    cfg = AttentionConfig(mode="active", log_path=str(tmp_path / "a.jsonl"), state_path=str(tmp_path / "s.json"))
    rt = AttentionRuntime(cfg)
    for i in range(1, 3):
        rt.before_cycle(i)
        rt.after_cycle(ThoughtEvent.build(cycle=i, thought=DEFERRAL))
    rt2 = AttentionRuntime(cfg)
    assert rt2.controller.state.consecutive_stalls == 2
    assert rt2.before_cycle(3).action in ("REVISIT", "EXPLORE")
