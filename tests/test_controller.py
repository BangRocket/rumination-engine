from dataclasses import replace

from attention.codelets import run_codelets
from attention.controller import AttentionController
from attention.models import (
    CONTINUE, EXPLORE, IDLE, REFRAME, REVISIT, SUSPENDED, AttentionState, Decision, ThoughtEvent,
)
from attention.storage import load_state, save_state
from conftest import load_fixture
from scripts.attention_eval import replay


def actions(rows):
    return [r["next"]["action"] for r in rows]


def drive(ctl, cfg, thought, cycle, **kw):
    """One full cycle: before -> event -> codelets -> after. Returns (applied, next)."""
    applied = ctl.before_cycle(cycle)
    e = ThoughtEvent.build(cycle=cycle, thought=thought, **kw)
    nxt, a, _ = ctl.after_cycle(e, run_codelets(e, ctl.state, cfg))
    return applied, nxt, a


# --- test matrix (plan §8) ------------------------------------------------------

def test_productive_fixture_stays_engaged(cfg):
    rows = replay(load_fixture("productive"), cfg)
    assert set(actions(rows)) == {CONTINUE}
    assert all(r["next"]["state"] == "ENGAGED" for r in rows)
    assert rows[-1]["state"]["interventions"] == 0


def test_repetitive_fixture_redirects_then_idles(cfg):
    rows = replay(load_fixture("repetitive"), cfg)
    acts = actions(rows)
    assert any(a in (REFRAME, REVISIT, EXPLORE) for a in acts)
    assert acts[-1] == IDLE
    assert any(r["assessment"]["stalled"] for r in rows[1:3])  # paraphrase noticed


def test_two_deferrals_get_one_cue_then_backoff(cfg):
    rows = replay(load_fixture("deferral"), cfg)
    acts = actions(rows)
    assert acts[0] == CONTINUE                       # one stall: no intervention
    assert acts[1] in (REVISIT, EXPLORE)             # two: one eligible cue
    assert acts[2] == IDLE                           # still unproductive: back off
    redirects = [a for a in acts if a in (REVISIT, EXPLORE)]
    assert len(redirects) == 1


def test_unresolved_question_wins_revisit(cfg):
    rows = replay(load_fixture("questions"), cfg)
    revisit = [r for r in rows if r["next"]["action"] == REVISIT]
    assert revisit, actions(rows)
    assert "saving fewer" in revisit[0]["next"]["cue"]
    assert any("questions" in c for c in revisit[0]["coalitions"])


def test_contradiction_with_new_evidence_not_rejected(cfg):
    rows = replay(load_fixture("contradiction"), cfg)
    assert rows[1]["assessment"]["contradiction"]
    assert rows[1]["assessment"]["productive"]
    assert rows[1]["next"]["action"] == CONTINUE


def test_no_cue_available_idles_without_redirect(cfg):
    c = replace(cfg, max_consecutive_redirects=0)
    rows = replay(load_fixture("deferral"), c)
    assert actions(rows)[1] == IDLE


def test_idle_backoff_doubles_and_caps(cfg):
    c = replace(cfg, idle_backoff_seconds=10, idle_backoff_max_seconds=35)
    ctl = AttentionController(c)
    waits = []
    t = "There is nothing new to think about right now. I'll come back to this later."
    cycle = 0
    while len(waits) < 4 and cycle < 50:
        cycle += 1
        d = ctl.before_cycle(cycle)
        if d.action == IDLE:
            waits.append(d.wait_seconds)
            d = ctl.before_cycle(cycle)            # wake probe
        e = ThoughtEvent.build(cycle=cycle, thought=t)
        ctl.after_cycle(e, run_codelets(e, ctl.state, c))
    assert waits == [10, 20, 35, 35]


def test_idle_decision_then_wake_probe_differs(cfg):
    ctl = AttentionController(cfg)
    t = "There is nothing new to think about right now. I'll come back to this later."
    seen = []
    for cycle in range(1, 6):
        d = ctl.before_cycle(cycle)
        seen.append(d.action)
        if d.action == IDLE:
            probe = ctl.before_cycle(cycle)
            assert probe.action != IDLE
            seen.append(probe.action)
        e = ThoughtEvent.build(cycle=cycle, thought=t)
        ctl.after_cycle(e, run_codelets(e, ctl.state, cfg))
    assert IDLE in seen


def test_self_prompt_competes_and_wins_when_productive(cfg):
    ctl = AttentionController(cfg)
    items = load_fixture("productive")
    sp_trace = [{"tool": "set_next_prompt", "input": {"prompt": "Test dated memories next"}, "output": '{"ok": true}'}]
    drive(ctl, cfg, items[0]["thought"], 1)
    _, nxt, _ = drive(ctl, cfg, items[1]["thought"], 2, tool_trace=items[1]["tool_trace"] + sp_trace,
                      self_prompt="Test dated memories next")
    assert nxt.action == CONTINUE
    assert nxt.source == "self_prompt"
    assert nxt.cue == "Test dated memories next"
    assert not nxt.overrode_self_prompt


def test_self_prompt_override_is_logged(cfg):
    rows = replay([dict(i, self_prompt="Keep musing about silence", self_prompt_set=False)
                   for i in load_fixture("questions")], cfg)
    redirected = [r for r in rows if r["next"]["action"] in (REVISIT, EXPLORE)]
    assert redirected and redirected[0]["next"]["overrode_self_prompt"]
    assert "self-prompt kept in storage" in redirected[0]["next"]["reason"]


def test_provider_errors_back_off(cfg):
    rows = replay(load_fixture("errors"), cfg)
    assert actions(rows)[:3] == [CONTINUE, CONTINUE, IDLE]
    assert not any(r["assessment"]["stalled"] for r in rows)


# --- budgets & suspension ------------------------------------------------------

def test_cycle_budget_suspends(cfg):
    c = replace(cfg, max_cycles=2)
    ctl = AttentionController(c)
    _, nxt, _ = drive(ctl, c, "A first thought about many different new things in detail.", 1)
    _, nxt, _ = drive(ctl, c, "A second thought, also adding several genuinely new ideas.", 2)
    assert nxt.action == SUSPENDED
    assert ctl.before_cycle(3).action == SUSPENDED


def test_token_budget_suspends(cfg):
    c = replace(cfg, max_approx_tokens=100)
    ctl = AttentionController(c)
    _, nxt, _ = drive(ctl, c, "Thought.", 1, approx_tokens=150)
    assert nxt.action == SUSPENDED and "token budget" in nxt.reason


def test_stop_file_overrides_everything(cfg, tmp_path):
    stop = tmp_path / "STOP"
    c = replace(cfg, stop_file=str(stop))
    ctl = AttentionController(c)
    assert ctl.before_cycle(1).action == CONTINUE
    stop.write_text("x")
    assert ctl.before_cycle(2).action == SUSPENDED


def test_runtime_budget(cfg):
    now = [1000.0]
    ctl = AttentionController(replace(cfg, max_runtime_seconds=60), clock=lambda: now[0])
    assert ctl.before_cycle(1).action == CONTINUE
    now[0] += 61
    assert ctl.before_cycle(2).action == SUSPENDED


def test_voluntary_stop(cfg):
    ctl = AttentionController(cfg)
    _, nxt, _ = drive(ctl, cfg, "I have said what I needed to. ::STOP::", 1)
    assert nxt.action == SUSPENDED


# --- determinism & persistence -------------------------------------------------

def test_replay_is_deterministic(cfg):
    a = replay(load_fixture("deferral"), cfg)
    b = replay(load_fixture("deferral"), cfg)
    assert [r["next"] for r in a] == [r["next"] for r in b]


def test_seed_changes_bank_cue_choice_only(cfg):
    a = replay(load_fixture("deferral"), cfg)
    b = replay(load_fixture("deferral"), replace(cfg, random_seed=99))
    assert actions(a) == actions(b)


def test_state_roundtrip(cfg, tmp_path):
    ctl = AttentionController(cfg)
    for r in load_fixture("questions"):
        drive(ctl, cfg, r["thought"], r["cycle"])
    p = tmp_path / "state.json"
    save_state(p, ctl.state)
    st = load_state(p)
    assert isinstance(st, AttentionState)
    assert st.as_dict() == ctl.state.as_dict()
    resumed = AttentionController(cfg, state=st)
    assert resumed.before_cycle(5).action == Decision.from_dict(ctl.state.pending_decision).action


def test_corrupt_state_snapshot_ignored(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{not json")
    assert load_state(p) is None
