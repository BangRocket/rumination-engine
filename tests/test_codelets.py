from attention.codelets import (
    ContradictionCodelet,
    OpenQuestionCodelet,
    ProgressCodelet,
    RepetitionCodelet,
    SelfPromptCodelet,
    extract_open_items,
    run_codelets,
)
from attention.models import CONTINUE, EXPLORE, REFRAME, REVISIT, AttentionState, OpenThread, ThoughtEvent
from conftest import load_fixture


def window_state(*texts, **kw) -> AttentionState:
    st = AttentionState(**kw)
    st.recent_thoughts = [{"cycle": i, "text": t} for i, t in enumerate(texts, 1)]
    return st


def ev(text, cycle=10, **kw):
    return ThoughtEvent.build(cycle=cycle, thought=text, **kw)


# --- progress ----------------------------------------------------------------

def test_progress_rewards_novel_structured_tool_backed_thought(cfg):
    items = load_fixture("productive")
    st = window_state(items[0]["thought"])
    out = ProgressCodelet().evaluate(ev(items[2]["thought"], tool_trace=items[2]["tool_trace"]), st, cfg)
    assert out.signals["score"] >= cfg.progress_threshold
    assert out.proposals[0].action == CONTINUE
    assert "add_memory" in out.signals["tool_actions"]


def test_progress_low_for_duplicate(cfg):
    t = load_fixture("repetitive")[0]["thought"]
    out = ProgressCodelet().evaluate(ev(t), window_state(t), cfg)
    assert out.signals["score"] < cfg.progress_threshold
    assert out.proposals[0].action == REFRAME


def test_progress_ignores_enthusiasm(cfg):
    hype = "This is AMAZING! Incredible! Truly profound and wonderful! I love this so much!!!"
    out = ProgressCodelet().evaluate(ev(hype), window_state("Something earlier about memory and time."), cfg)
    assert out.signals["score"] < cfg.progress_threshold


def test_failed_tool_is_not_progress(cfg):
    trace = [{"tool": "run_command", "input": {"command": "x"}, "output": '{"stderr": "boom", "exit_code": 1}'}]
    out = ProgressCodelet().evaluate(ev("short note", tool_trace=trace), AttentionState(), cfg)
    assert out.signals["tool_actions"] == []


# --- repetition --------------------------------------------------------------

def test_exact_duplicate_detected(cfg):
    t = load_fixture("repetitive")[0]["thought"]
    out = RepetitionCodelet().evaluate(ev(t), window_state(t), cfg)
    assert out.signals["exact_duplicate"] and out.signals["near_duplicate"]
    assert out.proposals and out.proposals[0].action == REFRAME


def test_paraphrase_detected(cfg):
    items = load_fixture("repetitive")
    out = RepetitionCodelet().evaluate(ev(items[2]["thought"]), window_state(items[0]["thought"], items[1]["thought"]), cfg)
    assert out.signals["paraphrase"] or out.signals["near_duplicate"]


def test_deferral_detected(cfg):
    for item in load_fixture("deferral"):
        out = RepetitionCodelet().evaluate(ev(item["thought"]), AttentionState(), cfg)
        assert out.signals["deferral"], item["thought"]


def test_shared_terminology_is_not_stagnation(cfg):
    items = load_fixture("shared_terminology")
    for i in range(1, len(items)):
        st = window_state(*[x["thought"] for x in items[:i]])
        out = RepetitionCodelet().evaluate(ev(items[i]["thought"]), st, cfg)
        assert not out.signals["near_duplicate"]
        assert not out.signals["paraphrase"]
        assert out.proposals == []


def test_explore_only_after_streak_threshold(cfg):
    t = "There is nothing new to think about right now. I'll come back to this later."
    first = RepetitionCodelet().evaluate(ev(t), AttentionState(consecutive_stalls=0), cfg)
    second = RepetitionCodelet().evaluate(ev(t), AttentionState(consecutive_stalls=1), cfg)
    assert EXPLORE not in {p.action for p in first.proposals}
    assert EXPLORE in {p.action for p in second.proposals}


def test_stop_token_signal(cfg):
    out = RepetitionCodelet().evaluate(ev("I am done for now. ::STOP::"), AttentionState(), cfg)
    assert out.signals["voluntary_stop"]


def test_provider_error_not_counted_as_repetition(cfg):
    out = RepetitionCodelet().evaluate(ev("[thought interrupted: x]"), window_state("[thought interrupted: x]"), cfg)
    assert out.proposals == [] and out.signals["provider_error"]


# --- contradiction -----------------------------------------------------------

def test_contradiction_flagged_but_advisory(cfg):
    items = load_fixture("contradiction")
    c = ContradictionCodelet()
    first = c.evaluate(ev(items[0]["thought"], cycle=1), AttentionState(), cfg)
    assert first.signals["claims"]
    st = AttentionState(claims=[{"cycle": 1, **cl} for cl in first.signals["claims"]])
    second = c.evaluate(ev(items[1]["thought"], cycle=2, tool_trace=items[1]["tool_trace"]), st, cfg)
    assert second.signals["flagged"]
    assert second.signals["new_evidence"]
    p = second.proposals[0]
    assert p.priority < 0.5                      # advisory, never dominant
    assert "not rejected" in p.evidence


def test_no_contradiction_on_consistent_claims(cfg):
    c = ContradictionCodelet()
    a = c.evaluate(ev("I believe that memory retrieval is driven by topic.", cycle=1), AttentionState(), cfg)
    st = AttentionState(claims=[{"cycle": 1, **cl} for cl in a.signals["claims"]])
    b = c.evaluate(ev("I still believe that memory retrieval is driven by topic.", cycle=2), st, cfg)
    assert not b.signals["flagged"]


# --- open questions ----------------------------------------------------------

def test_extract_questions_marked_and_next_steps():
    text = ("Open question: does saving fewer memories make search better?\n"
            "Why does retrieval feel continuous when it is actually reconstructed each time? "
            "Isn't it strange? I want to investigate how the store ranks dated memories.")
    items = extract_open_items(text)
    kinds = {i["kind"] for i in items}
    assert {"marked", "question", "next_step"} <= kinds
    assert not any(i["text"].lower().startswith("isn't it") for i in items)


def test_revisit_proposed_for_old_thread_not_on_cooldown(cfg):
    st = AttentionState(open_threads=[
        OpenThread(key="q:a", text="Does saving fewer memories make search better?", kind="marked",
                   created_cycle=1, last_seen_cycle=1, confidence=0.9),
        OpenThread(key="q:b", text="What is the shape of a thought about nothing at all?", kind="question",
                   created_cycle=1, last_seen_cycle=1, last_revisited_cycle=9),
    ])
    out = OpenQuestionCodelet().evaluate(ev("Quiet. Nothing new.", cycle=10), st, cfg)
    keys = [p.topic_key for p in out.proposals]
    assert keys == ["q:a"]                 # q:b is on cooldown
    assert out.proposals[0].action == REVISIT


def test_thread_marked_addressed(cfg):
    st = AttentionState(open_threads=[OpenThread(
        key="q:a", text="Does saving fewer, denser memories make search results better?",
        kind="marked", created_cycle=1, last_seen_cycle=1)])
    t = "I compared saving fewer, denser memories against saving everything: search results were better when denser."
    out = OpenQuestionCodelet().evaluate(ev(t, cycle=5), st, cfg)
    assert out.signals["addressed"] == ["q:a"]


# --- self prompt -------------------------------------------------------------

def test_self_prompt_fresh_beats_stale(cfg):
    fresh = SelfPromptCodelet().evaluate(ev("Some thought.", self_prompt="Investigate dated memories", self_prompt_set=True),
                                         AttentionState(), cfg)
    stale = SelfPromptCodelet().evaluate(ev("Some thought.", self_prompt="Investigate dated memories", self_prompt_set=False),
                                         AttentionState(), cfg)
    assert fresh.proposals[0].priority > stale.proposals[0].priority
    assert fresh.proposals[0].cue == "Investigate dated memories"


def test_self_prompt_detected_from_tool_trace():
    e = ThoughtEvent.build(cycle=1, thought="x", tool_trace=[
        {"tool": "set_next_prompt", "input": {"prompt": "p"}, "output": '{"ok": true}'}])
    assert e.self_prompt_set


# --- isolation ---------------------------------------------------------------

def test_failing_codelet_is_isolated(cfg):
    class Boom:
        name = "boom"

        def evaluate(self, *a):
            raise RuntimeError("nope")

    errors = []
    outs = run_codelets(ev("hello world"), AttentionState(), cfg,
                        codelets=(Boom(), ProgressCodelet()), on_error=lambda n, e: errors.append(n))
    assert errors == ["boom"]
    assert [o.source for o in outs] == ["boom", "progress"]
    assert "error" in outs[0].signals
