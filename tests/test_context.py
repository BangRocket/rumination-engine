from attention.context import TRUNCATION_MARK, apply_attention_cue, render_cue, trim_history_to_budget
from attention.models import CONTINUE, EXPLORE, IDLE, REFRAME, REVISIT, Decision


def legacy_messages(n_history: int, opener="[Begin]\n\nYour direction for this thought (written by you last time):\n\nold"):
    msgs = [{"role": "user", "content": opener}]
    for i in range(n_history):
        msgs += [{"role": "assistant", "content": f"t{i}"}, {"role": "user", "content": "[Continue]"}]
    return msgs


def test_render_none_is_legacy():
    assert render_cue(None) is None
    assert render_cue(Decision.passthrough(3)) is None
    assert render_cue(Decision(action=IDLE, cue="x")) is None


def test_render_each_action():
    assert render_cue(Decision(action=CONTINUE, cue="p", source="self_prompt")).endswith("last time):\n\np")
    assert "[Attention]" in render_cue(Decision(action=REFRAME, cue="angle"))
    assert "Pick it up now" in render_cue(Decision(action=REVISIT, cue="q?"))
    assert "Set the current thread aside" in render_cue(Decision(action=EXPLORE, cue="go"))


def test_apply_no_cue_is_identity():
    m = legacy_messages(2)
    assert apply_attention_cue(m, None, "init") is m


def test_apply_cue_first_cycle():
    out = apply_attention_cue(legacy_messages(0), "CUE", "init")
    assert out == [{"role": "user", "content": "[Begin]\n\nCUE"}]


def test_apply_cue_single_direction_with_history():
    m = legacy_messages(2)
    out = apply_attention_cue(m, "CUE", "init")
    assert out[0]["content"] == "[Begin]\n\nInitial contemplation: init"   # stale self-prompt removed
    assert out[-1] == {"role": "user", "content": "CUE"}                  # replaces trailing [Continue]
    assert sum("CUE" in str(x["content"]) for x in out) == 1
    assert len(out) == len(m)                                             # history_end unchanged
    assert m[-1]["content"] == "[Continue]"                               # input not mutated


def test_trim_disabled_or_fits():
    h = ["a" * 40, "b" * 40]
    assert trim_history_to_budget(h, 0) == h
    assert trim_history_to_budget(h, 10_000) == h


def test_trim_keeps_newest_whole_thoughts():
    h = ["x" * 400, "y" * 400, "z" * 400]        # ~100 tok each (+4)
    logs = []
    out = trim_history_to_budget(h, 220, logs.append)
    assert out == ["y" * 400, "z" * 400]
    assert logs and "kept 2/3" in logs[0]


def test_trim_truncates_oversized_newest():
    out = trim_history_to_budget(["q" * 4000], 100)
    assert len(out) == 1 and out[0].endswith(TRUNCATION_MARK)
    assert len(out[0]) <= 100 * 4
