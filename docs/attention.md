# Attention layer

A LIDA-*inspired* attention controller that sits between thought cycles. It borrows
attention codelets, competition and a global workspace from LIDA. It is not a full
LIDA implementation. It adds no model calls and no database.

Plan and rationale: [`plans/2026-10-08-lida-attention-plan.md`](plans/2026-10-08-lida-attention-plan.md).

## Modes

| `ATTENTION_MODE` | What happens |
|---|---|
| `off` (default) | Legacy behaviour, byte-for-byte. No attention files are written. |
| `observe` | Codelets and the controller run after every thought. Decisions are logged and streamed to the dashboard, but the model's context and the scheduler are never touched. |
| `active` | Decisions are applied. A controller fault falls back to the legacy flow for that cycle and is logged as a `fault`. |

Run `observe` first. Promote to `active` only after reviewing the logs (see *Evaluating* below).

## How a cycle flows

```
before_cycle ─► SUSPENDED? stop        (budgets, stop file, ::STOP::)
             ─► IDLE?      wait, no model call, then wake with one probe cue
             ─► otherwise  build_context(..., attention_cue=<one cue or None>)
provider.think (unchanged)
after_cycle  ─► codelets ─► workspace (score, coalitions) ─► controller (gates + FSM) ─► next decision
```

**Codelets** (`attention/codelets.py`) are pure and deterministic:

- `progress` scores novelty against recent thoughts, new vocabulary, structural markers (hypothesis, test, conclusion, evidence, artifact) and *successful* tool actions. Enthusiasm doesn't count, and neither does length beyond about 25 words.
- `repetition` flags exact or near duplicates (3-gram shingle Jaccard), likely paraphrases, deferral language ("nothing new to think about", "I'll come back to this later") and near-empty output. A paraphrase is only flagged when phrase overlap and low new vocabulary *both* support it, so staying on one topic isn't penalised.
- `contradiction` flags explicit changes of position and negated restatements of earlier "I believe that…" claims. It is advisory only (priority below 0.5) and notes when new tool evidence backs the change.
- `questions` tracks open questions, `Open question:` / `TODO:` lines and unfinished "I want to investigate…" steps. It nominates `REVISIT` for threads that aren't on cooldown and are under the revisit cap, and drops threads a productive thought has addressed.
- `self_prompt` turns `prompts/current.md` into one candidate. Fresh prompts score higher than stale ones, and prompts that repeat while stalled score lower.

**Workspace** (`attention/workspace.py`) scores each nomination as 0.6·priority + 0.25·confidence + 0.15·recency, minus a cooldown penalty. An action gets a bonus for each extra source backing it, plus coalition rules (`repetition + questions → REVISIT`, `progress + self_prompt → CONTINUE`). Ordering is fully deterministic. The seed only picks among built-in reframe and explore cues.

**Controller policy** (`attention/controller.py`) is deliberately conservative:

| Situation | Decision | FSM |
|---|---|---|
| Thought made progress | `CONTINUE` (with the self-prompt as cue if there is one) | ENGAGED |
| One weak or stalled thought | no intervention | REFLECTING / STALLED |
| Unproductive for 2+ cycles | one bounded `REFRAME`, competing with the self-prompt and open threads | REFLECTING |
| Stall streak ≥ `ATTENTION_DEFERRAL_THRESHOLD` | one `REVISIT`/`EXPLORE` redirect | EXPLORING |
| Redirect also failed, or no eligible cue | `IDLE`: backoff of `base·2^n` capped at `ATTENTION_IDLE_BACKOFF_MAX_SECONDS`, then wake with one probe | IDLE |
| 3 consecutive provider errors | `IDLE`, then retry the legacy prompt | IDLE |
| Budget hit, stop file present, or `::STOP::` | `SUSPENDED`, loop exits | SUSPENDED |

**Cue injection** (`attention/context.py`): the model sees at most one direction. When a cue is applied, the stale self-prompt is removed from the opener and the cue replaces the trailing `[Continue]`. Self-prompts are still archived in `prompts/`. When the controller defers one, the decision's `reason` says so (`overrode_self_prompt: true`).

## Files

| Path | What |
|---|---|
| `logs/attention.jsonl` | Append-only log of `start`, `decision`, `cycle` (signals, nominees, coalitions, next decision, state summary), `fault` and `stop` records |
| `logs/attention_state.json` | Controller snapshot, reloaded on restart (`ATTENTION_RESUME=false` disables this) |
| `logs/runs/run_*.jsonl` | Per-cycle run record (`RUN_RECORD=true`, or automatic with `BENCHMARK_MODE`) |

None of this is written to mem0. Controller state stays separate from the mind's memory.

## Configuration

See `.env.example` for the full list. Main knobs:

```env
ATTENTION_MODE=off                  # off | observe | active
ATTENTION_DUPLICATE_WINDOW=5        # recent thoughts compared against
ATTENTION_NEAR_DUPLICATE_THRESHOLD=0.55
ATTENTION_PARAPHRASE_THRESHOLD=0.5
ATTENTION_PROGRESS_THRESHOLD=0.35
ATTENTION_DEFERRAL_THRESHOLD=2      # stall streak before redirecting
ATTENTION_MAX_CONSECUTIVE_REDIRECTS=1
ATTENTION_IDLE_BACKOFF_SECONDS=120
ATTENTION_IDLE_BACKOFF_MAX_SECONDS=1800
ATTENTION_REVISIT_COOLDOWN_CYCLES=5
ATTENTION_MAX_THREAD_REVISITS=2
ATTENTION_MAX_CYCLES / ATTENTION_MAX_RUNTIME_SECONDS / ATTENTION_MAX_APPROX_TOKENS  # 0 = unlimited
ATTENTION_STOP_FILE=./logs/STOP     # touch to suspend an active run
ATTENTION_RANDOM_SEED=7
```

Baseline harness (mode-independent, every default leaves behaviour unchanged):
`RUN_MAX_THOUGHTS`, `RUN_MAX_SECONDS`, `RUN_RECORD`, `BENCHMARK_MODE` (withholds
`update_source`, `restart_self`, `add_mcp_server` and `remove_mcp_server` so an experiment
can't change its own control group), `HISTORY_TOKEN_BUDGET` (keeps the newest whole thoughts
that fit and truncates the newest one only if it alone exceeds the budget; full text stays in
`thoughts/`), and `LLM_TEMPERATURE`.

Token counts are a chars/4 estimate. The providers don't expose usage yet.

## Evaluating

```bash
# What would the controller have done on your existing archive?
python scripts/attention_eval.py replay thoughts/ --out logs/replay.jsonl

# Summarise a live observe/active log; export a sheet for manual rating
python scripts/attention_eval.py report logs/attention.jsonl --csv logs/rate.csv

# Paired comparison (e.g. a baseline run replayed vs an active run)
python scripts/attention_eval.py compare logs/replay.jsonl logs/attention.jsonl
```

Suggested paired experiment (plan PR 3): same `PERSONALITY_NAME`, `prompt.txt`, model and
`LLM_TEMPERATURE`, plus `BENCHMARK_MODE=true RUN_MAX_THOUGHTS=30`. Run once with
`ATTENTION_MODE=observe` and once with `active`, clearing `prompts/current.md` and
`logs/attention_state.json` between runs. Compare progress per 1k tokens, longest stall
streak, interventions and idle time. Rate the CSV by hand so the evaluator isn't
rewarding longer or more confident text.

## Tests

```bash
pip install pytest
python -m pytest
```

`tests/test_regressions.py` drives the real `ruminate.main()` loop with a fake provider and
no network. It checks that `off` sends exactly the legacy messages, that `observe` never
changes them, and that `active` idles without model calls, fails open on controller faults
and suspends on budgets. The fixtures in `tests/fixtures/` are synthetic. Add real examples
of the deferral loop there as you collect them.
