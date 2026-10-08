"""Attention controller: eligibility gates + arbitration + FSM + budgets.

Policy (conservative by design):
  * progress continuing            -> CONTINUE, respect the current thread       (ENGAGED)
  * one weak/stalled cycle         -> no intervention, keep watching           (REFLECTING/STALLED)
  * progress slow for 2+ cycles    -> one bounded REFRAME, competing fairly      (REFLECTING)
                                      with the self-prompt and open questions
  * stall streak >= threshold      -> one REVISIT/EXPLORE redirect               (EXPLORING)
  * redirect also failed           -> IDLE with capped exponential backoff, then (IDLE)
                                      wake with a single probe cue
  * hard budget / stop file / ::STOP:: -> SUSPENDED (overrides every nomination)
"""

from __future__ import annotations

import os
import random
import time

from .codelets import EXPLORE_CUES, REFRAME_CUES
from .config import AttentionConfig
from .evaluator import assess, stable_key
from .models import (
    CONTINUE, ENGAGED, EXPLORE, EXPLORING, IDLE, IDLING, REFLECTING, REFRAME, REVISIT, STALLED,
    SUSPENDED, SUSPENDED_STATE, Assessment, AttentionState, CodeletOutput, Decision, OpenThread,
    ThoughtEvent,
)
from .workspace import Nominee, Ranking, arbitrate

MAX_CONSECUTIVE_ERRORS = 3
MAX_CLAIMS = 50
MAX_SELF_PROMPTS = 5


class AttentionController:
    def __init__(self, config: AttentionConfig, state: AttentionState | None = None, clock=time.time):
        self.config = config
        self.state = state or AttentionState()
        self.clock = clock
        self.started_at = clock()
        self._last_self_prompt: str | None = None

    # ------------------------------------------------------------------ before
    def before_cycle(self, cycle: int) -> Decision:
        """Return the decision for `cycle`. Never calls a model."""
        st = self.state
        reason = self.budget_reason(cycle)
        if reason:
            return self._suspend(cycle, reason)

        pending = Decision.from_dict(st.pending_decision)
        if pending is None:
            d = Decision.passthrough(cycle, reason="first cycle: legacy opening prompt")
        elif pending.action == SUSPENDED:
            d = pending
        elif pending.action == IDLE:
            # Serve the idle once; next call wakes with the stashed probe.
            d = pending
            st.pending_decision = st.wake_decision or Decision.passthrough(cycle, reason="woke from idle").as_dict()
            st.wake_decision = None
        else:
            d = pending
            st.pending_decision = None
        d.cycle = cycle
        if d.action != IDLE:
            st.last_decision = d.as_dict()
        st.fsm_state = d.state
        return d

    def note_idle(self, seconds: float) -> None:
        self.state.idle_seconds_total += max(0.0, float(seconds))

    # ------------------------------------------------------------------- after
    def after_cycle(self, event: ThoughtEvent, outputs: list[CodeletOutput]) -> tuple[Decision, Assessment, Ranking]:
        cfg, st = self.config, self.state
        st.cycle = event.cycle
        self._last_self_prompt = event.self_prompt
        a = assess(outputs, cfg.progress_threshold)

        # --- counters
        st.consecutive_errors = st.consecutive_errors + 1 if a.provider_error else 0
        st.approx_tokens_total += max(0, int(event.approx_tokens))
        if a.productive:
            st.consecutive_stalls = 0
            st.consecutive_unproductive = 0
            st.consecutive_redirects = 0
            st.reframe_used = False
            st.idle_level = 0
        elif not a.provider_error:
            st.consecutive_unproductive += 1
            if a.stalled:
                st.consecutive_stalls += 1
        st.longest_stall_streak = max(st.longest_stall_streak, st.consecutive_stalls)

        proposals = [p for o in outputs for p in o.proposals]
        ranking = arbitrate(proposals, st, event.cycle + 1, cfg)
        decision = self._decide(a, ranking, event)

        self._fold_signals(event, outputs, a)
        self._commit(decision)
        return decision, a, ranking

    # ---------------------------------------------------------------- policy
    def _decide(self, a: Assessment, ranking: Ranking, event: ThoughtEvent) -> Decision:
        cfg, st = self.config, self.state
        nxt = event.cycle + 1

        reason = self.budget_reason(nxt)
        if reason:
            return self._suspend(nxt, reason, ranking)
        if a.voluntary_stop:
            return self._suspend(nxt, "model emitted ::STOP::", ranking)

        if st.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            probe = Decision.passthrough(nxt, reason="retry after provider errors")
            return self._idle(nxt, f"{st.consecutive_errors} consecutive provider errors", ranking, probe)

        if a.provider_error:
            return self._passthrough(nxt, "provider error; retrying legacy prompt", ranking, STALLED)

        if a.productive:
            n = ranking.pick(CONTINUE)
            return self._from(n, CONTINUE, ENGAGED, nxt, ranking, f"progress {a.progress_score:.2f}; continue thread")

        if st.consecutive_stalls >= cfg.deferral_threshold:
            why = f"stall streak {st.consecutive_stalls} ({', '.join(a.stall_reasons) or 'unproductive'})"
            redirect_actions = {REVISIT, EXPLORE}
            if st.consecutive_redirects < cfg.max_consecutive_redirects:
                action = ranking.best_action(redirect_actions)
                if action:
                    st.consecutive_redirects += 1
                    return self._from(ranking.pick(action), action, EXPLORING, nxt, ranking, why + "; redirect")
                return self._idle(nxt, why + "; no eligible cue", ranking, None)
            probe_action = ranking.best_action(redirect_actions)
            probe = self._from(ranking.pick(probe_action), probe_action, EXPLORING, nxt, ranking,
                               "probe after idle") if probe_action else None
            return self._idle(nxt, why + f"; {st.consecutive_redirects} redirect(s) failed", ranking, probe)

        why = ", ".join(a.stall_reasons) or f"progress {a.progress_score:.2f} below {cfg.progress_threshold}"
        slowing = st.consecutive_unproductive >= 2
        if slowing and not st.reframe_used:
            # Progress has slowed for 2+ cycles: one bounded REFRAME, competing
            # fairly with the self-prompt and any open thread.
            action = ranking.best_action({REFRAME, CONTINUE, REVISIT}) or REFRAME
            if action == REFRAME:
                st.reframe_used = True
            fsm = REFLECTING if action != CONTINUE else (STALLED if a.stalled else REFLECTING)
            return self._from(ranking.pick(action), action, fsm, nxt, ranking,
                              why + f"; unproductive x{st.consecutive_unproductive}")

        # A single weak or stalled cycle is not worth interrupting.
        action = ranking.best_action({CONTINUE, REVISIT} if slowing else {CONTINUE}) or CONTINUE
        suffix = "; reframe already tried" if st.reframe_used else "; watching (no intervention yet)"
        return self._from(ranking.pick(action), action, STALLED if a.stalled else REFLECTING, nxt, ranking,
                          why + suffix)

    # -------------------------------------------------------------- builders
    def _rng(self, cycle: int) -> random.Random:
        return random.Random(f"{self.config.random_seed}:{cycle}")

    def _pick_bank_cue(self, bank: tuple[str, ...], cycle: int) -> str:
        cool = self.config.revisit_cooldown_cycles
        fresh = [c for c in bank
                 if cycle - self.state.cooldowns.get(stable_key("cue", c), -10**9) >= cool]
        return self._rng(cycle).choice(fresh or list(bank))

    def _from(self, n: Nominee | None, action: str, fsm: str, cycle: int, ranking: Ranking, reason: str) -> Decision:
        p = n.proposal if n else None
        cue = p.cue if p else None
        topic = p.topic_key if p else None
        if action == REFRAME and not cue:
            cue = self._pick_bank_cue(REFRAME_CUES, cycle)
            topic = topic or stable_key("cue", cue)
        if action == EXPLORE and not cue:
            cue = self._pick_bank_cue(EXPLORE_CUES, cycle)
            topic = stable_key("cue", cue)
        d = Decision(
            action=action, state=fsm, cycle=cycle, cue=cue, source=p.source if p else None,
            topic_key=topic, priority=round(n.score, 4) if n else 0.0, evidence=p.evidence if p else "",
            reason=reason, nominees=ranking.as_list()[:8],
        )
        if action == REVISIT and p:
            d.evidence = p.evidence
            d.reason += f"; revisiting thought {self._thread_cycle(p.topic_key)}"
        sp = self._current_self_prompt()
        d.overrode_self_prompt = bool(sp) and d.cue is not None and d.source != "self_prompt"
        if d.overrode_self_prompt:
            d.reason += "; self-prompt kept in storage but deferred"
        return d

    def _passthrough(self, cycle: int, reason: str, ranking: Ranking, fsm: str = ENGAGED) -> Decision:
        d = Decision.passthrough(cycle, reason=reason)
        d.state, d.nominees = fsm, ranking.as_list()[:8]
        return d

    def _idle(self, cycle: int, reason: str, ranking: Ranking, probe: Decision | None) -> Decision:
        cfg, st = self.config, self.state
        wait = min(cfg.idle_backoff_seconds * (2 ** st.idle_level), cfg.idle_backoff_max_seconds)
        st.idle_level += 1
        d = Decision(action=IDLE, state=IDLING, cycle=cycle, reason=reason, wait_seconds=float(wait),
                     nominees=ranking.as_list()[:8])
        if probe is not None:
            self._mark_used(probe)
        st.wake_decision = (probe or Decision.passthrough(cycle, reason="woke from idle")).as_dict()
        return d

    def _suspend(self, cycle: int, reason: str, ranking: Ranking | None = None) -> Decision:
        return Decision(action=SUSPENDED, state=SUSPENDED_STATE, cycle=cycle, reason=reason,
                        nominees=ranking.as_list()[:8] if ranking else [])

    def budget_reason(self, cycle: int) -> str | None:
        cfg, st = self.config, self.state
        if cfg.max_cycles and cycle > cfg.max_cycles:
            return f"cycle budget reached ({cfg.max_cycles})"
        if cfg.max_runtime_seconds and self.clock() - self.started_at >= cfg.max_runtime_seconds:
            return f"runtime budget reached ({cfg.max_runtime_seconds}s)"
        if cfg.max_approx_tokens and st.approx_tokens_total >= cfg.max_approx_tokens:
            return f"token budget reached (~{st.approx_tokens_total}/{cfg.max_approx_tokens})"
        if cfg.stop_file and os.path.exists(cfg.stop_file):
            return f"stop file present ({cfg.stop_file})"
        return None

    # ------------------------------------------------------------- bookkeeping
    def _current_self_prompt(self) -> str | None:
        return self._last_self_prompt

    def _thread_cycle(self, key: str | None) -> int | str:
        for t in self.state.open_threads:
            if t.key == key:
                return t.created_cycle
        return "?"

    def _fold_signals(self, event: ThoughtEvent, outputs: list[CodeletOutput], a: Assessment) -> None:
        cfg, st = self.config, self.state
        sig = {o.source: o.signals or {} for o in outputs}

        if not a.provider_error:
            st.recent_thoughts.append({"cycle": event.cycle, "text": event.thought[:6000]})
            st.recent_thoughts = st.recent_thoughts[-max(cfg.duplicate_window, 1):]

        if event.self_prompt_set and event.self_prompt:
            st.recent_self_prompts.append(event.self_prompt[:1000])
            st.recent_self_prompts = st.recent_self_prompts[-MAX_SELF_PROMPTS:]

        for c in sig.get("contradiction", {}).get("claims", []) or []:
            st.claims.append({"cycle": event.cycle, **c})
        st.claims = st.claims[-MAX_CLAIMS:]

        q = sig.get("questions", {})
        addressed = set(q.get("addressed", []) or [])
        if a.productive and addressed:
            st.open_threads = [t for t in st.open_threads if t.key not in addressed]
        for t in st.open_threads:
            if t.key in addressed:
                t.last_seen_cycle = event.cycle
        existing = {t.key for t in st.open_threads}
        for item in q.get("new_items", []) or []:
            if item["key"] in existing:
                continue
            st.open_threads.append(OpenThread(
                key=item["key"], text=item["text"], kind=item["kind"], created_cycle=event.cycle,
                last_seen_cycle=event.cycle, confidence=float(item.get("confidence", 0.5)),
            ))
        st.open_threads = [t for t in st.open_threads
                           if event.cycle - t.last_seen_cycle <= cfg.thread_expiry_cycles
                           and t.revisits < cfg.max_thread_revisits + 1]
        st.open_threads = st.open_threads[-cfg.max_open_threads:]

        # expire stale cooldowns
        horizon = max(cfg.revisit_cooldown_cycles, 1) * 4
        st.cooldowns = {k: v for k, v in st.cooldowns.items() if event.cycle - v <= horizon}

    def _mark_used(self, d: Decision) -> None:
        """Charge cooldowns / revisit counts for a cue that will be shown."""
        st = self.state
        if d.action in (REVISIT, EXPLORE, REFRAME) and d.topic_key:
            st.cooldowns[d.topic_key] = d.cycle
        if d.cue and d.action in (REFRAME, EXPLORE):
            st.cooldowns[stable_key("cue", d.cue)] = d.cycle
        if d.action == REVISIT:
            for t in st.open_threads:
                if t.key == d.topic_key:
                    t.revisits += 1
                    t.last_revisited_cycle = d.cycle

    def _commit(self, d: Decision) -> None:
        st = self.state
        self._mark_used(d)
        if d.action in (REFRAME, REVISIT, EXPLORE, IDLE):
            st.interventions += 1
        st.fsm_state = d.state
        st.pending_decision = d.as_dict()
