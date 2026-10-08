"""Fail-open facade used by ruminate.py.

All attention work goes through here so the main loop has exactly three
seams (before_cycle, cue_text, after_cycle) and none of them can raise.

  off      -> every method is a cheap no-op; legacy behaviour is untouched.
  observe  -> codelets + controller run and log; returned decisions are
              always passthrough, so control flow never changes.
  active   -> decisions are applied. Any controller fault falls back to the
              legacy flow for that cycle.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import asdict

from .codelets import run_codelets
from .config import AttentionConfig
from .context import render_cue
from .controller import AttentionController
from .models import Decision, ThoughtEvent
from .storage import AttentionLog, load_state, save_state

Broadcast = Callable[[str, dict], None]


def _noop(*_a, **_k):
    return None


class AttentionRuntime:
    def __init__(self, config: AttentionConfig, broadcast: Broadcast | None = None,
                 log: Callable[[str], None] | None = None, clock=time.time):
        self.config = config
        self.mode = config.mode
        self._broadcast = broadcast or _noop
        self._log = log or _noop
        self.controller: AttentionController | None = None
        self.store: AttentionLog | None = None
        self.faults = 0
        if not config.enabled:
            return
        try:
            state = load_state(config.state_path) if config.resume else None
            self.controller = AttentionController(config, state=state, clock=clock)
            self.store = AttentionLog(config.log_path)
            self.store.append("start", {"mode": self.mode, "resumed": state is not None, "config": asdict(config)})
            self._log(f"Attention layer: mode={self.mode}, log={config.log_path}, resumed={state is not None}")
        except Exception as e:  # noqa: BLE001
            self._fault("init", e)
            self.controller = None

    # ---------------------------------------------------------------- flags
    @property
    def enabled(self) -> bool:
        return self.controller is not None

    @property
    def active(self) -> bool:
        return self.enabled and self.mode == "active"

    # ---------------------------------------------------------------- seams
    def before_cycle(self, cycle: int) -> Decision:
        if not self.enabled:
            return Decision.passthrough(cycle)
        try:
            d = self.controller.before_cycle(cycle)
            d.mode, d.applied = self.mode, self.active
            self._record("decision", d.as_dict())
            self._broadcast("attention_decision", self._public_decision(d))
            if not self.active:
                return Decision.passthrough(cycle, reason=f"observe: would {d.action}", mode=self.mode)
            return d
        except Exception as e:  # noqa: BLE001
            self._fault("before_cycle", e)
            return Decision.passthrough(cycle, reason="controller fault: legacy flow", mode=self.mode)

    def cue_text(self, decision: Decision | None) -> str | None:
        if not self.active or decision is None:
            return None
        try:
            return render_cue(decision)
        except Exception as e:  # noqa: BLE001
            self._fault("render_cue", e)
            return None

    def after_cycle(self, event: ThoughtEvent) -> Decision | None:
        if not self.enabled:
            return None
        try:
            outputs = run_codelets(event, self.controller.state, self.config,
                                   on_error=lambda name, e: self._fault(f"codelet:{name}", e))
            nxt, assessment, ranking = self.controller.after_cycle(event, outputs)
            nxt.mode, nxt.applied = self.mode, self.active
            self._record("cycle", {
                "cycle": event.cycle,
                "event": event.as_dict(),
                "assessment": assessment.as_dict(),
                "signals": {o.source: o.signals for o in outputs},
                "nominees": ranking.as_list(),
                "action_scores": ranking.action_scores,
                "coalitions": ranking.coalitions,
                "next": nxt.as_dict(),
                "state": self.controller.state.summary(),
            })
            self._broadcast("codelet_proposals", {
                "cycle": event.cycle,
                "assessment": assessment.as_dict(),
                "nominees": ranking.as_list()[:8],
                "action_scores": ranking.action_scores,
                "coalitions": ranking.coalitions,
            })
            self._broadcast("attention_state", self.snapshot())
            save_state(self.config.state_path, self.controller.state)
            return nxt
        except Exception as e:  # noqa: BLE001
            self._fault("after_cycle", e)
            return None

    def note_idle(self, seconds: float) -> None:
        if self.enabled:
            try:
                self.controller.note_idle(seconds)
                save_state(self.config.state_path, self.controller.state)
            except Exception as e:  # noqa: BLE001
                self._fault("note_idle", e)

    def close(self, reason: str) -> None:
        if not self.enabled:
            return
        try:
            self._record("stop", {"reason": reason, "state": self.controller.state.summary()})
            save_state(self.config.state_path, self.controller.state)
        except Exception as e:  # noqa: BLE001
            self._fault("close", e)

    # -------------------------------------------------------------- helpers
    def snapshot(self) -> dict:
        if not self.enabled:
            return {"mode": self.mode, "enabled": False}
        snap = self.controller.state.summary()
        pending = Decision.from_dict(self.controller.state.pending_decision)
        if pending is not None:
            pending.mode, pending.applied = self.mode, self.active
        snap.update({"mode": self.mode, "enabled": True, "faults": self.faults,
                     "pending": self._public_decision(pending)})
        return snap

    @staticmethod
    def _public_decision(d: Decision | None) -> dict | None:
        if d is None:
            return None
        out = d.as_dict()
        out["nominees"] = out.get("nominees", [])[:5]
        if out.get("cue") and len(out["cue"]) > 400:
            out["cue"] = out["cue"][:400] + "…"
        return out

    def _record(self, kind: str, rec: dict) -> None:
        if self.store:
            try:
                self.store.append(kind, rec)
            except Exception as e:  # noqa: BLE001
                self._log(f"Attention log write failed: {e}")

    def _fault(self, where: str, e: BaseException) -> None:
        self.faults += 1
        msg = f"ATTENTION FAULT in {where}: {type(e).__name__}: {e}"
        self._log(msg)
        self._log(traceback.format_exc())
        if self.store:
            try:
                self.store.append("fault", {"where": where, "error": f"{type(e).__name__}: {e}"})
            except Exception:
                pass
