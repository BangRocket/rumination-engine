"""Append-only JSONL decision log and an atomic state snapshot.

Kept separate from mem0 and from the thought archive on purpose: this is the
controller's bookkeeping, not the mind's memory.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from .models import AttentionState


class AttentionLog:
    def __init__(self, path: str | os.PathLike | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, record: dict) -> None:
        if not self.path:
            return
        line = json.dumps({"ts": round(time.time(), 3), "kind": kind, **record}, default=str, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def tail(self, n: int = 50, kinds: set[str] | None = None) -> list[dict]:
        if not self.path or not self.path.exists():
            return []
        out: list[dict] = []
        with open(self.path, encoding="utf-8") as f:
            lines = f.readlines()
        for line in reversed(lines):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kinds and rec.get("kind") not in kinds:
                continue
            out.append(rec)
            if len(out) >= n:
                break
        return list(reversed(out))


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def save_state(path: str | os.PathLike | None, state: AttentionState) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(state.as_dict(), default=str, ensure_ascii=False, indent=1)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, p)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        p.write_text(data, encoding="utf-8")


def load_state(path: str | os.PathLike | None) -> AttentionState | None:
    if not path or not Path(path).exists():
        return None
    try:
        return AttentionState.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    except Exception:
        return None


class RunRecorder:
    """Per-cycle record of a run, for baselines and offline replay (PR 0).

    One JSONL file per process: a `manifest` line, then one `cycle` line per
    thought, then a `end` line. Thought text is stored in full; tool outputs
    are truncated.
    """

    TOOL_OUTPUT_CHARS = 500

    def __init__(self, directory: str | os.PathLike, manifest: dict):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = d / f"run_{stamp}_{os.getpid()}.jsonl"
        self._log = AttentionLog(self.path)
        self._log.append("manifest", manifest)

    def record_cycle(self, *, cycle: int, thought: str, tool_trace: list[dict], **extra) -> None:
        trace = []
        for e in tool_trace or []:
            out = str(e.get("output", ""))
            trace.append({
                "tool": e.get("tool"),
                "input": {k: (v if len(str(v)) <= 300 else str(v)[:300] + "…") for k, v in dict(e.get("input") or {}).items()},
                "output": out[: self.TOOL_OUTPUT_CHARS] + ("…" if len(out) > self.TOOL_OUTPUT_CHARS else ""),
                "output_chars": len(out),
            })
        self._log.append("cycle", {"cycle": cycle, "thought": thought, "tool_trace": trace, **extra})

    def end(self, reason: str, **extra) -> None:
        self._log.append("end", {"reason": reason, **extra})
