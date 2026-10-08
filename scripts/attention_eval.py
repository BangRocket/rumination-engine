#!/usr/bin/env python3
"""Offline tooling for the attention layer. No model calls, no network.

  replay  SOURCE [--out FILE]   Run codelets + controller over a captured run
                                (logs/runs/run_*.jsonl) or a thoughts/ archive
                                and show what the controller would decide.
  report  FILE [--csv OUT]      Summarise an attention log (logs/attention.jsonl)
                                or a replay output. --csv writes a sheet of
                                interventions for manual rating.
  compare A B                   Side-by-side metrics for two logs (e.g. a
                                baseline run vs an active-attention run).

Replay answers "what would the controller have done on this baseline?" — it
cannot show how the model would have responded to the redirect.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from attention.codelets import run_codelets  # noqa: E402
from attention.config import AttentionConfig  # noqa: E402
from attention.controller import AttentionController  # noqa: E402
from attention.models import ThoughtEvent  # noqa: E402
from attention.storage import read_jsonl  # noqa: E402

INTERVENTIONS = {"REFRAME", "REVISIT", "EXPLORE", "IDLE"}


# --------------------------------------------------------------------- loading
def load_source(src: Path) -> list[dict]:
    """Return [{"cycle", "thought", "tool_trace", "self_prompt", "self_prompt_set", "approx_tokens"}]."""
    if src.is_dir():
        items = []
        for i, f in enumerate(sorted(src.glob("*.md")), 1):
            text = f.read_text(encoding="utf-8")
            body = re.sub(r"\A# Thought .*?\n\n?", "", text, count=1)
            items.append({"cycle": i, "thought": body.strip(), "tool_trace": [], "source_file": f.name})
        return items
    items = []
    prev_prompt = None
    for rec in read_jsonl(src):
        if rec.get("kind") != "cycle" or "thought" not in rec:
            continue
        after = rec.get("self_prompt_after")
        items.append({
            "cycle": len(items) + 1,
            "thought": rec.get("thought", ""),
            "tool_trace": rec.get("tool_trace", []),
            "self_prompt": after,
            "self_prompt_set": bool(after) and after != prev_prompt,
            "approx_tokens": rec.get("approx_tokens", 0),
        })
        prev_prompt = after
    return items


# ---------------------------------------------------------------------- replay
def replay(items: list[dict], config: AttentionConfig) -> list[dict]:
    ctl = AttentionController(config)
    out = []
    for it in items:
        cycle = it["cycle"]
        applied = ctl.before_cycle(cycle)
        if applied.action == "IDLE":   # replay doesn't wait; serve the wake probe
            ctl.note_idle(applied.wait_seconds)
            applied = ctl.before_cycle(cycle)
        ev = ThoughtEvent.build(
            cycle=cycle, thought=it["thought"], tool_trace=it.get("tool_trace") or [],
            self_prompt=it.get("self_prompt"), approx_tokens=it.get("approx_tokens") or 0,
            **({"self_prompt_set": it["self_prompt_set"]} if "self_prompt_set" in it else {}),
        )
        outputs = run_codelets(ev, ctl.state, config)
        nxt, a, ranking = ctl.after_cycle(ev, outputs)
        out.append({
            "kind": "cycle", "cycle": cycle, "event": ev.as_dict(), "assessment": a.as_dict(),
            "signals": {o.source: o.signals for o in outputs}, "nominees": ranking.as_list(),
            "coalitions": ranking.coalitions, "next": nxt.as_dict(), "state": ctl.state.summary(),
            "replay_applied": applied.action,
        })
    return out


def print_replay(rows: list[dict]) -> None:
    print(f"{'cyc':>4} {'prog':>5} {'sim':>5} {'stall':<26} {'next':<9} {'src':<12} reason")
    for r in rows:
        a, n = r["assessment"], r["next"]
        stall = ",".join(a["stall_reasons"])[:26] or "-"
        print(f"{r['cycle']:>4} {a['progress_score']:>5.2f} {a['max_similarity']:>5.2f} {stall:<26} "
              f"{n['action']:<9} {str(n.get('source') or '-'):<12} {n.get('reason', '')[:70]}")


# ---------------------------------------------------------------------- report
def summarize(records: list[dict]) -> dict:
    cycles = [r for r in records if r.get("kind") == "cycle"]
    decisions = [r for r in records if r.get("kind") == "decision"]
    stats = {
        "cycles": len(cycles), "productive": 0, "stalled": 0, "provider_errors": 0,
        "mean_progress": 0.0, "longest_stall_streak": 0, "approx_tokens": 0,
        "progress_per_1k_tokens": None, "self_modifying_cycles": 0,
        "next_actions": {}, "applied_actions": {}, "idle_seconds": 0.0, "faults": 0,
    }
    streak = 0
    prog_sum = 0.0
    for r in cycles:
        a = r.get("assessment", {})
        ev = r.get("event", {})
        prog_sum += float(a.get("progress_score", 0.0))
        stats["productive"] += bool(a.get("productive"))
        stats["provider_errors"] += bool(a.get("provider_error"))
        if a.get("stalled"):
            stats["stalled"] += 1
            streak += 1
            stats["longest_stall_streak"] = max(stats["longest_stall_streak"], streak)
        elif a.get("productive"):
            streak = 0
        stats["approx_tokens"] += int(ev.get("approx_tokens") or 0)
        if any(c.get("tool") in {"update_source", "restart_self", "add_mcp_server", "remove_mcp_server"}
               for c in ev.get("tool_calls", [])):
            stats["self_modifying_cycles"] += 1
        act = r.get("next", {}).get("action")
        stats["next_actions"][act] = stats["next_actions"].get(act, 0) + 1
    for d in decisions:
        if d.get("applied"):
            stats["applied_actions"][d["action"]] = stats["applied_actions"].get(d["action"], 0) + 1
            if d["action"] == "IDLE":
                stats["idle_seconds"] += float(d.get("wait_seconds") or 0)
    stats["faults"] = sum(1 for r in records if r.get("kind") == "fault")
    if cycles:
        stats["mean_progress"] = round(prog_sum / len(cycles), 3)
    if stats["approx_tokens"]:
        stats["progress_per_1k_tokens"] = round(1000 * prog_sum / stats["approx_tokens"], 4)
    stats["interventions"] = sum(v for k, v in stats["next_actions"].items() if k in INTERVENTIONS)
    return stats


def write_rating_csv(records: list[dict], out: Path) -> int:
    """One row per cycle; the cycle *after* an intervention is what gets rated."""
    cycles = [r for r in records if r.get("kind") == "cycle"]
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cycle", "applied_before", "progress", "stall_reasons", "next_action", "next_source",
                    "reason", "thought_excerpt", "human_useful_0_3", "false_interruption_y_n", "notes"])
        for r in cycles:
            ev, a, nx = r.get("event", {}), r.get("assessment", {}), r.get("next", {})
            w.writerow([r.get("cycle"), ev.get("applied_action") or r.get("replay_applied", ""),
                        a.get("progress_score"), "; ".join(a.get("stall_reasons", [])), nx.get("action"),
                        nx.get("source"), nx.get("reason"), (ev.get("thought") or "")[:400].replace("\n", " "),
                        "", "", ""])
            n += 1
    return n


def print_stats(stats: dict, title: str = "") -> None:
    if title:
        print(f"== {title}")
    for k, v in stats.items():
        print(f"  {k:<24} {v}")


# ------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("replay")
    p.add_argument("source", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--quiet", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("log", type=Path)
    p.add_argument("--csv", type=Path)
    p = sub.add_parser("compare")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "replay":
        env = dict(os.environ)
        env["ATTENTION_MODE"] = "observe"
        cfg = AttentionConfig.from_env(env)
        items = load_source(args.source)
        if not items:
            print(f"No thoughts found in {args.source}", file=sys.stderr)
            return 1
        rows = replay(items, cfg)
        if not args.quiet:
            print_replay(rows)
        print_stats(summarize(rows), f"replay of {len(items)} thoughts")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
            print(f"wrote {args.out}")
        return 0

    if args.cmd == "report":
        recs = read_jsonl(args.log)
        print_stats(summarize(recs), str(args.log))
        if args.csv:
            n = write_rating_csv(recs, args.csv)
            print(f"wrote {n} rows for manual rating to {args.csv}")
        return 0

    if args.cmd == "compare":
        sa, sb = summarize(read_jsonl(args.a)), summarize(read_jsonl(args.b))
        print(f"{'metric':<24} {'A':>18} {'B':>18}")
        for k in sa:
            print(f"{k:<24} {str(sa[k]):>18} {str(sb[k]):>18}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
