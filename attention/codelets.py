"""Attention codelets.

Each codelet has one narrow responsibility: look at a completed thought (plus
the controller's working state) and nominate zero or more next actions. They
are pure functions of their inputs — they never mutate state, call a model,
or touch the filesystem. The controller folds their `signals` into state.
"""

from __future__ import annotations

import re
from typing import Protocol

from .config import AttentionConfig
from .evaluator import (
    containment,
    content_words,
    normalize,
    sentences,
    shingles,
    similarity_to_window,
    stable_key,
    topic_key,
    words,
)
from .models import (
    CONTINUE,
    EXPLORE,
    REFRAME,
    REVISIT,
    AttentionProposal,
    AttentionState,
    CodeletOutput,
    ThoughtEvent,
)

# Tools whose successful use is evidence of doing something rather than musing.
_ACTION_TOOLS = {
    "run_command", "add_memory", "update_memory", "write_file", "edit_file", "create_directory",
    "fetch", "search", "add_mcp_server", "update_source",
}
_LOOKUP_TOOLS = {"search_memories", "get_memories", "read_file", "list_directory", "list_mcp_servers"}

# Structural signals of a substantive step (not of enthusiasm).
_STRUCTURE_PATTERNS = {
    "hypothesis": re.compile(r"\b(hypothes[ie]s|i (?:suspect|predict|conjecture)|my guess is|if .{3,60} then)\b"),
    "test": re.compile(r"\b(test(?:ed|ing)?|experiment|measure[ds]?|check(?:ed)?|verif(?:y|ied)|tried)\b"),
    "conclusion": re.compile(r"\b(therefore|so it follows|conclu(?:de|sion)|this (?:means|shows|implies)|which means)\b"),
    "evidence": re.compile(r"\b(because|evidence|the result|it turns out|found that|according to)\b"),
    "artifact": re.compile(r"```|^\s*(?:\d+\.|[-*])\s+\S", re.M),
}

DEFERRAL_PATTERNS = [
    re.compile(p) for p in (
        r"\bnothing (?:new |more |much )?(?:to|i can) (?:think|say|add|explore)\b",
        r"\b(?:i )?(?:don't|do not) have anything (?:new|more|to)\b",
        r"\bi(?:'ll| will) (?:come back|return) to (?:this|that|it) (?:later|next time|another time)\b",
        r"\b(?:let me|i(?:'ll| will)|i should) (?:just )?(?:pause|rest|wait|sit with this|be still)\b",
        r"\bfor now,? i(?:'ll| will) (?:just )?(?:rest|wait|pause|sit|leave it)\b",
        r"\b(?:defer|postpone|set (?:this|it) aside) (?:until|for) (?:later|now|next)\b",
        r"\bnext (?:time|thought),? i(?:'ll| will| should)\b",
        r"\bi(?:'ll| will) (?:continue|pick (?:this|it) (?:up|back up)) (?:later|next time|in the next)\b",
        r"\b(?:i'm|i am) (?:not sure what|at a loss for what) to (?:think|do) (?:about )?next\b",
        r"\bno (?:particular|clear) direction\b",
    )
]

STOP_TOKEN = "::STOP::"

REFRAME_CUES = (
    "State the strongest objection to where you've landed, then answer it or concede.",
    "Make one concrete, checkable prediction from your current idea — and say how you'd check it.",
    "Say plainly what would change your mind here, then go look for it.",
    "Stop describing the idea; do one small thing with a tool that tests it.",
    "Explain the idea to someone who disagrees, in five sentences, without metaphors.",
)

EXPLORE_CUES = (
    "Look something up you've been curious about but never actually checked.",
    "Search your memories for an older idea you never followed up on, and pick it back up.",
    "Pick one small, concrete thing to build or try with your tools, and start it.",
    "Write down a question you genuinely don't know the answer to, then make a first attempt at it.",
    "Notice something about your own recent behaviour that surprises you, and investigate it.",
)


class Codelet(Protocol):
    name: str

    def evaluate(self, event: ThoughtEvent, state: AttentionState, config: AttentionConfig) -> CodeletOutput: ...


def paraphrase_similarity(text: str, window: list[dict]) -> tuple[float, int | None]:
    cw = set(content_words(text))
    best, best_cycle = 0.0, None
    for item in window:
        other = set(content_words(item.get("text", "")))
        if not cw or not other:
            continue
        s = len(cw & other) / len(cw | other)
        if s > best:
            best, best_cycle = s, item.get("cycle")
    return best, best_cycle


def _current_topic(event: ThoughtEvent) -> str | None:
    return topic_key(event.thought)


# --- Progress -----------------------------------------------------------------

class ProgressCodelet:
    """Does this thought add something? Structural + novelty + tool outcomes."""
    name = "progress"

    def evaluate(self, event, state, config):
        out = CodeletOutput(self.name)
        if event.provider_error:
            out.signals = {"score": 0.0, "provider_error": True}
            return out

        text = event.thought
        window = state.recent_thoughts[-config.duplicate_window:]
        sim, _ = similarity_to_window(text, window)
        novelty = 1.0 - sim

        cw = set(content_words(text))
        seen: set[str] = set()
        for item in window:
            seen |= set(content_words(item.get("text", "")))
        new_vocab = (len(cw - seen) / len(cw)) if cw else 0.0

        norm = normalize(text)
        hits = sorted(k for k, pat in _STRUCTURE_PATTERNS.items() if pat.search(norm if k != "artifact" else text))
        structure = min(1.0, len(hits) / 3.0)

        ok_actions = [c.tool for c in event.tool_calls if c.ok and c.tool in _ACTION_TOOLS]
        ok_lookups = [c.tool for c in event.tool_calls if c.ok and c.tool in _LOOKUP_TOOLS]
        tool_score = min(1.0, 0.5 * len(ok_actions) + 0.15 * len(ok_lookups))

        n_words = len(words(text))
        substantial = n_words >= 25 or bool(ok_actions)

        score = 0.40 * novelty + 0.20 * new_vocab + 0.20 * structure + 0.20 * tool_score
        if not substantial:
            score *= 0.5
        score = round(min(1.0, max(0.0, score)), 3)

        evidence = (
            f"novelty={novelty:.2f} new_vocab={new_vocab:.2f} structure={hits or '-'} "
            f"tools_ok={ok_actions + ok_lookups or '-'} words={n_words}"
        )
        out.signals = {
            "score": score, "novelty": round(novelty, 3), "new_vocab": round(new_vocab, 3),
            "structure": hits, "tool_actions": ok_actions, "words": n_words, "provider_error": False,
        }
        topic = _current_topic(event)
        if score >= config.progress_threshold:
            out.proposals.append(AttentionProposal(
                source=self.name, action=CONTINUE, priority=round(0.4 + 0.5 * score, 3), topic_key=topic,
                cue=None, evidence=f"progress {score:.2f}: {evidence}", confidence=0.6,
            ))
        else:
            gap = config.progress_threshold - score
            out.proposals.append(AttentionProposal(
                source=self.name, action=REFRAME, priority=round(min(0.8, 0.35 + gap), 3), topic_key=topic,
                cue=None, evidence=f"progress slowed to {score:.2f}: {evidence}", confidence=0.5,
            ))
        return out


# --- Repetition / deferral ----------------------------------------------------

class RepetitionCodelet:
    """Near-duplicate phrasing and repeated deferral. Shared vocabulary alone is not stagnation."""
    name = "repetition"

    def evaluate(self, event, state, config):
        out = CodeletOutput(self.name)
        text = event.thought
        if event.provider_error:
            out.signals = {"provider_error": True}
            return out

        window = state.recent_thoughts[-config.duplicate_window:]
        sim, sim_cycle = similarity_to_window(text, window)
        norm = normalize(text)
        exact = any(normalize(w.get("text", "")) == norm for w in window) and bool(norm)
        near = exact or sim >= config.near_duplicate_threshold

        deferral_hits = [p.pattern for p in DEFERRAL_PATTERNS if p.search(norm)]
        n_words = len(words(text))
        low_content = n_words < 15 and not any(c.ok for c in event.tool_calls)
        stop = STOP_TOKEN in text

        # Likely paraphrase: same ideas reworded. Guarded so that merely sharing
        # vocabulary with recent thoughts (staying on topic) does not count.
        para_sim, para_cycle = paraphrase_similarity(text, window)
        acted = any(c.ok and c.tool in _ACTION_TOOLS for c in event.tool_calls)
        cw = set(content_words(text))
        seen = set().union(*(set(content_words(w.get("text", ""))) for w in window)) if window else set()
        new_vocab = (len(cw - seen) / len(cw)) if cw else 0.0
        paraphrase = (not near and not acted and para_sim >= config.paraphrase_threshold
                      and sim >= 0.12 and new_vocab < 0.35)
        if paraphrase and (sim_cycle is None or para_sim > sim):
            sim_cycle = para_cycle

        stalled = near or paraphrase or bool(deferral_hits) or low_content
        streak = (state.consecutive_stalls + 1) if stalled else 0

        out.signals = {
            "max_similarity": round(sim, 3), "similar_to_cycle": sim_cycle, "exact_duplicate": exact,
            "near_duplicate": near, "paraphrase": paraphrase, "paraphrase_similarity": round(para_sim, 3),
            "new_vocab": round(new_vocab, 3), "deferral": bool(deferral_hits), "deferral_patterns": deferral_hits[:3],
            "low_content": low_content, "voluntary_stop": stop, "stall_streak": streak, "provider_error": False,
        }
        if not stalled:
            return out

        why = []
        if exact:
            why.append(f"exact repeat of thought {sim_cycle}")
        elif near:
            why.append(f"{sim:.2f} phrase overlap with thought {sim_cycle}")
        elif paraphrase:
            why.append(f"likely paraphrase of thought {sim_cycle} ({para_sim:.2f} idea overlap, "
                       f"{new_vocab:.0%} new words)")
        if deferral_hits:
            why.append("deferral language")
        if low_content:
            why.append(f"only {n_words} words")
        evidence = "; ".join(why) + f" (stall streak {streak})"
        topic = _current_topic(event)

        out.proposals.append(AttentionProposal(
            source=self.name, action=REFRAME, priority=round(min(0.9, 0.5 + 0.1 * streak), 3),
            topic_key=topic, cue=None, evidence=evidence, confidence=0.8 if (exact or sim > 0.75) else 0.6,
        ))
        if streak >= config.deferral_threshold:
            out.proposals.append(AttentionProposal(
                source=self.name, action=EXPLORE, priority=round(min(0.95, 0.55 + 0.1 * streak), 3),
                topic_key=None, cue=None, evidence=evidence, confidence=0.7,
            ))
        return out


# --- Contradiction ------------------------------------------------------------

_POSITION_CHANGE = [
    re.compile(p) for p in (
        r"\bi (?:was|had been) wrong\b",
        r"\bi no longer (?:think|believe|hold)\b",
        r"\bcontrary to what i (?:said|thought|wrote)\b",
        r"\bi take (?:that|it) back\b",
        r"\bi (?:previously|earlier) (?:said|thought|believed|claimed)\b",
        r"\bi've changed my mind\b|\bi have changed my mind\b",
        r"\bactually,? (?:no|that's (?:not right|wrong))\b",
    )
]
_CLAIM_RE = re.compile(
    r"\bi (?:believe|think|conclude|am convinced|now think)(?: that)? (?P<claim>[^.?!\n]{8,140})"
)
_NEG_RE = re.compile(r"\b(?:not|no|never|isn't|aren't|doesn't|don't|can't|cannot|won't|wasn't)\b")


def _claim_core(claim: str) -> tuple[str, bool]:
    """Strip negation; return (core, polarity) where polarity False = negated."""
    c = normalize(claim)
    negated = bool(_NEG_RE.search(c))
    core = _NEG_RE.sub(" ", c)
    core = " ".join(w for w in core.split() if w not in {"that", "really", "actually"})
    return core.strip(), not negated


def extract_claims(text: str) -> list[dict]:
    claims = []
    for m in _CLAIM_RE.finditer(normalize(text)):
        core, pol = _claim_core(m.group("claim"))
        if len(core.split()) >= 3:
            claims.append({"claim": core, "polarity": pol})
    return claims


class ContradictionCodelet:
    """Flags explicit position changes and negated restatements. Advisory only."""
    name = "contradiction"

    def evaluate(self, event, state, config):
        out = CodeletOutput(self.name)
        if event.provider_error:
            return out
        norm = normalize(event.thought)
        explicit = [p.pattern for p in _POSITION_CHANGE if p.search(norm)]
        claims = extract_claims(event.thought)

        conflicts = []
        for c in claims:
            cs = set(c["claim"].split())
            for prior in state.claims:
                ps = set(prior["claim"].split())
                overlap = len(cs & ps) / max(1, len(cs | ps))
                if overlap >= 0.7 and prior["polarity"] != c["polarity"]:
                    conflicts.append({"now": c["claim"], "before": prior["claim"], "cycle": prior["cycle"]})
                    break

        has_new_evidence = any(cl.ok and cl.tool not in {"search_memories", "get_memories"} for cl in event.tool_calls)
        flagged = bool(conflicts) or bool(explicit)
        out.signals = {
            "flagged": flagged, "explicit_change": explicit[:2], "conflicts": conflicts[:3],
            "claims": claims[:5], "new_evidence": has_new_evidence,
        }
        if not flagged:
            return out

        high_conf = bool(conflicts)
        if conflicts:
            c = conflicts[0]
            evidence = f"now '{c['now'][:70]}' vs thought {c['cycle']} '{c['before'][:70]}'"
            cue = (f"Earlier (thought {c['cycle']}) you held the opposite of what you just said. "
                   "Say explicitly what changed your mind, and whether the earlier view had something right.")
        else:
            evidence = "explicit change of position"
            cue = "You changed position just now. Spell out what evidence moved you, and what you'd now predict."
        if has_new_evidence:
            evidence += " (with new tool evidence — not rejected)"
        out.proposals.append(AttentionProposal(
            source=self.name, action=REFRAME, priority=0.4 if high_conf else 0.3,
            topic_key=_current_topic(event), cue=cue, evidence=evidence,
            confidence=0.7 if high_conf else 0.4,
        ))
        return out


# --- Open questions -----------------------------------------------------------

_MARKED_RE = re.compile(r"^\s*(?:open question|question|todo|next step|unresolved)\s*[:\-]\s*(?P<t>.{8,240})$", re.I | re.M)
_NEXT_STEP_RE = re.compile(
    r"\bi (?:want|need|should|ought|plan|intend) to (?:explore|investigate|look into|figure out|find out|test|try|build|check|understand) (?P<t>[^.?!\n]{6,160})",
    re.I,
)
_RHETORICAL_START = ("isn't it", "isn't that", "who knows", "why not", "what else", "right?", "or is it")


def extract_open_items(text: str) -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()

    def add(t: str, kind: str, conf: float):
        t = t.strip().strip("\"'")
        k = stable_key("q", t)
        if k in seen or len(words(t)) < 5:
            return
        seen.add(k)
        items.append({"key": k, "text": t[:240], "kind": kind, "confidence": conf})

    for m in _MARKED_RE.finditer(text or ""):
        add(m.group("t"), "marked", 0.9)
    for s in sentences(text):
        if s.endswith("?"):
            low = s.lower()
            if low.startswith(_RHETORICAL_START) or len(words(s)) > 45:
                continue
            add(s, "question", 0.5)
    for m in _NEXT_STEP_RE.finditer(text or ""):
        add("Unfinished: " + m.group(0), "next_step", 0.6)
    return items[:6]


def thread_addressed(thread_text: str, thought: str) -> bool:
    tw = set(content_words(thread_text))
    if len(tw) < 2:
        return False
    return containment(tw, set(content_words(thought))) >= 0.6


class OpenQuestionCodelet:
    """Tracks unanswered questions / unfinished steps; nominates REVISITs."""
    name = "questions"

    def evaluate(self, event, state, config):
        out = CodeletOutput(self.name)
        if event.provider_error:
            return out
        new_items = extract_open_items(event.thought)
        existing = {t.key for t in state.open_threads}
        addressed = [t.key for t in state.open_threads
                     if t.created_cycle < event.cycle and thread_addressed(t.text, event.thought)]
        out.signals = {
            "new_items": [i for i in new_items if i["key"] not in existing],
            "addressed": addressed,
        }

        cycle = event.cycle
        stall_boost = 0.1 * min(3, state.consecutive_stalls + (1 if _looks_stalled(event, state, config) else 0))
        candidates = []
        for t in state.open_threads:
            if t.key in addressed:
                continue
            if t.revisits >= config.max_thread_revisits:
                continue
            if t.last_revisited_cycle is not None and cycle - t.last_revisited_cycle < config.revisit_cooldown_cycles:
                continue
            if cycle - t.created_cycle < 1:
                continue
            age = cycle - t.created_cycle
            recency = max(0.0, 1.0 - age / max(1, config.thread_expiry_cycles))
            prio = 0.35 + 0.15 * t.confidence + stall_boost - 0.05 * t.revisits
            candidates.append((prio, recency, t))

        candidates.sort(key=lambda x: (-x[0], -x[1], x[2].key))
        for prio, recency, t in candidates[:2]:
            out.proposals.append(AttentionProposal(
                source=self.name, action=REVISIT, priority=round(min(0.9, prio), 3), topic_key=t.key,
                cue=t.text, evidence=f"open {t.kind} from thought {t.created_cycle}, revisited {t.revisits}x",
                confidence=t.confidence, recency=round(recency, 3),
            ))
        return out


def _looks_stalled(event: ThoughtEvent, state: AttentionState, config: AttentionConfig) -> bool:
    sim, _ = similarity_to_window(event.thought, state.recent_thoughts[-config.duplicate_window:])
    norm = normalize(event.thought)
    return sim >= config.near_duplicate_threshold or any(p.search(norm) for p in DEFERRAL_PATTERNS)


# --- Model-authored self-prompt -------------------------------------------------

class SelfPromptCodelet:
    """Turns prompts/current.md into one candidate cue among others."""
    name = "self_prompt"

    def evaluate(self, event, state, config):
        out = CodeletOutput(self.name)
        prompt = (event.self_prompt or "").strip()
        if not prompt:
            return out
        prev = state.recent_self_prompts[-3:]
        repeat_sim = max((_sim(prompt, p) for p in prev), default=0.0)
        stalled = _looks_stalled(event, state, config) if not event.provider_error else False
        fresh = event.self_prompt_set

        if fresh and repeat_sim < config.near_duplicate_threshold:
            prio, conf, why = 0.65, 0.6, "freshly set by the model"
        elif fresh:
            prio, conf, why = (0.3 if stalled else 0.5), 0.5, f"freshly set but {repeat_sim:.2f} similar to a recent self-prompt"
        else:
            prio, conf, why = (0.25 if stalled else 0.45), 0.4, "carried over from an earlier cycle"
        if stalled:
            why += "; last thought stalled under it"

        out.signals = {"fresh": fresh, "repeat_similarity": round(repeat_sim, 3), "prompt": prompt[:300]}
        out.proposals.append(AttentionProposal(
            source=self.name, action=CONTINUE, priority=prio, topic_key=stable_key("prompt", prompt),
            cue=prompt, evidence=why, confidence=conf,
        ))
        return out


def _sim(a: str, b: str) -> float:
    sa, sb = shingles(a, 2), shingles(b, 2)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


DEFAULT_CODELETS: tuple = (
    ProgressCodelet(), RepetitionCodelet(), ContradictionCodelet(), OpenQuestionCodelet(), SelfPromptCodelet(),
)


def run_codelets(event: ThoughtEvent, state: AttentionState, config: AttentionConfig,
                 codelets=DEFAULT_CODELETS, on_error=None) -> list[CodeletOutput]:
    """Run each codelet in isolation; one failing codelet never sinks the others."""
    outputs = []
    for c in codelets:
        try:
            outputs.append(c.evaluate(event, state, config))
        except Exception as e:  # noqa: BLE001 — fault isolation is the point
            if on_error:
                on_error(c.name, e)
            outputs.append(CodeletOutput(c.name, signals={"error": f"{type(e).__name__}: {e}"}))
    return outputs
