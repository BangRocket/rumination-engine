"""Lightweight, deterministic text heuristics and cycle assessment.

No model calls, no embeddings. These are deliberately simple so that every
score can be explained from the text that produced it.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable

from .models import Assessment, CodeletOutput

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")
_MD_RE = re.compile(r"[`*_>#\[\]()|~]+")

STOPWORDS = frozenset("""
a about above after again against all also am an and any are aren't as at be because been before being
below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down
during each few for from further had hadn't has hasn't have haven't having he her here hers herself him
himself his how i i'd i'll i'm i've if in into is isn't it it's its itself just let's like me more most
much must my myself no nor not now of off on once only or other ought our ours ourselves out over own
perhaps really same she should shouldn't so some something such than that that's the their theirs them
themselves then there there's these they they're this those through thing things think thought thoughts
to too under until up upon us very was wasn't we we're were weren't what what's when where which while
who whom why will with won't would wouldn't yet you your yours yourself maybe even still might may one
again way kind sort seems seem feel feels quite rather already
""".split())


def normalize(text: str) -> str:
    text = _MD_RE.sub(" ", (text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return _WORD_RE.findall(normalize(text))


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if len(w) > 2 and w not in STOPWORDS and not w.isdigit()]


def shingles(text: str, k: int = 3) -> set[tuple[str, ...]]:
    ws = words(text)
    if len(ws) < k:
        return {tuple(ws)} if ws else set()
    return {tuple(ws[i:i + k]) for i in range(len(ws) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a: set, b: set) -> float:
    """Fraction of `a` contained in `b` (asymmetric)."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


def approx_tokens(text_or_chars: str | int) -> int:
    n = text_or_chars if isinstance(text_or_chars, int) else len(text_or_chars or "")
    return max(0, (n + 3) // 4)


def stable_key(prefix: str, text: str) -> str:
    h = hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}:{h}"


def topic_key(text: str, n: int = 3) -> str | None:
    """Coarse topic label: the most frequent content words, alphabetised."""
    counts = Counter(content_words(text))
    if not counts:
        return None
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return "topic:" + "-".join(sorted(w for w, _ in top))


def sentences(text: str) -> list[str]:
    # Keep code blocks out of sentence splitting.
    text = re.sub(r"```.*?```", " ", text or "", flags=re.S)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -*•\t") for p in parts if p and p.strip(" -*•\t")]


def similarity_to_window(text: str, window: Iterable[dict]) -> tuple[float, int | None]:
    """Max shingle-Jaccard similarity against recent thoughts. Returns (score, cycle)."""
    sh = shingles(text)
    best, best_cycle = 0.0, None
    for item in window:
        s = jaccard(sh, shingles(item.get("text", "")))
        if s > best:
            best, best_cycle = s, item.get("cycle")
    return best, best_cycle


# --- Assessment ---------------------------------------------------------------

def assess(outputs: list[CodeletOutput], progress_threshold: float) -> Assessment:
    """Fold codelet signals into one Assessment. Missing signals default safe."""
    sig: dict = {}
    for o in outputs:
        sig[o.source] = o.signals or {}

    prog = sig.get("progress", {})
    rep = sig.get("repetition", {})
    con = sig.get("contradiction", {})

    a = Assessment()
    a.provider_error = bool(prog.get("provider_error") or rep.get("provider_error"))
    a.progress_score = float(prog.get("score", 0.0))
    a.max_similarity = float(rep.get("max_similarity", 0.0))
    a.deferral = bool(rep.get("deferral"))
    a.contradiction = bool(con.get("flagged"))
    a.voluntary_stop = bool(rep.get("voluntary_stop"))

    reasons: list[str] = []
    if rep.get("exact_duplicate"):
        reasons.append("exact duplicate")
    elif rep.get("near_duplicate"):
        reasons.append(f"near duplicate ({a.max_similarity:.2f})")
    elif rep.get("paraphrase"):
        reasons.append(f"likely paraphrase ({float(rep.get('paraphrase_similarity', 0)):.2f})")
    if a.deferral:
        reasons.append("deferral language")
    if rep.get("low_content"):
        reasons.append("low content")
    if a.provider_error:
        reasons.append("provider error")
    a.stall_reasons = reasons
    a.stalled = bool(reasons) and not a.provider_error  # errors are tracked separately
    a.productive = (not a.stalled) and (not a.provider_error) and a.progress_score >= progress_threshold
    return a
