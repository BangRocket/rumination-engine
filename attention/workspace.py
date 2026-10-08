"""Global workspace: collect nominations, score them, form coalitions, rank.

The workspace only *ranks*. Which actions are eligible right now is the
controller's call (it knows the FSM state and budgets).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import AttentionConfig
from .models import ACTION_RANK, CONTINUE, EXPLORE, REFRAME, REVISIT, AttentionProposal, AttentionState

W_PRIORITY, W_CONFIDENCE, W_RECENCY = 0.6, 0.25, 0.15
COOLDOWN_PENALTY = 0.3
SUPPORT_BONUS = 0.08       # per extra distinct source backing the same action

# (action that gets the bonus, sources that must all be present, bonus)
COALITIONS = (
    (REVISIT, frozenset({"repetition", "questions"}), 0.15),
    (CONTINUE, frozenset({"progress", "self_prompt"}), 0.10),
    (REFRAME, frozenset({"progress", "repetition"}), 0.05),
)


@dataclass
class Nominee:
    proposal: AttentionProposal
    score: float
    penalties: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.proposal.as_dict()
        d["score"] = round(self.score, 4)
        if self.penalties:
            d["penalties"] = self.penalties
        if d.get("cue") and len(d["cue"]) > 240:
            d["cue"] = d["cue"][:240] + "…"
        return d


@dataclass
class Ranking:
    nominees: list[Nominee]                    # all, best first
    action_scores: dict[str, float]            # action -> coalition score
    coalitions: list[str]

    def best(self, actions: set[str] | None = None) -> Nominee | None:
        for n in self.nominees:
            if actions is None or n.proposal.action in actions:
                return n
        return None

    def pick(self, action: str) -> Nominee | None:
        """Nominee to carry out `action`. CONTINUE is a coalition: the progress
        codelet says *whether* to continue, a cue-bearing nominee (usually the
        self-prompt) says *where* — so prefer the best one that has a cue."""
        if action == CONTINUE:
            for n in self.nominees:
                if n.proposal.action == CONTINUE and n.proposal.cue:
                    return n
        return self.best({action})

    def best_action(self, actions: set[str]) -> str | None:
        cands = [(s, a) for a, s in self.action_scores.items() if a in actions]
        if not cands:
            return None
        cands.sort(key=lambda x: (-x[0], ACTION_RANK.get(x[1], 99)))
        return cands[0][1]

    def as_list(self) -> list[dict]:
        return [n.as_dict() for n in self.nominees]


def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def score_proposal(p: AttentionProposal, state: AttentionState, cycle: int, config: AttentionConfig) -> Nominee:
    s = W_PRIORITY * _clip(p.priority) + W_CONFIDENCE * _clip(p.confidence) + W_RECENCY * _clip(p.recency)
    penalties = []
    if p.topic_key and p.action in (REVISIT, EXPLORE, REFRAME):
        last = state.cooldowns.get(p.topic_key)
        if last is not None and cycle - last < config.revisit_cooldown_cycles:
            s -= COOLDOWN_PENALTY
            penalties.append(f"cooldown({cycle - last}/{config.revisit_cooldown_cycles})")
    return Nominee(p, s, penalties)


def arbitrate(proposals: list[AttentionProposal], state: AttentionState, cycle: int,
              config: AttentionConfig) -> Ranking:
    nominees = [score_proposal(p, state, cycle, config) for p in proposals]
    # Deterministic: score desc, action rank, source, cue text.
    nominees.sort(key=lambda n: (-round(n.score, 6), ACTION_RANK.get(n.proposal.action, 99),
                                 n.proposal.source, n.proposal.cue or ""))

    sources_by_action: dict[str, set[str]] = {}
    best_by_action: dict[str, float] = {}
    for n in nominees:
        a = n.proposal.action
        sources_by_action.setdefault(a, set()).add(n.proposal.source)
        best_by_action[a] = max(best_by_action.get(a, -1.0), n.score)

    all_sources = {n.proposal.source for n in nominees}
    action_scores: dict[str, float] = {}
    for a, best in best_by_action.items():
        action_scores[a] = best + SUPPORT_BONUS * (len(sources_by_action[a]) - 1)

    fired = []
    for action, needed, bonus in COALITIONS:
        if needed <= all_sources and action in action_scores:
            action_scores[action] += bonus
            fired.append(f"{'+'.join(sorted(needed))}→{action}")

    return Ranking(nominees=nominees, action_scores={k: round(v, 4) for k, v in action_scores.items()},
                   coalitions=fired)
