"""Vote-based acceptance with an explicit confidence level.

Votes per candidate (``loop.judge_models``): Qwen k seeded samples x 2 orders, GPT-OSS 1 x 2
orders, Gemma (optional) 1 x 2 orders.  A vote with ``certainty == "low"`` counts as ``equal``.

    c     = votes preferring the candidate / all votes (planned votes; failed calls count, as non-votes)
    veto  = a vote for the current text, or "both_wrong"
            (non-inferiority: "introduces an error")
    high     c >= accept_high   and no veto
    medium   c >= accept_medium and at most max_veto_medium vetoes, none from the producer's family
             (producer of an LLM edit: its model; code candidates have no family, so any veto counts
             as an ordinary veto)
    rejected otherwise
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from ..types import Verdict

VETO_VOTES = frozenset({"current", "both_wrong"})
ORDERS = ("current_first", "candidate_first")
_EPS = 1e-9


@dataclass(frozen=True)
class AcceptRule:
    high: float = 0.9
    medium: float = 0.75
    max_veto_medium: int = 1


def pairwise_vote(raw: str | None, certainty: str | None, cand_label: str) -> tuple[str | None, str]:
    """(un-blinded preference, effective vote) of one pairwise answer."""
    if raw is None:
        return None, "failed"
    if raw in ("equal", "both_wrong"):
        pref = raw
    else:
        pref = "candidate" if raw == cand_label else "current"
    vote = "equal" if certainty == "low" and pref != "equal" else pref
    return pref, vote


def noninf_vote(introduces_error: bool | None, certainty: str | None) -> str:
    if introduces_error is None:
        return "failed"
    if certainty == "low":
        return "equal"
    return "current" if introduces_error else "candidate"


def producer_family(producer: str) -> str | None:
    return None if producer in ("code", "", None) else producer


def _family(v: Verdict) -> str:
    return v.family or v.judge


def _counts(votes: Iterable[Verdict]) -> Counter:
    return Counter(v.vote or "failed" for v in votes)


def decide(votes: list[Verdict], producer: str, rule: AcceptRule, total: int | None = None) -> dict:
    """Confidence record ``{level, score, votes}`` of a (possibly short-circuited) vote list."""
    n = max(total or 0, len(votes))
    counts = _counts(votes)
    cand = counts.get("candidate", 0)
    vetoes = [v for v in votes if v.vote in VETO_VOTES]
    fam = producer_family(producer)
    c = cand / n if n else 0.0
    if n and c + _EPS >= rule.high and not vetoes:
        level = "high"
    elif (n and c + _EPS >= rule.medium and len(vetoes) <= rule.max_veto_medium
          and all(_family(v) != fam for v in vetoes)):
        level = "medium"
    else:
        level = "rejected"
    return {
        "level": level,
        "score": round(c, 4),
        "votes": {
            "n": n,
            "cast": len(votes),
            "complete": len(votes) >= n,
            **{k: counts.get(k, 0) for k in ("candidate", "current", "equal", "both_wrong", "failed")},
            "by_family": {f: dict(_counts(v for v in votes if _family(v) == f))
                          for f in sorted({_family(v) for v in votes})},
            "vetoes": [{"judge": v.judge, "family": _family(v), "order": v.order, "sample": v.sample,
                        "vote": v.vote} for v in vetoes],
            "producer": producer,
        },
    }


def rejection_certain(votes: list[Verdict], producer: str, rule: AcceptRule, total: int) -> bool:
    """True when no outcome of the remaining ``total - len(votes)`` votes can reach high or medium."""
    remaining = max(0, total - len(votes))
    cand = sum(1 for v in votes if v.vote == "candidate")
    vetoes = [v for v in votes if v.vote in VETO_VOTES]
    c_max = (cand + remaining) / total if total else 0.0
    fam = producer_family(producer)
    high_possible = not vetoes and c_max + _EPS >= rule.high
    medium_possible = (c_max + _EPS >= rule.medium and len(vetoes) <= rule.max_veto_medium
                       and all(_family(v) != fam for v in vetoes))
    return not (high_possible or medium_possible)


LEVEL_RANK = {"rejected": 0, "medium": 1, "high": 2, "deterministic": 3}
