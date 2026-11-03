"""Shared record types for the refinement loop.

Every record is a Pydantic model so that it can be validated on load and dumped
to JSON for the append-only, resumable state logs.  Field keys are strings:
``"question"`` or ``"option:<LABEL>"`` (e.g. ``"option:C"``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FieldKey = str  # "question" | "option:A" | ...

Category = Literal[
    "accuracy/mistranslation",
    "accuracy/omission",
    "accuracy/addition",
    "terminology",
    "fluency/grammar",
    "fluency/spelling",
    "source_issue",
]
Severity = Literal["minor", "major", "critical"]
FindingSource = Literal[
    "gptoss", "qwen", "numbers", "leakage", "corrupt", "obligation", "decoy",
    # deterministic detectors
    "drug", "units", "gender", "spelling",
]
# "termbase" is the plain term-base code substitution (still accepted by the schema);
# "drug" (NPL INN substitution + integration), "term_a" (term-base integration repair),
# "units" (secondary unit inserted by code), "llm_edit" (span repair), "retranslate".
CandidateKind = Literal["termbase", "llm_edit", "retranslate", "drug", "term_a", "units"]
Model = Literal["gptoss", "qwen", "gemma"]
# routing classes of a finding; "" = not routed (generic detector, decoys)
Route = Literal["", "DRUG", "TERM-A", "LEAK", "ACCURACY", "DET", "GRAMMAR", "SV-NOEVIDENCE", "PROTECTED", "REPORTED"]
ConfidenceLevel = Literal["high", "medium", "rejected", "deterministic"]
ItemStatus = Literal[
    "clean",
    "refined",
    "disputed_critical",
    "disputed_minor",
    "mechanical_only",
    "failed",
]


def option_key(label: str) -> FieldKey:
    return f"option:{label}"


def field_label(key: FieldKey) -> str | None:
    return key.split(":", 1)[1] if key.startswith("option:") else None


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Change(_Model):
    """One deterministic (mechanical) edit, logged so it can be reverted."""

    field: FieldKey
    rule: str  # e.g. "nbhyphen", "nbsp", "control_char", "decimal_comma", "tail_strip"
    start: int  # offset in the text *before* this change
    before: str
    after: str


class TermHit(_Model):
    """A term-base entry found in an English field."""

    entry_id: str
    field: FieldKey
    en_span: str
    start: int
    end: int
    sv_forms: list[str]  # accepted Swedish forms (lemma-level)
    level: Literal["A", "B"]
    single_sense: bool
    source: str  # "npl_atc", "swemesh", "ksh97p", "npl_lexicon", ...


class VerifiedPair(_Model):
    """An English term whose accepted Swedish rendering is present in the aligned Swedish field."""

    entry_id: str
    field: FieldKey
    en_span: str
    sv_span: str
    sv_start: int
    sv_end: int
    level: Literal["A", "B"]


class Obligation(_Model):
    """An English level-A term with no accepted Swedish form in the aligned field."""

    entry_id: str
    field: FieldKey
    en_span: str
    sv_forms: list[str]
    level: Literal["A", "B"]


class Finding(_Model):
    id: str  # stable within the item, e.g. "r1-f3"
    round: int
    field: FieldKey
    sv_quote: str
    en_quote: str = ""
    category: Category
    severity: Severity
    explanation: str = ""
    source: FindingSource
    # filled by the loop
    status: Literal["open", "dismissed", "vetoed", "ungrounded", "resolved", "reported",
                    "rejected_by_judges"] = "open"
    status_reason: str = ""
    # --- routing, confidence and tier fields (all defaulted) ---
    pass_name: str = ""  # "meaning" | "terminology" | "language"
    route: Route = ""
    support: list[str] = Field(default_factory=list)  # detector families that reported it
    confirmations: dict[str, str] = Field(default_factory=dict)  # verifier -> outcome
    tb_form: str = ""  # authoritative Swedish form (DRUG: NPL INN, TERM-A: level-A term-base form)
    target: str = ""  # English token replaced by code (DRUG / TERM-A substitution)


class SpanOp(_Model):
    old: str
    new: str


class Candidate(_Model):
    id: str  # e.g. "r1-c2"
    round: int
    kind: CandidateKind
    field: FieldKey
    finding_ids: list[str]
    ops: list[SpanOp] = Field(default_factory=list)  # empty for retranslate
    new_text: str  # full field text after applying ops (or the retranslation)
    producer: Literal["gptoss", "qwen", "gemma", "code"]
    gate_ok: bool | None = None
    gate_reasons: list[str] = Field(default_factory=list)
    # --- routing, confidence and tier fields (all defaulted) ---
    route: str = ""
    op_finding_ids: list[list[str]] = Field(default_factory=list)  # finding ids per op (parallel to ops)
    dropped_ops: list[dict] = Field(default_factory=list)  # ops removed by the per-op gate, with reasons
    base_text: str | None = None  # DRUG/TERM-A: text after the code substitution, before integration
    required_forms: list[str] = Field(default_factory=list)  # authoritative forms that must stay
    evidence: str = ""  # evidence line shown to the judges (DRUG, TERM-A)
    # {level: high|medium|rejected|deterministic, score: c, votes: {...}}; set once judged
    confidence: dict | None = None


class Verdict(_Model):
    """One judge call, already un-blinded to current/candidate."""

    judge: Model
    task: Literal["pairwise", "noninferiority", "item_final"]
    order: Literal["current_first", "candidate_first"]
    preferred: Literal["current", "candidate", "equal", "both_wrong"] | None = None
    introduces_error: bool | None = None  # non-inferiority task
    reason: str = ""
    request_hash: str = ""
    # --- routing, confidence and tier fields (all defaulted) ---
    family: str = ""  # model that actually answered (GPT-OSS may fall back to Qwen)
    sample: int = 0  # k of the seeded sample
    certainty: Literal["low", "medium", "high"] | None = None
    # effective vote after the certainty rule (low -> equal); "failed" when the call failed
    vote: Literal["candidate", "current", "equal", "both_wrong", "failed"] | None = None


class Item(_Model):
    """Prepared item (output of `prepare`), read-only for the loop."""

    prompt_id: str
    dataset: str
    answer: str | None
    en: dict[FieldKey, str]  # English fields; MedXpertQA question without its tail
    sv_draft: dict[FieldKey, str]  # raw GPT-5.4 draft fields (layer 1)
    sv_l0: dict[FieldKey, str]  # after hygiene (+ Swedish tail stripped)
    sv_l1: dict[FieldKey, str]  # after locale -> starting text of the loop
    en_tail: str | None = None  # MedXpertQA English "Answer Choices:" tail (verbatim)
    tail_format: str | None = None  # how to rebuild the Swedish tail
    tail_exception: bool = False  # English tail did not match its options
    corrupt_fields: list[FieldKey] = Field(default_factory=list)
    changes_l0: list[Change] = Field(default_factory=list)
    changes_l1: list[Change] = Field(default_factory=list)
    shard: int = 0
