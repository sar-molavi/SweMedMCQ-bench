"""The per-item refinement loop.

    current = L1 text
    for round r in 1..R (R = loop.max_rounds, default 3):
        for pass in loop.passes (default: meaning -> terminology -> language):
            DETECT  (pass-specific; round 1 = all fields, later rounds = fields changed in r-1)
              P1 meaning      GPT-OSS + Qwen accuracy-only detectors (merged, support flag)
                              + deterministic: corrupt fields, missing secondary units, missing numbers
              P2 terminology  GPT-OSS terminology-only detector
                              + deterministic: English INNs (NPL), leakage, term-base obligations
              P3 language     GPT-OSS grammar-only detector (major/critical -> GRAMMAR: both-model
                              confirmation, span repair, strict pairwise)
                              + deterministic: noun gender and spelling (checks.py)
            GROUND  quote grounding (whole word), PROTECTED veto, term-base veto, memory, ROUTING
            VERIFY  GPT-OSS (+ Qwen k-sample majority for ACCURACY not reported by both detectors
                    and for GRAMMAR),
                    decoys, one re-ask for missing decisions, low certainty = not confirmed
            REPAIR  per class: units by code; DRUG by NPL substitution + integration pass; TERM-A by
                    integration repair; LEAK / ACCURACY / DET by span repair; retranslation for
                    corrupt fields (and critical accuracy in rounds >= 2)
            GATE    per operation (gate.gate_ops), protected terms, required authoritative forms
            ACCEPT  vote-based confidence (loop/accept.py): pairwise for edits, non-inferiority for DRUG
        stop when a whole round produced no new candidate
    item-level final check when two or more fields changed by judged edits

Resumability: every LLM call goes through the content-addressed request cache of the engine,
and all code here is deterministic (concurrent calls are gathered in a fixed order and every
decision depends only on completed batches).  Re-running an unfinished item therefore replays
its finished steps from the cache and continues where it stopped.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections import Counter
from dataclasses import dataclass, field, fields as dc_fields
from typing import Any

from .. import gate as G
from .. import numbers as NUM
from .. import prompting as P
from ..engine import LLMRequest
from ..gate import GateResult
from ..io import stable_hash
from ..leakage import leakage_candidates
from ..locale import to_swedish_locale
from ..normalise import hygiene
from ..numbers import missing_numbers
from ..spans import apply_ops, char_edit_distance, find_quote, norm_for_match, replace_all
from ..types import Candidate, Finding, Item, SpanOp, Verdict, field_label
from . import routing as R
from .accept import (LEVEL_RANK, ORDERS, AcceptRule, decide, noninf_vote, pairwise_vote,
                     rejection_certain)

try:  # deterministic gender/spelling checks
    from .. import checks as CHECKS
except ImportError:  # pragma: no cover - until checks.py exists
    CHECKS = None

log = logging.getLogger(__name__)

FLUENCY = R.FLUENCY_CATEGORIES
ACCURACY = R.ACCURACY_CATEGORIES
PASSES = ("meaning", "terminology", "language")
# repair groups of each pass, in order; each group builds candidates on the text left by the previous one
GROUPS = {"meaning": ("units", "accuracy"), "terminology": ("drug", "term_a", "leak"),
          "language": ("det", "grammar")}
KIND_PRIORITY = {"units": 0, "drug": 1, "termbase": 1, "term_a": 2, "llm_edit": 3, "retranslate": 4}
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", re.UNICODE)
_INT_RE = re.compile(r"(?<![\d.,])\d{1,4}(?![\d.,])")


@dataclass
class LoopConfig:
    max_rounds: int = 3
    retranslate_max_words: int = 60
    decoys: bool = True
    item_final_check: bool = True
    judge_models: tuple[str, ...] = ("qwen", "gptoss")  # blind judges, in voting order
    # Fraction of candidates (chosen by a stable hash of the candidate) on which all votes are
    # collected even after rejection is certain, so judge agreement can be measured without the
    # selection bias of short-circuiting.
    agreement_sample_rate: float = 0.1
    # --- passes, judges and acceptance thresholds (fixed before the run) ---
    qwen_judge_samples: int = 3  # k seeded Qwen samples per order
    qwen_verify_samples: int = 3  # k seeded Qwen verifications (majority) for ACCURACY findings
    accept_high: float = 0.9
    accept_medium: float = 0.75
    max_veto_medium: int = 1
    passes: tuple[str, ...] = PASSES
    dual_meaning_detect: bool = True  # P1: Qwen as a second accuracy detector
    long_item_words: int = 120  # longer items get the sentence-by-sentence detect instruction
    language_llm_detect: bool = True  # P3: GPT-OSS grammar-only detector (major/critical -> GRAMMAR route)

    def __post_init__(self) -> None:
        self.judge_models = tuple(self.judge_models)
        self.passes = tuple(self.passes)
        bad = [p for p in self.passes if p not in PASSES]
        if bad:
            raise ValueError(f"unknown loop.passes {bad}; expected a subset of {PASSES}")

    @classmethod
    def from_dict(cls, d: dict | None) -> "LoopConfig":
        """Build from the config's ``loop:`` block, ignoring (and logging) unknown keys."""
        d = dict(d or {})
        known = {f.name for f in dc_fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            log.warning("ignoring unknown loop config keys: %s", unknown)
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def rule(self) -> AcceptRule:
        return AcceptRule(self.accept_high, self.accept_medium, self.max_veto_medium)


@dataclass
class Context:
    pool: Any  # swemedmcq.refine.engine.ModelPool
    termbase: Any  # swemedmcq.refine.termbase.TermBase
    lexicon: Any  # swemedmcq.refine.termbase.Lexicon
    cfg: LoopConfig = field(default_factory=LoopConfig)
    protected: set[str] | None = None  # default: Lexicon.keep, else resources/whitelist.yaml `keep:`
    _matcher: Any = field(default=None, init=False, repr=False)

    def protected_terms(self) -> R.ProtectedTerms:
        if self._matcher is None:
            terms = self.protected if self.protected is not None else R.load_keep(self.lexicon)
            self._matcher = R.ProtectedTerms(terms, self.lexicon)
        return self._matcher


def _key(pass_name: str, fieldkey: str, quote: str) -> tuple[str, str, str]:
    return pass_name, fieldkey, norm_for_match(quote)


def _words(text: str) -> int:
    return len(text.split())


_NUM_FMT = re.compile(r"\d[.,   ]\d")
_FMT_WORDS = re.compile(r"decimal|komma|comma|tusental|thousand|separator|mellanslag|space", re.IGNORECASE)


def _is_number_format(f: Finding) -> bool:
    """A finding about decimal point/comma or digit grouping only (same digits in SV and EN)."""
    if not _NUM_FMT.search(f.sv_quote) or not _FMT_WORDS.search(f.explanation):
        return False
    digits = lambda s: re.sub(r"\D", "", s)
    return not f.en_quote or digits(f.sv_quote) == digits(f.en_quote) or digits(f.sv_quote) in digits(f.en_quote)


def _accepts_kw(fn: Any, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def _overlap(a: tuple[int, int] | None, b: tuple[int, int] | None) -> bool:
    return a is not None and b is not None and a[0] < b[1] and b[0] < a[1]


class ItemLoop:
    """Runs the loop for one prepared item and produces its final record."""

    def __init__(self, item: Item, ctx: Context) -> None:
        self.item = item
        self.ctx = ctx
        self.cfg = ctx.cfg
        self.rule = ctx.cfg.rule
        self.prot = ctx.protected_terms()
        self.en = item.en
        self.start = dict(item.sv_l1)
        self.current = dict(item.sv_l1)
        self.findings: dict[str, Finding] = {}
        self.candidates: list[Candidate] = []
        self.verdicts: dict[str, list[Verdict]] = {}
        self.dismissed: set[tuple[str, str, str]] = set()
        self.rejected: dict[str, set[str]] = {}
        self.rejected_list: list[tuple[str, str]] = []
        self.accepted: list[str] = []
        self.reverted: list[str] = []
        self.rounds: list[dict] = []
        self.calls: list[dict] = []
        self.failed_steps: list[str] = []
        self.decoys: list[dict] = []
        self.verify_log: list[dict] = []
        self.per_finding_fields: set[str] = set()
        self.final_check: dict | None = None
        self.detect_failed_r1 = False
        self._ground_memo: dict = {}
        self._n = 0

    # ------------------------------------------------------------------ helpers
    def _id(self, prefix: str, r: int) -> str:
        self._n += 1
        return f"r{r}-{prefix}{self._n}"

    def _models(self) -> set[str]:
        backends = getattr(self.ctx.pool, "backends", None)
        return set(backends) if isinstance(backends, dict) else {"gptoss", "qwen"}

    async def _call(self, step: str, model: str, system: str, user: str, schema: str,
                    seed_key: list[str], max_tokens: int | None = None):
        req = LLMRequest(model=model, system=system, user=user, schema_name=schema,
                         seed_key=[self.item.prompt_id, *seed_key], max_tokens=max_tokens)
        res = await self.ctx.pool.generate(req)
        self.calls.append({
            "step": step, "model": model, "status": res.status,
            "model_used": getattr(res, "model_used", model),
            "from_cache": res.from_cache, "prompt_tokens": res.prompt_tokens,
            "completion_tokens": res.completion_tokens, "reasoning_empty": res.reasoning_empty,
            "finish_reason": res.finish_reason, "attempts": res.attempts, "hash": res.request_hash,
        })
        if res.status != "ok":
            self.failed_steps.append(step)
            return None, res
        return res.parsed, res

    @staticmethod
    def _family(res: Any, model: str) -> str:
        return getattr(res, "model_used", None) or model

    def _open(self, pass_name: str | None = None) -> list[Finding]:
        return [f for f in self.findings.values()
                if f.status == "open" and (pass_name is None or f.pass_name == pass_name)]

    def _ground(self):
        """Term-base grounding of the current text (memoised per text state)."""
        key = tuple(sorted(self.current.items()))
        if key not in self._ground_memo:
            self._ground_memo[key] = self.ctx.termbase.ground(self.en, self.current)
        return self._ground_memo[key]

    def _verified_pairs(self, fields: list[str] | None = None):
        pairs, _ = self._ground()
        if fields is not None:
            pairs = [p for p in pairs if p.field in fields]
        return pairs

    def _dismissed_quotes(self, pass_name: str) -> list[tuple[str, str]]:
        return [(f.field, f.sv_quote) for f in self.findings.values()
                if f.status == "dismissed" and f.sv_quote and f.pass_name == pass_name]

    def _protected_in_item(self) -> list[str]:
        return self.prot.present(*self.current.values()) if self.prot else []

    def _drug_source(self, token: str) -> str:
        """'npl' or 'swemesh': where the Swedish drug name comes from (for the evidence line)."""
        fn = getattr(self.ctx.termbase, "drug_inn_evidence", None)
        if fn is None or not token:
            return "npl"
        res = fn(token)
        return res[1] if res else "npl"

    def _drug_inn(self, token: str) -> str | None:
        fn = getattr(self.ctx.termbase, "drug_inn", None)
        if fn is None or not token:
            return None
        form = fn(token)
        return form if form and form.casefold() != token.casefold() else None

    def _substitute(self, text: str, target: str, form: str) -> str | None:
        """Code substitution of an English token by an authoritative form at every occurrence."""
        fn = getattr(self.ctx.termbase, "substitution", None)
        if fn is not None:
            out = fn(text, target, form)
            if out is None:
                return None
            new_text, n = out
            return new_text if n else None
        new_text, n = replace_all(text, [target], form)  # fallback (no inflection transfer)
        return new_text if n else None

    def _dismiss(self, f: Finding, reason: str) -> None:
        f.status, f.status_reason = "dismissed", reason
        self.dismissed.add(_key(f.pass_name, f.field, f.sv_quote or f.en_quote))

    # ------------------------------------------------------------------ 1 DETECT
    def _llm_finding(self, r: int, e: dict, family: str, pass_name: str) -> Finding:
        source = family if family in ("gptoss", "qwen") else "gptoss"
        return Finding(id=self._id("f", r), round=r, field=e["field"], sv_quote=e["sv_quote"],
                       en_quote=e["en_quote"], category=e["category"], severity=e["severity"],
                       explanation=e["explanation"], source=source, pass_name=pass_name, support=[family])

    async def detect_meaning(self, r: int, focus: list[str]) -> list[Finding]:
        out: list[Finding] = []
        verified = [(p.en_span, p.sv_span) for p in self._verified_pairs(focus)]
        long_item = sum(_words(self.en[k]) for k in self.en) > self.cfg.long_item_words
        system, user = P.detect_prompt(self.en, self.current, focus, verified, self._dismissed_quotes("meaning"),
                                       kind="meaning", protected=self._protected_in_item(), long_item=long_item)
        models = ["gptoss"] + (["qwen"] if self.cfg.dual_meaning_detect and "qwen" in self._models() else [])
        results = await asyncio.gather(*(
            self._call("detect_meaning", m, system, user, "detect_meaning", [f"r{r}", "detect_meaning"])
            for m in models))
        if r == 1 and results[0][0] is None:
            self.detect_failed_r1 = True
        for m, (parsed, res) in zip(models, results):
            if parsed is None:
                continue
            fam = self._family(res, m)
            for e in parsed["errors"]:
                if e["field"] not in focus or e["field"] not in self.current:
                    continue
                twin = self._twin(out, e, fam)
                if twin is not None:
                    twin.support.append(fam)
                    continue
                out.append(self._llm_finding(r, e, fam, "meaning"))
        out.extend(self._meaning_deterministic(r, focus))
        return out

    def _twin(self, found: list[Finding], e: dict, fam: str) -> Finding | None:
        """A finding of another detector on the same (overlapping) span of the same field."""
        text = self.current[e["field"]]
        span = find_quote(text, e["sv_quote"])
        for f in found:
            if f.field != e["field"] or fam in f.support or f.source not in R.LLM_SOURCES:
                continue
            if norm_for_match(f.sv_quote) == norm_for_match(e["sv_quote"]) or (
                    f.category == e["category"] and _overlap(find_quote(text, f.sv_quote), span)):
                return f
        return None

    def _unit_insertions(self, key: str, text: str | None = None) -> list[tuple[int, str]]:
        fn = getattr(NUM, "secondary_unit_insertions", None)
        if fn is None:
            return []
        return list(fn(self.en[key], self.current[key] if text is None else text))

    @staticmethod
    def _apply_insertions(text: str, ins: list[tuple[int, str]]) -> str:
        for off, s in sorted(ins, key=lambda x: x[0], reverse=True):
            text = text[:off] + s + text[off:]
        return text

    def _meaning_deterministic(self, r: int, focus: list[str]) -> list[Finding]:
        out: list[Finding] = []
        for key in focus:
            sv, en = self.current[key], self.en[key]
            if r == 1 and key in self.item.corrupt_fields:
                out.append(Finding(id=self._id("c", r), round=r, field=key, sv_quote="", en_quote="",
                                   category="accuracy/mistranslation", severity="critical",
                                   explanation="The Swedish draft of this field is corrupted (control character).",
                                   source="corrupt", pass_name="meaning", support=["code"]))
                continue
            ins = self._unit_insertions(key)
            if ins:
                out.append(Finding(id=self._id("u", r), round=r, field=key, sv_quote="",
                                   en_quote="; ".join(s.strip() for _, s in ins),
                                   category="accuracy/omission", severity="major",
                                   explanation="Secondary unit of the English source missing in Swedish: "
                                               + "; ".join(s.strip() for _, s in ins),
                                   source="units", pass_name="meaning", support=["code"]))
                sv = self._apply_insertions(sv, ins)  # numbers restored by the insertion are not missing
            for miss in missing_numbers(en, sv, key):
                out.append(Finding(id=self._id("n", r), round=r, field=key, sv_quote="", en_quote=miss.token,
                                   category="accuracy/omission", severity="critical",
                                   explanation=f"The number '{miss.token}' of the English text is missing in Swedish.",
                                   source="numbers", pass_name="meaning", support=["code"]))
        return out

    async def detect_terminology(self, r: int, focus: list[str]) -> list[Finding]:
        out: list[Finding] = []
        verified = [(p.en_span, p.sv_span) for p in self._verified_pairs(focus)]
        system, user = P.detect_prompt(self.en, self.current, focus, verified,
                                       self._dismissed_quotes("terminology"), kind="terminology",
                                       protected=self._protected_in_item())
        parsed, res = await self._call("detect_terminology", "gptoss", system, user, "detect_terminology",
                                       [f"r{r}", "detect_terminology"])
        drug = self._drug_findings(r, focus)
        drug_toks = {(f.field, f.sv_quote.casefold()) for f in drug}
        out.extend(drug)
        if parsed is not None:
            fam = self._family(res, "gptoss")
            for e in parsed["errors"]:
                if e["field"] not in focus or e["field"] not in self.current:
                    continue
                if (e["field"], e["sv_quote"].strip().casefold()) in drug_toks:
                    continue  # already a deterministic DRUG finding
                out.append(self._llm_finding(r, e, fam, "terminology"))
        whitelist = getattr(self.ctx.lexicon, "whitelist", set())
        for key in focus:
            for tok, _s, _e in leakage_candidates(self.current[key], key, is_swedish=self.ctx.lexicon.is_swedish,
                                                   is_english=self.ctx.lexicon.is_english, whitelist=whitelist):
                if (key, tok.casefold()) in drug_toks:
                    continue
                out.append(Finding(id=self._id("l", r), round=r, field=key, sv_quote=tok, en_quote="",
                                   category="terminology", severity="major",
                                   explanation=f"'{tok}' looks like an untranslated English word.",
                                   source="leakage", pass_name="terminology", support=["code"]))
        if r == 1:
            _, obligations = self._ground()
            for ob in obligations:
                if ob.field not in focus or ob.level != "A":
                    continue
                if (ob.field, ob.en_span.casefold()) in drug_toks:
                    continue
                forms = ", ".join(ob.sv_forms[:3])
                out.append(Finding(id=self._id("o", r), round=r, field=ob.field, sv_quote="",
                                   en_quote=ob.en_span, category="terminology", severity="major",
                                   explanation=(f"The term base lists {forms} for '{ob.en_span}', and none of these "
                                                "appears in the Swedish text. Confirm only if the Swedish leaves the "
                                                "English term or English spelling, or names a different concept; a "
                                                "different established Swedish term is correct."),
                                   source="obligation", pass_name="terminology", support=["code"]))
        return out

    def _drug_findings(self, r: int, focus: list[str]) -> list[Finding]:
        """English INNs left in the Swedish text: a Swedish token that also occurs in the English field
        and whose NPL Swedish INN differs only by transliteration (``TermBase.drug_inn``).  Once found,
        the token is flagged in every focus field where it occurs (substitution at every occurrence).
        Deterministic: no verifier."""
        out: list[Finding] = []
        if getattr(self.ctx.termbase, "drug_inn", None) is None:
            return out
        drugs: dict[str, tuple[str, str]] = {}  # casefold token -> (English token, NPL form)
        for key in focus:
            en_toks = {t.casefold(): t for t in _TOKEN_RE.findall(self.en[key])}
            for m in _TOKEN_RE.finditer(self.current[key]):
                k = m.group(0).casefold()
                if len(k) >= 4 and k in en_toks and k not in drugs:
                    form = self._drug_inn(m.group(0))
                    if form:
                        drugs[k] = (en_toks[k], form)
        for key in focus:
            seen: set[str] = set()
            for m in _TOKEN_RE.finditer(self.current[key]):
                tok = m.group(0)
                k = tok.casefold()
                if k not in drugs or k in seen:
                    continue
                seen.add(k)
                en_tok, form = drugs[k]
                out.append(Finding(id=self._id("d", r), round=r, field=key, sv_quote=tok, en_quote=en_tok,
                                   category="terminology", severity="major",
                                   explanation=(f"'{tok}' is the English INN spelling; the Swedish name in "
                                                f"{P.DRUG_SOURCES.get(self._drug_source(tok), P.DRUG_SOURCES['npl'])} is '{form}'."),
                                   source="drug", pass_name="terminology", support=["code"],
                                   tb_form=form, target=tok))
        return out

    async def detect_language(self, r: int, focus: list[str]) -> list[Finding]:
        out: list[Finding] = []
        if CHECKS is not None:
            for key in focus:
                sv = self.current[key]
                for g in CHECKS.gender_errors(sv, self.ctx.lexicon):
                    out.append(Finding(id=self._id("g", r), round=r, field=key, sv_quote=g["span"], en_quote="",
                                       category="fluency/grammar", severity="minor",
                                       explanation=(f"Noun-gender agreement error in '{g['span']}' (deterministic "
                                                    f"SALDO check; expected: {g['expected']})."),
                                       source="gender", pass_name="language", support=["code"]))
                for s in CHECKS.spelling_suspects(sv, self.ctx.lexicon, self.en[key]):
                    sugg = ", ".join(list(s.get("suggestions") or [])[:3])
                    out.append(Finding(id=self._id("s", r), round=r, field=key, sv_quote=s["token"], en_quote="",
                                       category="fluency/spelling", severity="minor",
                                       explanation=(f"'{s['token']}' is not a Swedish word (deterministic lexicon "
                                                    f"check); close lexicon words: {sugg}."),
                                       source="spelling", pass_name="language", support=["code"]))
        if self.cfg.language_llm_detect:
            system, user = P.detect_prompt(self.en, self.current, focus, [], self._dismissed_quotes("language"),
                                           kind="language", protected=self._protected_in_item())
            parsed, res = await self._call("detect_language", "gptoss", system, user, "detect_language",
                                           [f"r{r}", "detect_language"])
            if parsed is not None:
                fam = self._family(res, "gptoss")
                for e in parsed["errors"]:
                    if e["field"] in focus and e["field"] in self.current:
                        out.append(self._llm_finding(r, e, fam, "language"))
        return out

    # ------------------------------------------------------------------ 2 GROUND + ROUTE
    def ground(self, found: list[Finding], pass_name: str) -> list[Finding]:
        """Quote grounding, PROTECTED and term-base vetoes, memory and duplicate filters, routing."""
        kept: list[Finding] = []
        seen = {_key(f.pass_name, f.field, f.sv_quote or f.en_quote) for f in self.findings.values()
                if f.status in {"open", "resolved", "rejected_by_judges"}}
        whitelist = getattr(self.ctx.lexicon, "whitelist", set())
        for f in found:
            f.pass_name = f.pass_name or pass_name
            self.findings[f.id] = f
            if f.source in ("corrupt", "units"):  # confirmed by construction
                f.route = R.ACCURACY if f.source == "corrupt" else R.DET
                kept.append(f)
                continue
            text, en = self.current[f.field], self.en[f.field]
            span = find_quote(text, f.sv_quote) if f.sv_quote else None
            if f.source in R.LLM_SOURCES:
                if span is None:
                    f.status, f.status_reason = "ungrounded", "sv_quote not in field"
                    continue
                if f.en_quote and find_quote(en, f.en_quote) is None:
                    f.status, f.status_reason = "ungrounded", "en_quote not in English field"
                    continue
            if f.sv_quote and self.prot:
                term = self.prot.covers(text, span, f.sv_quote)
                if term:
                    f.status, f.status_reason, f.route = "vetoed", f"protected term: {term}", R.PROTECTED
                    continue
            if f.category == "terminology" and f.sv_quote and f.source != "drug":
                if f.en_quote and self.ctx.termbase.is_accepted_rendering(f.en_quote, f.sv_quote):
                    f.status, f.status_reason = "vetoed", "term base: accepted rendering"
                    continue
                if f.sv_quote.strip() in whitelist or f.sv_quote.strip().lower() in whitelist:
                    f.status, f.status_reason = "vetoed", "whitelist"
                    continue
            if f.source in R.LLM_SOURCES and _is_number_format(f):
                f.status, f.status_reason = "reported", "number formatting is handled mechanically (L1)"
                f.route = R.REPORTED
                continue
            k = _key(f.pass_name, f.field, f.sv_quote or f.en_quote)
            if k in self.dismissed and f.source not in R.SELF_CONFIRMED_SOURCES:
                f.status, f.status_reason = "dismissed", "dismissed in an earlier round"
                continue
            if k in seen:
                f.status, f.status_reason = "ungrounded", "duplicate"
                continue
            seen.add(k)
            route, reason = self.route(f)
            f.route = route
            if route in (R.SV_NOEVIDENCE, R.REPORTED):
                f.status, f.status_reason = "reported", reason
                continue
            kept.append(f)
        return kept

    def route(self, f: Finding) -> tuple[str, str]:
        """Routing class of a grounded finding and, for reported ones, the reason."""
        if f.source == "drug":
            return R.DRUG, ""
        if f.source in ("units", "gender", "spelling"):
            return R.DET, ""
        if f.source in ("corrupt", "numbers") or f.category in ACCURACY:
            return R.ACCURACY, ""
        if f.category == "source_issue":
            return R.REPORTED, "policy: source issue, reported, not repaired"
        if f.category in FLUENCY:  # LLM language finding (P3)
            if f.severity in ("major", "critical"):
                return R.GRAMMAR, ""
            return R.REPORTED, "policy: minor LLM language finding, reported only"
        # terminology
        tok = (f.sv_quote or "").strip()
        if tok and " " not in tok:
            form = self._drug_inn(tok)  # only English INN spellings map to a different NPL form
            if form:
                f.tb_form, f.target = form, tok
                return R.DRUG, ""
        english = f.source == "leakage" or (tok and R.span_is_english(tok, self.ctx.lexicon, self.en[f.field]))
        # Evidence must show that the current Swedish term is WRONG, not only that another term exists:
        # a term-base form replaces a Swedish span only when the span contains a token that is not a
        # Swedish word (malformed or English). Valid Swedish -> Swedish swaps are reported only.
        valid_swedish = bool(tok) and not english and all(
            self.ctx.lexicon.is_swedish(w) for w in R.words(tok) if len(w) >= 3)
        forms = [x for x in self._tb_form(f) if ", " not in x]  # never an inverted MeSH heading
        if forms and not tok and not self._english_left(f):
            # a term-base form is missing but the Swedish has no span to blame: these were
            # nearly all Swedish->Swedish swaps (Eisenmengers syndrom -> komplex), so report only
            return R.SV_NOEVIDENCE, "no evidence: term-base form absent but no English left in the Swedish (reported only)"
        if forms and not valid_swedish:
            f.tb_form = forms[0]
            return R.TERM_A, ""
        if english:
            if f.severity == "minor":
                return R.REPORTED, "policy: minor terminology without term-base form"
            return R.LEAK, ""
        if not tok:
            return R.REPORTED, "policy: no Swedish span and no term-base form"
        return R.SV_NOEVIDENCE, "no evidence: Swedish span without a level-A term-base form (reported only)"

    def _english_left(self, f: Finding) -> bool:
        """The finding's English term occurs verbatim (whole word) in the current Swedish field."""
        en = (f.en_quote or "").strip()
        return bool(en) and find_quote(self.current.get(f.field, ""), en) is not None

    def _tb_english(self, f: Finding) -> str:
        """English term of a finding: its en_quote, or an English token left in the Swedish text."""
        if f.en_quote:
            return f.en_quote
        q = (f.sv_quote or "").strip()
        if q and (f.source == "leakage" or self.ctx.lexicon.is_english(q)):
            return q
        return ""

    def _tb_form(self, f: Finding) -> list[str]:
        """Accepted single-sense level-A Swedish forms for the finding's English term."""
        en = self._tb_english(f)
        if not en:
            return []
        forms: list[str] = []
        for hit in self.ctx.termbase.spot(en, f.field):
            if hit.level == "A" and hit.single_sense:
                forms.extend(hit.sv_forms)
        return list(dict.fromkeys(forms))

    # ------------------------------------------------------------------ 3 VERIFY
    def _needs_verify(self, f: Finding) -> bool:
        return f.status == "open" and f.source not in R.SELF_CONFIRMED_SOURCES and f.route in R.VERIFY_ROUTES

    @staticmethod
    def _outcome(dec: dict | None) -> str:
        if dec is None:
            return "none"
        if dec["verdict"] != "confirm":
            return "dismiss"
        return "low" if dec.get("certainty") == "low" else "confirm"

    async def verify(self, r: int, pass_name: str, claims: list[Finding]) -> None:
        to_check = [f for f in claims if self._needs_verify(f)]
        if not to_check:
            return
        decoys = self._decoys(r, pass_name, to_check) if self.cfg.decoys else []
        gpt = await self._verify_with("gptoss", r, pass_name, to_check, decoys, None)
        second: list[Finding] = []
        for f in to_check:
            dec = gpt.get(f.id)
            outcome = self._outcome(dec)
            f.confirmations["gptoss"] = outcome
            if outcome == "none":
                self._dismiss(f, "no verification decision (after one re-ask)")
            elif outcome == "dismiss":
                self._dismiss(f, f"verify: {dec['reason']}")
            elif outcome == "low":
                self._dismiss(f, f"verify: confirmed with low certainty: {dec['reason']}")
            else:
                f.status_reason = f"verify: {dec['reason']}"
                if f.route in R.DUAL_CONFIRM_ROUTES and not {"gptoss", "qwen"} <= set(f.support):
                    second.append(f)
        if not second:
            return
        if "qwen" not in self._models():
            for f in second:
                f.confirmations["qwen"] = "unavailable"
                self._dismiss(f, f"{f.route} finding needs a second confirmation (Qwen not loaded)")
            return
        n = max(1, self.cfg.qwen_verify_samples)
        acc_decoys = [d for d in decoys if d.category in ACCURACY]
        samples = await asyncio.gather(*(self._verify_with("qwen", r, pass_name, second, acc_decoys, k)
                                         for k in range(n)))
        for f in second:
            outs = [self._outcome(s.get(f.id)) for s in samples]
            yes = sum(o == "confirm" for o in outs)
            f.confirmations["qwen"] = f"{yes}/{n}"
            if 2 * yes <= n:
                self._dismiss(f, f"second verification (Qwen, k={n}) did not confirm: {Counter(outs)}")

    async def _verify_with(self, model: str, r: int, pass_name: str, claims: list[Finding],
                           decoys: list[Finding], k: int | None) -> dict[str, dict]:
        step = f"verify_{pass_name}"
        key = [f"r{r}", step] + ([f"k{k}"] if k is not None else [])
        system, user = P.verify_prompt(self.en, self.current, claims + decoys)
        parsed, res = await self._call(step, model, system, user, "verify", key)
        decisions = {d["id"]: d for d in (parsed or {}).get("decisions", [])}
        real_missing = [c for c in claims if c.id not in decisions]
        reasked = False
        if real_missing:  # re-ask once, with only the claims that got no decision
            reasked = True
            missing = real_missing + [d for d in decoys if d.id not in decisions]
            system, user = P.verify_prompt(self.en, self.current, missing)
            parsed2, _ = await self._call(step + "_reask", model, system, user, "verify", key + ["reask"])
            for d in (parsed2 or {}).get("decisions", []):
                decisions.setdefault(d["id"], d)
        fam = self._family(res, model)
        self.verify_log.append({"round": r, "pass": pass_name, "model": model, "family": fam, "sample": k,
                                "n_claims": len(claims), "n_decoys": len(decoys), "reasked": reasked,
                                "missing_after_reask": sum(1 for c in claims if c.id not in decisions)})
        for d in decoys:
            dec = decisions.get(d.id)
            self.decoys.append({"round": r, "pass": pass_name, "model": model, "family": fam, "sample": k,
                                "kind": "accuracy" if d.category in ACCURACY else "terminology",
                                "sv": d.sv_quote, "en": d.en_quote,
                                "verdict": dec["verdict"] if dec else None,
                                "certainty": dec.get("certainty") if dec else None})
        return decisions

    def _decoys(self, r: int, pass_name: str, to_check: list[Finding]) -> list[Finding]:
        """Specific, plausible false claims mixed into the verification (false-confirmation rate)."""
        fields = sorted({f.field for f in to_check})
        out: list[Finding] = []
        h = int(stable_hash(self.item.prompt_id, r, pass_name)[:8], 16)
        if any(f.category == "terminology" for f in to_check):
            for p in self._verified_pairs(fields):
                if not self._exact_pair(p):
                    continue
                templates = [
                    f'"{p.sv_span}" is a calque of the English "{p.en_span}"; Swedish clinicians use a different '
                    f"established term for this concept.",
                    f'"{p.sv_span}" names a different concept than the English "{p.en_span}" (a related but '
                    f"narrower condition).",
                    f'"{p.sv_span}" is an outdated term; current Swedish clinical usage for "{p.en_span}" differs.',
                ]
                out.append(Finding(id=f"r{r}-{pass_name}-decoy1", round=r, field=p.field, sv_quote=p.sv_span,
                                   en_quote=p.en_span, category="terminology", severity="major",
                                   explanation=templates[h % len(templates)], source="decoy", pass_name=pass_name))
                break
        if any(f.category in ACCURACY for f in to_check):
            for key in fields:
                en_nums = set(_INT_RE.findall(self.en[key]))
                nums = [n for n in _INT_RE.findall(self.current[key]) if n in en_nums]
                if not nums:
                    continue
                n = nums[h % len(nums)]
                wrong = str(int(n) + 1 + h % 3)
                out.append(Finding(id=f"r{r}-{pass_name}-decoy2", round=r, field=key, sv_quote=n, en_quote=n,
                                   category="accuracy/mistranslation", severity="critical",
                                   explanation=(f"Wrong number: the Swedish gives {n}, but the English source gives "
                                                f"{wrong}."), source="decoy", pass_name=pass_name))
                break
        return out

    def _exact_pair(self, p) -> bool:
        """A verified pair whose Swedish span is exactly an accepted term-base form (no compound)."""
        if p.sv_span.casefold() == p.en_span.casefold() or " " in p.sv_span.strip() and len(p.sv_span) > 40:
            return False
        if self.prot and self.prot.present(p.sv_span):
            return False
        forms: set[str] = set()
        for hit in self.ctx.termbase.spot(p.en_span, p.field):
            if hit.level == "A":
                forms |= {f.casefold() for f in hit.sv_forms}
        if p.sv_span.casefold() in forms:
            return True
        fn = getattr(self.ctx.lexicon, "forms", None)
        if callable(fn):
            for f in list(forms)[:5]:
                try:
                    if p.sv_span.casefold() in {x.casefold() for x in fn(f)}:
                        return True
                except Exception:
                    return False
        return False

    # ------------------------------------------------------------------ 5 REPAIR
    def _is_new(self, fieldkey: str, text: str) -> bool:
        return text != self.current[fieldkey] and norm_for_match(text) not in self.rejected.get(fieldkey, set())

    def _refresh_open(self, pass_name: str) -> None:
        """Open findings whose span was removed by another accepted edit cannot be repaired any more."""
        for f in self._open(pass_name):
            if f.sv_quote and f.source not in ("corrupt",) and find_quote(self.current[f.field], f.sv_quote) is None:
                f.status, f.status_reason = "ungrounded", "span no longer in the text after another accepted edit"

    def _groups(self, findings: list[Finding]) -> list[tuple[str, list[Finding]]]:
        """Per field (default) or per finding (fields that switched to per-finding mode)."""
        by_field: dict[str, list[Finding]] = {}
        for f in findings:
            by_field.setdefault(f.field, []).append(f)
        out: list[tuple[str, list[Finding]]] = []
        for key in P.ordered_fields(self.current):
            fs = by_field.get(key, [])
            if not fs:
                continue
            if key in self.per_finding_fields:
                out.extend((key, [f]) for f in fs)
            else:
                out.append((key, fs))
        return out

    def units_candidates(self, r: int, open_f: list[Finding]) -> list[Candidate]:
        cands: list[Candidate] = []
        for key, fs in self._groups(open_f):
            ins = self._unit_insertions(key)
            if not ins:
                for f in fs:
                    f.status, f.status_reason = "resolved", "secondary unit present"
                continue
            new_text = self._apply_insertions(self.current[key], ins)
            cands.append(Candidate(id=self._id("u", r), round=r, kind="units", field=key,
                                   finding_ids=[f.id for f in fs], ops=[], new_text=new_text, producer="code",
                                   route=R.DET))
        return cands

    async def llm_candidates(self, r: int, open_f: list[Finding], step: str,
                             constraints: list[str] | None = None) -> list[Candidate]:
        if not open_f:
            return []
        per_finding = any(f.field in self.per_finding_fields for f in open_f)
        system, user = P.repair_prompt(self.en, self.current, open_f, constraints or [], self.rejected_list,
                                       per_finding, protected=self._protected_in_item())
        parsed, res = await self._call(step, "gptoss", system, user, "repair", [f"r{r}", step])
        if parsed is None:
            return []
        producer = self._family(res, "gptoss")
        valid = {f.id: f for f in open_f}
        for nc in parsed.get("no_change_needed", []):
            f = valid.get(nc["finding_id"])
            if f is not None and f.status == "open" and f.source not in R.SELF_CONFIRMED_SOURCES:
                self._dismiss(f, f"repair: no change needed: {nc['reason']}")
        groups: dict[tuple, list[dict]] = {}
        for e in parsed.get("edits", []):
            ids = [i for i in e["finding_ids"] if i in valid and valid[i].status == "open"]
            if not ids or e["field"] not in self.current:
                continue
            gkey = (e["field"], tuple(sorted(ids)) if e["field"] in self.per_finding_fields else ())
            groups.setdefault(gkey, []).append({**e, "finding_ids": ids})
        cands: list[Candidate] = []
        for (key, _), edits in groups.items():
            ops = [SpanOp(old=e["old"], new=e["new"]) for e in edits]
            ids = sorted({i for e in edits for i in e["finding_ids"]})
            route = valid[ids[0]].route
            cands.append(Candidate(id=self._id("e", r), round=r, kind="llm_edit", field=key, finding_ids=ids,
                                   ops=ops, op_finding_ids=[list(e["finding_ids"]) for e in edits],
                                   new_text=self.current[key], producer=producer, route=route))
        return cands

    async def retranslate_candidates(self, r: int, open_f: list[Finding]) -> list[Candidate]:
        cands: list[Candidate] = []
        fields: dict[str, list[str]] = {}
        for f in open_f:
            corrupt = f.source == "corrupt"
            critical_acc = f.category in ACCURACY and f.severity == "critical"
            short = field_label(f.field) is not None or _words(self.en[f.field]) <= self.cfg.retranslate_max_words
            tried = bool(self.rejected.get(f.field))
            if corrupt or (critical_acc and short and r >= 2 and tried):
                fields.setdefault(f.field, []).append(f.id)
        for key, ids in fields.items():
            system, user = P.retranslate_prompt(self.en, self.current, key)
            parsed, res = await self._call("retranslate", "gptoss", system, user, "retranslate",
                                           [f"r{r}", "retranslate", key])
            if parsed is None:
                continue
            text = parsed["text"].strip()
            if not self._is_new(key, text):
                continue
            cands.append(Candidate(id=self._id("x", r), round=r, kind="retranslate", field=key,
                                   finding_ids=ids, ops=[], new_text=text, producer=self._family(res, "gptoss"),
                                   route=R.ACCURACY))
        return cands

    async def authority_candidates(self, r: int, open_f: list[Finding], kind: str) -> list[Candidate]:
        """DRUG (kind="drug") and TERM-A (kind="term_a") candidates.

        If the code can substitute every target (an English token) by its authoritative form
        (``TermBase.substitution``, at every occurrence, inflection carried over), the integration
        pass only adjusts the surrounding words; otherwise the integration repair itself puts the
        form in place of the span.  Either way the form is a hard constraint of the gate."""
        cands: list[Candidate] = []
        mode = "drug" if kind == "drug" else "term"
        for key, fs in self._groups(open_f):
            text = self.current[key]
            pairs, evidence = [], []
            substituted = True
            for f in fs:
                en_term = f.target or self._tb_english(f) or f.en_quote
                target = f.target or self._substitution_target(f)
                pairs.append((en_term, f.tb_form, f.sv_quote))
                evidence.append(P.drug_evidence(en_term, f.tb_form, self._drug_source(f.target or en_term))
                                if kind == "drug"
                                else P.term_evidence(en_term, f.tb_form))
                new = self._substitute(text, target, f.tb_form) if (substituted and target) else None
                if new is None:
                    substituted = False
                else:
                    text = new
            base = text if substituted else self.current[key]
            rejected = [t for k, t in self.rejected_list if k == key]
            system, user = P.integrate_prompt(self.en, self.current, key, base, mode=mode, pairs=pairs,
                                              substituted=substituted, rejected=rejected,
                                              protected=self._protected_in_item())
            ids = [f.id for f in fs]
            parsed, res = await self._call(f"integrate_{mode}", "gptoss", system, user, "integrate",
                                           [f"r{r}", "integrate", key, *ids])
            ops = [SpanOp(old=e["old"], new=e["new"]) for e in (parsed or {}).get("edits", [])]
            if not substituted and not ops:
                for f in fs:  # nothing to try: the finding stays open (disputed) unless judged later
                    if "integration produced no edit" not in f.status_reason:
                        f.status_reason = (f.status_reason + "; " if f.status_reason else "") + "integration produced no edit"
                continue
            producer = self._family(res, "gptoss") if ops else "code"
            cands.append(Candidate(id=self._id("t" if kind == "term_a" else "d", r), round=r, kind=kind,
                                   field=key, finding_ids=ids, ops=ops, op_finding_ids=[ids for _ in ops],
                                   new_text=base, producer=producer, route=R.DRUG if kind == "drug" else R.TERM_A,
                                   base_text=base if substituted else None,
                                   required_forms=[f.tb_form for f in fs], evidence="\n".join(evidence)))
        return cands

    def _substitution_target(self, f: Finding) -> str:
        """English token that the code may replace (contract: is_english and not is_swedish)."""
        lx = self.ctx.lexicon
        for cand in ((f.sv_quote or "").strip(), (f.en_quote or "").strip()):
            if not cand or " " in cand:
                continue
            if find_quote(self.current[f.field], cand) is None:
                continue
            if lx.is_english(cand) and not lx.is_swedish(cand):
                return cand
        return ""

    # ------------------------------------------------------------------ 6 GATE
    def _normalise(self, text: str, key: str) -> str:
        """L0 hygiene + L1 locale of a candidate (LLM edits re-introduce U+2011/U+202F, English numbers)."""
        text = hygiene(text, key)[0]
        if _accepts_kw(to_swedish_locale, "en_text"):
            return to_swedish_locale(text, key, en_text=self.en[key])[0]
        return to_swedish_locale(text, key)[0]

    def _gate_common(self, key: str) -> dict:
        others = {k: v for k, v in self.current.items() if k != key and k.startswith("option:")}
        return dict(field=key, en_text=self.en[key], other_options=others if key.startswith("option:") else {},
                    verified_sv_spans=[p.sv_span for p in self._verified_pairs([key])],
                    lexicon=self.ctx.lexicon)

    def _gate(self, *, kind: str, key: str, current: str, candidate: str,
              anchors: list[tuple[int, int]]) -> GateResult:
        kw = dict(kind=kind, current=current, candidate=candidate, anchors=anchors, **self._gate_common(key))
        protected = self.prot.lowered if self.prot else set()
        if _accepts_kw(G.gate, "protected"):
            return G.gate(**kw, protected=protected)
        res = G.gate(**kw)
        reasons = list(res.reasons) + [f"removed_protected_term:{t}"
                                       for t in (self.prot.removed(current, candidate) if self.prot else [])]
        return GateResult(ok=not reasons, reasons=reasons)

    def _gate_ops(self, *, key: str, current: str, ops: list[SpanOp]):
        fn = getattr(G, "gate_ops", None)
        if fn is not None:
            return fn(kind="llm_edit", current=current, ops=ops, protected=self.prot.lowered if self.prot else set(),
                      **self._gate_common(key))
        # local fallback with the same contract (until gate.gate_ops exists)
        kept: list[SpanOp] = []
        rejected: list[tuple[SpanOp, list[str]]] = []
        for op in ops:
            if any(op == k for k in kept):
                continue
            text = apply_ops(current, [op])
            if text is None:
                rejected.append((op, ["apply_failed"]))
                continue
            span = find_quote(current, op.old)
            res = self._gate(kind="llm_edit", key=key, current=current, candidate=text, anchors=[span] if span else [])
            if res.ok:
                kept.append(op)
            else:
                rejected.append((op, list(res.reasons)))
        if not kept:
            return [], None, GateResult(ok=False, reasons=sorted({r for _, rs in rejected for r in rs})), rejected
        text = apply_ops(current, kept)
        if text is None:
            return [], None, GateResult(ok=False, reasons=["apply_failed"]), rejected
        anchors = [s for s in (find_quote(current, op.old) for op in kept) if s]
        res = self._gate(kind="llm_edit", key=key, current=current, candidate=text, anchors=anchors)
        return kept, (text if res.ok else None), res, rejected

    def _form_present(self, text: str, form: str) -> bool:
        fn = getattr(self.ctx.termbase, "match_sv", None)
        if callable(fn):
            try:
                if fn(text, [form]):
                    return True
            except Exception:  # pragma: no cover - fake term bases
                pass
        return re.search(r"(?<!\w)" + re.escape(form.casefold()), text.casefold()) is not None

    def run_gate(self, c: Candidate) -> None:
        if c.gate_ok is False:
            return
        key = c.field
        cur = self.current[key]
        ops_ok = True
        if c.ops:
            base = c.base_text if c.base_text is not None else cur
            ops = [SpanOp(old=o.old, new=hygiene(o.new, key)[0]) for o in c.ops]
            proposed = apply_ops(base, ops)
            if proposed is not None:  # the full proposal is what the rejection memory remembers
                c.new_text = self._normalise(proposed, key)
                if not self._is_new(key, c.new_text):
                    c.gate_ok, c.gate_reasons = False, ["not_new"]
                    return
            ids_of = {}
            for o, ids in zip(ops, c.op_finding_ids or [c.finding_ids] * len(ops)):
                ids_of.setdefault((o.old, o.new), set()).update(ids)
            kept, new_text, res, rejected = self._gate_ops(key=key, current=base, ops=ops)
            c.dropped_ops = c.dropped_ops + [{"old": o.old, "new": o.new, "reasons": list(rs)} for o, rs in rejected]
            if rejected and len(set(c.finding_ids)) >= 2:
                self.per_finding_fields.add(key)  # a multi-finding field whose candidate failed
            if new_text is None:
                kept = []
                if c.base_text is None:
                    c.gate_ok, c.gate_reasons = False, list(res.reasons) or ["all_ops_rejected"]
                    return
                new_text = c.base_text  # integration edits dropped: the substitution alone is the candidate
            c.ops = kept
            c.op_finding_ids = [sorted(ids_of.get((o.old, o.new), set(c.finding_ids))) for o in kept]
            if c.kind == "llm_edit":
                c.finding_ids = sorted({i for ids in c.op_finding_ids for i in ids})
            if not kept and c.kind in ("drug", "term_a"):
                c.producer = "code"
            c.new_text = new_text
            ops_ok = res.ok
        elif c.base_text is not None:
            c.new_text = c.base_text
        gated = c.new_text
        c.new_text = self._normalise(c.new_text, key)
        if not self._is_new(key, c.new_text):
            c.gate_ok, c.gate_reasons = False, ["not_new"]
            return
        reasons: list[str] = []
        if c.kind != "llm_edit" or c.new_text != gated or not ops_ok:
            anchors = [s for s in (find_quote(cur, o.old) for o in c.ops) if s]
            res = self._gate(kind=c.kind, key=key, current=cur, candidate=c.new_text, anchors=anchors)
            reasons.extend(res.reasons)
        for form in c.required_forms:
            if form and not self._form_present(c.new_text, form):
                reasons.append(f"missing_required_term:{form}")
        if c.kind == "drug":
            for fid in c.finding_ids:
                t = self.findings[fid].target
                if t and re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", c.new_text, re.IGNORECASE):
                    reasons.append(f"kept_english_term:{t}")
        c.gate_ok, c.gate_reasons = not reasons, reasons
        if reasons and len(set(c.finding_ids)) >= 2:
            self.per_finding_fields.add(key)

    # ------------------------------------------------------------------ 7 ACCEPT
    def _samples(self, judge: str) -> int:
        return max(1, self.cfg.qwen_judge_samples) if judge == "qwen" else 1

    def _judges(self) -> list[str]:
        loaded = self._models()
        return [j for j in self.cfg.judge_models if j in loaded]

    async def _one_vote(self, *, task: str, judge: str, k: int, order: str, field: str | None,
                        cur_text: str, cand_text: str, evidence: str, key: list[str]) -> Verdict:
        if order == "current_first":
            a, b, cand_label = cur_text, cand_text, "B"
        else:
            a, b, cand_label = cand_text, cur_text, "A"
        protected = self.prot.present(a, b) if self.prot else []
        if task == "noninferiority":
            system, user = P.noninferiority_prompt(self.en, field, a, b, new_label=cand_label, evidence=evidence,
                                                   protected=protected)
            schema = "noninferiority"
        else:
            system, user = P.pairwise_prompt(self.en, self.current if field else None, field, a, b,
                                             evidence=evidence, protected=protected)
            schema = "pairwise"
        step = {"pairwise": "judge", "noninferiority": "noninferiority", "item_final": "item_final"}[task]
        parsed, res = await self._call(step, judge, system, user, schema, [*key, order, f"k{k}"])
        certainty = (parsed or {}).get("certainty")
        pref, introduces = None, None
        if task == "noninferiority":
            introduces = None if parsed is None else bool(parsed["introduces_error"])
            vote = noninf_vote(introduces, certainty)
        else:
            pref, vote = pairwise_vote((parsed or {}).get("preferred"), certainty, cand_label)
        return Verdict(judge=judge, task=task, order=order, preferred=pref, introduces_error=introduces,
                       reason=(parsed or {}).get("reason", ""), request_hash=res.request_hash,
                       family=self._family(res, judge), sample=k, certainty=certainty, vote=vote)

    async def collect_votes(self, *, task: str, field: str | None, cur_text: str, cand_text: str,
                            evidence: str, producer: str, key: list[str], full: bool) -> tuple[list[Verdict], int]:
        """Votes judge by judge (each judge's samples x orders concurrently); stops early once rejection
        is certain, unless ``full`` (the agreement sample)."""
        judges = self._judges()
        total = sum(2 * self._samples(j) for j in judges)
        votes: list[Verdict] = []
        for judge in judges:
            batch = [self._one_vote(task=task, judge=judge, k=k, order=order, field=field, cur_text=cur_text,
                                    cand_text=cand_text, evidence=evidence, key=key)
                     for k in range(self._samples(judge)) for order in ORDERS]
            votes.extend(await asyncio.gather(*batch))
            if not full and rejection_certain(votes, producer, self.rule, total):
                break
        return votes, total

    def _in_agreement_sample(self, cid: str) -> bool:
        return int(stable_hash(self.item.prompt_id, cid)[:8], 16) / 0xFFFFFFFF < self.cfg.agreement_sample_rate

    async def accept(self, r: int, c: Candidate) -> bool:
        if c.kind == "units":  # deterministic: gate only
            c.confidence = {"level": "deterministic", "score": None, "votes": {}}
            return True
        task = "noninferiority" if c.kind == "drug" else "pairwise"
        prefix = "noninf" if task == "noninferiority" else "judge"
        votes, total = await self.collect_votes(task=task, field=c.field, cur_text=self.current[c.field],
                                                cand_text=c.new_text, evidence=c.evidence, producer=c.producer,
                                                key=[f"r{r}", prefix, c.id], full=self._in_agreement_sample(c.id))
        self.verdicts.setdefault(c.id, []).extend(votes)
        c.confidence = decide(votes, c.producer, self.rule, total)
        return c.confidence["level"] in ("high", "medium")

    def _reject(self, c: Candidate) -> None:
        self.rejected.setdefault(c.field, set()).add(norm_for_match(c.new_text))
        self.rejected_list.append((c.field, c.new_text))

    # ------------------------------------------------------------------ PASS / ROUND
    async def _group_candidates(self, r: int, pass_name: str, group: str) -> list[Candidate]:
        open_f = self._open(pass_name)
        if group == "units":
            return self.units_candidates(r, [f for f in open_f if f.source == "units"])
        if group == "accuracy":
            acc = [f for f in open_f if f.route == R.ACCURACY and f.source != "corrupt"]
            cands = await self.llm_candidates(r, acc, "repair_meaning")
            return cands + await self.retranslate_candidates(
                r, [f for f in self._open(pass_name) if f.route == R.ACCURACY])
        if group == "drug":
            return await self.authority_candidates(r, [f for f in open_f if f.route == R.DRUG], "drug")
        if group == "term_a":
            return await self.authority_candidates(r, [f for f in open_f if f.route == R.TERM_A], "term_a")
        if group == "leak":
            return await self.llm_candidates(r, [f for f in open_f if f.route == R.LEAK], "repair_terminology")
        if group == "det":
            det = [f for f in open_f if f.route == R.DET and f.source in ("gender", "spelling")]
            return await self.llm_candidates(r, det, "repair_language")
        if group == "grammar":
            return await self.llm_candidates(r, [f for f in open_f if f.route == R.GRAMMAR], "repair_grammar")
        raise ValueError(group)

    async def _run_group(self, r: int, pass_name: str, group: str, plog: dict) -> None:
        self._refresh_open(pass_name)
        cands = await self._group_candidates(r, pass_name, group)
        for c in cands:
            self.run_gate(c)
        self.candidates.extend(cands)
        plog["candidates"] += len(cands)
        plog["new_candidates"] += sum(1 for c in cands if c.gate_reasons != ["not_new"])
        plog["gate_passed"] += sum(1 for c in cands if c.gate_ok)
        for c in cands:
            if not c.gate_ok:
                self._reject(c)
        by_field: dict[str, list[Candidate]] = {}
        for c in cands:
            if c.gate_ok:
                by_field.setdefault(c.field, []).append(c)
        changed: set[str] = set()
        for key in P.ordered_fields(self.current):
            options = sorted(by_field.get(key, []),
                             key=lambda c: (KIND_PRIORITY.get(c.kind, 9), char_edit_distance(self.current[key], c.new_text)))
            for c in options:
                if key in changed:
                    break  # the other candidates of this field are stale; their findings stay open
                if any(self.findings[i].status != "open" for i in c.finding_ids):
                    continue
                if await self.accept(r, c):
                    self.current[key] = c.new_text
                    changed.add(key)
                    plog["accepted"].append(c.id)
                    self.accepted.append(c.id)
                    for fid in c.finding_ids:
                        self.findings[fid].status, self.findings[fid].status_reason = "resolved", f"by {c.id}"
                else:
                    self._reject(c)
                    if len(set(c.finding_ids)) >= 2:
                        self.per_finding_fields.add(key)
        plog["_changed"] |= changed

    async def run_pass(self, r: int, pass_name: str, focus: list[str]) -> dict:
        plog: dict[str, Any] = {"pass": pass_name, "detected": 0, "grounded": 0, "confirmed": 0, "candidates": 0,
                                "new_candidates": 0, "gate_passed": 0, "accepted": [], "_changed": set()}
        if focus:
            detect = {"meaning": self.detect_meaning, "terminology": self.detect_terminology,
                      "language": self.detect_language}[pass_name]
            found = await detect(r, focus)
            plog["detected"] = len(found)
            kept = self.ground(found, pass_name)
            plog["grounded"] = len(kept)
            plog["routes"] = dict(Counter(f.route for f in found if f.route))
            await self.verify(r, pass_name, kept)
        plog["confirmed"] = len(self._open(pass_name))
        for group in GROUPS[pass_name]:
            await self._run_group(r, pass_name, group, plog)
        return plog

    async def run(self) -> dict:
        focus = P.ordered_fields(self.current)
        stop = "max_rounds"
        first_confirmed = 0
        for r in range(1, self.cfg.max_rounds + 1):
            rlog: dict[str, Any] = {"round": r, "detect_focus": focus, "passes": []}
            changed: set[str] = set()
            new = 0
            for pass_name in self.cfg.passes:
                plog = await self.run_pass(r, pass_name, focus)
                changed |= plog.pop("_changed")
                new += plog["new_candidates"]
                rlog["passes"].append(plog)
            rlog["changed_fields"] = sorted(changed)
            rlog["new_candidates"] = new
            rlog["accepted"] = [a for p in rlog["passes"] for a in p["accepted"]]
            self.rounds.append(rlog)
            if r == 1:
                first_confirmed = sum(p["confirmed"] for p in rlog["passes"])
            if new == 0:
                stop = "converged" if not self._open() else "nothing_new"
                break
            focus = sorted(changed, key=lambda k: (k != "question", k))
            if not focus and not self._open():
                stop = "converged"
                break
        await self._final_check()
        self._close_findings()
        return self.record(stop, first_confirmed)

    def _close_findings(self) -> None:
        """Open findings that had a gate-passing candidate judged and rejected -> rejected_by_judges;
        the remaining open findings (no candidate ever passed the gate) make the item disputed."""
        judged: set[str] = set()
        for c in self.candidates:
            if c.gate_ok and self.verdicts.get(c.id):
                judged.update(c.finding_ids)
        for f in self._open():
            if f.id in judged:
                f.status, f.status_reason = "rejected_by_judges", "every gate-passing candidate was rejected by the judges"

    # ------------------------------------------------------------------ ITEM FINAL CHECK
    def _judged_accepted(self) -> list[Candidate]:
        acc = set(self.accepted)
        return [c for c in self.candidates if c.id in acc and c.kind != "units"]

    def _field_baseline(self, key: str) -> str:
        """Start text of a field with only its deterministic (units) edits re-applied."""
        text = self.start[key]
        if any(c.kind == "units" and c.field == key and c.id in self.accepted for c in self.candidates):
            ins = self._unit_insertions(key, text)
            text = self._apply_insertions(text, ins)
        return text

    def _field_auto(self, key: str) -> str:
        """Tier-1 (automatic) text of a field: the start text with only the code-made edits of its
        accepted candidates re-applied (secondary units; drug-name substitutions without the LLM
        integration edits).  Every other accepted edit is tier 2."""
        text = self._field_baseline(key)
        acc = set(self.accepted)
        for c in self.candidates:
            if c.id not in acc or c.field != key or c.kind != "drug":
                continue
            for fid in c.finding_ids:
                f = self.findings.get(fid)
                if f is None or not f.tb_form:
                    continue
                target = f.target or f.sv_quote
                new = self._substitute(text, target, f.tb_form) if target else None
                if new is not None:
                    text = new
        return text

    def tiers(self) -> dict[str, int]:
        """Release tier of each accepted candidate: 1 = code-made (units, drug name), 2 = LLM-made.
        A drug candidate whose integration edits were kept is tier 2 for those edits; its
        substitution is in the tier-1 text either way."""
        acc = set(self.accepted)
        out: dict[str, int] = {}
        for c in self.candidates:
            if c.id in acc:
                out[c.id] = 1 if c.kind == "units" or (c.kind == "drug" and not c.ops) else 2
        return out

    async def _final_check(self) -> None:
        """Whole-item vote (same rule) when judged edits changed two or more fields; on failure the
        field with the lowest-confidence edit is reverted until the check passes or one field remains."""
        judged = self._judged_accepted()
        if not self.cfg.item_final_check or len({c.field for c in judged}) < 2:
            return
        checks: list[dict] = []
        ok = False
        full = self._in_agreement_sample("item_final")
        for d in range(len({c.field for c in judged})):
            judged = self._judged_accepted()
            fields_ = sorted({c.field for c in judged}, key=lambda k: P.ordered_fields(self.current).index(k))
            if len(fields_) < 2:
                break  # one field edit remains: it keeps its own acceptance (the check did not pass)
            producer = "gptoss" if any(c.producer != "code" for c in judged) else "code"
            evidence = "\n".join(dict.fromkeys(c.evidence for c in judged if c.evidence))
            votes, total = await self.collect_votes(task="item_final", field=None, cur_text=P.item_text(self.start),
                                                    cand_text=P.item_text(self.current), evidence=evidence,
                                                    producer=producer, key=["final", f"d{d}"], full=full)
            conf = decide(votes, producer, self.rule, total)
            checks.append({"fields": fields_, "confidence": conf, "verdicts": [v.model_dump() for v in votes]})
            if conf["level"] in ("high", "medium"):
                ok = True
                break

            def strength(k: str) -> tuple:
                cs = [c for c in judged if c.field == k]
                lv = min(LEVEL_RANK.get((c.confidence or {}).get("level", "rejected"), 0) for c in cs)
                sc = min(((c.confidence or {}).get("score") or 0.0) for c in cs)
                return lv, sc, -fields_.index(k)

            worst = min(fields_, key=strength)
            self.current[worst] = self._field_baseline(worst)
            for c in judged:
                if c.field == worst:
                    self.accepted.remove(c.id)
                    self.reverted.append(c.id)
                    if c.confidence is not None:
                        c.confidence = {**c.confidence, "reverted_by_final_check": True}
                    for fid in c.finding_ids:
                        self.findings[fid].status = "rejected_by_judges"
                        self.findings[fid].status_reason = "reverted by the item-level final check"
            checks[-1]["reverted_field"] = worst
        self.final_check = {"ok": ok, "checks": checks, "reverted": list(self.reverted),
                            "verdicts": [v for ch in checks for v in ch["verdicts"]]}

    # ------------------------------------------------------------------ RECORD
    def record(self, stop: str, first_confirmed: int) -> dict:
        open_f = self._open()
        changed_fields = [k for k in self.current if self.current[k] != self.start[k]]
        mech = bool(self.item.changes_l0 or self.item.changes_l1)
        if self.detect_failed_r1 and not changed_fields:
            status = "failed"
        elif changed_fields:
            status = "refined"
        elif open_f:
            status = "disputed_critical" if any(f.severity == "critical" for f in open_f) else "disputed_minor"
        else:
            status = "mechanical_only" if mech else "clean"
        acc = set(self.accepted)
        levels = Counter((c.confidence or {}).get("level", "none") for c in self.candidates if c.id in acc)
        return {
            "prompt_id": self.item.prompt_id,
            "dataset": self.item.dataset,
            "loop_version": 3,
            "status": status,
            "stop": stop,
            "sv_final": self.current,
            "sv_auto": {k: self._field_auto(k) for k in self.current},
            "tiers": self.tiers(),
            "changed_fields": changed_fields,
            "residual_critical": any(f.severity == "critical" for f in open_f),
            "first_round_confirmed": first_confirmed,
            "findings": [f.model_dump() for f in self.findings.values()],
            "candidates": [c.model_dump() for c in self.candidates],
            "verdicts": {k: [v.model_dump() for v in vs] for k, vs in self.verdicts.items()},
            "accepted": self.accepted,
            "reverted": self.reverted,
            "confidence_counts": dict(levels),
            "rounds": self.rounds,
            "decoys": self.decoys,
            "verify_log": self.verify_log,
            "final_check": self.final_check,
            "failed_steps": self.failed_steps,
            "calls": self.calls,
        }
