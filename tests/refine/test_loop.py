"""End-to-end tests of the per-item loop with the fake engine (no GPU).

The deterministic helpers of other modules (TermBase.drug_inn / substitution, checks.py,
numbers.secondary_unit_insertions, gate.gate_ops) are provided by small fakes / monkeypatches,
so these tests do not depend on their implementation.
"""

from __future__ import annotations

import asyncio
import re
import types

import pytest
import yaml

from swemedmcq.paths import CONFIG_DIR
from swemedmcq.refine.io import resources_dir
from swemedmcq.refine import numbers as NUM
from swemedmcq.refine.cli.prepare import prepare_item
from swemedmcq.refine.engine import FakePool
from swemedmcq.refine.loop import item_loop as IL
from swemedmcq.refine.loop.accept import AcceptRule, decide, noninf_vote, pairwise_vote, rejection_certain
from swemedmcq.refine.loop.item_loop import Context, ItemLoop, LoopConfig
from swemedmcq.refine.loop.routing import ProtectedTerms, load_keep
from swemedmcq.refine.schemas import SCHEMAS
from swemedmcq.refine.spans import replace_all
from swemedmcq.refine.types import TermHit, Verdict, VerifiedPair

CONFIG = CONFIG_DIR / "refine"

ROW = {
    "prompt_id": "medqa_test_1",
    "question": "A patient receives vancomycin and gentamicin. What is the next step?",
    "options": {"A": "Stop vancomycin", "B": "Continue both"},
    "sv_question": "En patient får vankomycin and gentamicin. Vilket är nästa steg?",
    "sv_options": {"A": "Sätt ut vankomycin", "B": "Fortsätt med båda"},
}


def make_item(**kw):
    return prepare_item({**ROW, **kw}, "A", 1)


# ------------------------------------------------------------------------------------ fakes
class FakeTermBase:
    """Term base with vancomycin (verified), jaundice -> ikterus (level A) and an NPL INN table."""

    TERMS = {"vancomycin": ["vankomycin"], "jaundice": ["ikterus"]}
    INN = {"phenytoin": "fenytoin", "vancomycin": "vankomycin"}

    def spot(self, en_text, field):
        hits = []
        for en, forms in self.TERMS.items():
            for m in re.finditer(en, en_text, flags=re.I):
                hits.append(TermHit(entry_id=f"x:{en}", field=field, en_span=m.group(0), start=m.start(),
                                    end=m.end(), sv_forms=forms, level="A", single_sense=True, source="manual"))
        return hits

    def ground(self, en, sv):
        pairs = []
        for key, text in en.items():
            for h in self.spot(text, key):
                for form in h.sv_forms:
                    s = sv[key].lower().find(form)
                    if s >= 0:
                        pairs.append(VerifiedPair(entry_id=h.entry_id, field=key, en_span=h.en_span,
                                                  sv_span=sv[key][s:s + len(form)], sv_start=s,
                                                  sv_end=s + len(form), level="A"))
        return pairs, []

    def is_accepted_rendering(self, en_quote, sv_quote):
        return any(en in en_quote.lower() and any(f in sv_quote.lower() for f in forms)
                   for en, forms in self.TERMS.items())

    def drug_inn(self, tok):
        return self.INN.get(tok.lower())

    def substitution(self, sv_text, target, form):
        new, n = replace_all(sv_text, [target], form)
        return (new, n) if n else None


class FakeLexicon:
    whitelist: set[str] = set()
    ENGLISH = {"and", "vancomycin", "jaundice", "with", "foo"}
    keep = {"hereditet", "påverkad"}

    def is_swedish(self, tok):
        return tok.lower() not in self.ENGLISH

    def is_english(self, tok):
        return tok.lower() in self.ENGLISH


def claims(user: str) -> list[tuple[str, str]]:
    return re.findall(r'- id=(\S+) \[[^\]]+\][^"]*SV: "([^"]*)"', user)


def versions(user: str) -> tuple[str, str]:
    a = user.split("Version A:", 1)[1].split("Version B:", 1)[0].strip()
    b = user.split("Version B:", 1)[1].split("\n\n", 1)[0].strip()
    return a, b


BAD = ("and", "jaundice", "Phenytoin", " with ", "foo", "familjehistoria", "näst steg", "ett båda", "båda dem")


GOOD = ("Vad är",)


def bad_score(text: str) -> int:
    return sum(text.count(t) for t in BAD) - sum(text.count(t) for t in GOOD)


class Responder:
    """Configurable fake model behaviour.  Every call is recorded in ``self.log``."""

    def __init__(self, *, meaning=None, meaning_qwen=None, terminology=None, language=None,
                 verify=None, repairs=None, integrate=None, judge=None, noninf=None, retranslate=None):
        self.meaning, self.meaning_qwen = meaning or [], meaning_qwen or []
        self.terminology, self.language = terminology or [], language or []
        self.verify = verify or (lambda model, cid, quote, req: ("dismiss" if "decoy" in cid else "confirm", "high"))
        self.repairs = {"and": "och", "with": "med", "hereditet för diabetes": "familjehistoria av diabetes",
                        **(repairs or {})}
        self.integrate = integrate or (lambda req: [])
        self.judge = judge or self.default_judge
        self.noninf = noninf or (lambda model, k, order: (False, "high"))
        self.retranslate = retranslate
        self.log: list = []

    @staticmethod
    def default_judge(model, a, b, k, order, req):
        sa, sb = bad_score(a), bad_score(b)
        return ("A" if sa < sb else "B" if sb < sa else "equal"), "high"

    def __call__(self, req):
        self.log.append(req)
        s = req.schema_name
        if s == "detect_meaning":
            return {"errors": self.meaning_qwen if req.model == "qwen" else self.meaning}
        if s == "detect_terminology":
            return {"errors": self.terminology}
        if s == "detect_language":
            return {"errors": self.language}
        if s == "verify":
            out = []
            for cid, quote in claims(req.user):
                d = self.verify(req.model, cid, quote, req)
                if d is not None:
                    out.append({"id": cid, "reason": "test", "certainty": d[1], "verdict": d[0]})
            return {"decisions": out}
        if s == "repair":
            edits = []
            for m in re.finditer(r'- id=(\S+) \[([^\]]+)\][^"]*SV: "([^"]*)"', req.user):
                cid, fld, quote = m.groups()
                if quote in self.repairs:
                    edits.append({"finding_ids": [cid], "field": fld, "old": quote, "new": self.repairs[quote]})
            return {"edits": edits, "no_change_needed": []}
        if s == "integrate":
            return {"reason": "test", "edits": self.integrate(req)}
        if s == "pairwise":
            a, b = versions(req.user)
            pref, cert = self.judge(req.model, a, b, req.seed_key[-1], req.seed_key[-2], req)
            return {"reason": "test", "certainty": cert, "preferred": pref}
        if s == "noninferiority":
            intro, cert = self.noninf(req.model, req.seed_key[-1], req.seed_key[-2])
            return {"reason": "test", "certainty": cert, "introduces_error": intro}
        if s == "retranslate":
            return {"text": self.retranslate(req)}
        raise AssertionError(s)

    def steps(self, schema=None):
        return [r for r in self.log if schema is None or r.schema_name == schema]


def run(item, responder, tmp_path, cfg=None, models=("gptoss", "qwen"), tb=None):
    async def go():
        pool = FakePool(responder, SCHEMAS, tmp_path / "cache", models=models)
        ctx = Context(pool=pool, termbase=tb or FakeTermBase(), lexicon=FakeLexicon(), cfg=cfg or LoopConfig())
        return await ItemLoop(item, ctx).run()
    return asyncio.run(go())


def finding(rec, quote):
    return next(f for f in rec["findings"] if f["sv_quote"] == quote and f["source"] != "decoy")


def err(quote, en="", cat="terminology", sev="major", field="question", expl="test"):
    return {"field": field, "sv_quote": quote, "en_quote": en, "category": cat, "severity": sev, "explanation": expl}


@pytest.fixture
def item():
    return make_item()


# ------------------------------------------------------------------------------------ acceptance rule (pure)
def V(vote, family="qwen", order="current_first", k=0):
    return Verdict(judge=family, task="pairwise", order=order, family=family, sample=k, vote=vote)


RULE = AcceptRule(0.9, 0.75, 1)


def test_acceptance_levels():
    eight = [V("candidate")] * 6 + [V("candidate", "gptoss")] * 2
    assert decide(eight, "gptoss", RULE)["level"] == "high"
    # one equal: c = 7/8 < 0.9, no veto -> medium
    assert decide(eight[:-1] + [V("equal", "gptoss")], "gptoss", RULE)["level"] == "medium"
    # one veto from Qwen on a GPT-OSS edit -> medium
    assert decide([V("current")] + eight[1:], "gptoss", RULE)["level"] == "medium"
    # the same veto from the producer's own family -> rejected
    assert decide(eight[:-1] + [V("current", "gptoss")], "gptoss", RULE)["level"] == "rejected"
    # code candidates have no family: a GPT-OSS veto is an ordinary veto
    assert decide(eight[:-1] + [V("current", "gptoss")], "code", RULE)["level"] == "medium"
    # two vetoes, or both_wrong counted as a veto
    assert decide([V("current"), V("both_wrong")] + eight[2:], "code", RULE)["level"] == "rejected"
    # c < 0.75
    assert decide([V("equal")] * 3 + eight[3:], "code", RULE)["level"] == "rejected"
    # failed calls count in the denominator
    conf = decide(eight[:-1], "gptoss", RULE, total=8)
    assert conf["level"] == "medium" and conf["score"] == 0.875


def test_certainty_low_counts_as_equal_and_short_circuit():
    assert pairwise_vote("B", "low", "B") == ("candidate", "equal")
    assert pairwise_vote("A", "high", "B") == ("current", "current")
    assert pairwise_vote(None, None, "B") == (None, "failed")
    assert noninf_vote(True, "low") == "equal" and noninf_vote(True, "medium") == "current"
    assert noninf_vote(False, "high") == "candidate"
    assert not rejection_certain([V("current")], "gptoss", RULE, 8)
    assert rejection_certain([V("current"), V("current")], "gptoss", RULE, 8)
    assert rejection_certain([V("current", "gptoss")], "gptoss", RULE, 8)
    assert rejection_certain([V("equal")] * 3, "gptoss", RULE, 8)  # c_max = 5/8


def test_loop_config_from_dict_ignores_unknown_keys():
    cfg = LoopConfig.from_dict({"max_rounds": 2, "judge_models": ["qwen", "gptoss", "gemma"], "bogus": 1})
    assert cfg.max_rounds == 2 and cfg.judge_models == ("qwen", "gptoss", "gemma")
    for name in ("default.yaml", "with_gemma.yaml"):
        LoopConfig.from_dict(yaml.safe_load((CONFIG / name).read_text()).get("loop"))


# ------------------------------------------------------------------------------------ routing classes
def test_leak_is_repaired_and_accepted_high(item, tmp_path):
    resp = Responder()
    rec = run(item, resp, tmp_path)
    assert rec["status"] == "refined"
    assert " och gentamicin" in rec["sv_final"]["question"]
    f = finding(rec, "and")
    assert f["route"] == "LEAK" and f["status"] == "resolved"
    cand = next(c for c in rec["candidates"] if c["id"] in rec["accepted"])
    assert cand["confidence"]["level"] == "high" and cand["confidence"]["votes"]["n"] == 8
    votes = rec["verdicts"][cand["id"]]
    assert {(v["judge"], v["order"], v["sample"]) for v in votes} == {
        (j, o, k) for j, ks in (("qwen", 3), ("gptoss", 1)) for k in range(ks)
        for o in ("current_first", "candidate_first")}
    # Qwen samples are seeded per k
    seeds = {r.seed_key[-1] for r in resp.steps("pairwise") if r.model == "qwen"}
    assert seeds == {"k0", "k1", "k2"}


def test_term_a_integration_with_evidence(tmp_path):
    it = make_item(question="He has jaundice and fever.", sv_question="Han har jaundice och feber.",
                   options={"A": "Yes", "B": "No"}, sv_options={"A": "Ja", "B": "Nej"})
    resp = Responder()
    rec = run(it, resp, tmp_path)
    f = finding(rec, "jaundice")
    assert f["route"] == "TERM-A" and f["tb_form"] == "ikterus" and f["status"] == "resolved"
    assert rec["sv_final"]["question"] == "Han har ikterus och feber."
    cand = next(c for c in rec["candidates"] if c["id"] in rec["accepted"])
    assert cand["kind"] == "term_a" and cand["required_forms"] == ["ikterus"]
    integ = resp.steps("integrate")
    assert integ and "authoritative medical term base" in integ[0].user
    judged = [r for r in resp.steps("pairwise") if "WHOLE" not in r.user]
    assert judged and all('the term base lists "ikterus"' in r.user and 'not a requirement' in r.user for r in judged)


def test_drug_path_noninferiority_phenytoin_accepted(tmp_path):
    """Regression: 'Phenytoin' -> 'fenytoin' is accepted even though GPT-OSS vetoes once."""
    it = make_item(question="A patient with seizures receives phenytoin. What is the next step?",
                   sv_question="En patient med kramper får Phenytoin. Vilket är nästa steg?",
                   options={"A": "Stop phenytoin", "B": "Continue"},
                   sv_options={"A": "Sätt ut fenytoin", "B": "Fortsätt"})

    def noninf(model, k, order):  # GPT-OSS calls the Swedish INN a misspelling in one order
        return (model == "gptoss" and order == "current_first"), "high"

    resp = Responder(noninf=noninf)
    rec = run(it, resp, tmp_path)
    f = finding(rec, "Phenytoin")
    assert f["source"] == "drug" and f["route"] == "DRUG" and f["status"] == "resolved"
    assert "Fenytoin" in rec["sv_final"]["question"] and "Phenytoin" not in rec["sv_final"]["question"]
    cand = next(c for c in rec["candidates"] if c["id"] in rec["accepted"])
    assert cand["kind"] == "drug" and cand["producer"] == "code"
    assert cand["confidence"]["level"] == "medium" and cand["confidence"]["score"] == 0.875
    # deterministic: the drug finding never went to the verifier
    assert all("Phenytoin" not in r.user.split("Claimed errors:")[1] for r in resp.steps("verify"))
    ni = resp.steps("noninferiority")
    assert len(ni) == 8 and {r.seed_key[-2] for r in ni} == {"current_first", "candidate_first"}
    assert all("national drug register NPL" in r.user for r in ni)
    assert all("Original Swedish" not in r.user and "Modified" not in r.user for r in ni)
    news = {re.search(r"Version (\w) differs", r.user).group(1) for r in ni}
    assert news == {"A", "B"}  # neutral labels, both positions
    assert "NPL" in resp.steps("integrate")[0].user and "cannot be wrong" in resp.steps("integrate")[0].user


def test_drug_substituted_at_every_occurrence(tmp_path):
    it = make_item(question="A patient receives phenytoin. What is the next step?",
                   sv_question="En patient får Phenytoin. Vilket är nästa steg?",
                   options={"A": "Stop the drug", "B": "Continue"},
                   sv_options={"A": "Sätt ut phenytoin", "B": "Fortsätt"})
    rec = run(it, Responder(), tmp_path)
    assert "Fenytoin" in rec["sv_final"]["question"] and rec["sv_final"]["option:A"] == "Sätt ut fenytoin"
    drug = [f for f in rec["findings"] if f["source"] == "drug"]
    assert {f["field"] for f in drug} == {"question", "option:A"}
    assert all(f["status"] == "resolved" for f in drug)


def test_drug_integration_vetoed_by_producer_family_is_rejected(tmp_path):
    it = make_item(question="A patient with seizures receives phenytoin. What is the next step?",
                   sv_question="En patient med kramper får Phenytoin. Vilket är nästa steg?",
                   options={"A": "Yes", "B": "No"}, sv_options={"A": "Ja", "B": "Nej"})

    def integrate(req):  # GPT-OSS adjusts a surrounding word -> producer gptoss
        return [{"old": "får", "new": "behandlas med"}] if "får Fenytoin" in req.user else []

    def noninf(model, k, order):
        return (model == "gptoss" and order == "current_first"), "high"

    rec = run(it, Responder(noninf=noninf, integrate=integrate), tmp_path)
    drug = [c for c in rec["candidates"] if c["kind"] == "drug"]
    assert drug[0]["producer"] == "gptoss" and drug[0]["confidence"]["level"] == "rejected"
    assert drug[0]["confidence"]["votes"]["vetoes"][0]["family"] == "gptoss"


def test_accuracy_needs_both_models(tmp_path, item):
    acc = err("Vilket är nästa steg", "What is the next step", cat="accuracy/mistranslation")

    def verify(model, cid, quote, req):
        if "decoy" in cid:
            return "dismiss", "high"
        if model == "qwen":  # only 1 of 3 Qwen samples confirms
            return ("confirm" if req.seed_key[-1] == "k0" else "dismiss"), "high"
        return "confirm", "high"

    rec = run(item, Responder(meaning=[acc], verify=verify), tmp_path)
    f = finding(rec, "Vilket är nästa steg")
    assert f["route"] == "ACCURACY" and f["status"] == "dismissed"
    assert f["confirmations"] == {"gptoss": "confirm", "qwen": "1/3"}


def test_accuracy_confirmed_by_majority_is_repaired(tmp_path, item):
    acc = err("Vilket är nästa steg", "What is the next step", cat="accuracy/mistranslation")

    def verify(model, cid, quote, req):
        if "decoy" in cid:
            return "dismiss", "high"
        return ("dismiss" if model == "qwen" and req.seed_key[-1] == "k2" else "confirm"), "high"

    resp = Responder(meaning=[acc], verify=verify, repairs={"Vilket är nästa steg": "Vad är nästa steg"})
    rec = run(item, resp, tmp_path)
    f = finding(rec, "Vilket är nästa steg")
    assert f["confirmations"]["qwen"] == "2/3" and f["status"] == "resolved"
    assert rec["sv_final"]["question"].startswith("En patient får vankomycin och gentamicin. Vad är")


def test_accuracy_reported_by_both_detectors_needs_one_confirmation(tmp_path, item):
    acc = err("Vilket är nästa steg", "What is the next step", cat="accuracy/mistranslation")
    resp = Responder(meaning=[acc], meaning_qwen=[acc], repairs={"Vilket är nästa steg": "Vad är nästa steg"})
    rec = run(item, resp, tmp_path)
    f = finding(rec, "Vilket är nästa steg")
    assert sorted(f["support"]) == ["gptoss", "qwen"] and f["status"] == "resolved"
    assert "qwen" not in f["confirmations"]
    assert not [r for r in resp.steps("verify") if r.model == "qwen"]


def test_low_certainty_confirmation_is_not_confirmed(tmp_path, item):
    resp = Responder(verify=lambda m, cid, q, req: ("dismiss", "high") if "decoy" in cid else ("confirm", "low"))
    rec = run(item, resp, tmp_path)
    f = finding(rec, "and")
    assert f["status"] == "dismissed" and "low certainty" in f["status_reason"]
    assert rec["sv_final"] == item.sv_l1


def test_det_units_are_inserted_by_code_without_judges(tmp_path, monkeypatch):
    it = make_item(question="Temperature is 37.5 °C (99.5 °F). What next?",
                   sv_question="Temperaturen är 37,5 °C. Vad nu?",
                   options={"A": "Wait", "B": "Treat"}, sv_options={"A": "Vänta", "B": "Behandla"})

    def ins(en, sv):
        i = sv.find("°C")
        return [(i + 2, " (99,5 °F)")] if i >= 0 and "°F" not in sv and "°F" in en else []

    monkeypatch.setattr(NUM, "secondary_unit_insertions", ins, raising=False)
    resp = Responder()
    rec = run(it, resp, tmp_path)
    assert rec["sv_final"]["question"] == "Temperaturen är 37,5 °C (99,5 °F). Vad nu?"
    f = next(f for f in rec["findings"] if f["source"] == "units")
    assert f["route"] == "DET" and f["status"] == "resolved"
    cand = next(c for c in rec["candidates"] if c["kind"] == "units")
    assert cand["id"] in rec["accepted"] and cand["confidence"]["level"] == "deterministic"
    assert cand["id"] not in rec["verdicts"]
    # the missing 99.5 is not also raised as a numbers finding
    assert not [f for f in rec["findings"] if f["source"] == "numbers"]


def test_det_gender_and_spelling_are_repaired(tmp_path, monkeypatch):
    it = make_item(sv_question="En patient får vankomycin och gentamicin. Vilket är näst steg?",
                   sv_options={"A": "Sätt ut vankomycin", "B": "Fortsätt med ett båda"})

    def spelling(sv, lx, en=""):
        i = sv.find(" näst ")
        return [{"start": i + 1, "end": i + 5, "token": "näst", "suggestions": ["nästa"]}] if i >= 0 else []

    def gender(sv, lx):
        i = sv.find("ett båda")
        return [{"start": i, "end": i + 8, "span": "ett båda", "expected": "plural"}] if i >= 0 else []

    fake = types.SimpleNamespace(gender_errors=gender, spelling_suspects=spelling)
    monkeypatch.setattr(IL, "CHECKS", fake)
    resp = Responder(repairs={"näst": "nästa", "ett båda": "båda"},
                     language=[err("Vilket är", cat="fluency/grammar", sev="major")])
    rec = run(it, resp, tmp_path)
    f = finding(rec, "näst")
    assert f["source"] == "spelling" and f["route"] == "DET" and f["status"] == "resolved"
    assert "Vilket är nästa steg?" in rec["sv_final"]["question"]
    g = finding(rec, "ett båda")
    assert g["source"] == "gender" and g["route"] == "DET" and g["status"] == "resolved"
    assert rec["sv_final"]["option:B"] == "Fortsätt med båda"
    llm = finding(rec, "Vilket är")  # major LLM grammar finding: GRAMMAR route (no repair edit here)
    assert llm["route"] == "GRAMMAR" and llm["confirmations"]["qwen"] == "3/3"


def test_grammar_llm_finding_needs_both_models_and_is_repaired(tmp_path):
    it = make_item(sv_options={"A": "Sätt ut vankomycin", "B": "Fortsätt med båda dem"})
    g = err("båda dem", cat="fluency/grammar", sev="major", field="option:B")
    resp = Responder(language=[g], repairs={"båda dem": "båda"})
    rec = run(it, resp, tmp_path)
    f = finding(rec, "båda dem")
    assert f["route"] == "GRAMMAR" and f["confirmations"] == {"gptoss": "confirm", "qwen": "3/3"}
    assert f["status"] == "resolved" and rec["sv_final"]["option:B"] == "Fortsätt med båda"
    assert any(r.model == "qwen" and r.seed_key[-1] == "k2" for r in resp.steps("verify"))

    def verify(model, cid, quote, req):  # Qwen confirms only 1 of 3 samples -> not confirmed
        if "decoy" in cid:
            return "dismiss", "high"
        return ("confirm" if model == "gptoss" or req.seed_key[-1] == "k0" else "dismiss"), "high"

    rec2 = run(it, Responder(language=[g], repairs={"båda dem": "båda"}, verify=verify), tmp_path / "b")
    f2 = finding(rec2, "båda dem")
    assert f2["status"] == "dismissed" and f2["confirmations"]["qwen"] == "1/3"
    assert rec2["sv_final"]["option:B"] == it.sv_l1["option:B"]


def test_sv_noevidence_is_reported_not_verified(tmp_path, item):
    sv = err("Fortsätt med båda", "Continue both", field="option:B")
    resp = Responder(terminology=[sv])
    rec = run(item, resp, tmp_path)
    f = finding(rec, "Fortsätt med båda")
    assert f["route"] == "SV-NOEVIDENCE" and f["status"] == "reported"
    assert all("Fortsätt med båda" not in r.user.split("Claimed errors:")[1] for r in resp.steps("verify"))
    assert rec["sv_final"]["option:B"] == item.sv_l1["option:B"]


def test_protected_term_is_vetoed_hereditet_not_replaced(tmp_path):
    """Regression: 'hereditet' -> 'familjehistoria' is never accepted (judges would prefer it)."""
    it = make_item(question="He has a family history of diabetes. What is the risk?",
                   sv_question="Han har hereditet för diabetes. Vad är risken?",
                   options={"A": "High", "B": "Low"}, sv_options={"A": "Hög", "B": "Låg"})

    def judge(model, a, b, k, order, req):  # anglicised judges prefer the literal rendering
        return ("A" if "familjehistoria" in a else "B" if "familjehistoria" in b else "equal"), "high"

    resp = Responder(terminology=[err("hereditet", "family history")],
                     meaning=[err("hereditet", "family history", cat="accuracy/mistranslation")],
                     repairs={"hereditet": "familjehistoria"}, judge=judge)
    rec = run(it, resp, tmp_path)
    assert rec["sv_final"]["question"] == it.sv_l1["question"]
    fs = [f for f in rec["findings"] if f["sv_quote"] == "hereditet"]
    assert len(fs) == 2 and all(f["status"] == "vetoed" and f["route"] == "PROTECTED" for f in fs)
    # the protected list reaches the detector prompts
    assert '"hereditet"' in resp.steps("detect_terminology")[0].user


def test_gate_rejects_removal_of_protected_term(tmp_path):
    it = make_item(question="He has a family history of diabetes. What is the risk?",
                   sv_question="Han har hereditet för diabetes. Vad är risken?",
                   options={"A": "High", "B": "Low"}, sv_options={"A": "Hög", "B": "Låg"})
    acc = err("hereditet för diabetes", "family history of diabetes", cat="accuracy/omission")
    rec = run(it, Responder(meaning=[acc], meaning_qwen=[acc]), tmp_path)
    assert rec["sv_final"]["question"] == it.sv_l1["question"]
    cand = next(c for c in rec["candidates"] if c["kind"] == "llm_edit")
    reasons = cand["gate_reasons"] + [r for d in cand["dropped_ops"] for r in d["reasons"]]
    assert any(r.startswith("removed_protected_term") for r in reasons)
    assert finding(rec, "hereditet för diabetes")["status"] == "open"
    assert rec["status"] == "disputed_minor"


def test_protected_matcher_inflections():
    p = ProtectedTerms(["påverkad", "stomi", "persisterande ductus arteriosus"])
    assert p.present("Patienten är opåverkad.") == []
    assert p.present("Patienten är påverkade") == ["påverkad"]
    assert p.present("stomin läcker") == ["stomi"]
    text = "Barnet har persisterande ductus arteriosus."
    assert p.covers(text, (11, 23), "persisterande") == "persisterande ductus arteriosus"
    assert p.removed(text, "Barnet har öppen ductus arteriosus.") == ["persisterande ductus arteriosus"]


def test_keep_list_matches_style_guide():
    keep = load_keep(None, resources_dir() / "whitelist.yaml")
    guide = (resources_dir() / "style_guide_sv.md").read_text(encoding="utf-8")
    assert {"hereditet", "påverkad", "opåverkad", "oöm", "Blockmönster", "S-albumin"} <= keep
    missing = [t for t in keep if f"`{t}`" not in guide]
    assert not missing, missing


# ------------------------------------------------------------------------------------ verification, statuses, gate
def test_verifier_reask_on_missing_decision(tmp_path, item):
    def verify(model, cid, quote, req):
        if "decoy" in cid:
            return "dismiss", "high"
        return None if req.seed_key[-1] != "reask" else ("confirm", "high")

    resp = Responder(verify=verify)
    rec = run(item, resp, tmp_path)
    assert finding(rec, "and")["status"] == "resolved"
    assert any(c["step"].endswith("_reask") for c in rec["calls"])
    assert any(v["reasked"] for v in rec["verify_log"])


def test_verifier_missing_twice_is_dismissed(tmp_path, item):
    resp = Responder(verify=lambda m, cid, q, req: ("dismiss", "high") if "decoy" in cid else None)
    rec = run(item, resp, tmp_path)
    f = finding(rec, "and")
    assert f["status"] == "dismissed" and "after one re-ask" in f["status_reason"]


def test_rejected_by_judges_is_not_disputed(tmp_path, item):
    def judge(model, a, b, k, order, req):
        return ("A" if " and " in a else "B"), "high"  # judges keep the current text

    rec = run(item, Responder(judge=judge), tmp_path)
    f = finding(rec, "and")
    assert f["status"] == "rejected_by_judges"
    assert rec["status"] in {"clean", "mechanical_only"}
    assert rec["sv_final"] == item.sv_l1
    rej = [c for c in rec["candidates"] if c.get("confidence")]
    assert rej and all(c["confidence"]["level"] == "rejected" for c in rej)


def test_short_circuit_and_agreement_sample(tmp_path, item):
    def judge(model, a, b, k, order, req):
        return ("A" if " and " in a else "B"), "high"

    resp = Responder(judge=judge)
    run(item, resp, tmp_path, LoopConfig(agreement_sample_rate=0.0, max_rounds=1))
    assert not [r for r in resp.steps("pairwise") if r.model == "gptoss"]  # rejection certain after Qwen
    resp2 = Responder(judge=judge)
    rec = run(item, resp2, tmp_path / "full", LoopConfig(agreement_sample_rate=1.0, max_rounds=1))
    cid = next(c["id"] for c in rec["candidates"] if c.get("confidence"))
    assert len(rec["verdicts"][cid]) == 8 and rec["candidates"][0]["confidence"]["votes"]["complete"]


def test_per_finding_fallback_after_gate_failure(tmp_path):
    it = make_item(question="A patient with fever receives vancomycin and gentamicin.",
                   sv_question="En patient with feber får vankomycin and gentamicin.",
                   options={"A": "Yes", "B": "No"}, sv_options={"A": "Ja", "B": "Nej"})
    resp = Responder(repairs={"with": "foo"})  # the 'with' fix introduces an English token -> gate fails
    rec = run(it, resp, tmp_path)
    first = next(c for c in rec["candidates"] if c["kind"] == "llm_edit")
    assert [d["old"] for d in first["dropped_ops"]] == ["with"]
    assert first["gate_ok"] and first["ops"] == [{"old": "and", "new": "och"}]
    assert first["id"] in rec["accepted"]
    repairs = resp.steps("repair")
    assert len(repairs) >= 2 and "separate, independent edit" in repairs[1].user
    assert "separate, independent edit" not in repairs[0].user
    assert finding(rec, "with")["status"] == "open" and rec["status"] == "refined"


def test_resume_replays_from_cache(item, tmp_path):
    first = run(item, Responder(), tmp_path)
    second = run(item, Responder(), tmp_path)
    assert second["sv_final"] == first["sv_final"] and second["status"] == first["status"]
    assert all(c["from_cache"] for c in second["calls"])
    strip = lambda rec: [{k: v for k, v in c.items() if k != "from_cache"} for c in rec["calls"]]
    assert strip(first) == strip(second)


def test_misquoted_finding_is_ungrounded(item, tmp_path):
    rec = run(item, Responder(terminology=[err("blåssjud")]), tmp_path)
    assert finding(rec, "blåssjud")["status"] == "ungrounded"


def test_correct_term_is_vetoed_not_verified(item, tmp_path):
    rec = run(item, Responder(terminology=[err("vankomycin", "vancomycin")]), tmp_path)
    f = next(f for f in rec["findings"] if f["sv_quote"] == "vankomycin" and f["source"] == "gptoss")
    assert f["status"] == "vetoed"


def test_number_format_findings_are_not_repaired(tmp_path):
    it = make_item(question=ROW["question"] + " The ratio is 0.7.", sv_question=ROW["sv_question"] + " Kvoten är 0.7.")
    fmt = err("0.7", "0.7", cat="accuracy/mistranslation", sev="minor", expl="Swedish uses a decimal comma")
    rec = run(it, Responder(meaning=[fmt]), tmp_path)
    # L1 already converts "0.7" (verbatim in the English) to "0,7", so the finding no longer grounds;
    # either way a pure number-format finding must never be repaired by the LLM.
    assert finding(rec, "0.7")["status"] in {"reported", "ungrounded"}
    assert "0,7" in rec["sv_final"]["question"]
    f07 = finding(rec, "0.7")["id"]
    assert not [c for c in rec["candidates"] if f07 in c["finding_ids"]]


def test_candidate_typography_is_normalised(item, tmp_path):
    rec = run(item, Responder(repairs={"and": "och "}), tmp_path)
    assert " " not in rec["sv_final"]["question"]


def test_detect_failure_falls_back_to_qwen(item, tmp_path):
    base = Responder()

    def responder(req):
        if req.schema_name == "detect_meaning" and req.model == "gptoss":
            return "not json"
        return base(req)

    rec = run(item, responder, tmp_path)
    det = [c for c in rec["calls"] if c["step"] == "detect_meaning" and c["model"] == "gptoss"]
    assert det[0]["model_used"] == "qwen" and det[0]["status"] == "ok"
    assert rec["status"] == "refined"


def test_loop_terminates_within_max_rounds(item, tmp_path):
    def judge(model, a, b, k, order, req):
        return ("A" if " and " in a else "B"), "high"
    rec = run(item, Responder(judge=judge), tmp_path, LoopConfig(max_rounds=3))
    assert len(rec["rounds"]) <= 3 and rec["stop"] in {"nothing_new", "converged", "max_rounds"}


def test_item_final_check_reverts_lowest_confidence_field(tmp_path):
    it = make_item(question="He has fever and cough.", sv_question="Han har feber and hosta.",
                   options={"A": "Stop vancomycin", "B": "Continue"},
                   sv_options={"A": "Sätt ut vancomycin", "B": "Fortsätt"})
    tb = FakeTermBase()
    tb.INN = {}  # vancomycin in option A: TERM-A (ikterus-style) path, not DRUG

    def judge(model, a, b, k, order, req):
        if "WHOLE" in req.user:  # the whole-item check prefers the start text
            return ("A" if a.count("and") > b.count("and") else "B"), "high"
        cand_is_a = bad_score(a) < bad_score(b) or ("vankomycin" in a and "vancomycin" in b)
        if "Sätt ut" in a and model == "gptoss":
            return "equal", "high"  # option A edit gets only medium confidence
        return ("A" if cand_is_a else "B"), "high"

    rec = run(it, Responder(judge=judge), tmp_path, tb=tb)
    fc = rec["final_check"]
    assert fc is not None and fc["ok"] is False and fc["checks"][0]["reverted_field"] == "option:A"
    assert rec["sv_final"]["option:A"] == it.sv_l1["option:A"]
    assert " och " in rec["sv_final"]["question"]
    assert rec["reverted"] and all(r not in rec["accepted"] for r in rec["reverted"])


def test_gemma_judge_adds_two_votes(item, tmp_path):
    cfg = LoopConfig(judge_models=("qwen", "gptoss", "gemma"), max_rounds=1)
    rec = run(item, Responder(), tmp_path, cfg, models=("gptoss", "qwen", "gemma"))
    cid = rec["accepted"][0]
    assert sorted({v["judge"] for v in rec["verdicts"][cid]}) == ["gemma", "gptoss", "qwen"]
    assert len(rec["verdicts"][cid]) == 10


def test_protected_participle_forms_and_multiword_english():
    from swemedmcq.refine.loop.routing import ProtectedTerms, span_is_english
    prot = ProtectedTerms(["medtagen", "påverkad", "ljusstela pupiller"])
    assert prot.present("Barnet verkar medtaget.") == ["medtagen"]
    assert prot.present("Han är påverkat") == ["påverkad"]
    assert prot.removed("Barnet verkar medtaget.", "Barnet verkar sjukt.") == ["medtagen"]
    assert prot.present("Påverkar behandlingen") == []

    class Lex:
        whitelist = {"in situ"}
        def is_english(self, t): return False
        def is_swedish(self, t): return True
    assert span_is_english("peer support", Lex(), "Integrated peer support model")
    assert not span_is_english("in situ", Lex(), "carcinoma in situ")
    assert not span_is_english("EMR", Lex(), "EMR")


# ------------------------------------------------------------------------------- routing rules
def _route(it, f):
    ctx = Context(pool=None, termbase=FakeTermBase(), lexicon=FakeLexicon(), cfg=LoopConfig())
    return ItemLoop(it, ctx).route(f)


def test_obligation_without_english_left_is_reported_not_term_a():
    """Run-2 loophole: a term-base obligation with no Swedish span became TERM-A and replaced valid
    Swedish (Eisenmengers syndrom -> komplex).  Only an English leftover may be replaced."""
    from swemedmcq.refine.types import Finding
    ob = dict(id="r1-o1", round=1, field="question", sv_quote="", en_quote="jaundice", category="terminology",
              severity="major", explanation="x", source="obligation", pass_name="terminology", support=["code"])
    swedish = make_item(question="He has jaundice.", sv_question="Han har gulsot.")
    assert _route(swedish, Finding(**ob))[0] == "SV-NOEVIDENCE"
    leftover = make_item(question="He has jaundice.", sv_question="Han har jaundice.")
    f = Finding(**ob)
    assert _route(leftover, f)[0] == "TERM-A" and f.tb_form == "ikterus"


def test_term_evidence_is_a_reference_drug_evidence_authoritative():
    from swemedmcq.refine import prompting as P
    t = P.term_evidence("jaundice", "ikterus")
    assert "not a requirement" in t and "correct by definition" not in t
    assert "correct by definition" in P.drug_evidence("phenytoin", "fenytoin")


def test_gate_rejects_respelling_of_valid_swedish_word():
    from swemedmcq.refine.gate import gate, respellings

    class Lex:
        whitelist: set[str] = set()
        def is_swedish(self, t): return t.lower() not in {"occlusal"}
        def is_english(self, t): return t.lower() in {"occlusal"}

    assert respellings("i ocklusala planet", "i okklusala planet", "occlusal plane", Lex()) == [("ocklusala", "okklusala")]
    # inflection / agreement at the end, case only, and English leftovers are not respellings
    assert respellings("inte dislocerat", "inte förskjutet", "", Lex()) == []
    assert respellings("benet är förskjuten", "benet är förskjutet", "", Lex()) == []
    assert respellings("Rokuronium", "rokuronium", "", Lex()) == []
    assert respellings("occlusal yta", "ocklusal yta", "occlusal surface", Lex()) == []
    res = gate(kind="llm_edit", field="question", current="i ocklusala planet", candidate="i okklusala planet",
               en_text="in the occlusal plane", other_options={}, anchors=[(2, 11)], verified_sv_spans=[], lexicon=Lex())
    assert not res.ok and "respelled_swedish_word:ocklusala->okklusala" in res.reasons
    res = gate(kind="drug", field="question", current="i ocklusala planet", candidate="i okklusala planet",
               en_text="in the occlusal plane", other_options={}, anchors=[], verified_sv_spans=[], lexicon=Lex())
    assert not any(r.startswith("respelled") for r in res.reasons)


def test_tier1_text_has_code_edits_only(tmp_path, item):
    """sv_auto keeps the drug substitution (tier 1) but not the LLM leak repair (tier 2)."""
    it = make_item(question="A patient receives phenytoin and gentamicin. What is the next step?",
                   sv_question="En patient får Phenytoin and gentamicin. Vilket är nästa steg?",
                   options={"A": "Stop phenytoin", "B": "Continue"},
                   sv_options={"A": "Sätt ut fenytoin", "B": "Fortsätt"})
    rec = run(it, Responder(), tmp_path)
    q_final, q_auto = rec["sv_final"]["question"], rec["sv_auto"]["question"]
    assert "Fenytoin" in q_final and " och gentamicin" in q_final
    assert "Fenytoin" in q_auto and " and gentamicin" in q_auto
    kinds = {c["id"]: c["kind"] for c in rec["candidates"]}
    assert {kinds[c]: t for c, t in rec["tiers"].items()} == {"drug": 1, "llm_edit": 2}
    assert rec["loop_version"] == 3
