#!/usr/bin/env python
"""Build the term base and lexicons into the resources folder (paths.yaml ``resources``).

    python -m swemedmcq.refine.resources.build_termbase [--force] [--no-coverage]

Raw inputs live under paths.yaml ``termbase_raw`` (override: ``SWEMEDMCQ_TERMBASE_RAW``):
``svdict/raw/`` (SALDO, SALDOM, NPL zip and product XMLs, KSH97-P, hunspell sv_SE, Socialstyrelsen
termbank, KVÅ), ``svdict/processed/`` (one normalised JSONL per Swedish source, e.g.
``ksh97p.jsonl``, ``saldo.jsonl``), ``freedict/en_sv.sqlite3`` and the downloads of
``fetch_wikidata_atc`` / ``fetch_swemesh`` (``wikidata_atc/``, ``swemesh/``, ``nlm_mesh/``).
The hand-written ``whitelist.yaml`` is read from the resources folder itself.

Sources
  * NPL (Läkemedelsverket): product XMLs (``atc-code-lx`` + ``substance-description-lx``) give
    ATC code -> Swedish substance name; Lexicondata gives real sv/en pairs for pharmaceutical
    forms, routes of administration, dosage units and quantity units.
  * Wikidata (``raw/wikidata_atc``): ATC -> English label / INN / aliases, Swedish label.
    Joined with NPL on the ATC code -> ``npl_atc`` rows (EN INN -> NPL Swedish name).
  * Svensk MeSH 2026 (``raw/swemesh``): EN descriptor + entry terms <-> SV preferred term + synonyms.
  * KSH97-P: 967 EN <-> SV diagnosis labels (processed svdict jsonl).
  * SALDO/SALDOM, hunspell sv_SE, Socialstyrelsen termbank, KVÅ: Swedish validity / inflection.
  * FreeDict EN-SV: English headwords only (English word list).

Outputs (resources folder): termbase.parquet, sv_lexicon.txt.gz, sv_lemma_forms.json.gz,
sv_compound_parts.txt.gz, en_wordlist.txt.gz, whitelist.yaml (copy), MANIFEST.json and, with
``--coverage``, coverage_report.json.

Authority levels
  A  npl_atc (single-substance ATC codes, EN label/INN/lowercase aliases similar to the INN or
     confirmed by MeSH), swemesh preferred term (+ un-inverted variants) for the English terms of
     the descriptor's preferred concept (NLM concept structure), Swedish MeSH synonyms that are
     orthographic cognates of an English term (cognate-promoted), ksh97p (vague NOS/other labels
     dropped), npl_lexicon, and a handful of hand-written ``manual_terms`` (whitelist.yaml).
  B  npl_atc rows that fail the single-substance / similarity checks, swemesh Swedish synonyms,
     swemesh terms of narrower (non-preferred) concepts and trade names, the head of inverted
     Swedish headings ('Rekombination, genetisk' -> rekombination), wikidata_sv (Wikidata's own
     Swedish label).

single_sense=False when (documented heuristic):
  1. en_key is listed in whitelist.yaml ``ambiguous_en`` (general English / polysemous words);
  2. the same source maps en_key to >= 2 concepts whose Swedish form sets are disjoint
     (e.g. a MeSH entry term shared by two descriptors with different Swedish names);
  3. the English term is a short all-caps acronym (matched case-sensitively, still ambiguous);
  4. npl_lexicon single-word English terms that are ordinary English headwords (``tablet``);
  5. a single-word English term that is an ordinary English headword (FreeDict) and none of its
     level-A Swedish forms is an orthographic cognate (``will`` -> testamente, ``chills`` ->
     köldfrossa, ``rash`` -> utslag): general vocabulary with several everyday renderings.

obligation_ok (extra column, used by TermBase.ground): an A row creates an obligation only if it is
single-sense AND (source is npl_atc, i.e. an official Swedish substance name, OR at least one
Swedish A form is an orthographic cognate of the English term, cognate_similarity >= 0.6).
Non-cognate translations (``physical examination`` -> hälsokontroll, ``oxygen saturation`` ->
syremättnad) still verify, hint and veto, but Swedish has too many valid alternatives for a
missing MeSH form to be evidence of an error.

Idempotent: if MANIFEST.json records the same input fingerprint and all outputs exist, nothing
is rebuilt (``--force`` to rebuild).
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

from swemedmcq.paths import path as repo_path
from swemedmcq.refine.termbase import (  # noqa: E402
    SV_WORD_RE,
    TermBase,
    cognate_similarity,
    Lexicon,
    en_key,
    is_acronym,
)

RAW = repo_path("termbase_raw")
SVDICT = RAW / "svdict"
RAW_SVDICT = SVDICT / "raw"
PROC_SVDICT = SVDICT / "processed"
NPL_DIR = RAW_SVDICT / "lakemedelsverket_npl_xml"
FREEDICT = RAW / "freedict" / "en_sv.sqlite3"
INPUT_JSONL = repo_path("draft")  # coverage report: English fields of the drafts

CACHE = RAW / "cache"
BUILD = repo_path("resources")

MESH_TREES = ("A", "C", "D", "E", "F", "G", "B03", "B04", "B01.300")
NPL_LEXICONS = (
    "pharmaceutical-form-lx",
    "route-of-administration-lx",
    "dosage-unit-lx",
    "quantity-unit-lx",
)
KSH_VAGUE_RE = re.compile(r"\b(NOS|UNS|unspecified|ospecificerad\w*|other|others|övrig\w*|annan|annat|andra)\b", re.I)
COMBO_RE = re.compile(r"\+|\bkombination|\bcombination|\band\b|\boch\b|/|,| with ", re.I)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_lines_gz(path: Path, lines) -> int:
    lines = sorted(set(lines))
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as g:
        g.write("\n".join(lines).encode("utf-8"))
    atomic_write_bytes(path, buf.getvalue())
    return len(lines)


def write_json_gz(path: Path, obj) -> None:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as g:
        g.write(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    atomic_write_bytes(path, buf.getvalue())


def cached(name: str, inputs: list[Path], fn):
    """Cache an intermediate parse result under raw/cache keyed by input size+mtime."""
    key = hashlib.sha256(
        json.dumps([(str(p), p.stat().st_size, int(p.stat().st_mtime)) for p in inputs]).encode()
    ).hexdigest()[:16]
    path = CACHE / f"{name}.{key}.json.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    obj = fn()
    write_json_gz(path, obj)
    return obj


def lname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_tsv(path: Path) -> list[list[str]]:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        header = True
        for line in f:
            line = line.rstrip("\r\n")
            if header:
                header = False
                continue
            if not line:
                continue
            rows.append([c[1:-1].replace('""', '"') if len(c) >= 2 and c[0] == c[-1] == '"' else c for c in line.split("\t")])
    return rows


def sv_case(term: str, sv_words: set[str] | None = None) -> str:
    """Lower-case the first letter of a Swedish term unless it looks like an acronym or eponym."""
    term = " ".join(term.split())
    if not term:
        return term
    first = term.split(" ", 1)[0]
    letters = [c for c in first if c.isalpha()]
    if len(letters) >= 2 and sum(c.isupper() for c in letters) >= 2:
        return term  # HIV-infektioner, pH-värde, ...
    if (
        " " in term
        and first.endswith("s")
        and first[:1].isupper()
        and sv_words is not None
        and first.lower() not in sv_words
    ):
        return term  # eponym genitive: Alzheimers sjukdom, Crohns sjukdom
    return term[:1].lower() + term[1:]


def uninvert(term: str) -> list[str]:
    """'Diabetes mellitus, typ 2' -> ['diabetes mellitus typ 2', 'typ 2 diabetes mellitus']."""
    if "," not in term:
        return []
    parts = [p.strip() for p in term.split(",") if p.strip()]
    if len(parts) < 2 or any(len(p.split()) > 4 for p in parts):
        return []
    low = [p if (len(p) > 1 and sum(c.isupper() for c in p.split()[0]) >= 2) else p[:1].lower() + p[1:] for p in parts]
    return [" ".join(low), " ".join(reversed(low))]


similar = cognate_similarity


# ---------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------


def parse_npl_lexicon(name: str) -> list[dict]:
    path = NPL_DIR / "Lexicondata" / f"{name}.xml"
    out = []
    for _, el in ET.iterparse(path, events=("end",)):
        if lname(el.tag) != "commonTerm":
            continue
        rec = {"id": None, "status": None, "term": {}, "short": {}}
        for ch in el:
            t = lname(ch.tag)
            lang = ch.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
            if t == "mpaId":
                rec["id"] = (ch.text or "").strip()
            elif t == "term-name" and lang:
                rec["term"][lang] = ch.attrib.get("v", "").strip()
            elif t == "short-name" and lang:
                rec["short"][lang] = ch.attrib.get("v", "").strip()
            elif t == "status":
                rec["status"] = ch.attrib.get("status-code")
        out.append(rec)
        el.clear()
    return out


def parse_npl_products() -> dict:
    """ATC code -> {Swedish substance description: product count}."""
    subst = {r["id"]: r["term"].get("sv", "") for r in parse_npl_lexicon("substance-description-lx")}
    atc: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    n_files = n_with = 0
    for path in sorted((NPL_DIR / "Productdata").glob("*.xml")):
        n_files += 1
        codes, descs = [], []
        try:
            for _, el in ET.iterparse(path, events=("end",)):
                t = lname(el.tag)
                if t == "atc-code-lx":
                    codes.append(el.attrib.get("v", "").strip())
                elif t == "substance-description-lx":
                    descs.append(subst.get(el.attrib.get("v", ""), ""))
        except ET.ParseError:
            continue
        codes = [c for c in codes if c]
        descs = [d for d in descs if d]
        if codes and descs:
            n_with += 1
            for c in set(codes):
                for d in set(descs):
                    atc[c][d] += 1
    return {"n_files": n_files, "n_products_with_atc_and_substance": n_with, "atc": {k: dict(v) for k, v in atc.items()}}


def parse_saldom() -> dict:
    """lemma (lowercased) -> sorted forms; plus compound-part forms."""
    path = RAW_SVDICT / "saldom.xml"
    lemma_forms: dict[str, set] = collections.defaultdict(set)
    parts: set[str] = set()
    lemma = None
    forms: list[tuple[str, str]] = []
    for ev, el in ET.iterparse(path, events=("end",)):
        tag = el.tag
        if tag == "FormRepresentation":
            for f in el:
                if f.attrib.get("att") == "writtenForm":
                    lemma = f.attrib.get("val")
        elif tag == "WordForm":
            wf = msd = None
            for f in el:
                a = f.attrib.get("att")
                if a == "writtenForm":
                    wf = f.attrib.get("val")
                elif a == "msd":
                    msd = f.attrib.get("val")
            if wf:
                forms.append((wf, msd or ""))
            el.clear()
        elif tag == "LexicalEntry":
            if lemma and " " not in lemma.strip():
                key = lemma.lower()
                for wf, msd in forms:
                    if msd in ("c", "ci", "cm"):
                        p = wf.lower().strip("-")
                        if len(p) >= 2 and " " not in p:
                            parts.add(p)
                    elif msd == "sms":
                        continue
                    else:
                        lemma_forms[key].add(wf.lower())
                lemma_forms[key].add(key)
            lemma, forms = None, []
            el.clear()
    return {"lemma_forms": {k: sorted(v) for k, v in lemma_forms.items()}, "compound_parts": sorted(parts)}


_MORPH_SKIP_MSD = {"c", "ci", "cm", "sms"}


def parse_saldom_morph() -> dict:
    """Noun genders and inflection tables (msd -> forms) for nouns and adjectives, plus the parts
    of speech of every SALDOM word form (used by ``checks.gender_errors`` and
    ``TermBase.substitution``)."""
    path = RAW_SVDICT / "saldom.xml"
    nouns: dict[str, dict] = {}
    adjs: dict[str, dict] = {}
    pos_of: dict[str, set] = collections.defaultdict(set)
    lemma = pos = gender = None
    forms: list[tuple[str, str]] = []
    for _, el in ET.iterparse(path, events=("end",)):
        tag = el.tag
        if tag == "FormRepresentation":
            for f in el:
                a = f.attrib.get("att")
                if a == "writtenForm":
                    lemma = f.attrib.get("val")
                elif a == "partOfSpeech":
                    pos = f.attrib.get("val")
                elif a == "inherent":
                    gender = f.attrib.get("val")
        elif tag == "WordForm":
            wf = msd = None
            for f in el:
                a = f.attrib.get("att")
                if a == "writtenForm":
                    wf = f.attrib.get("val")
                elif a == "msd":
                    msd = f.attrib.get("val")
            if wf and msd not in _MORPH_SKIP_MSD:
                forms.append((wf.lower(), msd or ""))
            el.clear()
        elif tag == "LexicalEntry":
            if lemma and pos and " " not in lemma.strip():
                key = lemma.lower()
                for wf, _ in forms:
                    if " " not in wf:
                        pos_of[wf].add(pos)
                if pos == "nn":
                    rec = nouns.setdefault(key, {"g": set(), "f": collections.defaultdict(set)})
                    rec["g"].add(gender if gender in ("u", "n", "v") else "x")
                    for wf, msd in forms:
                        rec["f"][msd].add(wf)
                elif pos == "av":
                    rec = adjs.setdefault(key, collections.defaultdict(set))
                    for wf, msd in forms:
                        rec[msd].add(wf)
            lemma = pos = gender = None
            forms = []
            el.clear()

    def gender_of(gs: set) -> str:
        return gs.pop() if len(gs) == 1 else "v"

    return {
        "noun": {k: [gender_of(set(v["g"])), {m: sorted(fs) for m, fs in v["f"].items()}] for k, v in nouns.items()},
        "adj": {k: {m: sorted(fs) for m, fs in v.items()} for k, v in adjs.items()},
        "pos": {k: ",".join(sorted(v)) for k, v in pos_of.items()},
    }


def parse_npl_substances() -> list[str]:
    """Lower-cased single-substance names of the NPL substance lexicon ('Hydroxikarbamid', 'Primakin')."""
    out = set()
    for r in parse_npl_lexicon("substance-description-lx"):
        name = " ".join((r["term"].get("sv") or "").split())
        if name and not COMBO_RE.search(name):
            out.add(name.lower())
    return sorted(out)


# Hand-audited links of an English drug name to its NPL Swedish substance name where neither
# Wikidata nor Svensk MeSH connects them ("NPL takes priority over MeSH for drugs").
NPL_EXTRA_LINKS = {"tromethamine": "trometamol", "tham": "trometamol"}


def sv_singular(form: str, morph_noun_forms: dict, sv_words: set, en_term: str, head_ok=None) -> str | None:
    """Singular (lemma) of a plural Swedish MeSH heading ('Lacerationer' -> 'laceration',
    'Leverceller' -> 'levercell'); only the last word is changed.  None if it is not a plural
    or the singular is not certain."""
    words = form.split(" ")
    if len(words) > 1:
        return None  # 'dendritiska celler': the adjective would need its singular too
    last = words[-1]
    low = last.lower()
    head, _, tail = low.rpartition("-")
    prefix = (last[: len(head)] + "-") if head else ""
    word = tail
    sing = None
    for i in range(0, max(1, len(word) - 3)):
        part = word[i:]
        an = morph_noun_forms.get(part)
        if an is not None and (i == 0 or (i >= 3 and head_ok is not None and head_ok(word[:i]))):
            sg = {s for s, is_pl in an if is_pl}
            if sg and len(sg) == 1 and all(is_pl for _, is_pl in an):
                sing = word[:i] + sg.pop()
            break
    if sing is None and word.endswith("er") and len(word) >= 7 and word[-3] not in "aeiouyåäö":
        stem = word[:-2]
        en_last = en_term.lower().split()[-1] if en_term.split() else ""
        if not stem.endswith("er") and (stem in sv_words or cognate_similarity(en_last.rstrip("s"), stem) >= 0.75):
            sing = stem
    if sing is None or sing == word:
        return None
    new_last = prefix + (sing[:1].upper() + sing[1:] if not prefix and last[:1].isupper() else sing)
    return " ".join(words[:-1] + [new_last])


def parse_nlm_mesh() -> dict:
    """DescriptorUI -> {"pref": [terms of the preferred concept], "other": [terms of other concepts
    and trade names]} from the NLM descriptor XML."""
    path = RAW / "nlm_mesh" / "desc2026.xml"
    out = {}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag != "DescriptorRecord":
            continue
        dui = el.findtext("DescriptorUI")
        pref, other = [], []
        for concept in (el.find("ConceptList") if el.find("ConceptList") is not None else []):
            is_pref = concept.get("PreferredConceptYN") == "Y"
            for term in (concept.find("TermList") if concept.find("TermList") is not None else []):
                s = term.findtext("String")
                if not s:
                    continue
                (pref if is_pref and term.get("LexicalTag") != "TRD" else other).append(s)
        out[dui] = {"pref": pref, "other": other}
        el.clear()
    return out


def load_swemesh():
    meta = json.loads((RAW / "swemesh" / "swemesh.meta.json").read_text())
    d = RAW / "swemesh" / meta["extracted_dir"]
    terms = {r[0]: (r[1], r[2]) for r in read_tsv(d / "terms.csv") if len(r) >= 3}
    entries = collections.defaultdict(list)
    for r in read_tsv(d / "entryterms.csv"):
        if len(r) >= 2:
            entries[r[0]].append(r[1])
    syn = collections.defaultdict(list)
    for r in read_tsv(d / "synonyms.csv"):
        if len(r) >= 2:
            syn[r[1]].append(r[0])
    trees = collections.defaultdict(list)
    for r in read_tsv(d / "mesh_numbers.csv"):
        if len(r) >= 2:
            trees[r[1]].append(r[0])
    return meta, d, terms, entries, syn, trees


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


EN_JUNK_RE = re.compile(r"[\[\];:=*@#]|\d[,.]\d|\(.*\d.*\)")


def en_term_ok(t: str) -> bool:
    t = t.strip()
    if not t or len(t) > 60 or EN_JUNK_RE.search(t):
        return False
    k = en_key(t)
    if not k or len(k.split()) > 8:
        return False
    if not is_acronym(t) and len(k.replace(" ", "")) < 3:
        return False
    if sum(c.isdigit() for c in t) > 3:
        return False
    return any(c.isalpha() for c in t)


def en_variants(t: str) -> list[str]:
    """English surface variants to index: inverted MeSH headings are indexed only un-inverted."""
    t = " ".join(t.split())
    if "," in t:
        parts = [p.strip() for p in t.split(",") if p.strip()]
        if len(parts) >= 2 and all(len(p.split()) <= 4 and len(p) >= 2 for p in parts):
            v = " ".join(reversed(parts))
            if not re.match(r"(a|an|the)\s", v, re.I):
                return [v]
        return []
    return [t]


def sv_clean(term: str) -> list[str]:
    """Normalise a Swedish MeSH form; returns the form plus derived variants."""
    t = " ".join(term.replace(" - ", "-").split())
    out = [t]
    m = re.match(r"(?i)läkemedelskombination(?:en)? av (.+)$", t)
    if m:
        out.append(m.group(1))
    if t.lower().endswith("kombination") and len(t) > len("kombination") + 4 and " " not in t:
        out.append(t[: -len("kombination")])
    return out


def build(force: bool = False, do_coverage: bool = True) -> None:
    t_start = time.time()
    BUILD.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    import yaml

    whitelist_path = BUILD / "whitelist.yaml"
    whitelist = yaml.safe_load(whitelist_path.read_text(encoding="utf-8")) or {}
    ambiguous = {str(x).lower() for x in whitelist.get("ambiguous_en") or []}
    ambiguous_keys = {en_key(x) for x in ambiguous}

    # ---- fingerprint ----------------------------------------------------------------------
    fp_inputs = [
        Path(__file__),
        Path(__file__).resolve().parents[1] / "termbase.py",
        whitelist_path,
        RAW_SVDICT / "lakemedelsverket_npl.zip",
        RAW_SVDICT / "saldom.xml",
        PROC_SVDICT / "ksh97p.jsonl",
        FREEDICT,
    ]
    for p in (RAW / "wikidata_atc" / "wikidata_atc.meta.json", RAW / "swemesh" / "swemesh.meta.json", RAW / "nlm_mesh" / "nlm_mesh.meta.json"):
        if p.exists():
            fp_inputs.append(p)
    fingerprint = hashlib.sha256(
        json.dumps([(str(p), p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest() if p.stat().st_size < 5e6 else int(p.stat().st_mtime)) for p in fp_inputs]).encode()
    ).hexdigest()
    outputs = ["termbase.parquet", "sv_lexicon.txt.gz", "sv_lemma_forms.json.gz", "sv_compound_parts.txt.gz", "en_wordlist.txt.gz",
               "sv_morph.json.gz", "npl_substances.txt.gz", "whitelist.yaml", "MANIFEST.json"]
    man_path = BUILD / "MANIFEST.json"
    if not force and man_path.exists() and all((BUILD / o).exists() for o in outputs):
        try:
            if json.loads(man_path.read_text()).get("input_fingerprint") == fingerprint:
                log("build is up to date (input fingerprint matches); use --force to rebuild")
                if do_coverage and not (BUILD / "coverage_report.json").exists():
                    coverage()
                return
        except Exception:
            pass

    timings: dict[str, float] = {}
    manifest_sources: dict[str, dict] = {}

    # ---- Swedish lexicon ------------------------------------------------------------------
    t0 = time.time()
    log("parsing SALDOM (cached after the first run)")
    saldom = cached("saldom", [RAW_SVDICT / "saldom.xml"], parse_saldom)
    lemma_forms: dict[str, list[str]] = saldom["lemma_forms"]
    compound_parts = set(saldom["compound_parts"])
    sv_words: set[str] = set()
    for forms in lemma_forms.values():
        sv_words.update(forms)
    n_saldom_forms = len(sv_words)
    log("parsing SALDOM morphology (cached after the first run)")
    morph = cached("saldom_morph", [RAW_SVDICT / "saldom.xml"], parse_saldom_morph)
    # noun form -> [(singular form, form is 'pl indef nom')]: used to singularise plural MeSH headings
    morph_noun_forms: dict[str, list] = collections.defaultdict(list)
    for _lem, (_g, table) in morph["noun"].items():
        sg = (table.get("sg indef nom") or [_lem])[0]
        for msd, fs in table.items():
            for f in fs:
                morph_noun_forms[f].append((sg, msd == "pl indef nom"))
    timings["saldom"] = time.time() - t0

    word_ok = re.compile(r"^[a-zåäöéüæøàèáíóúç]+(?:-[a-zåäöéüæøàèáíóúç]+)*$")

    def add_tokens(text: str, into: set[str]) -> None:
        for tok in text.lower().split():
            tok = tok.strip(".,;:()[]\"'!?")
            if word_ok.match(tok):
                into.add(tok)
                if "-" in tok:
                    into.update(p for p in tok.split("-") if len(p) >= 2)

    # ---- English word list -----------------------------------------------------------------
    con = sqlite3.connect(FREEDICT)
    fd_words = {w.lower() for (w,) in con.execute("select distinct english from entries")}
    con.close()
    en_single = {w for w in fd_words if re.fullmatch(r"[a-z]+(?:-[a-z]+)*", w) and len(w) >= 2}
    freedict_headwords = en_single  # used for single_sense rule 4

    counts_lex = {}
    for res, fname in (
        ("saldo", "saldo.jsonl"),
        ("hunspell_sv_se", "hunspell_sv_se.jsonl"),
        ("socialstyrelsen_termbank", "socialstyrelsen_termbank.jsonl"),
        ("kva_procedures", "kva_procedures.jsonl"),
    ):
        before = len(sv_words)
        toks: set[str] = set()
        for r in read_jsonl(PROC_SVDICT / fname):
            add_tokens(r.get("sv_normalized") or r.get("sv_term") or "", toks)
        if res in ("socialstyrelsen_termbank", "kva_procedures"):
            # multi-word labels sometimes keep English words ('epidural blood patch')
            toks = {t for t in toks if t not in en_single}
        sv_words |= toks
        counts_lex[res] = len(sv_words) - before
    log(f"Swedish lexicon: {len(sv_words):,} forms (SALDOM {n_saldom_forms:,}; added {counts_lex})")

    def head_ok(h: str) -> bool:
        return h in sv_words or h in compound_parts or (h.endswith("s") and (h[:-1] in sv_words or h[:-1] in compound_parts))

    npl_substances = cached("npl_substances", [RAW_SVDICT / "lakemedelsverket_npl.zip"], parse_npl_substances)
    npl_names = set(npl_substances)

    # ---- NPL ----------------------------------------------------------------------------------
    t0 = time.time()
    log("parsing NPL product XMLs (cached after the first run)")
    npl = cached("npl_products", [RAW_SVDICT / "lakemedelsverket_npl.zip"], parse_npl_products)
    timings["npl_products"] = time.time() - t0
    npl_atc_primary: dict[str, dict] = {}
    for code, names in npl["atc"].items():
        if len(code) != 7:
            continue
        total = sum(names.values())
        single = {n: c for n, c in names.items() if not COMBO_RE.search(n)}
        if not single:
            continue
        primary, cnt = max(single.items(), key=lambda kv: (kv[1], -len(kv[0])))
        share = cnt / total
        npl_atc_primary[code] = {
            "name": primary,
            "share": share,
            "n_products": total,
            "n_names": len(names),
            "single_substance": share >= 0.6,
        }
    log(f"NPL: {len(npl['atc']):,} ATC codes, {len(npl_atc_primary):,} 7-char codes with a single-substance name")
    # official single-substance names (ATC primary names): these take priority over MeSH for drugs
    npl_primary = {v["name"].lower() for v in npl_atc_primary.values() if v["single_substance"]}

    rows: list[dict] = []

    def add_row(source, concept_id, en_term, sv_forms, level, extra_single=True, prepend=False, drug=False):
        sv_forms = [f for f in dict.fromkeys(" ".join(s.split()) for s in sv_forms) if f]
        if not sv_forms:
            return
        for v in en_variants(en_term):
            if not en_term_ok(v):
                continue
            k = en_key(v)
            rows.append(
                {
                    "source": source,
                    "concept_id": concept_id,
                    "en_term": v,
                    "en_key": k,
                    "sv_forms": sv_forms,
                    "level": level,
                    "_single": extra_single,
                    "_prepend": prepend,
                    "drug": bool(drug or source == "npl_atc"),
                }
            )

    mesh_sv_by_key: dict[str, set] = collections.defaultdict(set)
    # ---- Svensk MeSH -----------------------------------------------------------------------------
    mesh_stats = collections.Counter()
    if (RAW / "swemesh" / "swemesh.meta.json").exists():
        meta, mdir, terms, entries, syn, trees = load_swemesh()
        nlm_path = RAW / "nlm_mesh" / "desc2026.xml"
        nlm = cached("nlm_mesh", [nlm_path], parse_nlm_mesh) if nlm_path.exists() else {}
        if not nlm:
            log("WARNING: no NLM descriptor XML; all MeSH entry terms are treated as synonyms (level A)")
        for dui, (en_name, sv_pref) in terms.items():
            tn = trees.get(dui, [])
            if not any(t.startswith(MESH_TREES) for t in tn):
                mesh_stats["skipped_tree"] += 1
                continue
            if not sv_pref or "som ämne" in sv_pref.lower() or "as topic" in en_name.lower():
                mesh_stats["skipped_topic"] += 1
                continue
            mesh_stats["descriptors"] += 1
            sv_a = [sv_case(x, sv_words) for x in sv_clean(sv_pref)] + [sv_case(u, sv_words) for u in uninvert(sv_pref)]
            sv_b = [sv_case(x, sv_words) for s in syn.get(dui, []) if "som ämne" not in s.lower() for x in sv_clean(s)]
            if "," in sv_pref:  # 'Rekombination, genetisk': the head alone is a (B) rendering
                sv_b.append(sv_case(sv_pref.split(",", 1)[0], sv_words))
            # the singular (lemma) of a plural heading comes first ('Lacerationer' -> 'laceration'), so a
            # substitution never produces 'en lacerationer'; the plural stays accepted through inflection
            sing = []
            for x in sv_a:
                sg = sv_singular(x, morph_noun_forms, sv_words, en_name, head_ok)
                if sg and sg.lower() not in {y.lower() for y in sv_a}:
                    sing.append(sg)
            if sing:
                mesh_stats["plural_heading_singularised"] += 1
                sv_a = list(dict.fromkeys(sing + sv_a))
            is_drug = any(t.startswith("D") for t in tn)
            sv_a_pre = list(sv_a)
            sv_b_pre = [s for s in dict.fromkeys(sv_b) if s.lower() not in {x.lower() for x in sv_a_pre}]
            # NPL takes priority over MeSH for drugs: a MeSH synonym that is the NPL substance name
            # becomes the first level-A form (Hydroxyurea: 'hydroxikarbamid' before 'hydroxiurea')
            if is_drug and sv_a[0].lower() not in npl_primary:
                npl_hit = [x for x in sv_a + sv_b if x.lower() in npl_primary]
                if npl_hit:
                    mesh_stats["npl_name_promoted"] += 1
                    sv_a = [npl_hit[0]] + [x for x in sv_a if x.lower() != npl_hit[0].lower()]
            sv_b = [s for s in dict.fromkeys(sv_b) if s.lower() not in {x.lower() for x in sv_a}]
            if dui in nlm:
                en_a = [en_name] + nlm[dui]["pref"]
                en_other = [e for e in nlm[dui]["other"] if e not in set(en_a)]
            else:
                en_a, en_other = [en_name] + entries.get(dui, []), []
            for e in dict.fromkeys(en_a):
                add_row("swemesh", dui, e, sv_a, "A", drug=is_drug)
                if sv_b:
                    add_row("swemesh", dui, e, sv_b, "B", drug=is_drug)
            for e in dict.fromkeys(en_other):
                add_row("swemesh", dui, e, sv_a + sv_b, "B", drug=is_drug)
                mesh_stats["narrower_or_trade_terms_B"] += 1
            # cognate-promoted Swedish synonyms: 'Metabolic Acidosis' <-> 'Metabolisk acidos'
            for e in dict.fromkeys(en_a + en_other):
                ev = en_variants(e)
                if not ev:
                    continue
                for s_ in sv_b_pre:  # (before the NPL promotion, so promoted names keep their A rows)
                    if len(s_.split()) == len(ev[0].split()) and similar(ev[0], s_) >= 0.8 and similar(ev[0], s_) > max(similar(ev[0], x) for x in sv_a_pre) + 0.1:
                        sg = sv_singular(s_, morph_noun_forms, sv_words, en_name, head_ok)
                        add_row("swemesh", dui, e, [sg, s_] if sg else [s_], "A", prepend=True, drug=is_drug)
                        mesh_stats["cognate_promoted_A"] += 1
            if dui in nlm:
                mesh_vocab_en = en_a + en_other
            else:
                mesh_vocab_en = en_a
            for e in mesh_vocab_en:
                for v in en_variants(e):
                    mesh_sv_by_key[en_key(v)].update(x.lower() for x in sv_a + sv_b)
        if nlm:
            nm = json.loads((RAW / "nlm_mesh" / "nlm_mesh.meta.json").read_text())
            manifest_sources["nlm_mesh"] = {
                **{k: nm[k] for k in ("url", "retrieved", "licence", "sha256", "bytes")},
                "path": str((RAW / "nlm_mesh" / nm["file"]).relative_to(RAW)),
                "records": {"descriptors": len(nlm), "terms": sum(len(v["pref"]) + len(v["other"]) for v in nlm.values())},
            }
        manifest_sources["swemesh"] = {
            "url": meta["url"],
            "info_page": meta["info_page"],
            "retrieved": meta["retrieved"],
            "licence": meta["licence"],
            "sha256": meta["sha256"],
            "bytes": meta["bytes"],
            "path": str((RAW / "swemesh" / meta["file"]).relative_to(RAW)),
            "records": {"descriptors_total": len(terms), "entry_terms": sum(len(v) for v in entries.values()), "sv_synonyms": sum(len(v) for v in syn.values())},
            "used": {**dict(mesh_stats), "trees": list(MESH_TREES)},
            "note": "English descriptor names come from the KIB export; the NLM descriptor XML supplies the concept structure "
            "(preferred-concept terms = synonyms -> A; other concepts and trade names -> B). No crawling was needed.",
        }
    else:
        log("WARNING: no Svensk MeSH download")

    # ---- Wikidata x NPL ------------------------------------------------------------------------
    wd_path = RAW / "wikidata_atc" / "wikidata_atc.json"
    wd_stats = collections.Counter()
    if wd_path.exists():
        data = json.loads(wd_path.read_text())["results"]["bindings"]
        g = lambda b, k: (b.get(k) or {}).get("value")  # noqa: E731
        by_code: dict[str, dict] = collections.defaultdict(dict)
        for b in data:
            code, item = g(b, "atc"), g(b, "item")
            if not code or not item:
                continue
            it = by_code[code].setdefault(item, {"label": g(b, "enLabel"), "sv": g(b, "svLabel"), "inn": set(), "aliases": set()})
            if g(b, "innEn"):
                it["inn"].add(g(b, "innEn"))
            for a in (g(b, "aliases") or "").split("|"):
                if a:
                    it["aliases"].add(a)
        for code, items in sorted(by_code.items()):
            if len(code) != 7:
                continue
            subst = {i: it for i, it in items.items() if it["label"] and it["label"][:1].islower() and not COMBO_RE.search(it["label"])}
            npl_p = npl_atc_primary.get(code)
            for item, it in subst.items():
                label = it["label"]
                names_a = [label] + sorted(it["inn"])
                names_b = []
                for a in sorted(it["aliases"]):
                    if not re.fullmatch(r"[a-z][a-z\- ]{3,40}", a) or COMBO_RE.search(a):
                        continue
                    ref = max(similar(a, label), similar(a, npl_p["name"]) if npl_p else 0)
                    # an alias is also accepted when MeSH independently maps it to the NPL name
                    # (US adopted names: acetaminophen -> paracetamol, albuterol -> salbutamol)
                    mesh_ok = bool(npl_p) and npl_p["name"].lower() in mesh_sv_by_key.get(en_key(a), ())
                    if mesh_ok and ref < 0.7:
                        wd_stats["alias_confirmed_by_mesh"] += 1
                    (names_a if ref >= 0.7 or mesh_ok else names_b).append(a)
                if npl_p:
                    sv = sv_case(npl_p["name"], sv_words)
                    ok_a = npl_p["single_substance"] and (len(subst) == 1 or similar(label, npl_p["name"]) >= 0.6)
                    for n in names_a:
                        add_row("npl_atc", code, n, [sv], "A" if ok_a else "B")
                    for n in names_b:
                        add_row("npl_atc", code, n, [sv], "B")
                    wd_stats["npl_join_A" if ok_a else "npl_join_B"] += 1
                if it["sv"]:
                    add_row("wikidata_sv", item.rsplit("/", 1)[-1], label, [sv_case(it["sv"], sv_words)], "B")
                    for n in sorted(it["inn"]):
                        add_row("wikidata_sv", item.rsplit("/", 1)[-1], n, [sv_case(it["sv"], sv_words)], "B")
                    wd_stats["sv_label"] += 1
        wd_meta = json.loads((RAW / "wikidata_atc" / "wikidata_atc.meta.json").read_text())
        manifest_sources["wikidata_atc"] = {
            **{k: wd_meta[k] for k in ("url", "retrieved", "licence", "sha256", "bytes", "records")},
            "path": str(wd_path.relative_to(RAW)),
            "items": len({g(b, "item") for b in data}),
            "atc_codes": len(by_code),
            "join": dict(wd_stats),
        }
    else:
        log("WARNING: no Wikidata download; npl_atc rows will be missing")

    # ---- KSH97-P ------------------------------------------------------------------------------------
    n_ksh = n_ksh_vague = 0
    for r in read_jsonl(PROC_SVDICT / "ksh97p.jsonl"):
        if r.get("resource") == "ksh97p" and r.get("en_term") and r.get("sv_term"):
            if KSH_VAGUE_RE.search(r["en_term"]) or KSH_VAGUE_RE.search(r["sv_term"]):
                n_ksh_vague += 1
                continue
            n_ksh += 1
            add_row("ksh97p", r.get("code") or r.get("source_id"), r["en_term"], [sv_case(r["sv_term"], sv_words)], "A")

    # ---- NPL Lexicondata pairs ------------------------------------------------------------------------
    lex_stats = {}
    for name in NPL_LEXICONS:
        recs = parse_npl_lexicon(name)
        n = 0
        for rec in recs:
            if rec["status"] != "CURRENT" or not rec["id"] or rec["id"] == "-":
                continue
            pairs = []
            for kind in ("term", "short"):
                en, sv = rec[kind].get("en"), rec[kind].get("sv")
                if en and sv:
                    pairs.append((en, sv))
            for en, sv in dict.fromkeys(pairs):
                if en.lower() in ("unspecified", "ospecificerad"):
                    continue
                single = not (len(en.split()) == 1 and en.lower() in freedict_headwords)
                add_row("npl_lexicon", f"{name}:{rec['id']}", en, [sv if sv.isupper() else sv_case(sv, sv_words)], "A", extra_single=single)
                n += 1
        lex_stats[name] = n
    log(f"NPL lexicon pairs: {lex_stats}")

    # ---- manual terms (whitelist.yaml: manual_terms) --------------------------------------------------
    n_manual = 0
    for en_t, sv_list in (whitelist.get("manual_terms") or {}).items():
        add_row("manual", f"manual:{en_key(str(en_t))}", str(en_t), [str(x) for x in sv_list], "A")
        n_manual += 1

    for en_t, sv_name in NPL_EXTRA_LINKS.items():
        if sv_name in npl_names:
            add_row("manual", f"npl:{sv_name}", en_t, [sv_name], "A", drug=True)
            n_manual += 1

    # ---- single_sense + entry ids ----------------------------------------------------------------------
    by_src_key = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in rows:
        if r["level"] == "A":
            by_src_key[(r["source"], r["en_key"] if not is_acronym(r["en_term"]) else r["en_term"])][r["concept_id"]].update(f.lower() for f in r["sv_forms"])
    polysemous = set()
    for key, concepts in by_src_key.items():
        if len(concepts) < 2:
            continue
        sets = list(concepts.values())
        disjoint = any(not (sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets)))
        if disjoint:
            polysemous.add(key)
    # rule 5 is decided per en_key over ALL sources' level-A forms
    a_forms_by_key = collections.defaultdict(set)
    a_sources_by_key = collections.defaultdict(set)
    for r in rows:
        if r["level"] == "A":
            a_forms_by_key[r["en_key"]].update(r["sv_forms"])
            a_sources_by_key[r["en_key"]].add(r["source"])
    general_words = set()
    for r in rows:
        k = r["en_key"]
        if (
            r["level"] == "A"
            and k not in general_words
            and " " not in r["en_term"].strip()
            and r["en_term"].lower() in freedict_headwords
            and "npl_atc" not in a_sources_by_key[k]
            and max(cognate_similarity(r["en_term"], f) for f in a_forms_by_key[k]) < 0.6
        ):
            general_words.add(k)
    reasons = collections.Counter()
    final, seen = [], {}
    for r in rows:
        acro = is_acronym(r["en_term"])
        single = True
        if r["en_key"] in ambiguous_keys:
            single = False
            reasons["ambiguous_list"] += 1
        elif (r["source"], r["en_term"] if acro else r["en_key"]) in polysemous:
            single = False
            reasons["same_source_polysemy"] += 1
        elif acro:
            single = False
            reasons["acronym"] += 1
        elif not r["_single"]:
            single = False
            reasons["npl_lexicon_general_word"] += 1
        elif r["level"] == "A" and r["en_key"] in general_words:
            single = False
            reasons["general_english_word_non_cognate"] += 1
        obligation_ok = (
            single
            and r["level"] == "A"
            and (r["source"] in ("npl_atc", "manual") or max(cognate_similarity(r["en_term"], f) for f in r["sv_forms"]) >= 0.6)
            # a single short word (urin, urea) cannot be found inside compounds (stem >= 5 rule)
            and any(" " in f or len(f) >= 5 for f in r["sv_forms"])
        )
        eid = f"{r['source']}:{r['concept_id']}:{r['level']}:{hashlib.sha1((r['en_term'] if acro else r['en_key']).encode()).hexdigest()[:8]}"
        if eid in seen:  # same key from several surface variants / promoted synonyms: merge forms
            old = seen[eid]
            merged = (r["sv_forms"] + old["sv_forms"]) if r["_prepend"] else (old["sv_forms"] + r["sv_forms"])
            low, out = set(), []
            for f in merged:
                if f.lower() not in low:
                    low.add(f.lower())
                    out.append(f)
            old["sv_forms"] = out
            old["single_sense"] = old["single_sense"] and single
            old["obligation_ok"] = old["single_sense"] and (old["obligation_ok"] or obligation_ok)
            old["drug"] = old["drug"] or r["drug"]
            continue
        seen[eid] = {
                "entry_id": eid,
                "en_term": r["en_term"],
                "en_key": r["en_key"],
                "sv_forms": r["sv_forms"],
                "level": r["level"],
                "single_sense": single,
                "obligation_ok": obligation_ok,
                "source": r["source"],
                "concept_id": str(r["concept_id"]),
                "drug": r["drug"],
            }
        final.append(seen[eid])
    log(f"term base: {len(final):,} rows; single_sense=False reasons {dict(reasons)}")

    # Swedish forms of the term base join the lexicon, except tokens that are English headwords
    # (some Swedish MeSH/Wikidata labels keep English words: 'Blood', 'The', 'Heart')
    tb_tokens: set[str] = set()
    for r in final:
        for f in r["sv_forms"]:
            add_tokens(f, tb_tokens)
    n_tb_before = len(sv_words)
    sv_words |= {t for t in tb_tokens if t not in en_single}
    counts_lex["termbase_sv_forms"] = len(sv_words) - n_tb_before
    en_words = en_single - sv_words

    # ---- write -----------------------------------------------------------------------------------------
    import pyarrow as pa
    import pyarrow.parquet as pq

    final.sort(key=lambda r: (r["source"], r["concept_id"], r["level"], r["en_key"]))
    tbl = pa.Table.from_pylist(
        final,
        schema=pa.schema(
            [
                ("entry_id", pa.string()),
                ("en_term", pa.string()),
                ("en_key", pa.string()),
                ("sv_forms", pa.list_(pa.string())),
                ("level", pa.string()),
                ("single_sense", pa.bool_()),
                ("obligation_ok", pa.bool_()),
                ("source", pa.string()),
                ("concept_id", pa.string()),
                ("drug", pa.bool_()),
            ]
        ),
    )
    tmp = BUILD / "termbase.parquet.tmp"
    pq.write_table(tbl, tmp, compression="zstd")
    os.replace(tmp, BUILD / "termbase.parquet")
    n_sv = write_lines_gz(BUILD / "sv_lexicon.txt.gz", sv_words)
    write_json_gz(BUILD / "sv_lemma_forms.json.gz", lemma_forms)
    n_cp = write_lines_gz(BUILD / "sv_compound_parts.txt.gz", compound_parts)
    write_json_gz(BUILD / "sv_morph.json.gz", morph)
    n_npl_sub = write_lines_gz(BUILD / "npl_substances.txt.gz", npl_substances)
    n_en = write_lines_gz(BUILD / "en_wordlist.txt.gz", en_words)
    shutil.copyfile(whitelist_path, BUILD / "whitelist.yaml.tmp")
    os.replace(BUILD / "whitelist.yaml.tmp", BUILD / "whitelist.yaml")

    # ---- manifest ---------------------------------------------------------------------------------------
    log("hashing raw inputs for MANIFEST")
    counts = collections.Counter((r["source"], r["level"]) for r in final)
    ss = collections.Counter((r["source"], r["level"]) for r in final if r["single_sense"])
    ob = collections.Counter((r["source"], r["level"]) for r in final if r["obligation_ok"])
    manifest_sources.update(
        {
            "npl": {
                "provider": "Läkemedelsverket (Swedish Medical Products Agency), Nationellt produktregister för läkemedel (NPL)",
                "url": "https://catalog.lakemedelsverket.se/store/1/resource/9",
                "path": str(RAW_SVDICT / "lakemedelsverket_npl.zip"),
                "retrieved": dt.datetime.fromtimestamp((RAW_SVDICT / "lakemedelsverket_npl.zip").stat().st_mtime).date().isoformat(),
                "licence": "open government data (Läkemedelsverket open data)",
                "sha256": sha256_file(RAW_SVDICT / "lakemedelsverket_npl.zip"),
                "records": {
                    "product_xml_files": npl["n_files"],
                    "products_with_atc_and_substance": npl["n_products_with_atc_and_substance"],
                    "atc_codes": len(npl["atc"]),
                    "atc7_single_substance_name": len(npl_atc_primary),
                    "atc7_level_A_eligible": sum(1 for v in npl_atc_primary.values() if v["single_substance"]),
                    "lexicon_pairs": lex_stats,
                },
            },
            "ksh97p": {
                "provider": "Socialstyrelsen, KSH97-P (Swedish primary-care ICD-10)",
                "path": str(RAW_SVDICT / "ksh97p_svensk_engelsk.xls"),
                "processed_path": str(PROC_SVDICT / "ksh97p.jsonl"),
                "retrieved": dt.datetime.fromtimestamp((RAW_SVDICT / "ksh97p_svensk_engelsk.xls").stat().st_mtime).date().isoformat(),
                "licence": "open (Socialstyrelsen)",
                "sha256": sha256_file(RAW_SVDICT / "ksh97p_svensk_engelsk.xls"),
                "records": {"en_sv_labels_used": n_ksh, "skipped_vague_labels": n_ksh_vague},
            },
            "saldo_saldom": {
                "provider": "Språkbanken Text, University of Gothenburg",
                "path": [str(RAW_SVDICT / "saldo.xml"), str(RAW_SVDICT / "saldom.xml")],
                "licence": "CC BY 4.0",
                "sha256": {"saldo.xml": sha256_file(RAW_SVDICT / "saldo.xml"), "saldom.xml": sha256_file(RAW_SVDICT / "saldom.xml")},
                "records": {"lemmas": len(lemma_forms), "forms": n_saldom_forms, "compound_parts": len(compound_parts)},
            },
            "hunspell_sv_se": {
                "path": str(RAW_SVDICT / "hunspell_sv_SE.dic"),
                "licence": "LGPL/GPL (hunspell sv_SE dictionary)",
                "sha256": sha256_file(RAW_SVDICT / "hunspell_sv_SE.dic"),
                "records": {"new_forms": counts_lex["hunspell_sv_se"]},
            },
            "socialstyrelsen_termbank": {
                "path": str(RAW_SVDICT / "socialstyrelsen_termbank.xml"),
                "licence": "open (Socialstyrelsen)",
                "sha256": sha256_file(RAW_SVDICT / "socialstyrelsen_termbank.xml"),
                "records": {"new_forms": counts_lex["socialstyrelsen_termbank"]},
            },
            "kva": {
                "path": str(RAW_SVDICT / "kva_procedures.xlsx"),
                "licence": "open (Socialstyrelsen)",
                "sha256": sha256_file(RAW_SVDICT / "kva_procedures.xlsx"),
                "records": {"new_forms": counts_lex["kva_procedures"]},
            },
            "manual_terms": {"path": "resources/whitelist.yaml (manual_terms)", "licence": "project-internal", "records": {"terms": n_manual}},
            "freedict_en_sv": {
                "provider": "FreeDict / WikDict (Wiktionary), 2025-11-23",
                "path": str(FREEDICT),
                "licence": "CC BY-SA",
                "sha256": sha256_file(FREEDICT),
                "records": {"headwords": len(fd_words), "single_words": len(en_single), "en_wordlist_after_removing_swedish": n_en},
            },
        }
    )
    manifest = {
        "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "input_fingerprint": fingerprint,
        "builder": "resources/build_termbase.py",
        "outputs": {
            "termbase.parquet": {
                "rows": len(final),
                "by_source_level": {f"{s}:{l}": n for (s, l), n in sorted(counts.items())},
                "single_sense_by_source_level": {f"{s}:{l}": n for (s, l), n in sorted(ss.items())},
                "obligation_ok_by_source_level": {f"{s}:{l}": n for (s, l), n in sorted(ob.items())},
                "distinct_en_keys": len({r["en_key"] for r in final}),
                "drug_rows": sum(1 for r in final if r["drug"]),
                "single_sense_false_reasons": dict(reasons),
            },
            "sv_lexicon.txt.gz": {"forms": n_sv},
            "sv_lemma_forms.json.gz": {"lemmas": len(lemma_forms)},
            "sv_compound_parts.txt.gz": {"parts": n_cp},
            "sv_morph.json.gz": {"nouns": len(morph["noun"]), "adjectives": len(morph["adj"]), "forms_with_pos": len(morph["pos"])},
            "npl_substances.txt.gz": {"names": n_npl_sub},
            "en_wordlist.txt.gz": {"words": n_en},
            "whitelist.yaml": {"tokens": len(Lexicon(sv_words=(), en_words=(), whitelist=whitelist).whitelist), "ambiguous_en": len(ambiguous)},
        },
        "sources": manifest_sources,
        "timings_s": {k: round(v, 1) for k, v in timings.items()},
    }
    for o in outputs[:-1]:
        manifest["outputs"].setdefault(o, {})["sha256"] = sha256_file(BUILD / o)
    manifest["timings_s"]["total_build"] = round(time.time() - t_start, 1)
    atomic_write_bytes(man_path, json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))
    log(f"wrote {BUILD} in {time.time() - t_start:.0f}s")
    if do_coverage:
        coverage()


# ---------------------------------------------------------------------------
# coverage report
# ---------------------------------------------------------------------------

EN_TAIL_RE = re.compile(r"\s*Answer Choices:.*$", re.S)
SV_TAIL_RE = re.compile(r"\s*(Svarsalternativ|Answer Choices):.*$", re.S)


def coverage(sample_n: int = 30) -> None:
    import random

    t0 = time.time()
    lex = Lexicon.load(BUILD)
    tb = TermBase.load(BUILD, lexicon=lex)
    t_load = time.time() - t0
    log(f"loaded term base in {t_load:.1f}s ({len(tb.entries):,} entries)")

    hits_by = collections.Counter()
    hits_single_a = collections.Counter()
    items_with_hit = items_with_a = 0
    n_items = n_fields = 0
    verified_by = collections.Counter()
    obl_by_source = collections.Counter()
    obl_terms = collections.Counter()
    hit_terms = collections.Counter()
    obligations_all = []
    n_verified = n_obl = 0
    t1 = time.time()
    for row in read_jsonl(INPUT_JSONL):
        n_items += 1
        en = {"question": EN_TAIL_RE.sub("", row["question"] or "")}
        sv = {"question": SV_TAIL_RE.sub("", row.get("sv_question") or "")}
        for k, v in (row.get("options") or {}).items():
            en[f"option:{k}"] = v or ""
            sv[f"option:{k}"] = (row.get("sv_options") or {}).get(k) or ""
        n_fields += len(en)
        any_hit = any_a = False
        for f, t in en.items():
            for h in tb.spot(t, f):
                any_hit = True
                hits_by[(h.source, h.level)] += 1
                hit_terms[h.en_span.lower()] += 1
                if h.level == "A" and h.single_sense:
                    any_a = True
                    hits_single_a[h.source] += 1
        items_with_hit += any_hit
        items_with_a += any_a
        ver, obl = tb.ground(en, sv)
        n_verified += len(ver)
        n_obl += len(obl)
        for v in ver:
            verified_by[(tb.entries[v.entry_id]["source"], v.level)] += 1
        for o in obl:
            obl_by_source[tb.entries[o.entry_id]["source"]] += 1
            obl_terms[o.en_span.lower()] += 1
            obligations_all.append(
                {
                    "prompt_id": row["prompt_id"],
                    "field": o.field,
                    "en_span": o.en_span,
                    "sv_forms": o.sv_forms,
                    "entry_id": o.entry_id,
                    "en_text": en[o.field][:300],
                    "sv_text": sv[o.field][:300],
                }
            )
    t_run = time.time() - t1
    rng = random.Random(0)
    sample = rng.sample(obligations_all, min(sample_n, len(obligations_all)))

    def probe(en_t: str, sv_t: str) -> dict:
        hits = tb.spot(en_t, "question")
        return {
            "en": en_t,
            "sv": sv_t,
            "hits": [h.model_dump() for h in hits],
            "is_accepted_rendering": tb.is_accepted_rendering(en_t, sv_t),
            "ground": [x.model_dump() for x in tb.ground({"question": en_t}, {"question": sv_t})[0]],
        }

    # how often do key examples occur in the test set
    def count_pairs(en_word: str, sv_word: str) -> dict:
        n_en = n_sv_ok = n_sv_en = 0
        for row in read_jsonl(INPUT_JSONL):
            en_all = " ".join([row["question"] or ""] + list((row.get("options") or {}).values()))
            sv_all = " ".join([row.get("sv_question") or ""] + list((row.get("sv_options") or {}).values()))
            if re.search(rf"\b{en_word}", en_all, re.I):
                n_en += 1
                if re.search(rf"\b{sv_word}", sv_all, re.I):
                    n_sv_ok += 1
                if re.search(rf"\b{en_word}", sv_all, re.I):
                    n_sv_en += 1
        return {"items_with_en": n_en, "sv_has_swedish_form": n_sv_ok, "sv_has_english_form": n_sv_en}

    examples = {
        "IV vancomycin/intravenöst vankomycin (extra uncovered word -> no veto by design)": probe(
            "IV vancomycin", "intravenöst vankomycin"
        ),
        "vancomycin/vankomycin": {
            **probe("vancomycin", "vankomycin"),
            "test_set": count_pairs("vancomycin", "vankomycin"),
        },
        "acyclovir/aciklovir": {**probe("acyclovir", "aciklovir"), "test_set": count_pairs("acyclovir", "aciklovir")},
        "aciclovir/aciklovir": probe("aciclovir", "aciklovir"),
        "vincristine/vinkristin": probe("vincristine", "vinkristin"),
        "alteplase/alteplas": probe("alteplase", "alteplas"),
        "metabolic acidosis/metabol acidos": {
            **probe("metabolic acidosis", "metabol acidos"),
            "test_set": count_pairs("metabolic acidosis", "metabol acidos"),
        },
        "metabolic acidosis/metabolisk acidos": probe("metabolic acidosis", "metabolisk acidos"),
        "type 2 diabetes mellitus/typ 2-diabetes": probe("type 2 diabetes mellitus", "typ 2-diabetes"),
        "vancomycin-resistant/vankomycinresistenta": probe("vancomycin-resistant enterococci", "vankomycinresistenta enterokocker"),
        "resistance/resistens": probe("antibiotic resistance", "antibiotikaresistens"),
    }
    report = {
        "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "input": str(INPUT_JSONL),
        "note": "English fields of the GPT-5.4 test drafts (MedXpertQA 'Answer Choices:' tails removed); "
        "TermBase.spot per field, TermBase.ground against the aligned Swedish draft field.",
        "items": n_items,
        "fields": n_fields,
        "load_seconds": round(t_load, 2),
        "run_seconds": round(t_run, 1),
        "items_with_any_hit": items_with_hit,
        "share_items_with_any_hit": round(items_with_hit / max(n_items, 1), 4),
        "items_with_single_sense_A_hit": items_with_a,
        "share_items_with_single_sense_A_hit": round(items_with_a / max(n_items, 1), 4),
        "hits_by_source_level": {f"{s}:{l}": n for (s, l), n in hits_by.most_common()},
        "single_sense_A_hits_by_source": dict(hits_single_a.most_common()),
        "verified_pairs": n_verified,
        "verified_by_source_level": {f"{s}:{l}": n for (s, l), n in verified_by.most_common()},
        "obligations": n_obl,
        "obligations_by_source": dict(obl_by_source.most_common()),
        "verified_share_of_single_sense_A_hits": round(n_verified / max(n_verified + n_obl, 1), 4),
        "top_hit_terms": hit_terms.most_common(60),
        "top_obligation_terms": obl_terms.most_common(60),
        "obligation_sample": sample,
        "examples": examples,
    }
    atomic_write_bytes(BUILD / "coverage_report.json", json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"))
    log(f"coverage: {items_with_hit}/{n_items} items with a hit; {n_verified} verified, {n_obl} obligations ({t_run:.0f}s)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="rebuild even if the input fingerprint matches")
    p.add_argument("--no-coverage", action="store_true")
    p.add_argument("--coverage-only", action="store_true")
    a = p.parse_args()
    if a.coverage_only:
        coverage()
    else:
        build(force=a.force, do_coverage=not a.no_coverage)


if __name__ == "__main__":
    main()
