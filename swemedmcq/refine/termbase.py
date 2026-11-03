"""Term base and lexicons.

Build artefacts (made by ``resources/build_termbase.py`` into ``resources/build/``):

* ``termbase.parquet`` – one row per (English term, concept, level):
  ``entry_id, en_term, en_key, sv_forms (list[str]), level ("A"|"B"), single_sense (bool), source, concept_id``
  plus ``obligation_ok`` (bool, extra): whether a missing Swedish form may become an obligation
  (single-sense A row from NPL, or with a Swedish form that is a cognate of the English term)
* ``sv_lexicon.txt.gz`` – lowercased Swedish word forms (SALDO/SALDOM, hunspell, termbank, KVÅ, term base)
* ``sv_lemma_forms.json.gz`` – SALDOM lemma (lowercased) -> inflected word forms
* ``sv_compound_parts.txt.gz`` – SALDOM compound-part forms (msd c/ci/cm, e.g. ``hjärt``) [extra]
* ``en_wordlist.txt.gz`` – English single words (FreeDict headwords minus Swedish forms)
* ``whitelist.yaml`` – copy of ``resources/whitelist.yaml``

English matching
    Tokens are maximal runs of letters/digits (hyphens, commas, brackets are separators); each
    token is lowercased, a possessive ``'s`` is removed and a simple plural rule is applied
    (``ies->y``, ``(s|x|ch|sh|z)es -> \\1``, ``s`` dropped unless ``ss/us/is``).  The same
    normalisation builds ``en_key`` at build time, so matching is symmetric.  Short all-caps
    acronyms (``AIDS``, ``NIDDM``) are matched case-sensitively and only against acronym
    entries.  Matching is longest-n-gram first, non-overlapping.

Merging per span
    Several entries (sources) can share an ``en_key``.  ``spot`` returns ONE ``TermHit`` per
    matched span: level = best level present (A over B); ``sv_forms`` = union of the forms of all
    entries of that level; ``single_sense`` = all of them single-sense; ``entry_id`` = the
    entry of the highest-priority source; ``source`` = the contributing sources joined by ``+``.
    Level-B forms of the same span are still used by ``ground`` to recognise a verified rendering.

Swedish matching (``match_sv``)
    Every accepted form is expanded with ``Lexicon.forms`` (the last word is inflected via SALDOM;
    preceding adjectives get their agreement forms), then matched case-insensitively at word
    boundaries (whitespace and hyphen are flexible).  Compound rule: a token that starts OR ends
    with an accepted single-word form (>= 5 characters) matches when the remainder is a Swedish
    word (optionally with a linking ``s``).

Compound analysis, protected terms and drug names
    * ``Lexicon.is_swedish`` analyses compounds with linking ``s/e/a/o``, Latin/Greek derivations of
      attested stems (``-al/-ära/-iell/-ös/-isk/-erad/-erande``, ``-pati``, ``-oid`` ...) and Swedish
      inflection; English-looking tokens are accepted only by the lexicon.  ``Lexicon.keep`` is the
      protected-term list (``whitelist.yaml`` ``keep:``, source file first).  ``sv_morph.json.gz``
      (SALDOM genders and inflection tables) backs ``noun_gender`` / ``analyses``.
    * ``TermBase.drug_inn`` (transliteration-only INNs, NPL first) and ``TermBase.substitution``
      (English targets only, inflection carried over, never inside gene/protein names).
"""

from __future__ import annotations

import bisect
import difflib
import gzip
import itertools
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from swemedmcq.refine.types import FieldKey, Obligation, TermHit, VerifiedPair

# ---------------------------------------------------------------------------
# English normalisation (shared with resources/build_termbase.py)
# ---------------------------------------------------------------------------

EN_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
_POSSESSIVE_RE = re.compile(r"['’]s$")
MAX_NGRAM = 10

SOURCE_PRIORITY = {"manual": 0, "npl_atc": 1, "swemesh": 2, "ksh97p": 3, "npl_lexicon": 4, "wikidata_sv": 5}


def singularise(t: str) -> str:
    if len(t) <= 3 or not t.isalpha():
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith(("sses", "xes", "ches", "shes", "zes")):
        return t[:-2]
    if t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def norm_en_token(tok: str) -> str:
    t = _POSSESSIVE_RE.sub("", tok.lower()).replace("’", "'")
    return singularise(t)


def en_tokens(text: str) -> list[tuple[str, int, int, str]]:
    """(normalised token, start, end, raw token) for every word token of ``text``."""
    return [(norm_en_token(m.group()), m.start(), m.end(), m.group()) for m in EN_TOKEN_RE.finditer(text)]


def en_key(text: str) -> str:
    return " ".join(t[0] for t in en_tokens(text))


def is_acronym(term: str) -> bool:
    """Single all-caps token of 2–6 characters (e.g. AIDS, NIDDM, tPA is NOT one)."""
    t = term.strip()
    return (
        2 <= len(t) <= 6
        and EN_TOKEN_RE.fullmatch(t) is not None
        and any(c.isalpha() for c in t)
        and t == t.upper()
    )


def _cognate_norm(s: str) -> str:
    s = s.lower()
    for a, b in (("ph", "f"), ("th", "t"), ("c", "k"), ("y", "i"), ("z", "s"), ("qu", "kv"), ("x", "ks")):
        s = s.replace(a, b)
    s = re.sub(r"(ine|ide|ate|one|ole|ase)$", lambda m: m.group()[:-1], s)
    return re.sub(r"[^a-zåäöéü]", "", s)


_SV_SALT = re.compile(
    r"(di)?(hydroklorid|hydrobromid|klorid|bromid|propionat|furoat|maleat|tartrat|succinat|mesilat|besilat|"
    r"acetat|sulfat|fosfat|citrat|fumarat|valerat|natrium|kalium|kalcium|magnesium|monohydrat|dihydrat|"
    r"trihydrat|hemihydrat|natriumsalt|dinatrium|dikalium)$"
)
_EN_SALT = re.compile(
    r"\b(hydrochloride|hydrobromide|chloride|bromide|propionate|furoate|maleate|tartrate|succinate|mesylate|"
    r"besylate|acetate|sulfate|sulphate|phosphate|citrate|fumarate|valerate|sodium|potassium|calcium|magnesium|"
    r"hydrate|salt)\b", re.IGNORECASE)


_NEGATED_PREFIX = re.compile(r"\bnon[\s\-\u2010\u2011]?$", re.IGNORECASE)


def with_salt_base(en_span: str, forms: list[str]) -> list[str]:
    """Put the salt-free base name first when the English term names no salt.

    NPL substance names often carry the salt (``flutikasonpropionat``) while an exam item names
    the active moiety (*fluticasone*), whose correct Swedish rendering is ``flutikason``.
    """
    if not forms or _EN_SALT.search(en_span):
        return forms
    bases = []
    for f in forms:
        if " " in f.strip():
            continue
        b = _SV_SALT.sub("", f.strip())
        if b != f.strip() and len(b) >= 5:
            bases.append(b)
    return list(dict.fromkeys(bases + list(forms)))


def cognate_similarity(a: str, b: str) -> float:
    """Orthographic similarity of an English and a Swedish string after mapping the usual
    spelling correspondences (c->k, ph->f, th->t, y->i, ...): vancomycin/vankomycin = 1.0."""
    return difflib.SequenceMatcher(None, _cognate_norm(a), _cognate_norm(b)).ratio()


# ---------------------------------------------------------------------------
# Swedish helpers
# ---------------------------------------------------------------------------

SV_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
SV_HTOKEN_RE = re.compile(r"\w+(?:-\w+)*", re.UNICODE)
SV_FUNCTION_WORDS = frozenset(
    "och i av med en ett den det de till på för vid som om eller hos från mot under över utan".split()
)
_HEUR_SUFFIXES = ("", "en", "et", "n", "t", "er", "erna", "ar", "arna", "s", "ens", "ets", "ns", "ers", "ernas", "e", "a")
_MAX_PHRASE_VARIANTS = 600
_ADJ_SUFFIXES = ("isk", "al", "ell", "iv", "är", "ös", "ent", "ant", "ig", "ar")


def _deaccent(s: str) -> str:
    return s.replace("é", "e").replace("É", "E")


def _read_lines_gz(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return f.read().split("\n")


def _load_whitelist(build_dir: Path) -> dict:
    import yaml  # local import: only needed on load

    for p in (build_dir / "whitelist.yaml", build_dir.parent / "whitelist.yaml"):
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {}


def _load_keep(build_dir: Path) -> list[str] | None:
    """The `keep:` list of the SOURCE ``resources/whitelist.yaml`` (edited by hand between builds),
    falling back to the build copy.  None when neither file has the key."""
    import yaml

    for p in (build_dir.parent / "whitelist.yaml", build_dir / "whitelist.yaml"):
        if p.exists():
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if data.get("keep") is not None:
                return [str(x) for x in data.get("keep") or []]
    return None


# --- Swedish word analysis (Lexicon.is_swedish) -----------------------------------------------
# inflection / agreement endings (longest first)
_INFL = ("ernas", "arnas", "ornas", "erna", "arna", "orna", "ens", "ets", "ers", "ars", "ors",
         "er", "ar", "or", "en", "et", "na", "ns", "ts", "n", "t", "s", "e", "a")
# derivational endings of Latin/Greek medical vocabulary: adjectives, -era verbs and participles, nouns
_DERIV = ("erande", "erade", "erad", "erat", "erar", "ering", "era", "ell", "isk", "al", "ar", "är", "ös", "iv",
          "oid", "il", "ent", "ant", "ion", "tion", "itet", "ism", "ist", "it", "os", "om", "emi", "i",
          # Greek nouns of disease and procedure (koagulo|pati, odyno|fagi, nyktal|opi)
          "pati", "logi", "grafi", "skopi", "tomi", "ektomi", "stomi", "plastik", "uri", "algi", "fagi", "plegi",
          "trofi", "plasi", "megali", "penia", "peni", "cyt", "cyter", "blast", "opi")
# English-only endings: an unknown token with one of them is accepted only by the lexicon
_EN_ONLY_END = re.compile(r"(?:ated|ating|ization|izing|ized|ises|ised|izes|ness|ful|less|ous|ings|ed|ward|wards)$")
_DERIV_ADJ_AGREE = ("", "a", "t", "e")
# a stem is attested when stem + one of these is a lexicon word (atres -> atresi, arteri -> arteria)
_STEM_END = ("", "a", "e", "i", "o", "u", "um", "us", "ium", "ia", "ion", "is", "it", "os", "om", "er", "ar", "en",
             "isk", "al", "är", "ell", "ös", "iv", "ism", "in", "on", "ik", "ar", "s", "ell", "x")
_LINKS = ("s", "e", "a", "o")
# bound Latin/Greek prefixes of medical vocabulary (heads of 'periumbilikalt', 'retromammär')
_PREFIXES = frozenset(
    "peri para retro intra inter infra supra sub trans hyper hypo epi endo ekto exo meta pre post pro anti "
    "neo poly multi mikro makro pseudo semi uni tri extra kontra ante pan dys hemi iso homo hetero auto "
    "dia per ad ab kon kom de dis in im ir il non olig oligo pan pluri prae re sym syn tele ultra".split()
)
# spellings that do not occur in Swedish words outside the lexicon (English leftovers: 'dactylit',
# 'clinical', 'ethanol', 'tendinous'): such a token is never accepted by analysis, only by lexicon
_NON_SV_ORTHO = re.compile(r"c(?![eiyk])|ph|th|qu|w|sh|ou(?![mn])|ae|oe")
_VOWELS = "aeiouyåäöé"
_INDEF_NOUN_MSD = {"sg indef nom", "sg indef gen", "pl indef nom", "pl indef gen"}
_LATIN_NOUN_END = ("a", "e", "um", "us", "ia", "ium", "on", "is", "os", "ion")
# English inflections: a token whose English base is an English word is English ('relieving',
# 'deliveries', 'patterns'), whatever a compound analysis says
_EN_INFL = (("ies", "y"), ("ing", "e"), ("ing", ""), ("ed", "e"), ("ed", ""), ("es", ""), ("ers", ""), ("er", ""),
            ("est", ""), ("ly", ""), ("s", ""), ("d", ""))


class Lexicon:
    """Swedish validity, English word list, SALDOM inflection and the hand-written whitelist."""

    def __init__(
        self,
        sv_words: Iterable[str],
        en_words: Iterable[str],
        lemma_forms: dict[str, list[str]] | None = None,
        compound_parts: Iterable[str] = (),
        whitelist: dict | None = None,
        lemma_forms_path: Path | None = None,
        morph: dict | None = None,
        morph_path: Path | None = None,
        keep: Iterable[str] | None = None,
    ):
        self.sv_words: set[str] = set(sv_words)
        self.sv_words.discard("")
        self.en_words: set[str] = set(en_words)
        self.en_words.discard("")
        self.compound_parts: set[str] = set(compound_parts)
        self.compound_parts.discard("")
        self._lemma_forms = lemma_forms
        self._form_lemmas: dict[str, tuple[str, ...]] | None = None
        self._lemma_forms_path = lemma_forms_path
        self._morph = morph
        self._morph_path = morph_path
        self._form_analyses: dict[str, list[tuple[str, str, str, str]]] | None = None
        wl = whitelist or {}
        self.whitelist_data = wl
        toks: set[str] = set()
        for key in ("abbreviations", "latin_anatomy", "international_terms", "extra_tokens"):
            for t in wl.get(key, []) or []:
                t = str(t)
                toks.add(t)
                toks.add(t.lower())
        self.whitelist: set[str] = toks
        self.whitelist_patterns = [re.compile(p) for p in (wl.get("eponym_patterns") or [])]
        self.ambiguous_en: set[str] = {str(x).lower() for x in (wl.get("ambiguous_en") or [])}
        if keep is None:
            keep = wl.get("keep") or []
        # protected terms (whitelist.yaml `keep:`), lowercased, whitespace-normalised
        self.keep: set[str] = {" ".join(str(t).lower().split()) for t in keep if str(t).strip()}
        self._sorted_words: list[str] | None = None
        self._sv_cache: dict[str, bool] = {}

    # -- loading -----------------------------------------------------------
    @classmethod
    def load(cls, build_dir) -> "Lexicon":
        build_dir = Path(build_dir)
        cp = build_dir / "sv_compound_parts.txt.gz"
        return cls(
            sv_words=_read_lines_gz(build_dir / "sv_lexicon.txt.gz"),
            en_words=_read_lines_gz(build_dir / "en_wordlist.txt.gz"),
            compound_parts=_read_lines_gz(cp) if cp.exists() else (),
            whitelist=_load_whitelist(build_dir),
            lemma_forms_path=build_dir / "sv_lemma_forms.json.gz",
            morph_path=build_dir / "sv_morph.json.gz",
            keep=_load_keep(build_dir),
        )

    @property
    def lemma_forms(self) -> dict[str, list[str]]:
        """Lazily loaded SALDOM lemma -> forms map (about 1 s)."""
        if self._lemma_forms is None:
            if self._lemma_forms_path is not None and self._lemma_forms_path.exists():
                with gzip.open(self._lemma_forms_path, "rt", encoding="utf-8") as f:
                    self._lemma_forms = json.load(f)
            else:
                self._lemma_forms = {}
        return self._lemma_forms

    # -- SALDOM morphology (genders, inflection tables) --------------------
    @property
    def morph(self) -> dict:
        """Lazily loaded ``sv_morph.json.gz``: ``noun`` lemma -> [gender u|n|v|x, {msd: forms}],
        ``adj`` lemma -> {msd: forms}, ``pos`` form -> comma-joined SALDOM parts of speech."""
        if self._morph is None:
            if self._morph_path is not None and self._morph_path.exists():
                with gzip.open(self._morph_path, "rt", encoding="utf-8") as f:
                    self._morph = json.load(f)
            else:
                self._morph = {}
            self._morph.setdefault("noun", {})
            self._morph.setdefault("adj", {})
            self._morph.setdefault("pos", {})
        return self._morph

    def analyses(self, form: str) -> list[tuple[str, str, str, str]]:
        """SALDOM analyses of a (lowercased) noun or adjective form: ``(lemma, pos, gender, msd)``
        with pos ``nn``/``av`` (gender '' for adjectives)."""
        if self._form_analyses is None:
            idx: dict[str, list[tuple[str, str, str, str]]] = {}
            for lem, (g, table) in self.morph["noun"].items():
                for msd, fs in table.items():
                    for f in fs:
                        idx.setdefault(f, []).append((lem, "nn", g, msd))
            for lem, table in self.morph["adj"].items():
                for msd, fs in table.items():
                    for f in fs:
                        idx.setdefault(f, []).append((lem, "av", "", msd))
            self._form_analyses = idx
        return self._form_analyses.get(form.lower(), [])

    def pos(self, form: str) -> set[str]:
        """SALDOM parts of speech of a word form (empty if unknown)."""
        p = self.morph["pos"].get(form.lower())
        return set(p.split(",")) if p else set()

    def noun_gender(self, word: str) -> str | None:
        """Gender (``u``/``n``) of a singular indefinite noun form when SALDOM (or, for a compound,
        its last part) gives exactly one; None otherwise."""
        w = word.lower()
        if self.is_english(w) and not self.analyses(w):
            return None
        for i in range(max(1, len(w) - 3)):
            part = w[i:]
            an = self.analyses(part)
            if not an and part not in self.morph["pos"]:
                continue
            # compound: the head is a lexicon base form of >= 4 letters, the noun part >= 4 letters
            # ('ledstång' -> stång; but not 'bio|psi', 'dis|tal', 'kom|minut', 'vestibul|ärt')
            head = w[:i]
            if i and (i < 4 or len(part) < 4 or not (self._base_form(head) or (head.endswith("s") and self._base_form(head[:-1])))):
                continue
            nouns = [a for a in an if a[1] == "nn"]
            # an adjective reading that can follow 'en/ett' ('lila', 'yttre'); a definite one cannot ('värde')
            adj_indef = [a for a in an if a[1] == "av" and not a[3].startswith("pos def")]
            other = self.pos(part) - {"nn", "av", "pm"}
            if not nouns or adj_indef or other or part.endswith(("ande", "ende")):
                return None  # 'stor' (also an adjective), 'en' ... : not certainly a noun
            msds = {a[3] for a in nouns}
            if "sg indef nom" not in msds or not msds <= _INDEF_NOUN_MSD:
                return None  # 'kvoten' is not an indefinite singular; 'hus' (sg = pl) is fine
            genders = {a[2] for a in nouns}
            return genders.pop() if len(genders) == 1 and genders <= {"u", "n"} else None
        return None

    # -- validity ----------------------------------------------------------
    def is_whitelisted(self, token: str) -> bool:
        return token in self.whitelist or token.lower() in self.whitelist or any(
            p.fullmatch(token) for p in self.whitelist_patterns
        )

    def _is_part(self, s: str) -> bool:
        return len(s) >= 3 and (s in self.sv_words or s in self.compound_parts)

    def _word(self, s: str) -> bool:
        return s in self.sv_words or s in self.whitelist

    @property
    def sorted_words(self) -> list[str]:
        if self._sorted_words is None:
            self._sorted_words = sorted(self.sv_words | {w for w in self.whitelist if w.islower()})
        return self._sorted_words

    def _prefix_attested(self, stem: str, limit: int = 400, words_only: bool = False) -> bool:
        """Some lexicon word continues ``stem`` with a known ending or a word (``miktion`` ->
        ``miktionsbesvär``, ``sideroblast`` -> ``sideroblastisk``)."""
        words = self.sorted_words
        i = bisect.bisect_left(words, stem)
        for w in words[i: i + limit]:
            if not w.startswith(stem):
                break
            rest = w[len(stem):]
            if (rest in _STEM_END and not words_only) or (len(rest) >= 4 and (self._word(rest) or (rest[0] in _LINKS and self._word(rest[1:])))):
                return True
        return False

    def _stem_ok(self, stem: str) -> bool:
        """``stem`` is the stem of attested vocabulary (``mamm`` -> mamma, ``atres`` -> atresi)."""
        if len(stem) < 3 or not any(c in _VOWELS for c in stem):
            return False
        if self._base_form(stem) or any(self._word(stem + e) for e in _STEM_END if e):
            return True
        if stem.endswith("ot") and self._word(stem[:-1] + "s"):
            return True  # Greek -os/-ot- alternation: cyanot|isk <- cyanos, fibrot|isk <- fibros
        if stem.endswith("o") and len(stem) >= 5 and self._stem_ok(stem[:-1]):
            return True  # combining vowel before a Greek suffix: koagulo|pati
        return len(stem) >= 5 and self._prefix_attested(stem)

    def _derived(self, t: str) -> bool:
        """Latin/Greek derivation of an attested stem, optionally inflected
        (``mesangiell``, ``impakterad``, ``hyperekoisk``, ``mikrosomala``, ``sideroblaster``)."""
        for infl in ("",) + _INFL:
            if infl and not t.endswith(infl):
                continue
            r = t[: len(t) - len(infl)] if infl else t
            if len(r) < 4:
                continue
            if infl and infl in ("n", "t", "ns", "ts") and r[-1] not in _VOWELS and not r.endswith(("al", "ell", "är", "ös", "iv", "isk", "il")):
                continue
            if infl and infl in ("a", "e", "er", "ar", "or", "es") and r[-1] in _VOWELS:
                continue  # 'hematuri|a', 'neutropeni|a' are Latin/English forms, not Swedish inflection
            if infl and self._word(r) and not self._saldom(r) and len(r) >= 5:
                return True  # inflection of a (non-SALDOM) lexicon word missing from the lexicon (farmakoterapier)
            for d in _DERIV:
                if r.endswith(d) and len(r) - len(d) >= 4 and self._stem_ok(r[: -len(d)]):
                    return True
            if not infl and len(r) >= 6 and self._prefix_attested(r, words_only=True):
                return True  # attested as the head of lexicon compounds (miktion -> miktionsbesvär)
            if r.endswith("oid") and len(r) >= 7:
                return True  # Greek -oid adjective/noun (skafoid, sesamoid, haploid)
            if infl and len(r) >= 5 and (any(self._word(r + e) for e in _LATIN_NOUN_END)
                                         or (len(r) >= 7 and self._prefix_attested(r))):
                return True  # inflected Latin noun stem (arteri|erna, sideroblast|er)
        return False

    def _saldom(self, w: str) -> bool:
        """``w`` is a SALDOM word form (SALDOM lists all inflected forms of its lemmas)."""
        return w in self.morph["pos"] or w in self.form_lemmas

    def _base_form(self, h: str) -> bool:
        """A compound head in base form: a SALDOM compound part or lemma, or a non-SALDOM lexicon
        word (term-bank / MeSH vocabulary).  Inflected SALDOM forms (``ariar``) are not heads."""
        if h in self.compound_parts or h in self.lemma_forms or h in self.whitelist:
            return True
        return h in self.sv_words and not self._saldom(h)

    def _head_ok(self, h: str) -> bool:
        """A compound head: a lexicon base form or SALDOM compound part, an attested stem (``papillar``,
        ``skafoid``), or such a head plus a linking ``s/e/a/o/u`` (``kosto``, ``ilio``)."""
        if len(h) < 3:
            return False
        if self._base_form(h) or h in _PREFIXES:
            return True
        if len(h) >= 6 and any(h.endswith(d) and self._stem_ok(h[: -len(d)]) for d in ("al", "ar", "är", "isk", "ell")):
            return True
        if len(h) >= 7 and h.endswith("oid"):
            return True
        if len(h) >= 4 and h[-1] == "o" and self._prefix_attested(h):
            return True  # attested Greek/Latin combining form (ilio-psoas, kosto-sternal)
        if h[-1] in "oi" and len(h) >= 5 and self._prefix_attested(h):
            return True  # attested combining form ending in -i (hydroxi-apatit)
        if h[-1] in _LINKS and len(h) >= 4 and h[-2] != h[-1]:  # no double linking letter ('löks|s')
            x = h[:-1]
            if self._base_form(x) or (h[-1] in "oai" and len(x) >= 3 and self._stem_ok(x)):
                return True
            # a compound first element with linking s ('underbuk|s|smärta')
            return h[-1] == "s" and len(x) >= 7 and x[-1] != "s" and self._is_compound(x)
        return False

    def _bad_link(self, head: str, tail: str) -> bool:
        """A linking letter the lexicon contradicts: the compound without it is attested
        (``löks|skal`` -> lökskal, ``immuno|fluorescens`` -> immunfluorescens…)."""
        if head[-1] not in _LINKS or len(head) < 4 or head in self.lemma_forms:
            return False  # 'skalle|röntgen': the head is a lemma, not a stem plus a linking letter
        x = head[:-1]
        if not self._is_part(x) or len(x) > 6:
            return False  # a compound first element takes a linking s ('underbuk|s|smärta')
        if self._prefix_attested_exact(x + tail):
            return True
        for k in range(4, len(tail) + 1):
            p = tail[:k]
            if p in self.sv_words and (x + p) in self.sv_words:
                return True
        return False

    def _prefix_attested_exact(self, s: str) -> bool:
        """``s`` is a lexicon word or the beginning of one (``immunfluorescens…mikroskopi``)."""
        if s in self.sv_words:
            return True
        words = self.sorted_words
        i = bisect.bisect_left(words, s)
        return i < len(words) and words[i].startswith(s)

    def _analyse(self, t: str, depth: int, short_ok: bool = True) -> bool:
        """Compound / derivation analysis; at most ``depth`` further parts, and at most one part of
        three letters in the whole analysis (``ten|don|reflexerna`` is not a word)."""
        if self._word(t) and (len(t) >= 4 or (short_ok and self._saldom(t))):
            return True
        if len(t) >= 5 and self._derived(t):
            return True
        if depth <= 0:
            return False
        n = len(t)
        for i in range(3, n - 2):
            head, tail = t[:i], t[i:]
            short_head = len(head) <= 3
            if short_head and (not short_ok or len(tail) < 5):
                continue
            if len(tail) < 4 and not (len(head) >= 5 and self._saldom(tail) and self._base_form(head)):
                continue  # 'skafoid|ben' but not 'hydroxyu|rea' 
            if not self._head_ok(head):
                continue
            if self._bad_link(head, tail):
                continue
            if self._analyse(tail, depth - 1, short_ok and not short_head):
                return True
        return False

    def _english_inflection(self, t: str) -> bool:
        """``t`` is an inflected form of an English word list entry ('begins', 'diluted')."""
        for suf, rep_ in _EN_INFL:
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                base = t[: len(t) - len(suf)] + rep_
                if base in self.en_words:
                    return True
                if suf in ("ing", "ed", "er", "est") and len(base) >= 4 and base[-1] == base[-2] and base[:-1] in self.en_words:
                    return True  # 'mapping' -> map
        return False

    def attested(self, token: str) -> bool:
        """``token`` is listed as is (lexicon word list or a SALDOM word form), not only analysable."""
        t = token.lower().strip()
        return bool(t) and (t in self.sv_words or self._saldom(t))

    def is_swedish(self, token: str) -> bool:
        """A Swedish word: in the lexicon, a hyphenated compound of Swedish parts, or analysable as a
        compound (heads with linking ``s/e/a/o/u``) and/or a Latin/Greek derivation of an attested
        stem with Swedish inflection (``njurbikarbonatutsöndring``, ``iliolumbala``, ``mesangiell``,
        ``arterierna``).  Tokens in the English word list and spellings foreign to Swedish
        (``c`` before a/o/u/consonant, ``ph``, ``th``, ``qu``, ``w``) are only accepted by the
        lexicon.  A linking letter is rejected when the lexicon attests the compound without it
        (``löksskal``, ``immunofluorescens``)."""
        t = token.lower().strip()
        if not t:
            return False
        if t in self.sv_words:
            return True
        hit = self._sv_cache.get(t)
        if hit is not None:
            return hit
        if "-" in t:  # hyphenated compound: every non-empty part must be Swedish (or a number/abbrev.)
            parts = [p for p in t.split("-") if p]
            ok = bool(parts) and all(
                p in self.sv_words or p in self.compound_parts or p.isdigit() or self.is_swedish(p) for p in parts
            )
        elif self._english_inflection(t) or _EN_ONLY_END.search(t):
            ok = False  # 'relieving', 'deliveries', 'ligands'
        elif t in self.en_words or _NON_SV_ORTHO.search(t) or not t.isalpha():
            ok = self._is_compound(t, strict=True)
        else:
            ok = self._is_compound(t) or self._analyse(t, depth=3)
            if not ok and "kk" in t:  # 'okklusal' / 'bukkal' for ocklusal / buckal
                ok = self._analyse(t.replace("kk", "ck"), depth=3)
        if len(self._sv_cache) > 200_000:
            self._sv_cache.clear()
        self._sv_cache[t] = ok
        return ok

    def _is_compound(self, t: str, strict: bool = False) -> bool:
        """Two-part compound of lexicon words / SALDOM compound parts (basic rule).  ``strict`` (for
        English-looking tokens): both parts have at least four letters ('cancer|behandling', not
        'flo|wing')."""
        n = len(t)
        for i in range(3, n - 2):
            head, tail = t[:i], t[i:]
            if tail not in self.sv_words:
                continue
            if strict and (len(head) < 4 or len(tail) < 4 or not self._base_form(head.removesuffix("s"))):
                continue
            if self._bad_link(head, tail):
                continue
            if self._is_part(head):
                return True
            if head.endswith("s") and self._is_part(head[:-1]):
                return True
        return False

    def is_english(self, token: str) -> bool:
        return token.lower() in self.en_words

    # -- inflection --------------------------------------------------------
    @property
    def form_lemmas(self) -> dict[str, tuple[str, ...]]:
        """Reverse SALDOM index (form -> lemmas), built lazily from ``lemma_forms``."""
        if self._form_lemmas is None:
            d: dict[str, list[str]] = {}
            for lem, forms in self.lemma_forms.items():
                for f in forms:
                    d.setdefault(f, []).append(lem)
            self._form_lemmas = {k: tuple(v) for k, v in d.items()}
        return self._form_lemmas

    def _lemma_candidates(self, w: str) -> list[str]:
        """SALDOM lemmas of which ``w`` is a form (the word itself first if it is a lemma)."""
        out = [w] if w in self.lemma_forms else []
        out.extend(lem for lem in self.form_lemmas.get(w, ()) if lem != w)
        return out

    @lru_cache(maxsize=100_000)
    def word_forms(self, word: str) -> frozenset[str]:
        """Inflected forms of one (lowercased) word; includes the word itself."""
        w = word.lower()
        if not w or not any(c.isalpha() for c in w):
            return frozenset({w})
        if "-" in w.strip("-"):
            head, _, tail = w.rpartition("-")
            return frozenset({w} | {f"{head}-{f}" for f in self.word_forms(tail)})
        lf = self.lemma_forms
        lemmas = self._lemma_candidates(w)
        if lemmas:
            forms = {w}
            for lem in lemmas:
                forms.update(lf[lem])
            return frozenset(forms)
        # unknown word: union of (a) its longest known tail inflected (compound) and (b) heuristic
        # Swedish noun endings (drug names, Latin loans); a plural such as 'opioider' / 'stentar' /
        # 'ferritiner' is first reduced to its singular base
        forms = {w}
        for i in range(3, len(w) - 3):
            tail = w[i:]
            tl = self._lemma_candidates(tail)
            if tl:
                head = w[:i]
                for lem in tl:
                    forms.update(head + f for f in lf[lem])
                break
        bases = {w}
        for pl in ("erna", "arna", "er", "ar"):
            if w.endswith(pl) and len(w) - len(pl) >= 4:
                bases.add(w[: -len(pl)])
                break
        if w.endswith("um") and len(w) >= 6:  # myokardium -> myokardiet, endometrium -> endometriet
            forms.update(w[:-2] + s for s in ("et", "ets", "er", "erna"))
        forms.update(
            b + s
            for b in bases
            for s in _HEUR_SUFFIXES
            if not (s in ("n", "t", "ns", "ts") and b[-1:] not in "aeiouyåäö")
        )
        if w.endswith("a"):
            forms.update(w[:-1] + s for s in ("or", "orna", "an", "ans", "ors"))
        if w.endswith("e"):
            forms.update(w[:-1] + s for s in ("ar", "arna", "en", "et"))
        return frozenset(forms)

    def _modifier_forms(self, w: str) -> set[str] | None:
        """Agreement forms of a pre-modifying adjective (``neurologiska`` -> neurologisk/-t/-a)."""
        for lem in self._lemma_candidates(w):
            forms = self.lemma_forms[lem]
            if (lem + "a") in forms and len(forms) <= 20:
                return set(forms) | {w}
        for suf in _ADJ_SUFFIXES:  # unknown adjective (paraneoplastiska, brakial, ...)
            for base in (w, w[:-1]):
                if base.endswith(suf) and (base == w or w[-1] in "at") and len(base) >= 5:
                    return {w, base, base + "a", base + "t"}
        return None

    @lru_cache(maxsize=50_000)
    def _phrase_forms(self, term: str) -> frozenset[str]:
        words = term.lower().split()
        if not words:
            return frozenset()
        slots: list[set[str]] = []
        for i, w in enumerate(words):
            if i == len(words) - 1:
                slots.append(set(self.word_forms(w)))
            else:
                slots.append(self._modifier_forms(w) or {w})
        total = 1
        for s in slots:
            total *= len(s)
        if total > _MAX_PHRASE_VARIANTS:  # keep only the last word inflected
            slots = [{w} for w in words[:-1]] + [slots[-1]]
        return frozenset(" ".join(p) for p in itertools.product(*slots))

    def forms(self, sv_term: str) -> set[str]:
        """Lowercased full-phrase variants of ``sv_term``: last word inflected via SALDOM
        (heuristic endings for unknown words), preceding adjectives in their agreement forms."""
        return set(self._phrase_forms(" ".join(sv_term.split())))


# ---------------------------------------------------------------------------
# Term base
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# drug INNs: transliteration-only differences (DRUG route)
# ---------------------------------------------------------------------------


def translit_key(s: str) -> str:
    """Canonical spelling under the English->Swedish INN transliterations: ph->f, th->t, qu->k,
    kv->k, x->ks, c->k, ck->k, y->i, final -e dropped (phenytoin = fenytoin, primaquine = primakin,
    demeclocycline = demeklocyklin, methicillin = meticillin)."""
    t = re.sub(r"[^a-zåäöéü]", "", s.lower()).replace("é", "e")
    for a, b in (("ph", "f"), ("th", "t"), ("qu", "k"), ("kv", "k"), ("x", "ks"), ("c", "k"), ("y", "i")):
        t = t.replace(a, b)
    t = t.replace("kk", "k")
    return t.removesuffix("e")


def is_transliteration(en: str, sv: str) -> bool:
    """``sv`` differs from ``en`` only by INN transliteration (or not at all)."""
    return bool(en.strip()) and bool(sv.strip()) and " " not in sv.strip() and translit_key(en) == translit_key(sv)


# Swedish inflection suffixes that may follow an English term inside the Swedish text
# ('vancomycinet', 'lacerationen'), with the SALDOM msd they express (u/n: gender of the definite form)
_SV_SUFFIX_MSD = {
    "": "sg indef nom", "s": "sg indef gen",
    "en": "sg def nom", "n": "sg def nom", "et": "sg def nom", "t": "sg def nom",
    "ens": "sg def gen", "ns": "sg def gen", "ets": "sg def gen", "ts": "sg def gen",
    "er": "pl indef nom", "ar": "pl indef nom", "or": "pl indef nom",
    "ers": "pl indef gen", "ars": "pl indef gen", "ors": "pl indef gen",
    "erna": "pl def nom", "arna": "pl def nom", "orna": "pl def nom", "na": "pl def nom",
    "ernas": "pl def gen", "arnas": "pl def gen", "ornas": "pl def gen",
}
_SV_SUFFIX_ALT = "|".join(sorted((k for k in _SV_SUFFIX_MSD if k), key=len, reverse=True))
# consonant-final Latin/Greek noun endings whose Swedish plural is -er (laceration -> lacerationer)
_ER_PLURAL_END = ("ion", "it", "os", "om", "in", "id", "at", "ol", "yt", "ag", "an", "ad", "ant", "ent", "ism", "ist",
                  "or", "är", "ör", "al", "iv", "ul", "ell", "on", "en")
_NAME_FOLLOW_RE = re.compile(r"\s*[-‐‑]?\s*(?:\d|[IVX]{1,4}\b|[A-Z]{2,}\b|α|β|γ|alpha|beta|gamma)")


class TermBase:
    def __init__(self, rows: list[dict], lexicon: Lexicon, npl_substances: Iterable[str] = ()):
        self.lexicon = lexicon
        self.npl_substances: set[str] = {x.lower() for x in npl_substances if x}
        self.entries: dict[str, dict] = {}
        self.index: dict[str, list[str]] = {}
        self.acro_index: dict[str, list[str]] = {}
        max_n = 1
        for r in rows:
            r = dict(r)
            r["sv_forms"] = list(r["sv_forms"])
            self.entries[r["entry_id"]] = r
            if is_acronym(r["en_term"]):
                self.acro_index.setdefault(r["en_term"].strip(), []).append(r["entry_id"])
            else:
                k = r["en_key"]
                if not k:
                    continue
                self.index.setdefault(k, []).append(r["entry_id"])
                max_n = max(max_n, k.count(" ") + 1)
        self.max_n = min(max_n, MAX_NGRAM)

    @classmethod
    def load(cls, build_dir, lexicon: Lexicon | None = None) -> "TermBase":
        import pyarrow.parquet as pq

        build_dir = Path(build_dir)
        lexicon = lexicon or Lexicon.load(build_dir)
        tbl = pq.read_table(build_dir / "termbase.parquet")
        npl = build_dir / "npl_substances.txt.gz"
        return cls(tbl.to_pylist(), lexicon, npl_substances=_read_lines_gz(npl) if npl.exists() else ())

    # -- drugs -------------------------------------------------------------
    def drug_inn_evidence(self, en_token: str) -> tuple[str, str] | None:
        """``(Swedish INN, source)`` for a single English drug token whose preferred level-A Swedish
        name differs from it only by transliteration (see :func:`translit_key`).  ``source`` is
        ``"npl"`` when the name is an NPL substance name (or an ``npl_atc`` row), else ``"swemesh"``
        (a Svensk MeSH chemical/drug descriptor; e.g. meticillin, which NPL does not list).  When
        an NPL name exists for the drug and it is NOT a transliteration (hydroxyurea ->
        hydroxikarbamid), the answer is None: that is a TERM-A case, never a MeSH fallback."""
        tok = en_token.strip()
        if len(tok) < 4 or not re.fullmatch(r"[A-Za-z][a-z]+(?:-[a-z]+)?", tok):
            return None
        ids = self.index.get(en_key(tok))
        if not ids:
            return None
        rows = [self.entries[i] for i in ids]
        drug_a = [r for r in rows if r["level"] == "A" and r.get("drug", r["source"] == "npl_atc")]
        if not drug_a or not all(bool(r["single_sense"]) for r in drug_a):
            return None
        _info, a_forms, _ = self._merge(ids)
        forms = [f for f in with_salt_base(tok, a_forms) if f.strip() and " " not in f.strip()]
        if not forms:
            return None
        npl_rows = {f.lower() for r in drug_a if r["source"] == "npl_atc" for f in r["sv_forms"]}
        npl = [f for f in forms if f.lower() in self.npl_substances or f.lower() in npl_rows]
        if npl:
            name, source = npl[0], "npl"
        else:
            name, source = forms[0], "swemesh"
        if not is_transliteration(tok, name):
            return None
        return name.lower(), source

    def drug_inn(self, en_token: str) -> str | None:
        """Swedish INN when ``en_token`` is a single-substance drug of the term base (NPL first) and
        the INN differs from it only by transliteration (c->k, ph->f, th->t, y->i, final -e, qu->k,
        x->ks): Phenytoin -> fenytoin, clopidogrel -> klopidogrel, primaquine -> primakin.  None for
        non-drugs and for names that are not transliterations (hydroxyurea -> hydroxikarbamid)."""
        ev = self.drug_inn_evidence(en_token)
        return ev[0] if ev else None

    # -- code substitution ---------------------------------------------------
    def _inflect(self, form: str, msd: str, *, translit: bool, suffix: str) -> str | None:
        """``form`` (a Swedish lemma / singular, possibly multi-word) in the inflection ``msd`` of its
        last word; None when SALDOM gives no unambiguous form and no safe rule applies."""
        words = form.split(" ")
        last = words[-1]
        head, sep, word = last.rpartition("-")
        prefix = head + sep
        low = word.lower()
        if msd == "sg indef nom":
            return form
        new: str | None = None
        lx = self.lexicon
        nouns = [a for a in lx.analyses(low) if a[1] == "nn" and a[3] == "sg indef nom"] if hasattr(lx, "analyses") else []
        lemmas = {a[0] for a in nouns}
        if len(lemmas) == 1:
            table = lx.morph["noun"].get(lemmas.pop(), [None, {}])[1]
            # an analysis of the compound's last part carries the head over ('levercell' -> 'leverceller')
            cands = table.get(msd) or []
            if len(cands) == 1:
                new = cands[0]
        elif not lemmas and hasattr(lx, "analyses"):
            for i in range(3, len(low) - 2):  # compound: inflect the last SALDOM noun part
                tail = low[i:]
                an = [a for a in lx.analyses(tail) if a[1] == "nn" and a[3] == "sg indef nom"]
                if an:
                    tl = {a[0] for a in an}
                    if len(tl) == 1:
                        cands = lx.morph["noun"].get(tl.pop(), [None, {}])[1].get(msd) or []
                        if len(cands) == 1:
                            new = low[:i] + cands[0]
                    break
        if new is None and not lemmas:
            if translit and suffix:
                new = low + suffix  # transliterated drug name keeps the Swedish suffix it had
            elif msd == "pl indef nom" and low.endswith(_ER_PLURAL_END) and low[-1] not in _VOWELS:
                new = low + "er"
            elif msd == "pl def nom" and low.endswith(_ER_PLURAL_END) and low[-1] not in _VOWELS:
                new = low + "erna"
        if new is None:
            return None
        if word[:1].isupper() and not prefix:
            new = new[:1].upper() + new[1:]
        return " ".join(words[:-1] + [prefix + new])

    def _in_name(self, text: str, s: int, e: int) -> bool:
        """The occurrence ``text[s:e]`` is part of an English multi-word name (gene/protein names such
        as 'Hepatocyte nuclear factor-1', 'tumor necrosis factor alpha'): an adjacent English word,
        or a following number / Greek letter / all-caps symbol."""
        lx = self.lexicon
        nxt = re.match(r"[\s-]*([^\W\d_]+)", text[e:])
        prv = re.search(r"([^\W\d_]+)[\s-]*$", text[:s])
        for m in (nxt, prv):
            if m:
                w = m.group(1)
                if w.lower() not in SV_FUNCTION_WORDS and lx.is_english(w) and not lx.is_swedish(w):
                    return True
        return bool(_NAME_FOLLOW_RE.match(text, e)) and not re.match(r"\s*\d+(?:[.,]\d+)?\s*(?:mg|g|ml|mL|µg|mcg|%)", text[e:])

    def substitution(self, sv_text: str, target: str, form: str) -> tuple[str, int] | None:
        """Replace every whole-word occurrence of the English token ``target`` in ``sv_text`` by the
        Swedish ``form`` (a lemma / singular), carrying over inflection:

        * an English plural target ('lacerations') becomes the Swedish plural indefinite form;
        * a Swedish suffix attached to the English token ('vancomycinet', 'lacerationen') becomes the
          same SALDOM inflection of ``form`` (``-et`` -> sg def, ``-er`` -> pl indef, ...);
        * a transliterated drug name keeps its suffix verbatim (vancomycinet -> vankomycinet).

        Occurrences inside English multi-word names (gene/protein names) are left alone.  Returns
        ``(new_text, n_replaced)``; None when the target is a Swedish word (no Swedish->Swedish
        substitution without a drug transliteration), when an occurrence's inflection is ambiguous,
        or when every occurrence sits inside a name."""
        target = " ".join(target.split())
        form = " ".join(form.split())
        if not target or not form:
            return None
        lx = self.lexicon
        translit = " " not in target and is_transliteration(target, form)
        last = target.split(" ")[-1]
        if not translit and lx.is_swedish(last) and not lx.is_english(last):
            return None  # 'Mandibularkanalen', 'mitralisstenos': Swedish words are never replaced here
        is_wl = getattr(lx, "is_whitelisted", None)
        if not translit and callable(is_wl) and is_wl(last):
            return None  # whitelisted Latin / international terms ('maxilla', 'pancreas') stay
        en_plural = singularise(last.lower()) != last.lower() and last.lower().endswith("s")
        pat = re.compile(r"(?<![\w-])" + r"\s+".join(re.escape(w) for w in target.split(" "))
                         + r"(" + _SV_SUFFIX_ALT + r")?(?![\w])", re.IGNORECASE)
        out: list[str] = []
        pos = n = 0
        blocked = 0
        for m in pat.finditer(sv_text):
            suffix = (m.group(1) or "").lower()
            if self._in_name(sv_text, m.start(), m.end()):
                blocked += 1
                continue
            msd = _SV_SUFFIX_MSD[suffix]
            if not suffix and en_plural:
                msd = "pl indef nom"
            new = self._inflect(form, msd, translit=translit, suffix=suffix)
            if new is None:
                return None
            found = m.group(0)
            if found[:1].isupper():
                new = new[:1].upper() + new[1:]
            elif found[:1].islower() and not new[1:2].isupper():
                new = new[:1].lower() + new[1:]
            out.append(sv_text[pos: m.start()])
            out.append(new)
            pos = m.end()
            if new != found:
                n += 1
        if blocked and not n:
            return None
        out.append(sv_text[pos:])
        return "".join(out), n

    # -- English side ------------------------------------------------------
    def _candidates(self, en_text: str) -> list[tuple[int, int, list[str]]]:
        toks = en_tokens(en_text)
        cands: list[tuple[int, int, int, list[str]]] = []  # (n, i, j, ids)
        norms = [t[0] for t in toks]
        for i in range(len(toks)):
            for n in range(min(self.max_n, len(toks) - i), 0, -1):
                if n == 1 and is_acronym(toks[i][3]):
                    break  # 'PON' must not match 'pons' (-> 'pon'); acronyms use acro_index
                ids = self.index.get(" ".join(norms[i : i + n]))
                if ids:
                    cands.append((n, i, i + n, ids))
                    break  # longest match at this start
            raw = toks[i][3]
            acro = self.acro_index.get(raw) or (
                self.acro_index.get(raw[:-1]) if raw.endswith("s") and raw[:-1].isupper() else None
            )
            if acro and is_acronym(raw.rstrip("s") or raw):
                cands.append((1, i, i + 1, acro))
        cands.sort(key=lambda c: (-c[0], c[1]))
        taken = [False] * len(toks)
        out = []
        for n, i, j, ids in cands:
            if any(taken[i:j]):
                continue
            if _NEGATED_PREFIX.search(en_text[: toks[i][1]]):
                continue  # 'non-ST-elevation MI' must not match the 'ST-elevation MI' entry
            for k in range(i, j):
                taken[k] = True
            out.append((toks[i][1], toks[j - 1][2], ids))
        out.sort()
        return out

    def _merge(self, ids: list[str]) -> tuple[dict, list[str], list[str]]:
        ents = sorted((self.entries[e] for e in ids), key=lambda e: (e["level"], SOURCE_PRIORITY.get(e["source"], 9)))
        a = [e for e in ents if e["level"] == "A"]
        b = [e for e in ents if e["level"] == "B"]

        def union(es):
            seen, out = set(), []
            for e in es:
                for f in e["sv_forms"]:
                    if f.lower() not in seen:
                        seen.add(f.lower())
                        out.append(f)
            return out

        chosen = a or b
        info = {
            "entry_id": chosen[0]["entry_id"],
            "level": chosen[0]["level"],
            "single_sense": all(bool(e["single_sense"]) for e in chosen),
            "obligation_ok": any(bool(e.get("obligation_ok", True)) for e in chosen),
            "source": "+".join(sorted({e["source"] for e in chosen}, key=lambda s: SOURCE_PRIORITY.get(s, 9))),
        }
        return info, union(a), union(b)

    def _spot_full(self, en_text: str, field: FieldKey):
        out = []
        for s, e, ids in self._candidates(en_text):
            info, a_forms, b_forms = self._merge(ids)
            a_forms = with_salt_base(en_text[s:e], a_forms)
            hit = TermHit(
                entry_id=info["entry_id"],
                field=field,
                en_span=en_text[s:e],
                start=s,
                end=e,
                sv_forms=a_forms if info["level"] == "A" else b_forms,
                level=info["level"],
                single_sense=info["single_sense"],
                source=info["source"],
            )
            out.append((hit, a_forms, b_forms, info["obligation_ok"]))
        return out

    def spot(self, en_text: str, field: FieldKey) -> list[TermHit]:
        return [h for h, *_ in self._spot_full(en_text, field)]

    # -- Swedish side ------------------------------------------------------
    @lru_cache(maxsize=20_000)
    def _sv_matcher(self, forms_key: tuple[str, ...]):
        variants: set[str] = set()
        for f in forms_key:
            f = " ".join(f.split())
            if not f:
                continue
            variants.add(f.lower())
            variants.update(self.lexicon.forms(f))
            if "(" in f:  # also accept the form without a parenthetical qualifier
                g = " ".join(re.sub(r"\([^)]*\)", " ", f).split())
                if g:
                    variants.add(g.lower())
                    variants.update(self.lexicon.forms(g))
        if not variants:
            return None, None, (frozenset(), frozenset())
        variants = {_deaccent(v) for v in variants}

        def pat(v: str) -> str:
            p = re.escape(v).replace("e", "[eé]")
            return p.replace(r"\ ", r"\s+").replace(r"\-", r"[-‐‑\s]?")

        ordered = sorted(variants, key=len, reverse=True)
        rx = re.compile(r"(?<!\w)(?:" + "|".join(pat(v) for v in ordered) + r")(?!\w)", re.IGNORECASE)
        multi = [v for v in ordered if " " in v]
        # multi-word form whose last word continues as a compound: 'gastroesofageal refluxsjukdom'
        rx_tail = (
            re.compile(r"(?<!\w)(?:" + "|".join(pat(v) for v in multi) + r")([^\W\d_]{3,})(?!\w)", re.IGNORECASE)
            if multi
            else None
        )
        single = {v for v in variants if " " not in v and len(v) >= 5}
        single |= {v.replace("-", "") for v in single}
        # compound-initial stems: skalle -> skall(form), gallblåsa -> gallblås(e)rubbning, hjärta -> hjärt
        prefixes = set(single)
        for v in single:
            if v[-1] in "ae" and len(v) - 1 >= 5:
                prefixes.add(v[:-1])
                prefixes.add(v[:-1] + "e")
        return rx, rx_tail, (frozenset(single), frozenset(prefixes))

    def _remainder_ok(self, rem: str, *, rem_first: bool) -> bool:
        lx = self.lexicon
        if len(rem) < 3:
            return False
        if rem_first:
            if lx._is_part(rem) or lx.is_swedish(rem):
                return True
            return rem.endswith("s") and (lx._is_part(rem[:-1]) or rem[:-1] in lx.sv_words)
        if lx.is_swedish(rem):
            return True
        return rem.startswith("s") and len(rem) > 3 and lx.is_swedish(rem[1:])

    def match_sv(self, sv_text: str, sv_forms: list[str]) -> list[tuple[int, int, str]]:
        rx, rx_tail, single = self._sv_matcher(tuple(sv_forms))
        if rx is None:
            return []
        single, prefixes = single
        spans = [(m.start(), m.end(), m.group()) for m in rx.finditer(sv_text)]
        covered = [(s, e) for s, e, _ in spans]
        if rx_tail is not None:
            for m in rx_tail.finditer(sv_text):
                s, e = m.start(), m.end()
                if not any(cs < e and s < ce for cs, ce in covered) and self._remainder_ok(
                    m.group(1).lower(), rem_first=False
                ):
                    spans.append((s, e, m.group()))
                    covered.append((s, e))
        if prefixes:
            for m in SV_HTOKEN_RE.finditer(sv_text):
                s, e = m.start(), m.end()
                if any(cs < e and s < ce for cs, ce in covered):
                    continue
                full = _deaccent(m.group().lower())
                ok = False
                for tok in dict.fromkeys((full, full.replace("-", ""))):
                    if tok in single:  # 'alfa-fetoprotein' for the form 'alfafetoprotein'
                        ok = True
                        break
                    if len(tok) < 8:
                        continue
                    for k in range(5, len(tok) - 2):  # token starts with a form
                        if tok[:k] in prefixes and self._remainder_ok(tok[k:].lstrip("-"), rem_first=False):
                            ok = True
                            break
                    if not ok:
                        for k in range(3, len(tok) - 4):  # token ends with a form
                            if tok[k:] in single and self._remainder_ok(tok[:k].rstrip("-"), rem_first=True):
                                ok = True
                                break
                    if ok:
                        break
                if ok:
                    spans.append((s, e, m.group()))
        spans.sort()
        return spans

    def _kept_swedish_valid(self, en_span: str, sv_text: str) -> bool:
        """The English term itself occurs in the Swedish field and every word of it is a valid
        Swedish word ('intubation', 'hypertension'): not evidence of an error -> no obligation."""
        words = SV_WORD_RE.findall(en_span.lower())
        if not words or not all(w in self.lexicon.sv_words for w in words):
            return False
        pat = r"(?<!\w)" + r"[\s-]+".join(re.escape(w) for w in words) + r"(?!\w)"
        return re.search(pat, sv_text, re.IGNORECASE) is not None

    # -- grounding ---------------------------------------------------------
    def ground(
        self, en: dict[FieldKey, str], sv: dict[FieldKey, str]
    ) -> tuple[list[VerifiedPair], list[Obligation]]:
        verified: list[VerifiedPair] = []
        obligations: list[Obligation] = []
        seen_obl: set[tuple[str, str]] = set()
        for field, en_text in en.items():
            sv_text = sv.get(field) or ""
            used: dict[str, int] = {}
            for hit, a_forms, b_forms, obligation_ok in self._spot_full(en_text or "", field):
                match_level, matches = None, []
                if a_forms:
                    matches = self.match_sv(sv_text, a_forms)
                    match_level = "A"
                if not matches and b_forms:
                    matches = self.match_sv(sv_text, b_forms)
                    match_level = "B"
                if matches:
                    k = used.get(hit.entry_id, 0)
                    s, e, txt = matches[min(k, len(matches) - 1)]
                    used[hit.entry_id] = k + 1
                    verified.append(
                        VerifiedPair(
                            entry_id=hit.entry_id, field=field, en_span=hit.en_span, sv_span=txt,
                            sv_start=s, sv_end=e, level=match_level,
                        )
                    )
                elif (
                    hit.level == "A"
                    and hit.single_sense
                    and obligation_ok
                    and a_forms
                    and not self._kept_swedish_valid(hit.en_span, sv_text)
                ):
                    key = (hit.entry_id, field)
                    if key not in seen_obl:
                        seen_obl.add(key)
                        obligations.append(
                            Obligation(entry_id=hit.entry_id, field=field, en_span=hit.en_span, sv_forms=a_forms, level="A")
                        )
        return verified, obligations

    def is_accepted_rendering(self, en_quote: str, sv_quote: str) -> bool:
        """Concept-aware veto test.

        True only if ``en_quote`` contains at least one level-A term, every level-A term in it is
        single-sense and has an accepted form (inflection/compound-aware) in ``sv_quote``, and
        every other Swedish word of ``sv_quote`` is a function word, or a Swedish word that is an
        orthographic cognate (similarity >= 0.75) of an English word *inside* a matched term span
        (``metabolic acidosis`` -> ``metabol acidos`` where only ``acidos`` is the stored form).
        Word validity alone never vetoes."""
        if not en_quote or not sv_quote:
            return False
        hits = [(h, a) for h, a, *_ in self._spot_full(en_quote, "question") if h.level == "A" and a]
        if not hits or not all(h.single_sense for h, _ in hits):
            return False
        mask = [False] * len(sv_quote)
        en_words_in_hits: list[str] = []
        for h, a_forms in hits:
            m = self.match_sv(sv_quote, a_forms)
            if not m:
                # multi-word form: accept its head word; the modifiers must then pass the cognate
                # test below ('metabol acidos' for the stored form 'metabolisk acidos')
                heads = [f.split()[-1] for f in a_forms if len(f.split()) > 1]
                m = self.match_sv(sv_quote, heads) if heads else []
            if not m:
                return False
            for s, e, _ in m:
                for i in range(s, e):
                    mask[i] = True
            en_words_in_hits.extend(t[3] for t in en_tokens(h.en_span))
        for m in SV_WORD_RE.finditer(sv_quote):
            if all(mask[m.start() : m.end()]):
                continue
            w = m.group().lower()
            if w in SV_FUNCTION_WORDS:
                continue
            if self.lexicon.is_swedish(w) and any(
                len(ew) >= 4 and cognate_similarity(ew, w) >= 0.75 for ew in en_words_in_hits
            ):
                continue
            return False
        return True
