"""Render the LLM prompts (refine/prompts/*.md) for each loop step.

Templates use ``$name`` placeholders (string.Template) and two sections,
``## system`` and ``## user``.  Items are rendered as compact plain text,
never as JSON, and no JSON schema is pasted into a prompt.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template

from .io import resources_dir
from .types import Finding, FieldKey, field_label

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=None)
def _template(name: str) -> tuple[Template, Template]:
    text = (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
    system, user = text.split("## user", 1)
    system = system.replace("## system", "", 1).strip()
    return Template(system), Template(user.strip())


@lru_cache(maxsize=None)
def style_guide() -> str:
    return (resources_dir() / "style_guide_sv.md").read_text(encoding="utf-8").strip()


def render(name: str, **values: str) -> tuple[str, str]:
    system, user = _template(name)
    values = {"style_guide": style_guide(), **values}
    return system.substitute(values), user.substitute(values)


def field_name(key: FieldKey) -> str:
    label = field_label(key)
    return "question" if label is None else f"option {label}"


def ordered_fields(fields: dict[FieldKey, str]) -> list[FieldKey]:
    rest = sorted(k for k in fields if k != "question")
    return (["question"] if "question" in fields else []) + rest


def render_item(en: dict[FieldKey, str], sv: dict[FieldKey, str] | None = None) -> str:
    """Plain-text item: one block per field with EN and (optionally) SV lines."""
    blocks = []
    for key in ordered_fields(en):
        lines = [f"[{key}]", f"EN: {en[key]}"]
        if sv is not None:
            lines.append(f"SV: {sv.get(key, '')}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_sv_context(sv: dict[FieldKey, str], skip: FieldKey) -> str:
    others = [k for k in ordered_fields(sv) if k != skip]
    if not others:
        return ""
    body = "\n".join(f"[{k}] {sv[k]}" for k in others)
    return f"Other Swedish fields of the item (unchanged, for context):\n{body}"


def render_findings(findings: list[Finding], with_category: bool = True) -> str:
    lines = []
    for f in findings:
        cat = f" ({f.category}, {f.severity})" if with_category else ""
        en = f' | EN: "{f.en_quote}"' if f.en_quote else ""
        lines.append(f'- id={f.id} [{f.field}]{cat} SV: "{f.sv_quote}"{en} — {f.explanation}')
    return "\n".join(lines) if lines else "(none)"


# ---- detector / repair / judge helpers ------------------------------------------------


def protected_line(terms) -> str:
    """Protected terms (style guide `keep` list) that occur in the texts shown to a model."""
    terms = [t for t in dict.fromkeys(terms) if t]
    if not terms:
        return ""
    body = ", ".join(f'"{t}"' for t in terms)
    return (f"Protected terms in this text (established Swedish clinical terms; they are correct and must not "
            f"be reported, replaced or judged as errors): {body}.")


DRUG_SOURCES = {
    "npl": "the national drug register NPL (Läkemedelsverket)",
    "swemesh": "Svensk MeSH (Karolinska Institutet), the authoritative Swedish medical vocabulary",
}


def drug_evidence(target: str, form: str, source: str = "npl") -> str:
    where = DRUG_SOURCES.get(source, DRUG_SOURCES["npl"])
    return (f'Evidence: "{form}" is the Swedish name of "{target}" in {where}; '
            f"it is correct by definition — judge only the surrounding wording.")


def term_evidence(en_term: str, form: str) -> str:
    # Not "correct by definition": that wording made every judge pass term-base swaps
    # unanimously, including concept shifts (Eisenmengers syndrom -> komplex).  Only drug names
    # (a register name, transliteration only) keep the authoritative wording.
    return (f'Reference: the term base lists "{form}" as a Swedish term for "{en_term}". '
            f"It is a reference, not a requirement: judge both versions on every criterion, including "
            f"whether this term names the same concept and fits this context.")


def _join(*parts: str) -> str:
    return "\n".join(p for p in parts if p)


LONG_ITEM_RULE = ("8. This item is long: go through the Swedish sentence by sentence and, for each sentence, check "
                  "every clinical finding, qualifier, number and time of the English for additions and omissions.")


# ---- step-specific renderers ---------------------------------------------------


def detect_prompt(en, sv, focus: list[FieldKey], verified: list[tuple[str, str]],
                  dismissed: list[tuple[str, str]], *, kind: str = "generic",
                  protected: list[str] | tuple[str, ...] = (), long_item: bool = False) -> tuple[str, str]:
    """kind: "generic" (detect.md) or a pass: "meaning" | "terminology" | "language"."""
    hints = []
    if verified:
        pairs = "; ".join(f'"{s}" = verified Swedish rendering of "{e}"' for e, s in verified[:40])
        hints.append(f"Term-base check (authoritative, these renderings are correct): {pairs}.")
    if protected:
        hints.append(protected_line(protected))
    if dismissed:
        items = "; ".join(f'[{f}] "{q}"' for f, q in dismissed[:40])
        hints.append(f"Already reviewed and found correct (do not report again): {items}.")
    name = "detect" if kind == "generic" else f"detect_{kind}"
    values = dict(item=render_item(en, sv), hints="\n".join(hints), focus_fields=", ".join(focus))
    if kind == "meaning":
        values["long_item_rule"] = LONG_ITEM_RULE if long_item else ""
    return render(name, **values)


def verify_prompt(en, sv, claims: list[Finding]) -> tuple[str, str]:
    return render("verify", item=render_item(en, sv), claims=render_findings(claims))


def repair_prompt(en, sv, findings: list[Finding], constraints: list[str],
                  rejected: list[tuple[FieldKey, str]], per_finding: bool,
                  protected: list[str] | tuple[str, ...] = ()) -> tuple[str, str]:
    cons = ("Required terms (authoritative term base, use exactly these Swedish forms, inflected as needed):\n"
            + "\n".join(f"- {c}" for c in constraints)) if constraints else ""
    cons = _join(cons, protected_line(protected))
    rej = ("Earlier corrections that were NOT accepted — do not propose them again:\n"
           + "\n".join(f'- [{k}] "{t}"' for k, t in rejected[:20])) if rejected else ""
    mode = ("Propose a separate, independent edit for each error (each edit lists exactly one finding id)."
            if per_finding else
            "You may combine several errors of the same field into one edit when their spans are close.")
    return render("repair", item=render_item(en, sv), findings=render_findings(findings),
                  constraints=cons, rejected=rej, mode=mode)


def integrate_prompt(en, sv, field: FieldKey, text: str, *, mode: str, pairs: list[tuple[str, str, str]],
                     substituted: bool, rejected: list[str] = (),
                     protected: list[str] | tuple[str, ...] = ()) -> tuple[str, str]:
    """Integration pass for DRUG (mode="drug") and TERM-A (mode="term") repairs.

    ``pairs``: (English term, required Swedish form, Swedish span to replace or "").
    ``substituted``: the code already put the forms into ``text``; the model only adjusts the
    surrounding words.  Otherwise the model must replace the span by the form itself."""
    lines = []
    for en_term, form, span in pairs:
        if mode == "drug":
            src = ("the Swedish drug name was selected from an authoritative Swedish source (the national "
                   "drug register NPL or Svensk MeSH) and cannot be wrong")
        else:
            src = "the form comes from the authoritative medical term base"
        if substituted:
            lines.append(f'- "{en_term}" has been replaced by "{form}" in the text below; {src}.')
        else:
            where = f'the Swedish span "{span}"' if span else f'the rendering of the English "{en_term}"'
            lines.append(f'- Replace {where} by "{form}" (inflected as needed); {src}.')
    head = ("Required Swedish drug names (authoritative: national drug register NPL / Svensk MeSH):" if mode == "drug"
            else "Required Swedish terms (authoritative term base):")
    task = head + "\n" + "\n".join(lines)
    if rejected:
        task += "\nEarlier versions that were NOT accepted — do not propose them again:\n" + \
            "\n".join(f'- "{t}"' for t in list(rejected)[:10])
    task = _join(task, protected_line(protected))
    if substituted:
        requirement = ("Keep the required term exactly as it is now in the text (you may only change its "
                       "inflection ending if the grammar needs it); only adjust the surrounding words if needed.")
    else:
        requirement = ("Put the required term in place of the span, inflected as the sentence needs; the required "
                       "term is the only allowed replacement. Then adjust the surrounding words only if needed.")
    return render("integrate", item=render_item(en, sv), task=task, field_name=field_name(field),
                  text=text, requirement=requirement)


def retranslate_prompt(en, sv, field: FieldKey) -> tuple[str, str]:
    return render("retranslate", field=field_name(field), item=render_item(en, sv), en_text=en[field])


def pairwise_prompt(en, sv_context: dict[FieldKey, str] | None, field: FieldKey | None,
                    text_a: str, text_b: str, evidence: str = "",
                    protected: list[str] | tuple[str, ...] = ()) -> tuple[str, str]:
    ev = _join(evidence, protected_line(protected))
    if field is None:  # whole-item comparison
        return render("pairwise", en_item=render_item(en), sv_context="", evidence=ev,
                      target_label="Two Swedish versions of the WHOLE item:", text_a=text_a, text_b=text_b)
    return render("pairwise", en_item=render_item(en),
                  sv_context=render_sv_context(sv_context or {}, field), evidence=ev,
                  target_label=f"Two Swedish versions of the {field_name(field)}:",
                  text_a=text_a, text_b=text_b)


def noninferiority_prompt(en, field: FieldKey, text_a: str, text_b: str, new_label: str = "B",
                          evidence: str = "", protected: list[str] | tuple[str, ...] = ()) -> tuple[str, str]:
    """Neutral labels: ``new_label`` names the version with the authoritative term (asked in both orders)."""
    ref_label = "A" if new_label == "B" else "B"
    return render("noninferiority", en_item=render_item(en), evidence=_join(evidence, protected_line(protected)),
                  target_label=f"Two Swedish versions of the {field_name(field)}:",
                  text_a=text_a, text_b=text_b, new_label=new_label, ref_label=ref_label)


def item_text(sv: dict[FieldKey, str]) -> str:
    return "\n".join(f"[{k}] {sv[k]}" for k in ordered_fields(sv))
