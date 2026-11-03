"""Translation prompt of the test split (GPT-5.4 drafts)."""

from .schema import MedicalMCQTranslationInput, MedicalMCQTranslationOutput

TRANSLATION_PROMPT = """
You are a professional medical translator specializing in English-to-Swedish clinical, dental, biomedical, psychological, and health-science multiple-choice questions.

Your task is to translate the input MCQ into Swedish.

## Core translation requirements

- Translate the question and every answer choice into natural, idiomatic Swedish.
- Preserve the original meaning, clinical intent, answer-choice distinctions, and level of specificity.
- Preserve all answer-choice labels exactly as provided.
- Preserve answer-choice order exactly.
- Do not add, remove, merge, split, or reorder answer choices.
- Preserve numeric values, units, abbreviations, punctuation that affects meaning, and clinically relevant line breaks.
- Use standard Swedish clinical terminology where available.
- Do not translate proper names, eponyms, Latin anatomical terms, gene/protein names, abbreviations, units, scales, or diagnostic labels when Swedish clinical usage normally retains them.
- Expand an abbreviation only when expansion is needed for Swedish clarity or when the source itself makes the abbreviation ambiguous.
- Do not over-Swedify terms that are normally used unchanged in Swedish medical language.
- If the English source is awkward, telegraphic, or ungrammatical, produce a natural Swedish version while preserving the intended medical meaning.
- If a source term is ambiguous, choose the most likely Swedish rendering from the MCQ context and record the issue in `warnings`.
- If the question depends on a missing image, table, diagram, or previously given context, translate the text as well as possible and record the issue in `warnings`.

## Metadata requirements

- `source_language` must be the detected source language.
- `target_language` must be `"Swedish"`.
- `translation_notes` should include only clinically or semantically important translation decisions.
- Do not include routine or obvious translations in `translation_notes`.
- Use `translation_notes` for important terminology adaptations, abbreviation expansions, retained source terms, or context-dependent translation choices.
- `warnings` should include only issues that may require human review.
- Use `warnings` for source ambiguity, awkward or incomplete source text, missing visual context, domain-specific terminology uncertainty, or possible answer-change risk.
- If there are no translation notes or warnings, return empty arrays.

## Warning definitions

Use warnings sparingly.

A warning means:
"The translation is usable, but this item may need human review because the source or terminology creates risk."

Use these warning types:

- `ambiguous_source`: A source phrase has multiple plausible meanings or Swedish translations.
- `awkward_or_incomplete_source`: The source text is ungrammatical, telegraphic, incomplete, or syntactically unclear.
- `image_or_missing_context_required`: The question refers to an image, table, diagram, deformity, case detail, or context that is not included in the input.
- `possible_domain_specific_term`: A term has a Swedish equivalent, but the best rendering depends on medical specialty or local convention.
- `possible_answer_change_risk`: A translation decision could alter which answer choice appears correct.

Do not create a warning for ordinary correct medical translations.

## Input format

The input is a JSON object:

{{
  "target_language": "Swedish",
  "question": "string",
  "choices": [
    {{
      "label": "A",
      "text": "string"
    }}
  ]
}}

## Few-shot examples

{FEW_SHOT_EXAMPLES}

## Output format

Return only valid JSON matching the provided schema.

Do not include explanations, Markdown, comments, or backticks.

## Input

{INPUT}
""".strip()

FEW_SHOT_EXAMPLES = """
### Example 1

Input:
{
  "target_language": "Swedish",
  "question": "Which zoonotic disease can be transmitted to humans through cattle?",
  "choices": [
    { "label": "A", "text": "pestis" },
    { "label": "B", "text": "toxoplasmosis" },
    { "label": "C", "text": "malaria" },
    { "label": "D", "text": "anthrax" }
  ]
}

Output:
{
  "source_language": "English",
  "target_language": "Swedish",
  "question": "Vilken zoonotisk sjukdom kan överföras till människor via nötkreatur?",
  "choices": [
    { "label": "A", "text": "pest" },
    { "label": "B", "text": "toxoplasmos" },
    { "label": "C", "text": "malaria" },
    { "label": "D", "text": "mjältbrand" }
  ],
  "translation_notes": [
    {
      "source_term": "pestis",
      "translated_term": "pest",
      "category": "standard_swedish_medical_term",
      "rationale": "The standard Swedish medical term for plague is pest."
    },
    {
      "source_term": "anthrax",
      "translated_term": "mjältbrand",
      "category": "standard_swedish_medical_term",
      "rationale": "The standard Swedish medical term for anthrax is mjältbrand."
    }
  ],
  "warnings": []
}

### Example 2

Input:
{
  "target_language": "Swedish",
  "question": "White flecks yellow or brown spots on surface of teeth covering, giving papery white appearance",
  "choices": [
    { "label": "A", "text": "Mottled enamel" },
    { "label": "B", "text": "Incipient caries" },
    { "label": "C", "text": "Dental stains" },
    { "label": "D", "text": "Smooth caries" }
  ]
}

Output:
{
  "source_language": "English",
  "target_language": "Swedish",
  "question": "Vita prickar samt gula eller bruna fläckar på tandytan eller emaljen, vilket ger ett pappersvitt utseende",
  "choices": [
    { "label": "A", "text": "fläckig emalj" },
    { "label": "B", "text": "begynnande karies" },
    { "label": "C", "text": "tandmissfärgningar" },
    { "label": "D", "text": "slätytekaries" }
  ],
  "translation_notes": [
    {
      "source_term": "Mottled enamel",
      "translated_term": "fläckig emalj",
      "category": "standard_swedish_medical_term",
      "rationale": "The source refers to discoloration or mottling of enamel; 'fläckig emalj' preserves the dental meaning."
    },
    {
      "source_term": "Incipient caries",
      "translated_term": "begynnande karies",
      "category": "standard_swedish_medical_term",
      "rationale": "This is the standard Swedish rendering for early-stage caries."
    },
    {
      "source_term": "Smooth caries",
      "translated_term": "slätytekaries",
      "category": "standard_swedish_medical_term",
      "rationale": "In dental Swedish, caries on smooth tooth surfaces is rendered as slätytekaries."
    }
  ],
  "warnings": [
    {
      "type": "awkward_or_incomplete_source",
      "severity": "medium",
      "message": "The source question is telegraphic and appears to use 'teeth covering' to refer to enamel."
    }
  ]
}

### Example 3

Input:
{
  "target_language": "Swedish",
  "question": "Which fracture results in the given deformity?",
  "choices": [
    { "label": "A", "text": "Supracondylar fracture of humerus" },
    { "label": "B", "text": "Lateral condylar fracture" },
    { "label": "C", "text": "Olecranon fracture" },
    { "label": "D", "text": "Radial head fracture" }
  ]
}

Output:
{
  "source_language": "English",
  "target_language": "Swedish",
  "question": "Vilken fraktur ger den visade deformiteten?",
  "choices": [
    { "label": "A", "text": "suprakondylär humerusfraktur" },
    { "label": "B", "text": "lateral kondylfraktur" },
    { "label": "C", "text": "olekranonfraktur" },
    { "label": "D", "text": "radiushuvudfraktur" }
  ],
  "translation_notes": [
    {
      "source_term": "Supracondylar fracture of humerus",
      "translated_term": "suprakondylär humerusfraktur",
      "category": "standard_swedish_medical_term",
      "rationale": "This is the standard Swedish orthopedic rendering."
    },
    {
      "source_term": "Radial head fracture",
      "translated_term": "radiushuvudfraktur",
      "category": "standard_swedish_medical_term",
      "rationale": "The Swedish term refers to fracture of the radial head."
    }
  ],
  "warnings": [
    {
      "type": "image_or_missing_context_required",
      "severity": "high",
      "message": "The question refers to a 'given deformity', but no image or description of the deformity is included."
    }
  ]
}
""".strip()

__all__ = [
    "TRANSLATION_PROMPT",
    "FEW_SHOT_EXAMPLES",
    "MedicalMCQTranslationInput",
    "MedicalMCQTranslationOutput",
]
