"""Prompt templates of the benchmark (one user message each).

Explanation ("trace") prompt: the model returns ``{explanation, answer_index}``;
used for the benchmark models, the teacher traces and the trace students.
Answer-only prompt: the model returns ``{answer_index}``; used for the
answer-only students. ``{question}`` and ``{options}`` are filled with
``str.format``.
"""

RESPONSE_PROMPT = """
You are a highly knowledgeable and careful medical expert.

Your task is to analyze the following multiple-choice medical question and determine the single best answer based on established medical knowledge and reasoning.

Follow these steps carefully:
1. Read the question and all available answer options.
2. Identify key clinical findings, pathophysiology, or reasoning relevant to the question.
3. Select the single best answer from the listed options.
4. Provide a concise, evidence-based explanation for your choice.

---

Return your final output **only** in valid JSON format that matches this schema:

{{
  "explanation": "A brief medical reasoning explaining why this option is correct and others are not.",
  "answer_index": "The letter corresponding to the correct answer option (e.g., 'A', 'B', 'C', etc.)."
}}

---

## Input

**Question:**
{question}

**Options:**

{options}

---
""".strip()


SWE_RESPONSE_PROMPT = """
Du är en mycket kunnig och noggrann medicinsk expert.

Din uppgift är att analysera följande flervalsfråga och avgöra vilket alternativ som är det enda bästa svaret baserat på etablerad medicinsk kunskap och resonemang.

Följ dessa steg noggrant:
1. Läs frågan och alla tillgängliga svarsalternativ.
2. Identifiera viktiga kliniska fynd, patofysiologi eller relevant medicinskt resonemang.
3. Välj det enda bästa svaret bland de angivna alternativen.
4. Ge en kortfattad, evidensbaserad förklaring till ditt val.

---

Returnera ditt slutliga svar **endast** i giltigt JSON-format som följer detta schema:

{{
  "explanation": "En kort medicinsk motivering som förklarar varför detta alternativ är korrekt och varför de andra inte är det.",
  "answer_index": "Bokstaven som motsvarar det korrekta svarsalternativet (t.ex. 'A', 'B', 'C' osv.)."
}}

---

## Indata

**Fråga:**
{question}

**Alternativ:**

{options}

---
""".strip()


ANSWER_RESPONSE_PROMPT = """
You are a highly knowledgeable and careful medical expert.

Your task is to answer the following multiple-choice medical question by selecting the single best option.

Follow these steps carefully:
1. Read the question and all available answer options.
2. Select the single best answer from the listed options.

---

Return your final output **only** in valid JSON format that matches this schema:

{{
  "answer_index": "The letter corresponding to the correct answer option (e.g., 'A', 'B', 'C', etc.)."
}}

---

## Input

**Question:**
{question}

**Options:**

{options}

---
""".strip()


ANSWER_SWE_RESPONSE_PROMPT = """
Du är en mycket kunnig och noggrann medicinsk expert.

Din uppgift är att besvara följande flervalsfråga genom att välja det enda bästa alternativet.

Följ dessa steg noggrant:
1. Läs frågan och alla tillgängliga svarsalternativ.
2. Välj det enda bästa svaret bland de angivna alternativen.

---

Returnera ditt slutliga svar **endast** i giltigt JSON-format som följer detta schema:

{{
  "answer_index": "Bokstaven som motsvarar det korrekta svarsalternativet (t.ex. 'A', 'B', 'C' osv.)."
}}

---

## Indata

**Fråga:**
{question}

**Alternativ:**

{options}

---
""".strip()


__all__ = [
    "ANSWER_RESPONSE_PROMPT",
    "ANSWER_SWE_RESPONSE_PROMPT",
    "RESPONSE_PROMPT",
    "SWE_RESPONSE_PROMPT",
]
