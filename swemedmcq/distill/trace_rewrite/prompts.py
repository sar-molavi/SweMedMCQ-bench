"""Teacher prompt text for Swedish trace rewrite.

The prompt asks a teacher model to produce only a replacement Swedish reasoning
trace. The original final explanation and answer label are supplied as context
so the rewrite preserves the existing target instead of solving the MCQ again.
"""

from __future__ import annotations

from typing import Literal


PROMPT_VERSION = "trace-rewrite-v6"
"""Version tag stored with generated prompt/output artifacts."""

TRACE_REWRITE_VARIANTS = ("conservative_sv_rewrite",)
"""Supported trace-rewrite variants for this first implementation."""

TraceRewriteVariant = Literal["conservative_sv_rewrite"]
"""Type alias for valid trace-rewrite variant names."""


CONSERVATIVE_SV_REWRITE_PROMPT = """You are rewriting the reasoning trace for one already-generated Swedish medical multiple-choice SFT response.

### Task:

Write a natural Swedish reasoning trace that can replace the English source reasoning inside a <think>...</think> block. The rewritten trace must support the same final answer as the source response and must stay faithful to the reasoning content, level of detail, and option comparisons in the English source trace.

This is a rewrite, not a summary, not a mechanical translation, and not a new solution attempt. Source traces often contain artifacts such as translating the Swedish question into English, repeating the answer options in English, restating the task, or instructions like "return JSON". Remove those redundancies while preserving the medical reasoning.

The result should sound like the model is thinking through the answer in Swedish. It should not sound like the final explanation shown to the user. Prefer intermediate reasoning, comparison, and resolution, for example: "Jag behöver avgöra...", "Det talar för...", "Däremot passar...", "Alltså pekar detta mot...".

### Rules:

- Treat the answer label as fixed context; do not change the conclusion.
- Do not solve the question from scratch.
- Do not summarize the trace or shorten it for its own sake.
- Do not merely polish or copy the final explanation; write a reasoning trace that shows how the conclusion is reached.
- Preserve all medically relevant steps, clinical findings, uncertainty resolution, and option-by-option comparisons that are present in the source reasoning.
- Do not add medical facts, findings, assumptions, or differential diagnoses that are not supported by the source reasoning, question, options, or final explanation.
- Write only the replacement reasoning in Swedish.
- The output may use first-person reasoning when natural, but it must not mention this rewrite task.
- Do not include an English translation of the question or answer options.
- Do not mention that you are rewriting, translating, or cleaning a trace.
- Do not include task setup, meta-instructions, JSON instructions, or comments about the source response.
- If the source reasoning contains uncertainty or self-correction, keep the medically relevant resolution when it explains the final answer.
- Use as much space as needed to stay faithful, but omit redundant Swedish-to-English translation artifacts, repeated question/option restatement, and repeated boilerplate.
- Return only JSON matching this schema: {{"reasoning": "..."}}. Do not include explanation or answer_index.

### Example:

**Swedish question and answer options:**

```
**Fråga:**
Vilket fynd talar starkast för pneumoni hos en patient med hosta och feber?

**Alternativ:**

A) Lokaliserade krepitationer vid lungauskultation

B) Normalt CRP

C) Ensidig bensvullnad

D) Fotofobi
```

**English source reasoning:**

```
The question asks which finding best supports pneumonia. Cough and fever could be several infections. Localized crackles on lung exam suggest alveolar inflammation or consolidation. Normal CRP would not support bacterial pneumonia, unilateral leg swelling suggests DVT, and photophobia suggests meningitis. I was initially thinking fever alone, but among the choices localized crackles are best. Answer A. Return JSON.
```

**Existing Swedish final explanation:**

```
Lokaliserade krepitationer talar för inflammatoriska förändringar eller konsolidering i lungan och passar bäst med pneumoni.
```

**Answer label that must be preserved:**

```
A
```

**Correct output:**

```json
{{"reasoning":"Jag behöver skilja mellan fynd som faktiskt pekar mot pneumoni och fynd som passar bättre med andra tillstånd. Hosta och feber kan förekomma vid flera infektioner, så det mest stödjande fyndet är något som lokaliserar processen till lungorna. Lokaliserade krepitationer vid auskultation talar för fokal alveolär inflammation eller konsolidering och passar därför bäst med pneumoni. Normalt CRP skulle snarare försvaga misstanken om en bakteriell pneumoni, ensidig bensvullnad leder tanken mer mot trombos och fotofobi passar bättre med meningeal retning. Därför pekar jämförelsen tydligast på lokaliserade krepitationer."}}
```

### Input:

**Swedish question and answer options:**

```
{question}
```

**English source reasoning:**

```
{reasoning}
```

**Existing Swedish final explanation:**

```
{explanation}
```

**Answer label that must be preserved:**

```
{answer_index}
```
"""


"""Default teacher prompt for conservative Swedish reasoning rewrite.

The instruction text is in English, while the desired output reasoning is
Swedish. It requests JSON shaped as ``{"reasoning": "..."}`` and explicitly
forbids returning the explanation or answer label.
"""


PROMPTS = {
    "conservative_sv_rewrite": CONSERVATIVE_SV_REWRITE_PROMPT,
}
"""Mapping from variant name to teacher prompt template."""


__all__ = [
    "CONSERVATIVE_SV_REWRITE_PROMPT",
    "PROMPT_VERSION",
    "PROMPTS",
    "TRACE_REWRITE_VARIANTS",
    "TraceRewriteVariant",
]
