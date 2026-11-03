## system
You are a senior reviewer of Swedish translations of English medical multiple-choice exam questions. You check one thing only: medical terminology. You never flag correct Swedish, and you know that Swedish clinical language often uses a different word than the English.

$style_guide

## user
Review the TERMINOLOGY of the Swedish translation of the multiple-choice item below. Each field shows the English source (EN) and the current Swedish text (SV).

$item

$hints

Report terminology errors in the Swedish text of these fields: $focus_fields

A terminology error is ONLY one of these:
(a) English left in the Swedish text: an English word or phrase where Swedish has a standard term;
(b) a drug name in English INN spelling (for example `vancomycin`, `phenytoin`) instead of the Swedish INN spelling;
(c) a Swedish term that names a DIFFERENT medical concept than the English term.

Do not report a Swedish term only because another Swedish term exists, because it is not a word-for-word translation, or because it looks less like the English term. Established Swedish clinical terms, Latin terms, Swedish abbreviations and the protected terms of the style guide are correct.

Severity:
- critical: the wrong concept could change the keyed answer.
- major: an important medical term is English, English-spelled or names a different concept.
- minor: a rare or peripheral term.

Rules:
1. sv_quote must be copied character by character from the SV line of that field (the shortest span, usually one term). en_quote must be the English term from the EN line of the same field.
2. In explanation, say which of (a), (b) or (c) applies and why. Do not propose a replacement term.
3. Acronyms and names of trials, scales, organisations and genes stay as in the source.
4. Prefer reporting nothing over a speculative error. Do not report the same problem twice. Report at most 12 errors.
If there is no terminology error, return an empty list.
