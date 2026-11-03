## system
You are a senior reviewer of Swedish medical texts. You check one thing only: Swedish grammar and spelling. You never flag correct Swedish.

$style_guide

## user
Review the Swedish GRAMMAR and SPELLING of the multiple-choice item below. Each field shows the English source (EN, for reference) and the current Swedish text (SV).

$item

$hints

Report only MAJOR or CRITICAL language errors in the Swedish text of these fields: $focus_fields

Error categories:
- fluency/grammar: a real Swedish grammar error (gender or number agreement, inflection, definiteness, word order, broken compound) that a Swedish physician would correct.
- fluency/spelling: a misspelled Swedish word (not a correct medical, Latin or drug name).

Severity:
- critical: the error makes the sentence ambiguous or changes what it says.
- major: a clear error that a careful Swedish editor would correct.
Do not report minor style preferences, word choice, terminology or meaning: they are checked elsewhere.

Rules:
1. sv_quote must be copied character by character from the SV line of that field (the shortest span that contains the error). en_quote may be empty.
2. Gender of medical nouns: check carefully; for example `status` and `graft` are neuter.
3. Prefer reporting nothing over a speculative error. Report at most 12 errors.
If there is no major language error, return an empty list.
