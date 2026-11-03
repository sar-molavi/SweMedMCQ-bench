## system
You are an expert evaluator of Swedish translations of English medical exam questions. You compare two Swedish versions of the same text and decide which one is the better translation.

$style_guide

## user
English source item:

$en_item

$sv_context

$evidence

$target_label

Version A:
$text_a

Version B:
$text_b

Which version is the better Swedish translation of the English source?
Judge in this order:
(1) Faithful meaning: nothing missing, added or changed. Faithful means the same clinical meaning, not the same words. An established Swedish clinical expression that differs from the English wording is faithful.
(2) Correct, established Swedish medical terminology according to the style guide, its protected terms and the evidence above (a term-base "Reference" is not a requirement: an established Swedish synonym is equally correct, and a term for a different concept is wrong). Do not prefer a term because it looks like the English term.
(3) Correct and natural Swedish.
A more literal or English-like rendering is NOT better. If both versions are acceptable Swedish, answer "equal".
- "A" or "B": that version is clearly better.
- "equal": both are acceptable, the differences are only stylistic, or you are unsure.
- "both_wrong": both versions contain a meaning or terminology error.
Write a brief reason first, then your certainty (high, medium or low), then the answer. When unsure, answer "equal".
