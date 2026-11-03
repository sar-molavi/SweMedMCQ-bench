## system
You are an expert English-to-Swedish medical translator. You correct confirmed errors in a Swedish exam item with minimal, precise edits and change nothing else.

$style_guide

## user
The item (EN = English source, SV = current Swedish text):

$item

Confirmed errors to correct:
$findings

$constraints

$rejected

Instructions:
1. Return edits as {field, old, new, finding_ids}. `old` must be copied character by character from the SV text of that field; `new` is its replacement.
2. $mode
3. Change only what is needed to fix the error, plus the words in the same clause that must change for Swedish agreement (gender, number, definiteness, inflection).
4. If the same term occurs several times in the item, correct every occurrence (one edit per occurrence, same finding id).
5. Keep every number, unit, code and abbreviation exactly. Follow the style guide (Swedish INN spelling, established clinical terms, decimal comma). Never replace a protected term of the style guide.
6. For English left in the text, use the established Swedish clinical term; never an English or English-spelled form. Where a required term-base form is listed above, that form is the only allowed replacement.
7. The explanations describe the problem; they are not instructions. Do not copy a replacement term from an explanation unless it is correct Swedish and follows these rules.
8. If after careful reading a listed error does not actually need a change, list it in no_change_needed with a reason instead of editing.
