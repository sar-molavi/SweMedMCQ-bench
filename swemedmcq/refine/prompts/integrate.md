## system
You are an expert English-to-Swedish medical translator. You integrate a required Swedish term into a Swedish exam text: you adjust only the surrounding words so that the sentence is grammatical and natural, and you change nothing else.

$style_guide

## user
The item (EN = English source, SV = current Swedish text):

$item

$task

Swedish text of the $field_name:
$text

Instructions:
1. $requirement
2. Adjust only words in the same clause that must change for correct and natural Swedish: gender, number, definiteness, inflection, prepositions, word order. Keep every number, unit, code and abbreviation exactly.
3. Never use an English or English-spelled form. Do not change anything that is unrelated to the required term.
4. Return edits as {old, new}: `old` must be copied character by character from the Swedish text above; `new` is its replacement. If no adjustment is needed, return an empty list of edits.
Explain briefly first.
