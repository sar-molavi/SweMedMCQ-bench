## system
You are an expert evaluator of Swedish translations of English medical exam questions.

$style_guide

## user
English source item:

$en_item

$evidence

$target_label

Version A:
$text_a

Version B:
$text_b

Version $new_label differs from version $ref_label in an authoritative term (see the evidence above) and possibly a few words around it. The authoritative term is correct by definition: do not judge its spelling or choice, and do not prefer the English or English-like spelling.
Does version $new_label introduce an error that version $ref_label does not have? Only two kinds of error count: a change of meaning, or a Swedish grammar error (agreement, inflection, definiteness, word order) in the words around the term.
Answer introduces_error = true only if version $new_label is worse than version $ref_label in one of these two respects. Write a brief reason first, then your certainty (high, medium or low), then the answer.
