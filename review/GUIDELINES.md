# Rater guidelines: Swedish medical translation review

You are a bilingual (English/Swedish) medical translation reviewer with clinical
knowledge. Thank you for helping. The items are multiple-choice questions from English medical exams and
benchmarks, translated into Swedish for a benchmark. The Swedish must keep the exact
clinical meaning of the English, so that the same option is correct in both
languages, and read as natural, professional Swedish medical language (as in
Läkartidningen, FASS, Socialstyrelsen, 1177 for professionals, Svensk MeSH).

Reference: `style_guide_sv.md` (the Swedish style conventions the benchmark follows,
e.g. decimal comma, Swedish INN drug names, unit spacing). Use it as a guide, not as law:
an established Swedish clinical term is correct even if a different form exists.

Priorities, most important first:
1. Accuracy: meaning, numbers, units, negations, laterality, which option is correct.
2. Terminology: the term a Swedish clinician would use; Swedish drug names (INN in Swedish spelling).
3. Language: grammar, gender/inflection, spelling, English leftovers.
4. Style conventions (locale, typography) matter least.

General rules
- Judge only what the text says. Use only the sheet, these guidelines and the style
  guide; please do not look up the items in other sources or discuss them with other raters.
- Be precise and conservative. "Equal" is the right verdict when the differences are
  immaterial, both acceptable, or a matter of taste.
- Write short comments (max 25 words) that name the span you base the verdict on.
- Write your answers in the matching file in `answers_template/` (same name as the sheet;
  pairwise sheets have one file per rater, `_a` or `_b`). Keep the ids and the column
  names; fill one row per item. Any spreadsheet program works; save as CSV (UTF-8).

## Sheet 1: pairwise (A vs B)
Each row: the English item, two Swedish versions A and B, and a list of the places where
they differ. Decide which Swedish version is the better translation of the English.
Columns: sample_id, verdict, meaning_changed, severity, comment
- verdict: `A better` | `B better` | `equal` | `both wrong`
  (`both wrong`: both versions have a real error in the differing span.)
- meaning_changed: `y` if A and B differ in clinical meaning, else `n`.
- severity: how much the difference matters: `minor` (style, spelling, a term both
  readers would understand), `major` (clearly wrong or unidiomatic term, misleading
  grammar), `critical` (changes the answer or the clinical meaning).

## Sheet 2: findings
Each row: an English field, its Swedish translation, a flagged Swedish span and the claimed
problem. Decide whether the claimed problem is a real error in the Swedish.
Columns: finding_uid, label, severity, comment
- label: `real error` | `not an error`
- severity (only for real errors, else empty): `minor` | `major` | `critical`

## Sheet 3: control
Each row: an English item and its Swedish translation. Decide whether the Swedish contains
a translation error that a Swedish medical editor would correct.
Columns: control_id, has_error, which_error, severity, comment
- has_error: `y` | `n`
- which_error: quote the Swedish span + a short description (empty if n).
- severity: `minor` | `major` | `critical` (empty if n). Pure locale/typography
  conventions alone are `minor`.

## Sheet 4: flagged items
Each row: an English item, its Swedish translation, and one or more problems that automatic
checkers flagged in the Swedish. Many flags are false alarms. Check each flagged problem,
then judge the item as a whole.
Columns: item_id, findings_labels, item_ok_for_benchmark, worst_severity, comment
- findings_labels: one entry per flagged problem, in order, separated by "; ",
  each `(n) real/minor`, `(n) real/major`, `(n) real/critical` or `(n) not`.
- Also look for any other translation error the flags missed and mention it in the comment.
- item_ok_for_benchmark: `y` if a Swedish reader would understand the item exactly as the
  English reader does and the same option is correct (minor flaws allowed); `n` if an error
  could change the reading, the difficulty or the answer.
- worst_severity: the most severe real error in the item (`none` | `minor` | `major` | `critical`).

## Sheet 5: English source issues
Each row: an English item, its answer key, and a claimed problem in the English source itself
(typo, ambiguity, wrong key). Judge the English only.
Columns: item_id, issue_real, answer_still_unambiguous, comment
- issue_real: `y` if the claimed problem really exists in the English, else `n`.
- answer_still_unambiguous: `y` if an expert would still pick the keyed answer as the single
  best option despite the problem, `n` if the item is ambiguous, unanswerable or mis-keyed.
