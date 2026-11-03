# Blinded review of the Swedish test translations

This folder holds the blinded review of the refined test translations described in the paper: the review sheets, the guidelines for the raters, the labels of the LLM raters, the answer keys, and the per-edit audit. The review decided which refinement layer became the benchmark: layer 4a (code-made edits only) rather than layer 4 (also the model-written edits); see [`../data/README.md`](../data/README.md#the-final-benchmark-layer-4a-on-the-core-items).

**Blinded** means that the rater does not know which text is the edited one. A pairwise row shows the English item and two Swedish versions, A and B, in random order: one is the text before the edits, the other the text after them. The rater picks the better translation; only the key (`keys/key.json`), kept apart from the sheets, says which version was edited, and it is used only for scoring. The other sheets likewise do not show whether a finding was accepted or dismissed by the loop. The raters in the paper were LLMs (Claude Opus 5.5, a model family that the refinement loop does not use). The same sheets can be given to human raters, and their answers scored with the same code.

## Contents

| Path | What it is |
|---|---|
| `GUIDELINES.md` | What to judge and the allowed answers, sheet by sheet |
| `sheets/` | The items, in the batches the raters saw |
| `sheets_full/` | Sheets 1–3 as one file each, and `expert_review.xlsx` with the three sheets in one workbook |
| `answers_template/` | One empty answer file per sheet, with the ids filled in |
| `llm_labels/` | The answers of the LLM raters: pairwise raters `a` and `b`, the adjudication of their disagreements, the other sheets, `disagreements.txt` and `summary.json` (the scored result) |
| `keys/` | The answer keys: which version of each pairwise row is the edit, the strata and inclusion probabilities (`key.json`, sheets 1–3), and the keys of sheets 4–5 (`key_disputed.json`, `key_source.json`, `disputed_strata_pop.json`) |
| `edit_audit/` | The per-edit audit of 129 accepted edits (`labels.csv`; the "Check" columns of the edit tables in the paper) |

The Swedish style guide the raters used is `data/resources/style_guide_sv.md`.

## The sheets

| Sheet | Files | Items | Question for the rater |
|---|---|---|---|
| 1 Pairwise | `1_pairwise_1..4` | 200 | Which of two Swedish versions, A or B, translates the English better? A and B are in random order. |
| 2 Findings | `2_findings_1..4` | 371 | Is the flagged problem a real error in the Swedish? |
| 3 Control | `3_control_1..3` | 150 | Does this Swedish item contain an error a medical editor would correct? |
| 4 Flagged items | `4_disputed_1..2` | 106 | Are the flagged problems real, and can the item stay in the benchmark? |
| 5 English source | `5_source_1` | 55 | Is the claimed problem in the English source real, and is the keyed answer still unambiguous? |

- **Sheet 1** gives the primary endpoint, the harm rate per tier. Tier-1 items compare the draft with the tier-1 text (edits made by deterministic rules); tier-2 items compare the tier-1 text with the full refined text (edits written by a model). Both versions always have the same typography and number layers.
- **Sheets 2–3** measure the precision and the recall of the loop's findings.
- **Sheets 4–5** check the items the loop set aside and the claimed problems in the English source. They are extra checks; the paper reports sheets 1–3.

Sheets 1–3 are built from the refinement run with a fixed seed (20261006):

```bash
python -m swemedmcq.refine.evaluation.expert_sheet make --run-dir runs/refine/full
```

This writes the sheets and `expert/key.json` into the run folder (it needs the run's `prep/` and `release/` folders). The key records which version is which, the strata and the inclusion probabilities; it never goes into the sheets. With the released run, the default seed reproduces `sheets_full/` and `keys/key.json` byte for byte. `--seed <n>` draws a new sample with a new A/B order, for example for new raters.

**Raters.** Sheet 1 needs two independent raters per item (answer files `_a` and `_b`). A third pass settles their disagreements in `1_pairwise_adjudicated.csv`, with the same columns. The other sheets need one rater each. A rater should see only `GUIDELINES.md`, the style guide, `sheets/` (or `sheets_full/`) and `answers_template/`, never `llm_labels/`, `edit_audit/` or `keys/`. The released keys belong to the released sheets; give new raters sheets drawn with a new seed.

## Scoring

Put the filled answer files in one folder, with the file names of `answers_template/` plus `1_pairwise_adjudicated.csv`, and run:

```bash
python -m swemedmcq.review.score_review --labels <labels folder> --keys review/keys --out runs/review/<name>
```

`--labels` defaults to `review/llm_labels` and `--keys` to `review/keys` (`key.json` for sheets 1–3; `key_disputed.json`, `key_source.json` and `disputed_strata_pop.json` for sheets 4–5). `summary.json` in the `--out` folder then gives:
- the agreement between the two pairwise raters, with Cohen's κ;
- the harm rate per tier and per confidence level, with 95% Wilson intervals, the inverse-probability-weighted harm rate and the one-sided sign test;
- the precision of the accepted findings and the share of real errors among the dismissed findings;
- the error rates of the control sheet, with the severities;
- the summaries of sheets 4–5.

Scoring `llm_labels/` this way (`python -m swemedmcq.review.score_review --out runs/review/llm`) reproduces `llm_labels/summary.json` exactly. A tier passes if the upper Wilson bound of its harm rate is below 10%; the benchmark uses the full refined layer only if tier 2 passes, and the tier-1 layer otherwise.

**LLM raters, results.** The two pairwise raters agreed on 170 of 200 items (85%, κ = 0.78). No tier-1 item was worse than the draft. Tier-2 items were better 4.3 times as often as worse, but the upper bound of their harm rate, 17.1%, is above 10%, so the benchmark uses the tier-1 layer.

## Per-edit audit

Besides the item-level review, every accepted edit class was audited edit by edit: 129 accepted edits, sampled by class, each shown blind next to the text of its field just before the edit, with the English field as reference. The rater chose the better version or *equal*. The shares per class, weighted by the number of edits per class, give the "Check" columns of the paper's edit tables.

```bash
# Rebuild the blinded sample (key.json and three batches) from the released edit history
python -m swemedmcq.review.edit_audit make --out runs/review/edit_audit
# Score the labels
python -m swemedmcq.review.edit_audit score --labels review/edit_audit/labels.csv
```

`make` reproduces the 129 items of `edit_audit/labels.csv` exactly. `labels.csv` has one row per sampled edit: the item and field, the edit class (`stratum`) and its number of accepted edits, the confidence level, which version is the edit (`edit_is`), the rater's verdict, outcome, severity and reason, and the texts shown (`en`, `A`, `B`).
