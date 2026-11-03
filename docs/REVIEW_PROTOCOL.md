# Review protocol

The refined translations were reviewed blind before the benchmark layer was chosen. Everything needed to repeat or re-score the review is in [`review/`](../review/README.md): the sheets, the rater guidelines, the answer templates, the labels of the LLM raters (Claude Opus 5.5), the answer keys and the per-edit audit.

**Design.**

- **Sheet 1, pairwise (primary endpoint).** 200 items. The rater sees two Swedish versions in random order and picks the better translation of the English item. Tier-1 items compare the draft with the tier-1 text; tier-2 items compare the tier-1 text with the fully refined text. Two independent raters; a third pass settles their disagreements.
- **Sheet 2, findings.** 371 findings of the loop, accepted and dismissed. Is each a real error?
- **Sheet 3, control.** 150 items. Is there an error a medical editor would fix?
- **Sheets 4–5.** The items the loop set aside, and the claimed problems in the English source (extra checks; the paper reports sheets 1–3).

**Scoring and decision rule.**

- The harm rate is the share of items where the edited version is worse. Each tier is reported with a 95% Wilson interval, an inverse-probability-weighted estimate and a one-sided sign test.
- The rule was set before the review: a tier is accepted if the upper Wilson bound of its harm rate is below 10%. The benchmark uses the fully refined layer only if tier 2 passes, and the tier-1 layer otherwise.
- With the LLM raters, tier 1 had no harmful edit. Tier 2 improved 77 items and harmed 18 of 160; its upper bound of 17.1% is above 10%, so the benchmark uses the tier-1 layer.

**Re-scoring from the repository.**

```bash
python -m swemedmcq.review.score_review --out runs/review/llm      # reproduces review/llm_labels/summary.json
python -m swemedmcq.review.edit_audit score                        # per-edit audit by edit class and tier
```

**New sheets.** To draw a new sample (for example for human raters), build sheets from a refinement run with another seed. The new key decides the A/B order, so the released keys reveal nothing about the new sheets:

```bash
python -m swemedmcq.refine.evaluation.expert_sheet make --run-dir runs/refine/full --seed <new seed>
```
