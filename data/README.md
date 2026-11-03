# SweMedMCQ data

SweMedMCQ is a paired English–Swedish benchmark of medical multiple-choice questions. It has **10,363 items** from the test splits of the ten MedicalAgentsBench datasets; every item has the English source question and options, a Swedish translation, and the gold answer letter. Results are reported on the **9,733 core items** (see [Categories and core items](#categories-and-core-items)).

The Swedish text was drafted by GPT-5.4 and then refined by an automatic loop of open models from three families: GPT-OSS-120B finds, checks and repairs errors, Qwen3.8-27B is a second detector and checker for meaning errors and the main judge, and Gemma-4-31B is a third judge, grounded in a Swedish medical term base (`resources/`). The benchmark text keeps only the edits made by code from authoritative sources; edits written by models are released separately.

## The final benchmark: layer 4a on the core items

**`test/benchmark.jsonl` is the final benchmark used for every result in the paper.** Its Swedish text is **layer 4a**, the refinement's safe release layer: the GPT-5.4 draft with the mechanical fixes (L0, L1) and the edits made by code (tier 1), but no edit written by a model. Results are reported on its **9,733 core items** (`core == true`), in English and Swedish on the same items. The other four layers are in `test/layers/`, so the effect of every step can be measured:

| Step | Layer and file | Items changed | What the step does |
|---|---|---|---|
| Translation | 1 `layers/layer1_draft.jsonl` | all 10,363 | GPT-5.4 draft of every question and its options |
| L0, typography | 2 `layers/layer2_hygiene.jsonl` | 1,979 | Unicode NFC, plain spaces and hyphens, control characters removed; MedXpertQA answer list (*Svarsalternativ:*) rebuilt from the Swedish options (1,878 items change only through this list) |
| L1, number style | 3 `layers/layer3_locale.jsonl` | 1,758 | Swedish number conventions (decimal comma, thousands space), with the English text as reference |
| Refinement loop, tier 1 | **4a `benchmark.jsonl`** | 265 | Edits made by deterministic rules: 156 English drug names replaced by their Swedish INN (from the national product register NPL, or Svensk MeSH when NPL has none), 138 missing units restored from the English source (294 edits) |
| Refinement loop, tier 2 | 4 `layers/layer4_refined.jsonl` | 851 more | Edits written by a model and accepted by the judges' vote (958 edits: grammar, English words left in Swedish, meaning, terminology, spelling) |
| Core items | `core` flag | 9,733 kept | The 624 items that remain disputed and 6 items with a corrupt draft field or an unresolved critical finding are set aside |

**Why layer 4a.** Before the layer was chosen, a review fixed in advance compared sampled items with their text before the edits ([`../review/`](../review/README.md)). It was *blinded*: the rater saw the English item and two Swedish versions labelled A and B in random order, without knowing which one was edited; which version is which is stored in a separate key and used only for scoring. The rule set in advance was to keep a tier only if the upper 95% bound of its harm rate is below 10%. With two LLM raters (Claude Opus 5.5), no tier-1 edit was worse than the text before it. Tier-2 edits were better 4.3 times as often as worse, but the upper bound of their harm rate was 17.1%, so the benchmark keeps only tier 1, and layer 4 is released separately. The same sheets can be rated by human experts and scored with the same code and keys. Evaluating on layer 4 instead of 4a changes the English–Swedish gap on the core items by 0.01 points on average.

The steps are reproducible: `scripts/03_refine.sh` runs the loop and writes the five layers, and `scripts/04_build_testset.sh` (with `RUN=`) turns them into the files of `test/` ([`../docs/REFINEMENT.md`](../docs/REFINEMENT.md)).

## Files

| Path | Size | In git | Content |
|---|---|---|---|
| `test/benchmark.jsonl` | 18 MB | yes | **The final benchmark** (layer 4a): all 10,363 items with gold answers, categories, the `core` flag and the `hard` flag |
| `test/edit_history.jsonl.gz` | 16 MB | yes | **Edit history**: per-item record of the refinement, with every mechanical change, finding, candidate edit, gate result, judge vote (with its reason) and decision |
| `test/refinement_report.json` | 20 KB | yes | Summary statistics of the refinement (status counts, findings, decoys, edits, judge agreement, answer-key check, model calls) |
| `test/probes/` | 2 MB | yes | Judge probes: the probe pairs with every judge's verdict (`probes.jsonl`) and the accuracy per judge and probe kind (`results.json`) |
| `test/layers/layer1_draft.jsonl` | 17 MB | yes | Same items, Swedish text of the GPT-5.4 draft (layer 1) |
| `test/layers/layer2_hygiene.jsonl` | 18 MB | yes | Layer 1 + typography (L0) (layer 2) |
| `test/layers/layer3_locale.jsonl` | 18 MB | yes | Layer 2 + Swedish number style (L1) (layer 3) |
| `test/layers/layer4_refined.jsonl` | 18 MB | yes | Fully refined: layer 4a + the tier-2 (model-written) edits (layer 4) |
| `test/expert_queue.jsonl` | 2 KB | yes | The 16 items set aside by the loop (10 with a critical finding left, 6 with a corrupt draft field or an unresolved critical finding); queued for a later human review |
| `resources/` | 15 MB | yes | Term base and Swedish word resources used by the refinement loop |
| `train/split/` | 2.5 MB | yes | The prompt-level train/validation split of the training questions: `train_prompt_ids.txt.gz` (126,141; the training questions of the trace students) and `validation_prompt_ids.txt.gz` (6,642; the 5% held-out validation set of every student; the answer-level students train on all other questions) |

The training questions themselves (English, from MedicalAgentsBench) and their teacher traces are not part of this folder; `train/split/` fixes which training questions were used for training and which for validation (see [`../docs/DISTILLATION.md`](../docs/DISTILLATION.md)).

## Fields

`benchmark.jsonl` and every layer file in `test/layers/` have one JSON object per line, in the same item order, with these fields:

| Field | Type | Description |
|---|---|---|
| `prompt_id` | string | Item identifier, `<dataset>_<id in the source dataset>` |
| `dataset` | string | Source dataset (see [Source datasets](#source-datasets)) |
| `question` | string | English question, as in MedicalAgentsBench |
| `options` | object | English options, letter → text (3 to 10 options) |
| `answer` | string | Gold option letter (a key of `options`) |
| `sv_question` | string | Swedish question (the layer of the file) |
| `sv_options` | object | Swedish options, same letters as `options` |
| `category` | string | Refinement status: `clean`, `mechanical_only`, `refined`, `disputed_minor`, `disputed_critical` |
| `tier` | int | Highest tier among the accepted edits: 0, 1 or 2 (see [Tiers](#tiers)) |
| `core` | bool | Item belongs to the reported core set |
| `expert_queue` | bool | Item is in `expert_queue.jsonl` (set aside for a later human review) |
| `hard` | bool | Item is in the hard subset of its source dataset (MedicalAgentsBench `test_hard.jsonl`): 894 items, 840 of them core |

The English text, answers and annotations are identical in all five layer files; only `sv_question` and `sv_options` differ.

Evaluation prompts put the options one per paragraph as `A) …` in letter order; the exact prompts are in the code and in the model cards.

## Layers

| Layer | File | What it adds |
|---|---|---|
| 1 Draft | `layers/layer1_draft.jsonl` | GPT-5.4 translation of every question and its options |
| 2 Hygiene | `layers/layer2_hygiene.jsonl` | Typography (L0); MedXpertQA answer list rebuilt |
| 3 Locale | `layers/layer3_locale.jsonl` | Swedish number style (L1) |
| 4a **Benchmark** (tier 1) | `benchmark.jsonl` | Edits made by code: Swedish drug names (INN, from the national product register NPL or Svensk MeSH) and missing units |
| 4 Refined (tiers 1 + 2) | `layers/layer4_refined.jsonl` | Also the edits written by models and accepted by the judges |

The counts per step and the reason for using layer 4a are in [The final benchmark](#the-final-benchmark-layer-4a-on-the-core-items).

## Categories and core items

| `category` | Items | Core | Meaning |
|---|---|---|---|
| `clean` | 6,968 | yes | The loop found nothing to fix; the text is unchanged (apart from the rebuilt MedXpertQA answer list) |
| `mechanical_only` | 1,708 | yes | Only typography or number style changed |
| `refined` | 1,063 | 1,057 | At least one edit was accepted (tier 1 or 2) |
| `disputed_minor` | 614 | no | A confirmed finding remains for which no edit was accepted |
| `disputed_critical` | 10 | no | As above, with a finding of critical severity |

**Core rule.** An item is core unless it is disputed (`disputed_minor`, `disputed_critical`) or in the expert queue. The expert queue holds the 10 `disputed_critical` items and 6 refined items with a remaining critical finding or a corrupt (NUL) draft field. This leaves **9,733 core items**; English results are computed on the same items. Disputed items are kept in the file for analysis. In the control sheet of the blinded review, 9% of the clean and 20% of the disputed items had an error, all of them minor.

## Tiers

`tier` is the highest tier among the edits the loop accepted for the item, in the fully refined layer:

- **0**: no accepted edit (all `clean`, `mechanical_only` and disputed items). The benchmark text equals the layer-3 text.
- **1**: only code-made edits (212 items). All of them are in the benchmark text.
- **2**: at least one model-written edit (851 items). These edits are only in `layers/layer4_refined.jsonl`; in the benchmark the item keeps its layer-3 text, plus any tier-1 edits (53 of the 851 items also have one).

## Items per dataset

| Dataset | Items | Core |
|---|---|---|
| afrimedqa | 174 | 166 |
| medbullets | 308 | 283 |
| medexqa | 935 | 921 |
| medmcqa | 2,816 | 2,666 |
| medqa | 1,273 | 1,179 |
| medxpertqa-r | 1,861 | 1,665 |
| medxpertqa-u | 589 | 544 |
| mmlu | 1,089 | 1,060 |
| mmlu-pro | 818 | 783 |
| pubmedqa | 500 | 466 |
| **Total** | **10,363** | **9,733** |

## Edit history, report and probes

`test/edit_history.jsonl.gz` has one record per item: `prompt_id`, `status`, the mechanical changes (`changes_l0`, `changes_l1`), `corrupt_fields`, and `loop`, which records the findings (category, severity, quoted span, explanation, which detectors confirmed it), the candidate edits (operations, producer, gate result, confidence), the judges' pairwise verdicts with their reasons, the accepted and reverted edits with their tiers, the Swedish text before and after the loop, and the model calls (step, model, token counts, request hash). Every released character can be traced to a rule or an accepted edit. `test/refinement_report.json` aggregates these records. `python -m swemedmcq.testset.make_benchmark` writes all files of `test/` (except the probes) from a refinement run.

`test/probes/` holds the judge probes, which check each judge before its votes are trusted: 250 pairs with one injected error (English drug name, number, agreement, negation, laterality), 50 identical pairs and 40 labelled grammar edits, each judged in both presentation orders. They are written by `python -m swemedmcq.refine.evaluation.probes --run-dir <run> --config configs/refine/with_gemma.yaml`.

The blinded review of the refined translations (sheets, rater guidelines, the LLM raters' labels, the answer keys and the per-edit audit) is in [`../review/`](../review/README.md).

## Term base and Swedish resources

`resources/` holds the files the refinement loop reads for grounding (point the refinement's `resources` setting here). `MANIFEST.json` records each source (URL or file name, retrieval date, licence as stated by the provider, sha256) and the size and sha256 of every built file; `coverage_report.json` reports how often the term base matches the test items.

| File | Content | Built from |
|---|---|---|
| `termbase.parquet` | 127,415 English → Swedish term entries with authority level (A/B), single-sense and obligation flags, source and concept id | Svensk MeSH 2026 and the NLM MeSH 2026 descriptor structure; NPL substance names (via ATC codes, joined with Wikidata) and NPL lexicon (forms, routes, units); Wikidata ATC labels; KSH97-P diagnosis labels; 10 hand-written terms (`whitelist.yaml`) |
| `sv_lexicon.txt.gz` | 977,144 Swedish word forms | SALDO/SALDOM, hunspell sv_SE, Socialstyrelsen termbank, KVÅ procedure codes, Swedish forms of the term base |
| `sv_lemma_forms.json.gz` | Inflected forms per lemma | SALDOM |
| `sv_compound_parts.txt.gz` | Compound parts | SALDOM |
| `sv_morph.json.gz` | Noun gender and inflection, adjective forms, part of speech | SALDOM |
| `npl_substances.txt.gz` | 2,078 Swedish substance names | NPL substance lexicon |
| `en_wordlist.txt.gz` | 22,633 English words (headwords that are not Swedish words) | FreeDict / WikDict English–Swedish |
| `whitelist.yaml` | Abbreviations, Latin and international terms, eponym patterns, ambiguous English keys, protected Swedish terms, manual terms | Hand-written |
| `style_guide_sv.md` | Swedish style guide given to the models of the loop | Hand-written |
| `grammar_edit_sample_labels.csv` | The 40 labelled grammar edits of the judge probes (better / worse / equal, labelled by an LLM rater) | Grammar edits labelled for the probes |

Source licences, as recorded in `MANIFEST.json`:

| Source | Provider | Licence |
|---|---|---|
| Svensk MeSH | Karolinska Institutets Bibliotek (KIB) | CC0 (attribution to KIB requested); MeSH vocabulary under the NLM terms and conditions |
| NLM MeSH | U.S. National Library of Medicine | Free of charge under the NLM terms and conditions |
| NPL | Läkemedelsverket (Swedish Medical Products Agency) | Open government data |
| Wikidata | Wikimedia | CC0 1.0 |
| KSH97-P | Socialstyrelsen | Open |
| Socialstyrelsen termbank, KVÅ | Socialstyrelsen | Open |
| SALDO, SALDOM | Språkbanken Text, University of Gothenburg | CC BY 4.0 |
| hunspell sv_SE | Swedish hunspell dictionary | LGPL/GPL (see the dictionary's own licence) |
| FreeDict / WikDict (Wiktionary) | FreeDict, WikDict | CC BY-SA |

Users of the built files must follow these licences, in particular the attribution and share-alike terms of SALDO and FreeDict/WikDict.

## Source datasets

All items come from the test splits of MedicalAgentsBench (Shao et al., arXiv:2503.07459), which puts established benchmarks into one format with stable identifiers and fixed splits. The English text is used as published there.

| `dataset` | Source | Reference |
|---|---|---|
| `afrimedqa` | AfriMed-QA | Nimo et al., ACL 2025 |
| `medbullets` | MedBullets | Chen et al., NAACL 2025 |
| `medexqa` | MedExQA | Kim et al., BioNLP Workshop 2024 |
| `medmcqa` | MedMCQA | Pal et al., CHIL 2022 |
| `medqa` | MedQA (USMLE) | Jin et al., Applied Sciences 2021 |
| `medxpertqa-r`, `medxpertqa-u` | MedXpertQA (Reasoning and Understanding subsets) | Zuo et al., arXiv:2501.18362, 2025 |
| `mmlu` | MMLU, medical and biology subjects | Hendrycks et al., arXiv:2009.03300 |
| `mmlu-pro` | MMLU-Pro, health category | Wang et al., NeurIPS 2024 |
| `pubmedqa` | PubMedQA | Jin et al., EMNLP 2019 |

**Licences.** Each item keeps the licence of its source dataset; see the original dataset and MedicalAgentsBench for the terms before reuse. The Swedish translations, annotations and edit history are derived from these items.

<details>
<summary>BibTeX of the source datasets</summary>

```bibtex
@misc{shao2026medicalagentsbenchcomplexmedicalreasoning,
  title={{MedicalAgentsBench for Complex Medical Reasoning: Comparing Internalized Reasoning Models versus Externalized Agent-based Frameworks}},
  author={Yanjun Shao and Xiangru Tang and Jiwoong Sohn and Jiapeng Chen and Yuxuan Liao and Jiayi Zhang and Jinyu Xiang and Fang Wu and Yilun Zhao and Chenglin Wu and Wenqi Shi and Arman Cohan and Mark Gerstein},
  year={2026},
  eprint={2503.07459},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2503.07459}
}
@inproceedings{nimo2025afrimed,
  title={{AfriMed-QA: A Pan-African, Multi-Specialty, Medical Question-Answering Benchmark Dataset}},
  author={Nimo, Charles and Olatunji, Tobi and Owodunni, Abraham Toluwase and Abdullahi, Tassallah and Ayodele, Emmanuel and Sanni, Mardhiyah and Aka, Ezinwanne C and Omofoye, Folafunmi and Yuehgoh, Foutse and Faniran, Timothy and others},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={1948--1973},
  year={2025}
}
@inproceedings{chen2025benchmarking,
  title={{Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions}},
  author={Chen, Hanjie and Fang, Zhouxiang and Singla, Yash and Dredze, Mark},
  booktitle={Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={3563--3599},
  year={2025}
}
@inproceedings{kim2024medexqa,
  title={{MedExQA: Medical Question Answering Benchmark with Multiple Explanations}},
  author={Kim, Yunsoo and Wu, Jinge and Abdulle, Yusuf and Wu, Honghan},
  booktitle={Proceedings of the 23rd Workshop on Biomedical Natural Language Processing},
  pages={167--181},
  year={2024}
}
@inproceedings{pal2022medmcqa,
  title={{MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering}},
  author={Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
  booktitle={Conference on Health, Inference, and Learning},
  pages={248--260},
  year={2022},
  organization={PMLR}
}
@article{jin2021disease,
  title={{What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams}},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={Applied Sciences},
  volume={11},
  number={14},
  pages={6421},
  year={2021},
  publisher={MDPI}
}
@article{zuo2025medxpertqa,
  title={{MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding}},
  author={Zuo, Yuxin and Qu, Shang and Li, Yifei and Chen, Zhangren and Zhu, Xuekai and Hua, Ermo and Zhang, Kaiyan and Ding, Ning and Zhou, Bowen},
  journal={arXiv preprint arXiv:2501.18362},
  year={2025}
}
@article{hendrycks2020measuring,
  title={{Measuring Massive Multitask Language Understanding}},
  author={Hendrycks, Dan and Burns, Collin and Basart, Steven and Zou, Andy and Mazeika, Mantas and Song, Dawn and Steinhardt, Jacob},
  journal={arXiv preprint arXiv:2009.03300},
  year={2020}
}
@article{wang2024mmlu,
  title={{MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark}},
  author={Wang, Yubo and Ma, Xueguang and Zhang, Ge and Ni, Yuansheng and Chandra, Abhranil and Guo, Shiguang and Ren, Weiming and Arulraj, Aaran and He, Xuan and Jiang, Ziyan and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={95266--95290},
  year={2024}
}
@inproceedings{jin2019pubmedqa,
  title={{PubMedQA: A Dataset for Biomedical Research Question Answering}},
  author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={2567--2577},
  year={2019}
}
```
</details>

## Intended use and limitations

- For research on medical language models in Swedish and English: measuring the accuracy gap between the two languages and the effect of adaptation. **Not for clinical use**; a model's score on exam questions says nothing about its safety in care.
- The Swedish text is machine-translated and automatically refined. Some errors remain in all categories: in the blinded review by LLM raters, 9% of the clean and 20% of the disputed items had an error, all of them minor.
- Keep the test items out of training data. The items are public, so contamination of models trained on web data cannot be ruled out.

## Citation

```bibtex
@misc{anonymous2026swemedmcq,
  title  = {Benchmarking and Adapting {LLMs} for {English}--{Swedish} Medical Question Answering},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```
