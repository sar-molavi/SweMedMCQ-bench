# Pipeline

The work runs in eight stages, one script each. Every script runs from the repository root, resumes where it stopped, takes `STAGES="a b"` to run only some of its steps, and prints the commands without running them with `DRY_RUN=1`.

```
source benchmarks (EN)
 ├─ 1a  01_translate_test.sh   GPT-5.4 drafts of the test items ──────┐
 │                                                                     ▼
 │                                   2  03_refine.sh   refinement loop → release layers, edit history
 │                                                                     ▼
 │                                   3  04_build_testset.sh   benchmark.jsonl → evaluation prompt files
 │                                                                     ▼
 ├─ 1b  02_translate_train.sh  GPT-OSS-120B translation of the training questions
 │        ▼
 │   4a  05_teacher_traces.sh  teacher answers (EN and SV)
 │        ▼
 │   4b  06_build_sft_data.sh  filtering, split, Swedish rewrite, answer-level data, tokenisation
 │        ▼
 │   5   07_train.sh <student> fine-tuning ─────────────────────────────┐
 │                                                                     ▼
 └────────────────────────────────────────────── 6  08_evaluate.sh   evaluation and per-item results
```

## Which stages for which goal

| Goal | What to run | Needs |
|---|---|---|
| Use the benchmark | nothing: read `data/test/benchmark.jsonl` | – |
| Evaluate an open model | add it to `configs/models.yaml`; `bash scripts/04_build_testset.sh`; `GPUS=0 ONLY='<name>' bash scripts/08_evaluate.sh` | GPU |
| Run a released student | the model card, or the student entries of `configs/models.yaml` | GPU |
| Re-score the translation review | `python -m swemedmcq.review.score_review --out runs/review/llm`; `python -m swemedmcq.review.edit_audit score` | CPU |
| Rebuild the benchmark files from the refinement records | `RUN=<run> ANSWERS=<gold> bash scripts/04_build_testset.sh` (needs a refinement run folder) | CPU |
| Refine a new set of drafts | stages 2–3 ([REFINEMENT](REFINEMENT.md)) | 4 × 141 GB GPUs per shard |
| Retrain a student | stages 1b, 4a, 4b, 5 ([TRANSLATION](TRANSLATION.md), [DISTILLATION](DISTILLATION.md), [TRAINING](TRAINING.md)) | source benchmarks, GPUs |
| Everything | stages 1a to 6 in order | OpenAI API key (stage 1a), GPUs |

## Stages

| Stage | Script | Input | Output |
|---|---|---|---|
| 1a | `01_translate_test.sh` | `data/source/*/test.jsonl` | `work/translate/test/`, draft file → `data/test/layers/layer1_draft.jsonl` |
| 1b | `02_translate_train.sh` | `data/source/*/train.jsonl` | `data/train/sv_train_translation.jsonl` |
| 2 | `03_refine.sh` | draft, gold letters, `data/resources/` | `runs/refine/<RUN>/release/`, `probes/`, `expert/` |
| 3 | `04_build_testset.sh` | `data/test/` (or a refinement run with `RUN=`) | `runs/testset/data/` (evaluation prompt files, `items.csv`) |
| 4a | `05_teacher_traces.sh` | source benchmarks, training translation | `work/prompts/`, `work/teacher/{en,sv}/<run>/` |
| 4b | `06_build_sft_data.sh` | teacher runs, `data/train/split/` | `work/sft/`, `work/preprocessed/` |
| 5 | `07_train.sh <student>` | `work/preprocessed/`, `configs/train/<student>.json` | `runs/train/<student>/final_checkpoint` |
| 6 | `08_evaluate.sh` | `runs/testset/data/`, models | `runs/eval/` (raw generations, scores, `items.parquet`) |

## Resuming

GPU stages write one result file per item (a pickle per prompt for the vLLM runner, one result per item for the refinement shards) and skip the items that already have one, so a crashed or timed-out job continues when the same command is run again. Training resumes from the newest checkpoint.
