# SweMedMCQ: Benchmarking and Adapting LLMs for English--Swedish Medical Question Answering

SweMedMCQ is a paired English–Swedish benchmark of medical multiple-choice questions, together with the code that built it and a set of small Swedish medical models trained with it. This repository accompanies the paper *Benchmarking and Adapting LLMs for English–Swedish Medical Question Answering* (under review).

The work has two parts:

1. **A paired benchmark.** We translate the test sets of ten English medical datasets into Swedish. Each Swedish item keeps the options, their order and the gold answer of its English source, so a model answers the same question in both languages. A conservative refinement loop checks the translations: open models from three families find errors, ground them in a Swedish medical term base, and repair them, and an edit is kept only if blind judges vote for it.
2. **Adapting small models.** We translate the training sets, let two open teacher models write reasoning traces, and fine-tune two small students (LFM2.5-1.2B and EuroLLM-1.7B) on the gold answer alone or on the teachers' reasoning, in English, Swedish or both.

This repository contains:

| Component | Where |
|---|---|
| **The final benchmark** (refinement layer 4a, used for every result of the paper): 10,363 items, English and Swedish, with gold answers, the core flag and the hard flag | [`data/test/benchmark.jsonl`](data/README.md#the-final-benchmark-layer-4a-on-the-core-items) |
| The edit history of every item: each finding, candidate edit, judge vote and decision of the refinement | [`data/test/edit_history.jsonl.gz`](data/README.md#edit-history-report-and-probes) |
| The other four refinement layers: draft (1), typography (2), number style (3), fully refined (4) | [`data/test/layers/`](data/README.md#layers) |
| The blinded review of the translations: sheets, guidelines, the raters' labels, the answer keys, the per-edit audit | [`review/`](review/README.md) |
| The Swedish term base and word resources used by the refinement loop | [`data/resources/`](data/README.md#term-base-and-swedish-resources) |
| Code for every step: translation, refinement, test-set build, teacher traces and training data, training, evaluation | [`swemedmcq/`](swemedmcq), [`scripts/`](scripts), [`configs/`](configs) |
| The train/validation split of the training questions (held-out validation set of every student) | [`data/train/split/`](docs/DISTILLATION.md) |
| The 14 trained students | [Hugging Face](https://huggingface.co/Anonymiserad) (see [Models](#models)) |

## Main results

All numbers are from the paper: accuracy (%) on the **9,733 core items** (items whose translation the refinement loop could settle), with greedy decoding. EN is the English item and SV the Swedish one; Δ = EN − SV, positive when Swedish is worse.

### Large models lose little in Swedish, small models lose more

| Model | EN (reasoning on) | SV | Δ | EN (reasoning off) | SV | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Gemma-4-31B | 78.7 | 78.0 | 0.8 | 77.1 | 74.0 | 3.0 |
| Qwen3.6-35B-A3B | 74.7 | 73.5 | 1.3 | 72.2 | 68.5 | 3.7 |
| Qwen3.8-27B | 74.6 | 72.0 | 2.6 | 70.6 | 66.5 | 4.1 |
| GPT-OSS-120B | 71.8 | 70.4 | 1.4 | 64.3 | 56.9 | 7.4 |
| GPT-OSS-20B | 65.8 | 63.7 | 2.1 | 59.0 | 50.6 | 8.4 |
| MedGemma-27B | – | – | – | 64.5 | 60.5 | 4.0 |
| Qwen3-4B-Thinking | 59.3 | 57.4 | 1.8 | 53.9 | 43.3 | 10.5 |
| Qwen2.5-3B | – | – | – | 44.9 | 32.5 | 12.4 |

Every open model is less accurate in Swedish. The gap grows as models get smaller and shrinks when they reason before answering.

### Swedish reasoning traces close the gap for small students

| Supervision | Trained on | LFM2.5-1.2B EN | SV | Δ | EuroLLM-1.7B EN | SV | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| None (base model) | – | 36.3 | 23.8 | 12.4 | 14.7 | 21.3 | −6.6 |
| Answer-level | EN | 35.2 | 24.8 | 10.4 | 33.7 | 30.0 | 3.6 |
| | SV | 27.4 | 27.8 | −0.4 | 30.9 | **30.7** | 0.1 |
| | EN+SV | 33.2 | **30.7** | 2.5 | 31.6 | 29.4 | 2.2 |
| None (base model) | – | 37.2 | 23.5 | 13.8 | 17.5 | 23.5 | −6.0 |
| Reasoning trace | EN | 49.0 | 14.5 | 34.5 | 43.2 | 39.4 | 3.8 |
| | SV | 50.1 | 49.8 | 0.3 | 44.3 | **44.5** | −0.2 |
| | SV (rewritten) | 44.0 | 44.3 | −0.3 | 39.4 | 39.0 | 0.5 |
| | EN+SV | 50.5 | **52.0** | −1.5 | 39.4 | 42.6 | −3.2 |

Answer-level models (and the base model above them) use the answer-only prompt; reasoning-trace models use the explanation prompt and reason before answering. *SV (rewritten)*: Swedish data in which the teacher's reasoning was rewritten into Swedish.

Training on gold answers alone helps little. Reasoning traces that include Swedish close the gap and lift both languages. The best student, LFM2.5-1.2B trained on English and Swedish traces, answers 52.0% of the Swedish items. That is about as many as GPT-OSS-20B without reasoning (50.6%), from 2.3 GB of weights against 14 GB. The largest open models stay clearly more accurate, but their weights take 54–72 GB.

### Translation quality

- Of the 10,363 drafts, the loop confirms 6,968 as they are, applies only typography and number-style fixes to 1,708, and refines 1,063. The remaining 624 stay disputed and are not part of the core.
- A blinded review (the rater sees the text before and after the edit as A and B in random order, without knowing which is which) by LLM raters found no harmful edit among the edits made by code (tier 1). The edits written by models (tier 2) improved the text 77 times and harmed it 18 times in 160 reviewed items. The upper 95% bound of their harm rate (17.1%) is above the 10% bar set before the review.
- The benchmark therefore uses the tier-1 text: **layer 4a** (`data/test/benchmark.jsonl`) = GPT-5.4 draft (layer 1) + typography (layer 2) + Swedish number style (layer 3) + the 294 code-made edits. The fully refined text (layer 4, with the 958 model-written edits) is released in `data/test/layers/`; using it instead changes the gap on the core items by 0.01 points on average. [`data/README.md`](data/README.md#the-final-benchmark-layer-4a-on-the-core-items) gives every step with its counts.

## Quick start

Load the benchmark:

```python
import json

items = [json.loads(line) for line in open("data/test/benchmark.jsonl", encoding="utf-8")]
core = [it for it in items if it["core"]]
item = core[0]
print(item["sv_question"], item["sv_options"], item["answer"])
```

Each item has `prompt_id`, `dataset`, `question`, `options`, `sv_question`, `sv_options`, `answer`, `category`, `tier`, `core`, `expert_queue` and `hard`. The fields are described in [`data/README.md`](data/README.md). Report results on the core items (`core == true`) and compare both languages on the same items.

To evaluate an open model with the paper's protocol, add an entry to [`configs/models.yaml`](configs/models.yaml) (the file explains every field), then build the prompt files and run its jobs:

```bash
pip install -r requirements/inference.txt     # vLLM 0.28 environment
bash scripts/04_build_testset.sh              # evaluation prompts from data/test/benchmark.jsonl
GPUS=0 ONLY='my-model' bash scripts/08_evaluate.sh
```

The per-item results end up in `runs/eval/items.parquet`. See [`docs/EVALUATION.md`](docs/EVALUATION.md) for the protocol and the options.

## Reproducing the work

The whole pipeline runs from eight scripts, one per stage. Each one resumes where it stopped, takes `STAGES=` to run only some of its steps, and prints its commands without running them when `DRY_RUN=1` is set.

| Stage | Script | Needs | Docs |
|---|---|---|---|
| 1a. Translate the test sets (GPT-5.4 drafts) | `scripts/01_translate_test.sh` | OpenAI Batch API (paid) | [TRANSLATION](docs/TRANSLATION.md) |
| 1b. Translate the training sets (GPT-OSS-120B) | `scripts/02_translate_train.sh` | GPUs | [TRANSLATION](docs/TRANSLATION.md) |
| 2. Refine the test translations | `scripts/03_refine.sh` | 4 × 141 GB GPUs per shard | [REFINEMENT](docs/REFINEMENT.md), [REVIEW_PROTOCOL](docs/REVIEW_PROTOCOL.md) |
| 3. Build the benchmark and the evaluation prompts | `scripts/04_build_testset.sh` | CPU | [REFINEMENT](docs/REFINEMENT.md) |
| 4a. Teacher reasoning traces | `scripts/05_teacher_traces.sh` | GPUs | [DISTILLATION](docs/DISTILLATION.md) |
| 4b. Training data (filtering, splits, Swedish rewrite, answer-level, tokenisation) | `scripts/06_build_sft_data.sh` | CPU (GPUs for the rewrite) | [DISTILLATION](docs/DISTILLATION.md) |
| 5. Fine-tune the students | `scripts/07_train.sh <student>` | 4 GPUs | [TRAINING](docs/TRAINING.md) |
| 6. Evaluate and collect | `scripts/08_evaluate.sh` | GPUs | [EVALUATION](docs/EVALUATION.md) |

Start with [`docs/SETUP.md`](docs/SETUP.md): it covers the environments, the hardware, the data that is not shipped (the English MedicalAgentsBench source files and the Swedish training translation, which stage 1b regenerates) and running on a Slurm cluster. The [`configs/`](configs/README.md) and [`scripts/`](scripts/README.md) folders have their own overviews. [`docs/PIPELINE.md`](docs/PIPELINE.md) shows which stages you need for which goal.

The blinded review of the translations is scored with `swemedmcq.review`; [`review/README.md`](review/README.md) describes the sheets and the commands.

The released files are reproducible from the code. Rebuilding the benchmark from the refinement run's records gives `data/test/` byte for byte, the test-set build reproduces the evaluation prompt files, re-scoring the review from `review/` reproduces its summary, and the training pipeline uses the released train/validation split.

## Models

The 14 students are fine-tuned from [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) (loaded from its [`unsloth/` mirror](https://huggingface.co/unsloth/LFM2.5-1.2B-Instruct), which has the same weights, by all LFM2.5 students except Trace-SV-RW) and [EuroLLM-1.7B-Instruct](https://huggingface.co/utter-project/EuroLLM-1.7B-Instruct) and published on the Hugging Face Hub. Each one is the checkpoint with the best validation loss. Each model card gives the training data, settings, prompt format and how to run the model with vLLM.

| Model | Supervision | Trained on | EN | SV |
|---|---|---|---:|---:|
| [`SweMedMCQ-LFM2.5-1.2B-Answer-EN`](https://huggingface.co/Anonymiserad/SweMedMCQ-LFM2.5-1.2B-Answer-EN) | answer-level | EN | 35.2 | 24.8 |
| [`SweMedMCQ-LFM2.5-1.2B-Answer-SV`](https://huggingface.co/Anonymiserad/SweMedMCQ-LFM2.5-1.2B-Answer-SV) | answer-level | SV | 27.4 | 27.8 |
| [`SweMedMCQ-LFM2.5-1.2B-Answer-EN-SV`](https://huggingface.co/Anonymiserad/SweMedMCQ-LFM2.5-1.2B-Answer-EN-SV) | answer-level | EN+SV | 33.2 | 30.7 |
| [`SweMedMCQ-LFM2.5-1.2B-Trace-EN`](https://huggingface.co/Anonymiserad/SweMedMCQ-LFM2.5-1.2B-Trace-EN) | reasoning trace | EN | 49.0 | 14.5 |
| [`SweMedMCQ-LFM2.5-1.2B-Trace-SV`](https://huggingface.co/Anonymiserad/SweMedMCQ-LFM2.5-1.2B-Trace-SV) | reasoning trace | SV | 50.1 | 49.8 |
| [`SweMedMCQ-LFM2.5-1.2B-Trace-SV-RW`](https://huggingface.co/Anonymiserad/SweMedMCQ-LFM2.5-1.2B-Trace-SV-RW) | reasoning trace | SV (rewritten) | 44.0 | 44.3 |
| [`SweMedMCQ-LFM2.5-1.2B-Trace-EN-SV`](https://huggingface.co/Anonymiserad/SweMedMCQ-LFM2.5-1.2B-Trace-EN-SV) | reasoning trace | EN+SV | 50.5 | 52.0 |
| [`SweMedMCQ-EuroLLM-1.7B-Answer-EN`](https://huggingface.co/Anonymiserad/SweMedMCQ-EuroLLM-1.7B-Answer-EN) | answer-level | EN | 33.7 | 30.0 |
| [`SweMedMCQ-EuroLLM-1.7B-Answer-SV`](https://huggingface.co/Anonymiserad/SweMedMCQ-EuroLLM-1.7B-Answer-SV) | answer-level | SV | 30.9 | 30.7 |
| [`SweMedMCQ-EuroLLM-1.7B-Answer-EN-SV`](https://huggingface.co/Anonymiserad/SweMedMCQ-EuroLLM-1.7B-Answer-EN-SV) | answer-level | EN+SV | 31.6 | 29.4 |
| [`SweMedMCQ-EuroLLM-1.7B-Trace-EN`](https://huggingface.co/Anonymiserad/SweMedMCQ-EuroLLM-1.7B-Trace-EN) | reasoning trace | EN | 43.2 | 39.4 |
| [`SweMedMCQ-EuroLLM-1.7B-Trace-SV`](https://huggingface.co/Anonymiserad/SweMedMCQ-EuroLLM-1.7B-Trace-SV) | reasoning trace | SV | 44.3 | 44.5 |
| [`SweMedMCQ-EuroLLM-1.7B-Trace-SV-RW`](https://huggingface.co/Anonymiserad/SweMedMCQ-EuroLLM-1.7B-Trace-SV-RW) | reasoning trace | SV (rewritten) | 39.4 | 39.0 |
| [`SweMedMCQ-EuroLLM-1.7B-Trace-EN-SV`](https://huggingface.co/Anonymiserad/SweMedMCQ-EuroLLM-1.7B-Trace-EN-SV) | reasoning trace | EN+SV | 39.4 | 42.6 |

Accuracy is for the trace students with reasoning on.

**Running a student with vLLM.** The prompts are in each model's `prompts.json`. Use greedy decoding and `max_model_len=12288`. EuroLLM also needs `hf_overrides={"max_position_embeddings": 12288}`, and its trace students need `<think>\n` appended to the generation prompt. [`hf/README.md`](hf/README.md) has a complete example and the serving command.

## Repository layout

```
SweMedMCQ/
├── data/
│   ├── test/benchmark.jsonl   the final benchmark: layer 4a (10,363 items, English and Swedish)
│   ├── test/edit_history.jsonl.gz   per-item edit history of the refinement
│   ├── test/probes/           judge probes and their results
│   ├── test/layers/           refinement layers 1, 2, 3 and 4 (the benchmark is layer 4a)
│   ├── train/split/           train/validation prompt ids of the training questions
│   └── resources/             Swedish term base, word lists, protected terms, style guide
├── swemedmcq/                 Python package, one folder per stage
│   ├── common/                vLLM runner, reasoning parsers, prompts, answer schemas
│   ├── translate/             GPT-5.4 test drafts and GPT-OSS-120B training translation
│   ├── refine/                the refinement loop (detect, ground, verify, repair, gate, vote)
│   ├── testset/               benchmark file and evaluation prompt files
│   ├── distill/               teacher traces, filtering, splits, Swedish rewrite, answer-level data
│   ├── train/                 full fine-tuning with Unsloth and TRL
│   ├── evaluate/              evaluation jobs, scoring, result collection, paired statistics, question labels
│   └── review/                scoring of the blinded review and the per-edit audit
├── configs/                   paths, evaluated models, refinement settings, the 14 training configs
├── scripts/                   one runbook per stage, plus Slurm examples
├── review/                    review sheets, rater guidelines, LLM raters' labels, answer keys, per-edit audit
├── docs/                      how to run each stage
├── hf/                        model upload script and vLLM usage of the released students
├── tests/                     unit tests (pytest)
└── requirements/              inference (vLLM) and training (Unsloth) environments
```

## Licences

- **Code:** Apache License 2.0 ([`LICENSE`](LICENSE)).
- **Benchmark:** every item keeps the licence of its source dataset. The ten sources and their references are listed in [`data/README.md`](data/README.md#source-datasets).
- **Term base:** built from third-party sources with their own licences, including attribution and share-alike terms. See [`data/README.md`](data/README.md#term-base-and-swedish-resources).
- **Models:** the LFM2.5 students follow the LFM Open License v1.0 of their base model; the EuroLLM students follow Apache 2.0.

## Intended use

The benchmark and the models are for research on medical language understanding in Swedish. They are not medical devices and must not be used for clinical decisions.