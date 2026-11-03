# Evaluation

## Protocol

- **Items.** Every model answers each of the 10,363 items in English (`en`) and in Swedish (`sv_L4a`, the benchmark text). Report results on the 9,733 core items (`core == true`), in both languages on the same items. The set `sv_L4_delta` holds the 851 items whose fully refined text differs from the benchmark; it gives results on the refined layer without re-running everything.
- **Prompts.** One user message per item, no system prompt.
  - The explanation prompt asks for `{"explanation": ..., "answer_index": ...}` and is used by benchmark models and trace students.
  - The answer-only prompt asks for `{"answer_index": ...}` and is used by answer-level students.
  - The prompts are in `swemedmcq/common/prompts.py`. `scripts/04_build_testset.sh` writes them for every item into `runs/testset/data/`.
- **Decoding.** Greedy, with the JSON schema enforced by vLLM:
  - after the reasoning, when a model reasons;
  - from the first token, when it answers directly.
  
  Settings by model:
  - benchmark models: up to 30,720 output tokens;
  - students: 12,288-token context;
  - EuroLLM: with the RoPE override.
- **Scoring.** The answer letter is extracted with the same parser in both languages. A missing, unparsable or out-of-range answer counts as wrong.

## Running

```bash
bash scripts/04_build_testset.sh                          # prompt files (CPU, once)
GPUS=0,1,2,3 bash scripts/08_evaluate.sh                  # all jobs of configs/models.yaml, then collect
GPUS=0 ONLY='lfm-trace-ensv' bash scripts/08_evaluate.sh  # a subset (regex on the job id)
```

**Steps of `08_evaluate.sh`:**
- `jobs` writes the job lists to `runs/eval/jobs/`: one job per configuration and set, with Slurm array files for the GPU counts.
- `local` runs the jobs on the GPUs in `$GPUS`. It runs several in parallel, skips busy GPUs, and resumes per item.
- `score` scores the jobs whose generations are complete. It needs no GPU.
- `collect` merges every scored job into `runs/eval/items.parquet`, with one row per configuration, layer and item (correctness, status, token counts, core flag and category); join `hard` from `data/test/benchmark.jsonl` by `prompt_id` for the hard subset.
- `stats` writes the paired statistics (below) to `runs/eval/paired_core.csv` and `paired_hard.csv`.

To run one job by hand: `python -m swemedmcq.evaluate.run sv_L4a/<name> [--stage generate|score] [--dry-run]`.

## Paired statistics

`python -m swemedmcq.evaluate.stats` (step `stats` of `08_evaluate.sh`) reads `runs/eval/items.parquet` and gives, for every configuration, on the paired core items:
- the accuracy in each language with a 95% bootstrap interval (10,000 resamples of items, seed 13);
- the English–Swedish gap with a paired bootstrap interval;
- the number of items solved in only one language;
- an exact McNemar test.

`--subset hard` or `--subset rest` restricts the items to the hard subset or the rest, and `--layer L4` pairs English with layer 4 instead of layer 4a. On the paper's per-item results it reproduces the paired-statistics tables of the benchmark and the students.

## Question labels and features

For the analysis of the hard subset, every English test item gets labels and simple text features. Both commands write to `runs/eval/question_tags/`.

- **Labels:** `python -m swemedmcq.evaluate.question_tags.tag_items --gpus 0,1`.
  - Labels: a question type (10 classes), a specialty (17) and three flags (needs a calculation, has lab values, clinical vignette).
  - Model: GPT-OSS-120B, greedy, reasoning effort low, JSON output. It sees the question and options, never the answer; the prompt is `swemedmcq/evaluate/question_tags/prompt.md`.
  - The run is resumable; on two 40 GB GPUs all items take about 12 minutes.
  - `--export-only` rebuilds `tags.parquet` from the cache. `spotcheck_50.csv` is a random sample of 50 labels for a human check.
- **Features:** `python -m swemedmcq.evaluate.question_tags.features`.
  - Regular-expression features of the core items: words, options, vignette, numbers, lab units, likely calculation.
  - The subject, from the source dataset files where a dataset has one.
  - With `items.parquet`, the number of open benchmark configurations answering each item correctly per language.

## Adding a model

Add an entry to `configs/models.yaml`; the header of the file explains every field. The main fields:

| Field | Meaning |
|---|---|
| `name` | unique id, also the output folder |
| `model`, `revision` | Hugging Face id (pinned snapshot) or a local folder |
| `prompt` | `trace` (explanation prompt) or `answer` (answer-only prompt) |
| `mode` | how reasoning is switched:<br>`default`: the chat template's default<br>`no_think`: `enable_thinking=False`<br>`reasoning`: JSON only after the reasoning, with a reasoning parser<br>`forced_json`: JSON from the first token<br>`token_set:<recipe>`: pre-tokenised prompts for models without a clean switch |
| `parser`, `prefix` | the reasoning parser, and text added after the generation prompt (EuroLLM trace students: `<think>\n`) |
| `max_out`, `max_len`, `hf_overrides`, `gpus` | generation settings and tensor-parallel size |
| `sets` | which sets to run (`en`, `sv_L4a`, `sv_L4_delta`) |

`python -m swemedmcq.evaluate.check_parsers --out DIR --only <regex>` checks on a few items per configuration that the JSON comes right after the reasoning (or from the first token) and validates; `python -m swemedmcq.evaluate.smoke sv_L4a/<name> --out DIR --n 4` runs a few items of one job with its real command. Both need a GPU.

## The released students

The student entries of `configs/models.yaml` load the models from the Hugging Face Hub (`Anonymiserad/SweMedMCQ-*`) with the settings of the paper:
- LFM2.5 trace students use their own chat template (`enable_thinking` opens `<think>`);
- EuroLLM trace students get `<think>\n` (reasoning) or an empty block (direct answer) after the generation prompt, and the RoPE override `max_position_embeddings: 12288`.

Each model card shows the same setup with plain vLLM (`LLM.generate` and `vllm serve`).
