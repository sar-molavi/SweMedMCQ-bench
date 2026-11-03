# Teacher traces and training data

The students learn from the training splits of the ten MedicalAgentsBench datasets, in English and in Swedish (the GPT-OSS-120B translation of stage 1b, [TRANSLATION](TRANSLATION.md)). Two kinds of supervision are built: **reasoning traces** written by two open teachers, and **answer-level** data with only the gold letter.

## 4a. Teacher traces (`scripts/05_teacher_traces.sh`)

Each teacher answers every training question with the explanation prompt (`swemedmcq/common/prompts.py`; JSON `{explanation, answer_index}`), in English and in Swedish:

| Teacher | Runs (folder = `source_run`) | Decoding |
|---|---|---|
| GPT-OSS-120B | `*_gpt_oss_120b`, `*_gpt_oss_120b_t=0.7`, `*_gpt_oss_120b_t=0.7_v2` | greedy and twice at temperature 0.7; default reasoning effort; JSON after the analysis channel |
| Qwen3.6-35B-A3B | `*_qwen_3.6_35B_A3B`, `*_qwen_3.6_35B_A3B_t=0.7` | greedy and once at temperature 0.7; thinking on; JSON after `</think>` |

`*` is `en_response_train` or `sw_response_train`. Steps: `prompts` (`swemedmcq.distill.make_response_prompts_{en,sw}`), then `en` and `sv` (`swemedmcq.common.{en,sw}_response`, the vLLM runner; one result file per prompt, resumable). `RUNS="<run names>"` limits the runs. The output-token limit is the runner default (30,720).

## 4b. Training data (`scripts/06_build_sft_data.sh`)

| Step | What it does | Output |
|---|---|---|
| `en` | decode the English runs; merge them with the prompts and gold answers; keep a response only if its answer is correct and its explanation is valid, and drop length outliers per run (Tukey fences); keep GPT-OSS's reasoning as a `<think>…</think>` block and drop Qwen's; split by the released prompt ids; flatten to one example per kept response; draw the train-time validation set (2,048 prompts, stratified, seed 13) | `work/sft/en/flat/` |
| `sv` | the same for the Swedish runs, with the same prompt ids (train-time validation: 3,072 prompts) | `work/sft/sv/splits/flat/` |
| `sv_rw_prompts`, `sv_rw_generate`, `sv_rw_build` | **SV-RW:** GPT-OSS-120B rewrites the (mostly English) reasoning of the Swedish traces into Swedish (up to 8,192 output tokens, 12,288-token context); responses whose rewrite fails validation are dropped, responses without reasoning are kept as they are | `work/sft/sv_rw/` |
| `answer` | answer-level data: the answer-only prompt with the gold letter as the only target, in English, Swedish and both, with the same validation prompts | `work/sft/answer_level/` |
| `preprocess` | tokenise each student's data with its config (`configs/train/*.json`); EN+SV train sets are the union of the English and the Swedish set, shuffled (seed 2000) | `work/preprocessed/` |

**The released split.** `data/train/split/validation_prompt_ids.txt.gz` (6,642 prompt ids) is the 5% held-out validation set of every student, in both languages; `train_prompt_ids.txt.gz` (126,141) holds the training questions of the trace students. The split was drawn once at the prompt level, stratified by dataset, number of kept responses and presence of reasoning (seed 13), so all responses to a question and its Swedish version stay on the same side. The `en` step reads both files (`swemedmcq.distill.split_english_sft --train-ids --validation-ids`), and the `sv` step uses the same ids, so the rebuilt trace data has exactly the released students' questions; prompts listed in neither file are left out. Without the two options, `split_english_sft` draws a new stratified split. The answer-level data (`answer` step) uses only the validation file: every other training question with a gold answer is a training example (135,093 English, 134,993 Swedish questions).

Every step can be run alone with `STAGES=...`; `sv_rw_generate` needs GPUs (`CUDA_VISIBLE_DEVICES`).
