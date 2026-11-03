# Translation

## 1a. Test items: GPT-5.4 drafts

`scripts/01_translate_test.sh` translates the 10,363 test items with GPT-5.4 through the OpenAI Batch API (`OPENAI_API_KEY` in the environment or in `.env`). The `submit` step is the only one that calls the API and costs money.

| Step | Module | What it does |
|---|---|---|
| `prompts` | `swemedmcq.translate.test_gpt54.make_prompts` | one translation prompt per item (question and options, JSON output) |
| `requests` | `swemedmcq.translate.test_gpt54.build_requests` | Batch API request files |
| `submit`, `status`, `download` | `swemedmcq.translate.openai_batch.*` | submit the batches, poll, download the results |
| `retry` | (same, with `--prompt-ids-file`) | re-send items that failed (quota errors, truncation); list them in `retry_ids.txt` |
| `draft` | `swemedmcq.translate.test_gpt54.extract`, `to_draft` | extract the translations, join them with the English items → `draft.jsonl` |

Settings of the released drafts: `gpt-5.4-2026-03-05`, Responses API, reasoning effort medium, strict JSON schema, at most 8,192 output tokens (4,096 for a first batch of 500 items). The draft is layer 1 of the release (`data/test/layers/layer1_draft.jsonl`) and the input of the refinement loop.

## 1b. Training questions: GPT-OSS-120B

`scripts/02_translate_train.sh` translates the training questions of the ten datasets with GPT-OSS-120B on local GPUs (vLLM).

| Step | Module | What it does |
|---|---|---|
| `prompts` | `swemedmcq.translate.train_gptoss.make_prompts` | one prompt per training question |
| `generate` | `swemedmcq.translate.train_gptoss.translate` | greedy generation, JSON schema after the analysis channel (resumable) |
| `extract`, `rerun` | `...extract`, `...translate --prompt-ids-file` | extract; generate again the items whose extraction failed |
| `filter` | `swemedmcq.translate.train_gptoss.filter` | drop every translation with a `possible_answer_change_risk` warning |

Settings: greedy decoding, GPT-OSS reasoning effort medium, up to 30,720 output tokens. The result, `data/train/sv_train_translation.jsonl`, is the input of the Swedish teacher traces and training data ([DISTILLATION](DISTILLATION.md)).
