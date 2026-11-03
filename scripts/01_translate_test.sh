#!/usr/bin/env bash
# Stage 1a: GPT-5.4 draft translation of the test split (OpenAI Batch API).
#
#   bash scripts/01_translate_test.sh            # all stages
#   STAGES="prompts requests" bash scripts/01_translate_test.sh
#
# Needs OPENAI_API_KEY (environment or .env). Submitting a batch costs money; the
# `submit` stage is the only one that calls the API, `status` and `download` only read.
# Input: the English source benchmarks (paths.yaml `source_benchmark`, one folder per
# dataset with test.jsonl). Output: <translate_dir>/test/ and the draft file
# ($OUT/draft.jsonl, or $DRAFT if set; copy it to paths.yaml `draft`, the input of stage 2).
#
# Settings of the released drafts: model gpt-5.4 (gpt-5.4-2026-03-05), /v1/responses,
# reasoning effort medium with summary auto, strict JSON schema, max_output_tokens 8192
# (4096 for a first batch of 500 items),
# no temperature. Items that fail (quota errors, truncation) are re-sent with
# --prompt-ids-file (see the `retry` stage) and the valid extractions are merged.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SRC="${SRC:-$(path source_benchmark)}"
OUT="${OUT:-$(path translate_dir)/test}"
MODEL="${MODEL:-gpt-5.4}"
PROMPTS="$OUT/translate_test_test_prompt_set.jsonl"
mkdir -p "$OUT"

if stage prompts; then
  run "$PYTHON" -m swemedmcq.translate.test_gpt54.make_prompts -i "$SRC" --file-name test \
      --output-name test --output-dir "$OUT"
fi
if stage requests; then
  run "$PYTHON" -m swemedmcq.translate.test_gpt54.build_requests -i "$PROMPTS" -o "$OUT/requests.jsonl" \
      --model "$MODEL" --max-output-tokens 8192 --reasoning-effort medium
fi
if stage submit; then
  run "$PYTHON" -m swemedmcq.translate.openai_batch.submit_batch_job --requests-file "$OUT/requests.jsonl" \
      --save-metadata "$OUT/batch_main.json"
fi
if stage status; then
  run "$PYTHON" -m swemedmcq.translate.openai_batch.check_batch_status --metadata-file "$OUT/batch_main.json"
fi
if stage download; then   # once the batch is completed
  run "$PYTHON" -m swemedmcq.translate.openai_batch.download_batch_results --metadata-file "$OUT/batch_main.json" \
      --output "$OUT/main_results.jsonl" --errors-output "$OUT/main_errors.jsonl"
  run "$PYTHON" -m swemedmcq.translate.test_gpt54.extract -i "$OUT/main_results.jsonl" \
      -o "$OUT/main_extracted.jsonl" --task translation
fi
if stage retry; then
  # Re-send the failed items: list their prompt ids (one per line) in $OUT/retry_ids.txt,
  # e.g. from the *_failed.txt file of the extraction and the error file.
  run "$PYTHON" -m swemedmcq.translate.test_gpt54.build_requests -i "$PROMPTS" -o "$OUT/retry_requests.jsonl" \
      --model "$MODEL" --max-output-tokens 8192 --reasoning-effort medium --prompt-ids-file "$OUT/retry_ids.txt"
  run "$PYTHON" -m swemedmcq.translate.openai_batch.submit_batch_job --requests-file "$OUT/retry_requests.jsonl" \
      --save-metadata "$OUT/batch_retry.json"
  # then: check_batch_status / download_batch_results / extract as above, writing retry_extracted.jsonl
fi
if stage draft; then
  # The extractor writes <name>.jsonl (all rows), <name>_valid.jsonl and <name>_failed.txt.
  # Main batch first, retries after (a later file wins for a repeated prompt id).
  shopt -s nullglob
  valid=("$OUT"/main_extracted_valid.jsonl "$OUT"/retry*_extracted_valid.jsonl)
  run "$PYTHON" -m swemedmcq.translate.test_gpt54.to_draft --english "$(path benchmark)" \
      --translations "${valid[@]}" -o "${DRAFT:-$OUT/draft.jsonl}"
fi
