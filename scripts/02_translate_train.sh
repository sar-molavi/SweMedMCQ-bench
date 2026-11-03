#!/usr/bin/env bash
# Stage 1b: GPT-OSS-120B translation of the training split (vLLM, local GPUs).
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/02_translate_train.sh
#   STAGES="extract filter" bash scripts/02_translate_train.sh
#
# Input: the English source benchmarks (paths.yaml `source_benchmark`, train.jsonl per
# dataset). Output: <translate_dir>/train/ ; the final file (extraction + warning filter)
# is written to paths.yaml `train_translation`, the input of stage 4.
#
# Settings of the released translation: greedy, GPT-OSS reasoning effort medium (its chat
# template default), JSON schema MedicalMCQTranslationOutput after the analysis channel,
# max output tokens 30,720 (runner default). Items whose extraction failed are generated
# again with --prompt-ids-file and extracted again. Rows with a possible_answer_change_risk
# warning (any severity) are dropped.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SRC="${SRC:-$(path source_benchmark)}"
OUT="${OUT:-$(path translate_dir)/train}"
FINAL="${FINAL:-$(path train_translation)}"
MODEL="${MODEL:-openai/gpt-oss-120b}"
PROMPTS="$OUT/translate_train_train_prompt_set.jsonl"
mkdir -p "$OUT"

if stage prompts; then
  run "$PYTHON" -m swemedmcq.translate.train_gptoss.make_prompts -i "$SRC" --file-name train \
      --output-name train --output-dir "$OUT"
fi
if stage generate; then   # resumable: existing result_<id>.pkl files are skipped
  run "$PYTHON" -m swemedmcq.translate.train_gptoss.translate -i "$PROMPTS" -o "$OUT/pickles" \
      --model-name "$MODEL" --temperature 0
fi
if stage extract; then
  run "$PYTHON" -m swemedmcq.translate.train_gptoss.extract -i "$OUT/pickles" -o "$OUT/results.jsonl" \
      --model-name "$MODEL" --task translation
fi
if stage rerun; then      # regenerate the items listed in results_failed.txt, then extract again
  run "$PYTHON" -m swemedmcq.translate.train_gptoss.translate -i "$PROMPTS" -o "$OUT/pickles" \
      --model-name "$MODEL" --temperature 0 --prompt-ids-file "$OUT/results_failed.txt"
  run "$PYTHON" -m swemedmcq.translate.train_gptoss.extract -i "$OUT/pickles" -o "$OUT/results.jsonl" \
      --model-name "$MODEL" --task translation
fi
if stage filter; then
  run "$PYTHON" -m swemedmcq.translate.train_gptoss.filter -i "$OUT/results_valid.jsonl" -o "$FINAL" \
      --summary-path "$OUT/filter_summary.json" --warning-type possible_answer_change_risk --severity low
fi
