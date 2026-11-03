#!/usr/bin/env bash
# Stage 4b: SFT datasets from the teacher traces, and their tokenised versions.
#
#   bash scripts/06_build_sft_data.sh                         # every stage, in order
#   STAGES="en sv" bash scripts/06_build_sft_data.sh          # some stages
#   CUDA_VISIBLE_DEVICES=0,1,2,3 STAGES="sv_rw_generate" bash scripts/06_build_sft_data.sh
#
# Stages (in order):
#   en              decode the English teacher runs, merge with prompts and gold, source statistics,
#                   filter (correct answer, schema-valid explanation, per-source Tukey length limits;
#                   GPT-OSS reasoning kept, Qwen reasoning dropped), prompt-level split from the
#                   released prompt ids (data/train/split/), flatten, train-time validation
#                   (2,048 prompts, seed 13) -> work/sft/en/
#   sv              the same for Swedish; the split follows the same prompt ids -> work/sft/sv/
#   sv_rw_prompts   Swedish rewrite prompts for every response with (English) reasoning
#   sv_rw_generate  GPU: GPT-OSS-120B rewrites the reasoning into Swedish (resumable)
#   sv_rw_build     extract, apply the valid rewrites, flatten -> work/sft/sv_rw/
#   answer          answer-only splits (EN, SV, EN+SV; validation prompts = English validation split)
#                   -> work/sft/answer_level/
#   preprocess      tokenise every student's data with its own config (configs/train/*.json)
#                   and build the bilingual train sets -> work/preprocessed/
# The teacher runs are named by source run (e.g. en_response_train_gpt_oss_120b_t=0.7);
# the flatten/filter code selects responses by these names.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

W="$(path work_dir)"
SRC="$(path source_benchmark)"
WORKERS="${WORKERS:-8}"
NUM_PROC="${NUM_PROC:-1}"
BATCH_SIZE="${BATCH_SIZE:-10000}"
EN="$W/sft/en"; SV="$W/sft/sv"; RW="$W/sft/sv_rw"; AL="$W/sft/answer_level"
# The released prompt-level train/validation split (5 % validation, stratified, seed 13).
TRAIN_IDS="${TRAIN_IDS:-data/train/split/train_prompt_ids.txt.gz}"
VALIDATION_IDS="${VALIDATION_IDS:-data/train/split/validation_prompt_ids.txt.gz}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-128}"

GPT=openai/gpt-oss-120b
QWEN=Qwen/Qwen3.6-35B-A3B
EN_PREFERENCE=(en_response_train_gpt_oss_120b_t=0.7 en_response_train_gpt_oss_120b
               en_response_train_qwen_3.6_35B_A3B_t=0.7 en_response_train_qwen_3.6_35B_A3B
               en_response_train_gpt_oss_120b_t=0.7_v2)
SV_PREFERENCE=(sw_response_train_gpt_oss_120b_t=0.7 sw_response_train_gpt_oss_120b
               sw_response_train_qwen_3.6_35B_A3B_t=0.7 sw_response_train_qwen_3.6_35B_A3B
               sw_response_train_gpt_oss_120b_t=0.7_v2)
SV_SOURCES=(sw_response_train_gpt_oss_120b sw_response_train_gpt_oss_120b_t=0.7
            sw_response_train_gpt_oss_120b_t=0.7_v2 sw_response_train_qwen_3.6_35B_A3B
            sw_response_train_qwen_3.6_35B_A3B_t=0.7)
pref_args() { local out=() s; for s in "$@"; do out+=(--source-preference "$s"); done; echo "${out[@]}"; }

# ---------------------------------------------------------------- English traces
if stage en; then
  mkdir -p "$EN/unraveled"
  for spec in "en_response_train_gpt_oss_120b|$GPT|gpt_oss" "en_response_train_gpt_oss_120b_t=0.7|$GPT|gpt_oss" \
              "en_response_train_gpt_oss_120b_t=0.7_v2|$GPT|gpt_oss" "en_response_train_qwen_3.6_35B_A3B|$QWEN|qwen" \
              "en_response_train_qwen_3.6_35B_A3B_t=0.7|$QWEN|qwen"; do
    IFS='|' read -r name model fmt <<< "$spec"
    run "$PYTHON" -m swemedmcq.distill.unravel_en_responses --input-dir "$W/teacher/en/$name" \
        --output-path "$EN/unraveled/$name.jsonl" --model-name "$model" --response-format "$fmt"
  done
  run "$PYTHON" -m swemedmcq.distill.combine_unraveled_en_responses --input-dir "$EN/unraveled" \
      --output-path "$EN/merged_unraveled_en_responses.jsonl" --data-dir "$SRC" \
      --prompt-path "$W/prompts/en_train_prompt_set.jsonl" --workers "$WORKERS"
  run "$PYTHON" -m swemedmcq.distill.summarize_merged_en_responses --input-path "$EN/merged_unraveled_en_responses.jsonl" \
      --output-path "$EN/merged_en_response_source_stats.json"
  run "$PYTHON" -m swemedmcq.distill.filter_sft_en_responses --input-path "$EN/merged_unraveled_en_responses.jsonl" \
      --stats-path "$EN/merged_en_response_source_stats.json" --output-path "$EN/sft_filtered_en_responses.jsonl" \
      --report-path "$EN/sft_filtered_en_responses_report.json"
  run "$PYTHON" -m swemedmcq.distill.split_english_sft --input-path "$EN/sft_filtered_en_responses.jsonl" \
      --output-dir "$EN/splits" --train-ids "$TRAIN_IDS" --validation-ids "$VALIDATION_IDS"
  # shellcheck disable=SC2046
  run "$PYTHON" -m swemedmcq.distill.build_flat_splits --train-input "$EN/splits/train.jsonl" \
      --validation-input "$EN/splits/validation.jsonl" --output-dir "$EN/flat" --work-dir "$EN/flat/prompt" \
      --validation-cap 2048 --seed 13 $(pref_args "${EN_PREFERENCE[@]}")
fi

# ---------------------------------------------------------------- Swedish traces
if stage sv; then
  mkdir -p "$SV/unraveled"
  pos=0
  for spec in "sw_response_train_gpt_oss_120b|$GPT|gpt_oss" "sw_response_train_gpt_oss_120b_t=0.7|$GPT|gpt_oss" \
              "sw_response_train_gpt_oss_120b_t=0.7_v2|$GPT|gpt_oss" "sw_response_train_qwen_3.6_35B_A3B|$QWEN|qwen" \
              "sw_response_train_qwen_3.6_35B_A3B_t=0.7|$QWEN|qwen"; do
    IFS='|' read -r name model fmt <<< "$spec"
    run "$PYTHON" -m swemedmcq.distill.unravel_sw_responses --input-dir "$W/teacher/sv/$name" \
        --output-path "$SV/unraveled/$name.jsonl" --source-run "$name" --model-name "$model" \
        --response-format "$fmt" --sample-index 0 --timeout-seconds 120 --log-every 1000 --progress-position "$pos"
    pos=$((pos + 1))
  done
  run "$PYTHON" -m swemedmcq.distill.combine_unraveled_sw_responses --input-dir "$SV/unraveled" \
      --output-path "$SV/merged_unraveled_sw_responses.jsonl" --data-dir "$SRC" \
      --prompt-path "$W/prompts/sw_train_trans_filtered_prompt_set.jsonl" --workers "$WORKERS" \
      --source-runs "${SV_SOURCES[@]}"
  run "$PYTHON" -m swemedmcq.distill.summarize_merged_en_responses --input-path "$SV/merged_unraveled_sw_responses.jsonl" \
      --output-path "$SV/merged_sw_response_source_stats.json"
  run "$PYTHON" -m swemedmcq.distill.filter_sft_en_responses --input-path "$SV/merged_unraveled_sw_responses.jsonl" \
      --stats-path "$SV/merged_sw_response_source_stats.json" --output-path "$SV/sft_filtered_sw_responses.jsonl" \
      --report-path "$SV/sft_filtered_sw_responses_report.json"
  run "$PYTHON" -m swemedmcq.distill.split_sft_dataset --input-path "$SV/sft_filtered_sw_responses.jsonl" \
      --output-dir "$SV/splits" --english-train-path "$TRAIN_IDS" \
      --english-validation-path "$VALIDATION_IDS"
  # shellcheck disable=SC2046
  run "$PYTHON" -m swemedmcq.distill.flatten_sft_dataset --train-input "$SV/splits/train.jsonl" \
      --validation-input "$SV/splits/validation.jsonl" --output-dir "$SV/splits/flat" \
      $(pref_args "${SV_PREFERENCE[@]}")
  # shellcheck disable=SC2046
  run "$PYTHON" -m swemedmcq.distill.create_train_time_validation --validation-input "$SV/splits/validation.jsonl" \
      --prompt-output "$SV/splits/validation_train_time.jsonl" --flat-output "$SV/splits/flat/validation_train_time.jsonl" \
      --manifest-output "$SV/splits/validation_train_time_manifest.json" --cap 3072 --seed 13 \
      $(pref_args "${SV_PREFERENCE[@]}")
fi

# ---------------------------------------------------------------- Swedish rewritten reasoning (SV-RW)
if stage sv_rw_prompts; then
  run "$PYTHON" -m swemedmcq.distill.trace_rewrite.dataset --input-path "$SV/splits/train.jsonl" --output-dir "$RW"
  run "$PYTHON" -m swemedmcq.distill.trace_rewrite.rewrite_prompts --input-path "$RW/train_with_reasoning.jsonl" \
      --output-path "$RW/rewrite_prompts.jsonl" --manifest-path "$RW/rewrite_prompts_manifest.json"
  V="$RW/validation_train_time"
  run "$PYTHON" -m swemedmcq.distill.trace_rewrite.dataset --input-path "$SV/splits/validation_train_time.jsonl" \
      --output-dir "$V/source" --with-reasoning-path "$V/source/with_reasoning.jsonl" \
      --no-reasoning-path "$V/source/no_reasoning.jsonl" --manifest-path "$V/source/split_manifest.json"
  run "$PYTHON" -m swemedmcq.distill.trace_rewrite.rewrite_prompts --input-path "$V/source/with_reasoning.jsonl" \
      --output-path "$V/rewrite_prompts.jsonl" --manifest-path "$V/rewrite_prompts_manifest.json"
fi
if stage sv_rw_generate; then   # resumable
  run "$PYTHON" -m swemedmcq.distill.trace_rewrite.rewriter --input-path "$RW/rewrite_prompts.jsonl" \
      --output-dir "$RW/rewrite_results" --model-name "$GPT" --max-output-tokens 8192 --max-model-len 12288
  run "$PYTHON" -m swemedmcq.distill.trace_rewrite.rewriter --input-path "$RW/validation_train_time/rewrite_prompts.jsonl" \
      --output-dir "$RW/validation_train_time/rewrite_results" --model-name "$GPT" \
      --max-output-tokens $((12 * 1024)) --max-model-len $((14 * 1024))
fi
if stage sv_rw_build; then
  run env HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" "$PYTHON" -m swemedmcq.distill.trace_rewrite.extract_rewrites \
      -i "$RW/rewrite_results" -o "$RW/reports/rewrite_extractions.jsonl" -j "${N_JOBS:--1}"
  run "$PYTHON" -m swemedmcq.distill.trace_rewrite.build_dataset --source-path "$SV/splits/train.jsonl" \
      --rewrites-path "$RW/reports/rewrite_extractions_valid.jsonl" --output-path "$RW/final/clustered/train.jsonl" \
      --summary-path "$RW/reports/final_dataset_summary.json"
  run "$PYTHON" -m swemedmcq.distill.flatten_sft_dataset --train-input "$RW/final/clustered/train.jsonl" \
      --validation-input "$SV/splits/validation.jsonl" --output-dir "$RW/final/flat" --selection all
  V="$RW/validation_train_time"
  run env HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" "$PYTHON" -m swemedmcq.distill.trace_rewrite.extract_rewrites \
      -i "$V/rewrite_results" -o "$V/reports/rewrite_extractions.jsonl" -j "${N_JOBS:--1}"
  run "$PYTHON" -m swemedmcq.distill.trace_rewrite.build_dataset --source-path "$SV/splits/validation_train_time.jsonl" \
      --rewrites-path "$V/reports/rewrite_extractions_valid.jsonl" --output-path "$V/final/validation_train_time_clustered.jsonl" \
      --flat-output-path "$V/final/validation_train_time.jsonl" --summary-path "$V/reports/final_dataset_summary.json"
fi

# ---------------------------------------------------------------- answer-only data
if stage answer; then
  run "$PYTHON" -m swemedmcq.distill.answer_level.build_answer_level_dataset --language en --english-input "$SRC" \
      --validation-prompt-ids "$VALIDATION_IDS" --train-file-name train \
      --test-file-name test --output-dir "$AL"
  run "$PYTHON" -m swemedmcq.distill.answer_level.extract_swedish_answer_level_rows --input-path "$(path train_translation)" \
      --answer-key-data-dir "$SRC" --output-path "$AL/sv/source/train_trans_valid_filtered.jsonl" \
      --report-path "$AL/sv/source/extraction_manifest.json"
  run "$PYTHON" -m swemedmcq.distill.answer_level.build_answer_level_dataset --language sv --english-input "$SRC" \
      --swedish-input "$AL/sv/source/train_trans_valid_filtered.jsonl" \
      --validation-prompt-ids "$VALIDATION_IDS" --train-file-name train --no-test \
      --output-dir "$AL"
  run "$PYTHON" -m swemedmcq.distill.answer_level.build_bilingual_answer_level_dataset --mode splits \
      --english-train "$AL/en/splits/train.jsonl" --swedish-train "$AL/sv/splits/train.jsonl" \
      --english-validation "$AL/en/splits/validation.jsonl" --swedish-validation "$AL/sv/splits/validation.jsonl" \
      --output-dir "$AL/bilingual" --shuffle-seed 2000
fi

# ---------------------------------------------------------------- tokenisation
if stage preprocess; then
  P="$W/preprocessed"
  for base in LFM2.5-1.2B EuroLLM-1.7B; do
    for v in Trace-EN Trace-SV Trace-SV-RW Answer-EN Answer-SV; do
      cfg="configs/train/SweMedMCQ-$base-$v.json"
      out="$("$PYTHON" -c 'import json,sys,pathlib; print(pathlib.Path(json.load(open(sys.argv[1]))["dataset_path"]).parent)' "$cfg")"
      if [[ -f "$out/manifest.json" ]]; then echo "exists: $out"; continue; fi
      run "$PYTHON" -m swemedmcq.train.preprocess_dataset -c "$cfg" --output-dir "$out" \
          --num-proc "$NUM_PROC" --batch-size "$BATCH_SIZE"
    done
  done
  # Bilingual trace data and the EuroLLM answer-only EN+SV data: the two tokenised train sets,
  # concatenated and shuffled (seed 2000); validation stays per language.
  for pair in "lfm|trace" "eurollm|trace" "eurollm|answer"; do
    IFS='|' read -r s r <<< "$pair"
    if [[ -f "$P/$s/${r}_en_sv/manifest.json" ]]; then echo "exists: $P/$s/${r}_en_sv"; continue; fi
    run "$PYTHON" -m swemedmcq.distill.bilingual.build_dataset --train-dataset "en=$P/$s/${r}_en/train" \
        --train-dataset "sv=$P/$s/${r}_sv/train" --output-path "$P/$s/${r}_en_sv/train" \
        --manifest-path "$P/$s/${r}_en_sv/manifest.json" --shuffle-seed 2000
  done
  # LFM answer-only EN+SV: tokenised from the shuffled EN+SV JSONL mix (as in the released run).
  if [[ ! -f "$P/lfm/answer_en_sv/manifest.json" ]]; then
    run "$PYTHON" -m swemedmcq.train.preprocess_dataset -c configs/train/preprocess/SweMedMCQ-LFM2.5-1.2B-Answer-EN-SV.json \
        --output-dir "$P/lfm/answer_en_sv" --num-proc "$NUM_PROC" --batch-size "$BATCH_SIZE"
  fi
  run "$PYTHON" -m swemedmcq.distill.bilingual.check_prompt_id_overlap \
      --train-path "$EN/flat/train.jsonl" --train-path "$SV/splits/flat/train.jsonl" \
      --validation-path "$EN/flat/validation_train_time.jsonl" --validation-path "$SV/splits/flat/validation_train_time.jsonl"
fi
