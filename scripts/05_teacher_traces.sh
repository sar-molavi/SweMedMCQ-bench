#!/usr/bin/env bash
# Stage 4a: teacher reasoning traces on the training questions (vLLM, local GPUs).
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/05_teacher_traces.sh
#   STAGES="prompts" bash scripts/05_teacher_traces.sh
#   STAGES="en" RUNS="en_response_train_gpt_oss_120b" bash scripts/05_teacher_traces.sh
#
# Prompts: the explanation prompt (swemedmcq/common/prompts.py) on the English training
# questions (source benchmark, train.jsonl) and on their Swedish translations (stage 1b).
# Teachers, one run folder each (the folder name is the `source_run` used downstream):
#   GPT-OSS-120B     greedy (temperature 0) and twice at temperature 0.7 (_t=0.7, _t=0.7_v2)
#   Qwen3.6-35B-A3B  greedy and once at temperature 0.7
# Structured JSON {explanation, answer_index}; GPT-OSS at its template's default reasoning
# effort, JSON after the analysis channel; Qwen with thinking on, JSON after </think>.
# Output-token limit: the runner default (30,720). Sampled runs (temperature 0.7) differ
# from run to run; the released train/validation split is fixed by data/train/split/.
# Every run is resumable (existing result_<id>.pkl files are skipped).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

W="$(path work_dir)"
P="$W/prompts"
EN_PROMPTS="$P/en_train_prompt_set.jsonl"
SV_PROMPTS="$P/sw_train_trans_filtered_prompt_set.jsonl"
GPT=openai/gpt-oss-120b
QWEN=Qwen/Qwen3.6-35B-A3B
# name|model|temperature|extra runner flags
EN_RUNS=(
  "en_response_train_gpt_oss_120b|$GPT|0|"
  "en_response_train_gpt_oss_120b_t=0.7|$GPT|0.7|"
  "en_response_train_gpt_oss_120b_t=0.7_v2|$GPT|0.7|"
  "en_response_train_qwen_3.6_35B_A3B|$QWEN|0|"
  "en_response_train_qwen_3.6_35B_A3B_t=0.7|$QWEN|0.7|"
)
SV_RUNS=(
  "sw_response_train_gpt_oss_120b|$GPT|0|"
  "sw_response_train_gpt_oss_120b_t=0.7|$GPT|0.7|"
  "sw_response_train_gpt_oss_120b_t=0.7_v2|$GPT|0.7|"
  "sw_response_train_qwen_3.6_35B_A3B|$QWEN|0|"
  "sw_response_train_qwen_3.6_35B_A3B_t=0.7|$QWEN|0.7|"
)

if stage prompts; then
  run "$PYTHON" -m swemedmcq.distill.make_response_prompts_en -i "$(path source_benchmark)" --file-name train \
      --output-name train --output-dir "$P"
  run "$PYTHON" -m swemedmcq.distill.make_response_prompts_sw -i "$(path train_translation)" \
      --output-name train_trans_filtered --output-dir "$P"
fi

teacher() {  # lang module prompts spec
  local lang="$1" module="$2" prompts="$3" name model temp flags
  IFS='|' read -r name model temp flags <<< "$4"
  if [[ -n "${RUNS:-}" && " $RUNS " != *" $name "* ]]; then return; fi
  # shellcheck disable=SC2086
  run "$PYTHON" -m "$module" -i "$prompts" -o "$W/teacher/$lang/$name" --model-name "$model" \
      --temperature "$temp" $flags
}
if stage en; then
  for spec in "${EN_RUNS[@]}"; do teacher en swemedmcq.common.en_response "$EN_PROMPTS" "$spec"; done
fi
if stage sv; then
  for spec in "${SV_RUNS[@]}"; do teacher sv swemedmcq.common.sw_response "$SV_PROMPTS" "$spec"; done
fi
