#!/usr/bin/env bash
# Stage 5: fine-tune one student (full fine-tuning, 4 GPUs, accelerate + DeepSpeed ZeRO-2).
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/07_train.sh SweMedMCQ-EuroLLM-1.7B-Trace-SV
#   bash scripts/07_train.sh configs/train/SweMedMCQ-LFM2.5-1.2B-Answer-EN.json
#   for s in configs/train/SweMedMCQ-*.json; do bash scripts/07_train.sh "$s"; done
#
# Students: SweMedMCQ-<LFM2.5-1.2B|EuroLLM-1.7B>-<Answer|Trace>-<EN|SV|SV-RW|EN-SV>
# (Answer: EN, SV, EN-SV; Trace: EN, SV, SV-RW, EN-SV). The data must be tokenised first
# (scripts/06_build_sft_data.sh, stage preprocess). EN-SV configs use the bilingual trainer
# (one shuffled EN+SV train set, EN and SV validation losses, weighted aggregate for early
# stopping); the others the single-language trainer. Every config: 2 epochs, global batch
# 128 (4 GPUs x 32), lr 5e-5 cosine, warm-up 0.05, NLL loss, trained embeddings, bf16,
# padding-free, evaluation and checkpoint every 200 steps, early stopping (patience 4 for the
# trace students, 10 for the answer-level students), checkpoint with the best validation loss.
#
# Environment:
#   PYTHON                interpreter of the training environment (requirements/train.txt)
#   CUDA_VISIBLE_DEVICES  exactly four GPU ids (default 0,1,2,3; Slurm sets it)
#   ACCELERATE_CONFIG     default configs/train/accelerate_deepspeed_zero2_4gpu.yaml
#   RESUME                auto (default): continue from the newest checkpoint-*; 0: refuse if any exist
#   SKIP_CHECK=1          skip the CPU dry run (swemedmcq.train.check_config)
#   DRY_RUN=1             run the checks and print the launch command only
#
# Re-running after a crash, a time-out or a finished run is safe: it resumes from the
# newest complete checkpoint, or, if final_checkpoint exists, only redoes the post-step.
# Post-step: EuroLLM checkpoints get the full tokenizer file set (finalize_checkpoint);
# LFM trace checkpoints get the reasoning chat template (create_lfm_dynamic_tokenizer),
# which evaluation needs (enable_thinking opens or closes the <think> block).
# The final model is <output_dir>/final_checkpoint; to evaluate it, point the student's
# `model` in configs/models.yaml to that folder.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
set -E

TARGET="${1:?Give a student name or a config path}"
CONFIG="$TARGET"
[[ -f "$CONFIG" ]] || CONFIG="configs/train/$TARGET.json"
[[ -f "$CONFIG" ]] || { echo "Config not found: $TARGET" >&2; exit 2; }
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/train/accelerate_deepspeed_zero2_4gpu.yaml}"
RESUME="${RESUME:-auto}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false

log() { echo "[$(date --iso-8601=seconds)] $*"; }
IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#DEVICES[@]} != 4 )); then
  echo "CUDA_VISIBLE_DEVICES must list exactly four GPUs (got '$CUDA_VISIBLE_DEVICES')." >&2
  exit 1
fi

read_cfg() { "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' "$CONFIG" "$1"; }
OUTPUT_DIR="$(read_cfg output_dir)"
BASE_MODEL="$(read_cfg model_name_or_path)"
ANSWER_LEVEL="$(read_cfg answer_level_sft)"
FINAL="$OUTPUT_DIR/final_checkpoint"
if [[ -n "$(read_cfg train_dataset_paths)" ]]; then MODULE=swemedmcq.train.bilingual.train_unsloth_full
else MODULE=swemedmcq.train.train_unsloth_full; fi

finalize() {
  case "$BASE_MODEL" in
    *EuroLLM*)
      log "Post-step: EuroLLM tokenizer files and chat template"
      "$PYTHON" -m swemedmcq.train.finalize_checkpoint "$FINAL" --base "$BASE_MODEL" ;;
    *LFM2.5*)
      if [[ "$ANSWER_LEVEL" != True ]]; then
        log "Post-step: reasoning chat template for the LFM trace student"
        "$PYTHON" -m swemedmcq.train.create_lfm_dynamic_tokenizer "$FINAL"
      fi ;;
  esac
}

if [[ -f "$FINAL/model.safetensors" ]]; then
  log "final_checkpoint already exists: $FINAL"
  [[ "${DRY_RUN:-0}" == 1 ]] || finalize
  exit 0
fi
if [[ "${SKIP_CHECK:-0}" != 1 ]]; then
  log "CPU dry run: swemedmcq.train.check_config"
  "$PYTHON" -m swemedmcq.train.check_config --config "$CONFIG" --accelerate-config "$ACCELERATE_CONFIG"
fi

# The trainer resumes from the newest checkpoint-* (by mtime). A checkpoint cut off
# mid-save (no trainer_state.json) would crash the resume, so it is moved aside.
CKPTS=()
if [[ -d "$OUTPUT_DIR" ]]; then
  mapfile -t CKPTS < <(find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%T@ %p\n' | sort -rn | cut -d' ' -f2-)
fi
if [[ "$RESUME" == 0 && ${#CKPTS[@]} -gt 0 ]]; then
  echo "RESUME=0 but $OUTPUT_DIR already has checkpoint-* directories." >&2
  exit 1
fi
LATEST=""
for ckpt in "${CKPTS[@]}"; do
  if [[ -f "$ckpt/trainer_state.json" ]]; then LATEST="$ckpt"; break; fi
  AWAY="$OUTPUT_DIR/incomplete-$(basename "$ckpt")-$(date +%Y%m%dT%H%M%S)"
  log "Incomplete checkpoint $ckpt (no trainer_state.json); moving it to $AWAY"
  [[ "${DRY_RUN:-0}" == 1 ]] || mv "$ckpt" "$AWAY"
done
log "${LATEST:+Resuming from $LATEST}${LATEST:-Fresh run into $OUTPUT_DIR}"

COMMAND=("$PYTHON" -m accelerate.commands.launch --config_file "$ACCELERATE_CONFIG" -m "$MODULE" --config "$CONFIG")
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi

mkdir -p "$OUTPUT_DIR/logs"
STAMP="$(date +%Y%m%dT%H%M%S)"
cp "$CONFIG" "$OUTPUT_DIR/logs/train_config.json"
{
  echo "start=$STAMP"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "config=$CONFIG"
  echo "accelerate_config=$ACCELERATE_CONFIG"
  echo "resume_from=${LATEST:-none}"
} > "$OUTPUT_DIR/logs/run_info_$STAMP.txt"
log "Launching: ${COMMAND[*]}"
"${COMMAND[@]}" 2>&1 | tee -a "$OUTPUT_DIR/logs/train.log"
log "Training finished"
finalize
