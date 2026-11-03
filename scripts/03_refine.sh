#!/usr/bin/env bash
# Stage 2: refinement loop on the Swedish test drafts (vLLM, one node with 4 GPUs per shard).
#
#   RUN=full SHARDS=8 bash scripts/03_refine.sh                      # every stage, shards one after another
#   RUN=full SHARDS=8 STAGES=prepare bash scripts/03_refine.sh       # one stage
#   RUN=full SHARD=3 STAGES=shard bash scripts/03_refine.sh          # one shard (e.g. one slurm job)
#
# Stages, in order (every one is resumable and idempotent):
#   prepare   CPU: L0 hygiene, L1 number locale, MedXpertQA tails; shards (stable hash)
#   shard     GPU: the loop for each shard (both/all models loaded in-process)
#   probes    GPU: judge probes (can run next to the shards)
#   assemble  CPU: release layers, provenance, report (needs every shard DONE)
#   expert    CPU: blinded review sheets and their key (CSV + XLSX)
# On a cluster, run `prepare` once, then one 4-GPU job per shard plus one for `probes`
# (scripts/slurm/refine_shards.sbatch), then `assemble` and `expert`.
#
# Inputs: the draft (paths.yaml `draft`), gold letters (paths.yaml `benchmark`), the term base
# (paths.yaml `resources`; REFINE_RESOURCES overrides). The released run used
# configs/refine/with_gemma.yaml (GPT-OSS-120B, Qwen3.8-27B, Gemma-4-31B-it).
# GPT-OSS needs the openai_harmony vocab files; on offline nodes fetch them once with
# scripts/refine/fetch_offline.sh and set TIKTOKEN_ENCODINGS_BASE.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUN="${RUN:-full}"
SHARDS="${SHARDS:-8}"
CONFIG="${CONFIG:-configs/refine/with_gemma.yaml}"
RUN_DIR="$(path refine_runs)/$RUN"
LOGS="$RUN_DIR/logs"
mkdir -p "$LOGS"

if stage prepare; then
  run "$PYTHON" -m swemedmcq.refine.cli.prepare --run-dir "$RUN_DIR" --num-shards "$SHARDS" \
      --input "${DRAFT:-$(path draft)}" --answers "${ANSWERS:-$(path benchmark)}"
fi
if stage shard; then
  if [[ -n "${SHARD:-}" ]]; then list=("$SHARD"); else list=($(seq 0 $((SHARDS - 1)))); fi
  for k in "${list[@]}"; do
    if [[ -f "$RUN_DIR/shards/$(printf '%03d' "$k")/DONE" ]]; then echo "shard $k already DONE"; continue; fi
    run "$PYTHON" -m swemedmcq.refine.cli.run_shard --run-dir "$RUN_DIR" --shard "$k" --config "$CONFIG" \
        2>&1 | tee -a "$LOGS/shard_$(printf '%03d' "$k").log"
  done
fi
if stage probes; then
  run "$PYTHON" -m swemedmcq.refine.evaluation.probes --run-dir "$RUN_DIR" --config "$CONFIG" \
      2>&1 | tee -a "$LOGS/probes.log"
fi
if stage assemble; then
  run "$PYTHON" -m swemedmcq.refine.cli.assemble --run-dir "$RUN_DIR" --input "${DRAFT:-$(path draft)}"
fi
if stage expert; then
  run "$PYTHON" -m swemedmcq.refine.evaluation.expert_sheet make --run-dir "$RUN_DIR"
fi
echo "release layers: $RUN_DIR/release/ (layer4a_auto.jsonl = benchmark text, layer4_refined.jsonl = refined)"
