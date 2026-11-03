#!/usr/bin/env bash
#SBATCH --job-name=swemedmcq_refine
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=64G
#SBATCH --array=0-7
#SBATCH --output=logs/refine_%A_%a.log
#
# Refinement loop: one 4-GPU job per shard (array task k = shard k), the configuration of
# the released run (GPT-OSS-120B on GPU 0, Qwen3.8-27B on GPUs 1-2, Gemma-4-31B-it on GPU 3;
# one H200 each). Run `prepare` once before (CPU) and `probes`, `assemble`, `expert` after:
#   RUN=full SHARDS=8 STAGES=prepare bash scripts/03_refine.sh
#   mkdir -p logs && sbatch scripts/slurm/refine_shards.sbatch   # this file
#   RUN=full STAGES=probes bash scripts/03_refine.sh   (inside one more 4-GPU allocation)
#   RUN=full SHARDS=8 STAGES="assemble expert" bash scripts/03_refine.sh
# Resubmitting is safe: finished items are skipped and finished LLM calls are replayed
# from the request cache, so a restart continues where it stopped.
#
# Copy the repository with the data (and, for offline nodes, the HF cache with the model
# snapshots and the harmony vocab of scripts/refine/fetch_offline.sh) to <CLUSTER_DIR>, e.g.
#   tar -czf - data configs scripts swemedmcq | ssh <USER>@<HOST> "mkdir -p <CLUSTER_DIR> && tar -xzf - -C <CLUSTER_DIR>"
set -euo pipefail
cd "${PROJECT:-<CLUSTER_DIR>}"
if [[ -n "${ENV_SETUP:-}" ]]; then source "$ENV_SETUP"; fi   # activates the inference environment
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_refine_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export VLLM_USE_FLASHINFER_SAMPLER=0
# Offline nodes: export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TIKTOKEN_ENCODINGS_BASE=<dir>
RUN="${RUN:-full}" SHARDS="${SHARDS:-8}" SHARD="${SLURM_ARRAY_TASK_ID:?}" STAGES=shard \
  exec bash scripts/03_refine.sh
