#!/usr/bin/env bash
# Run the generation of one evaluation job on a cluster node (called by the
# <results_dir>/jobs/<n>gpu.sbatch arrays that swemedmcq.evaluate.make_jobs writes).
#
#   bash scripts/slurm/eval_job.sh sv_L4a/gpt-oss-120b
#
# Environment: activate the inference environment before (or set ENV_SETUP to a file
# that does it). HF_HUB_CACHE must hold the model snapshots (configs/models.yaml).
# Scoring needs no GPU: run `python -m swemedmcq.evaluate.run <job> --stage score` after.
set -eo pipefail
if [[ -n "${ENV_SETUP:-}" ]]; then source "$ENV_SETUP"; fi
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
# Compilation caches per task, so simultaneous array tasks do not race.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/triton_cache_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}}"
# torch.compile cache of vLLM's compiled mode: node-local, one per task.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${TMPDIR:-/tmp}/vllm_cache_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
exec python -m swemedmcq.evaluate.run "${1:?Expected a job id like sv_L4a/gpt-oss-120b}" --stage generate
