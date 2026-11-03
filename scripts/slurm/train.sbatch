#!/usr/bin/env bash
#SBATCH --job-name=swemedmcq_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --output=logs/train_%x_%j.log
#
# Train one student on one node with 4 GPUs (DeepSpeed ZeRO-2):
#   mkdir -p logs && sbatch --export=ALL,STUDENT=SweMedMCQ-EuroLLM-1.7B-Trace-EN,ENV_SETUP=<file> scripts/slurm/train.sbatch
# The tokenised data (work/preprocessed/...) must be on the cluster, e.g.
#   tar -czf - work/preprocessed/eurollm/trace_en | ssh <USER>@<HOST> "tar -xzf - -C <CLUSTER_DIR>"
# Resubmitting the same command resumes from the newest checkpoint. Bring the result back with
#   ssh <USER>@<HOST> "tar -czf - -C <CLUSTER_DIR> runs/train/$STUDENT/final_checkpoint" | tar -xzf -
set -Eeuo pipefail
cd "${PROJECT:-<CLUSTER_DIR>}"
if [[ -n "${ENV_SETUP:-}" ]]; then source "$ENV_SETUP"; fi   # activates the training environment
export PYTHON="${PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
echo "[$(date --iso-8601=seconds)] job ${SLURM_JOB_ID:-} GPUs ${CUDA_VISIBLE_DEVICES:-unset}"
exec bash scripts/07_train.sh "${STUDENT:?Set STUDENT (configs/train/<STUDENT>.json)}"
