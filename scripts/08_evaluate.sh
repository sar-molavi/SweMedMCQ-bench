#!/usr/bin/env bash
# Stage 6: evaluate every configuration on the test sets, then collect per-item results.
#
#   GPUS=0,1,2,3 bash scripts/08_evaluate.sh                  # job lists, all jobs on local GPUs, collect
#   GPUS=0,1 ONLY='euro-' bash scripts/08_evaluate.sh         # a subset (regex on the job id)
#   STAGES=jobs bash scripts/08_evaluate.sh                   # only the job lists (+ slurm arrays)
#   STAGES="score collect" bash scripts/08_evaluate.sh        # after generating on a cluster
#
# Stages:
#   jobs         <results_dir>/jobs/all.txt, <n>gpu.txt and <n>gpu.sbatch (configs/models.yaml x sets)
#   local        run the jobs on the GPUs in $GPUS (several jobs in parallel; busy GPUs are skipped;
#                resumable: finished items and finished jobs are skipped)
#   score        score every job whose generations are complete (no GPU), e.g. after a cluster run
#   collect      <results_dir>/items.parquet: one row per configuration, layer and item
#   stats        <results_dir>/paired_{core,hard}.csv: bootstrap intervals and McNemar tests (swemedmcq.evaluate.stats)
# Needs the test-set files (scripts/04_build_testset.sh). Students load from the Hugging Face Hub
# (configs/models.yaml) or from a local folder.
# One job by hand: python -m swemedmcq.evaluate.run sv_L4a/gpt-oss-20b [--stage generate|score] [--dry-run]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

R="$(path results_dir)"
ONLY_ARGS=(); [[ -n "${ONLY:-}" ]] && ONLY_ARGS=(--only "$ONLY")

if stage jobs; then
  run "$PYTHON" -m swemedmcq.evaluate.make_jobs
fi
if stage local; then
  run "$PYTHON" -m swemedmcq.evaluate.run_local --gpus "${GPUS:?Set GPUS, e.g. GPUS=0,1,2,3}" \
      --jobs "${JOBS:-$R/jobs/all.txt}" "${ONLY_ARGS[@]}"
fi
if stage score && [[ -f "${JOBS:-$R/jobs/all.txt}" ]]; then
  while read -r job; do
    [[ -n "$job" ]] || continue
    if [[ -n "${ONLY:-}" && ! "$job" =~ $ONLY ]]; then continue; fi
    run env CUDA_VISIBLE_DEVICES= "$PYTHON" -m swemedmcq.evaluate.run "$job" --stage score || true
  done < "${JOBS:-$R/jobs/all.txt}"
fi
if stage collect; then
  run "$PYTHON" -m swemedmcq.evaluate.collect
fi
if stage stats; then
  for subset in core hard; do
    run "$PYTHON" -m swemedmcq.evaluate.stats --subset "$subset" --out "$R/paired_$subset.csv"
  done
fi
