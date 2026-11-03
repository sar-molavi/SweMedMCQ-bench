#!/usr/bin/env bash
# Stage 3: the benchmark file and the evaluation prompt files.
#
#   bash scripts/04_build_testset.sh                    # prompt files from data/test/benchmark.jsonl
#   RUN=full bash scripts/04_build_testset.sh           # first rebuild data/test/ from a refinement run
#                                                       # (gold letters: ANSWERS, default data/test/benchmark.jsonl)
#
# `benchmark` (only with RUN): data/test/benchmark.jsonl, layers/layer{1_draft,2_hygiene,3_locale,4_refined}.jsonl,
#   expert_queue.jsonl,
#   edit_history.jsonl.gz and refinement_report.json
#   from <refine_runs>/<RUN>/release/, the gold letters (ANSWERS, JSON or JSONL) and the hard
#   subsets of the source benchmarks (paths.yaml `source_benchmark`). With the
#   released run this reproduces the released files byte for byte.
# `prompts`: <testset_dir>/data/ en_*, sv_L4a_* (benchmark text) and sv_L4_delta_* (the items
#   whose refined text differs), items.csv and manifest.json; the input of stage 6.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ -n "${RUN:-}" ]] && stage benchmark; then
  run "$PYTHON" -m swemedmcq.testset.make_benchmark --run-dir "$(path refine_runs)/$RUN" \
      --answers "${ANSWERS:-$(path benchmark)}"
fi
if stage prompts; then
  run "$PYTHON" -m swemedmcq.testset.build --refined "$(path refined)"
fi
