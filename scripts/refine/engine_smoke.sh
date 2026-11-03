#!/usr/bin/env bash
# Engine smoke test of the refinement loop: all models of a config loaded at once, non-empty
# reasoning, valid JSON, no Gemma thought-channel leakage, GPU placement, batching
# throughput and 100 % cache replay. Writes <refine_runs>/<run-name>/summary.json.
#
#   CONFIG=configs/refine/with_gemma.yaml bash scripts/refine/engine_smoke.sh --models gptoss,qwen,gemma --fresh
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
CONFIG="${CONFIG:-configs/refine/with_gemma.yaml}"
export TOKENIZERS_PARALLELISM=false
exec "$PYTHON" -m swemedmcq.refine.cli.engine_smoke --config "$CONFIG" "$@"
