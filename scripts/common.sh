# Shared settings of the runbook scripts (sourced, not run).
# Every script runs from the repository root with the interpreter in $PYTHON
# (default: `python` of the activated environment). Locations come from
# configs/paths.yaml (override any of them with SWEMEDMCQ_<KEY>, see swemedmcq/paths.py).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

# path KEY: a location from configs/paths.yaml
path() { "$PYTHON" -m swemedmcq.paths "$1"; }
# run CMD...: print, then run (DRY_RUN=1 prints only)
run() {
  printf '+ %s\n' "$*" >&2
  if [[ "${DRY_RUN:-0}" != 1 ]]; then "$@"; fi
}
# stage NAME: is NAME selected? (STAGES="a b c" limits a script to some of its stages)
stage() { [[ -z "${STAGES:-}" || " $STAGES " == *" $1 "* ]]; }
