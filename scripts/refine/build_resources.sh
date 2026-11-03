#!/usr/bin/env bash
# Rebuild the term base and lexicons of the refinement loop into paths.yaml `resources`
# (data/resources; the released build is already there, so this is only needed to rebuild).
#
#   bash scripts/refine/build_resources.sh            # fetch (if missing) + build (skipped if up to date) + coverage
#   bash scripts/refine/build_resources.sh --force    # re-download and rebuild everything
#   OFFLINE=1 bash scripts/refine/build_resources.sh  # no downloads; build from what is in termbase_raw
#
# Raw inputs: paths.yaml `termbase_raw` (see swemedmcq/refine/resources/build_termbase.py for the
# expected files: SALDO/SALDOM, NPL, KSH97-P, hunspell, termbank, KVÅ, FreeDict; Wikidata ATC and
# Svensk MeSH are downloaded here).
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
FORCE=()
[[ "${1:-}" == "--force" ]] && FORCE=(--force)

if [[ "${OFFLINE:-0}" != "1" ]]; then
  run "$PYTHON" -m swemedmcq.refine.resources.fetch_wikidata_atc "${FORCE[@]}" \
    || echo "WARNING: Wikidata fetch failed; building without npl_atc rows" >&2
  run "$PYTHON" -m swemedmcq.refine.resources.fetch_swemesh "${FORCE[@]}" \
    || echo "WARNING: Svensk MeSH / NLM fetch failed; building without them" >&2
fi
run "$PYTHON" -m swemedmcq.refine.resources.build_termbase "${FORCE[@]}"
ls -la "$(path resources)"
