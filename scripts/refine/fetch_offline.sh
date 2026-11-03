#!/usr/bin/env bash
# Download the vocab files that openai_harmony (GPT-OSS prompt rendering and output parsing)
# otherwise fetches at run time, for GPU nodes without internet. Idempotent.
#
#   bash scripts/refine/fetch_offline.sh [OUT_DIR]       # default <work_dir>/offline/tiktoken
#   export TIKTOKEN_ENCODINGS_BASE=<OUT_DIR>             # then, on the GPU nodes
source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
OUT="${1:-$(path work_dir)/offline/tiktoken}"
mkdir -p "$OUT"
declare -A SHA=(
  [o200k_base]=446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d
  [cl100k_base]=223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7
)
for name in "${!SHA[@]}"; do
  f="$OUT/$name.tiktoken"
  [[ -s "$f" ]] || curl -sSfL -o "$f" "https://openaipublic.blob.core.windows.net/encodings/$name.tiktoken"
  echo "${SHA[$name]}  $f" | sha256sum -c -
done
echo "export TIKTOKEN_ENCODINGS_BASE=$OUT"
