#!/usr/bin/env bash
# Upload the 14 student models in hf/models/ to the Hugging Face Hub as public model
# repositories, and group them in a public collection.
#   hf auth login            # once, with a write token
#   HF_USER=<user or org> bash hf/upload.sh [model folder ...]
# Without arguments every folder in hf/models/ is uploaded. Re-running is safe: existing
# repositories are kept and only changed files are uploaded.
set -euo pipefail
HF_USER="${HF_USER:-Anonymiserad}"
HF="${HF:-hf}"
PYTHON="${PYTHON:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ "$#" -gt 0 ]; then dirs=("$@"); else dirs=("$HERE"/models/SweMedMCQ-*/); fi
for dir in "${dirs[@]}"; do
  name="$(basename "$dir")"
  echo ">> $HF_USER/$name"
  "$HF" repos create "$HF_USER/$name" --repo-type model --public --exist-ok
  "$HF" upload "$HF_USER/$name" "$dir" . --repo-type model --commit-message "Upload model"
done

# Public collection with all models (needs a token that may manage collections; skipped otherwise).
HF_USER="$HF_USER" MODELS_DIR="$HERE/models" "$PYTHON" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import add_collection_item, create_collection

user = os.environ["HF_USER"]
try:
    collection = create_collection(
        title="SweMedMCQ students",
        namespace=user,
        description="Small LFM2.5-1.2B and EuroLLM-1.7B models fine-tuned for English and Swedish medical multiple-choice questions.",
        private=False,
        exists_ok=True,
    )
    for path in sorted(Path(os.environ["MODELS_DIR"]).glob("SweMedMCQ-*")):
        add_collection_item(collection.slug, item_id=f"{user}/{path.name}", item_type="model", exists_ok=True)
    print(collection.url)
except Exception as exc:  # e.g. a fine-grained token without collection rights
    print(f"collection not created: {exc}")
PY
