"""Give a EuroLLM final_checkpoint the full EuroLLM file set.

A complete EuroLLM checkpoint contains:
    chat_template.jinja config.json generation_config.json model.safetensors
    tokenizer.json tokenizer_config.json tokenizer.model training_args.bin
    (+ special_tokens_map.json)
The trainer saves the model and `tokenizer.save_pretrained(...)`; depending on
the transformers version the sentencepiece `tokenizer.model` and
`special_tokens_map.json` are not written. This script copies any missing
tokenizer file from the base model (the tokenizer used in training), writes
chat_template.jinja from the base tokenizer_config if it is missing, and checks
that the saved chat template equals the base template.

    python -m swemedmcq.train.finalize_checkpoint <run>/final_checkpoint [--base utter-project/EuroLLM-1.7B-Instruct] [--check-only]

CPU only; no torch/transformers import. Exits non-zero if a required file is
still missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REQUIRED = [
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "training_args.bin",
]
COPY_FROM_BASE = ["tokenizer.model", "special_tokens_map.json"]


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def base_file(repo: str, name: str) -> Path | None:
    local = Path(repo)
    if local.is_dir():
        return local / name if (local / name).is_file() else None
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo, name))
    except Exception as exc:  # offline or file absent upstream
        print(f"  could not fetch {name} from {repo}: {exc.__class__.__name__}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--base", default="utter-project/EuroLLM-1.7B-Instruct")
    ap.add_argument("--check-only", action="store_true", help="report only; do not copy anything")
    args = ap.parse_args()
    ckpt: Path = args.checkpoint
    if not ckpt.is_dir():
        print(f"not a directory: {ckpt}")
        return 1

    for name in COPY_FROM_BASE:
        target = ckpt / name
        if target.is_file():
            continue
        src = None if args.check_only else base_file(args.base, name)
        if src is None:
            print(f"  missing {name}" + ("" if args.check_only else " (no base copy available)"))
            continue
        shutil.copy2(src, target)
        print(f"  copied {name} from base {args.base}")

    base_cfg = base_file(args.base, "tokenizer_config.json")
    base_template = None
    if base_cfg is not None:
        base_template = json.loads(base_cfg.read_text()).get("chat_template")
    template_path = ckpt / "chat_template.jinja"
    if not template_path.is_file() and not args.check_only and base_template:
        template_path.write_text(base_template)
        print("  wrote chat_template.jinja from the base tokenizer_config")
    if template_path.is_file() and base_template is not None:
        same = template_path.read_text() == base_template
        print(f"  chat_template.jinja {'matches' if same else 'DIFFERS FROM'} the base chat template")
    if (ckpt / "tokenizer.model").is_file():
        base_sp = base_file(args.base, "tokenizer.model")
        if base_sp is not None:
            same = md5(ckpt / "tokenizer.model") == md5(base_sp)
            print(f"  tokenizer.model {'matches' if same else 'DIFFERS FROM'} the base tokenizer.model")

    missing = [n for n in REQUIRED if not (ckpt / n).is_file()]
    present = sorted(p.name for p in ckpt.iterdir())
    print(f"  {ckpt}: {', '.join(present)}")
    if missing:
        print(f"  MISSING required files: {', '.join(missing)}")
        return 1
    print("  final_checkpoint has the full EuroLLM file set")
    return 0


if __name__ == "__main__":
    sys.exit(main())
