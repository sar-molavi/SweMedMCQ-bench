"""Merge Swedish unraveled MCQ responses with prompts and gold labels."""

from __future__ import annotations

from .combine_unraveled_en_responses import (
    combine_files,
    discover_input_files,
    load_prompt_metadata,
)
from .combine_unraveled_en_responses import parse_args as _parse_args


def main() -> None:
    """Run the language-neutral merge with Swedish command defaults."""
    args = _parse_args()
    input_files = discover_input_files(args.input_dir, args.output_path)
    if not input_files:
        raise FileNotFoundError(f"No JSONL files found in {args.input_dir}")
    combine_files(
        input_files=input_files,
        output_path=args.output_path,
        prompt_metadata=load_prompt_metadata(args.data_dir, args.prompt_path),
        workers=max(args.workers, 1),
        include_empty_prompts=args.include_empty_prompts,
    )


if __name__ == "__main__":
    main()
