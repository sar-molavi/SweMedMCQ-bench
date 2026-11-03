"""Decode one selected completion from Swedish MCQ vLLM result pickles.

This is the Swedish counterpart to :mod:`swemedmcq.distill.unravel_en_responses`.  It
intentionally writes one JSONL row per input prompt, so a multi-sample vLLM
run can be represented as separate logical source runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import signal
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from transformers import AutoTokenizer

from .unravel_en_responses import prompt_id, unravel_text

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for CLI ergonomics.
    tqdm = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


@contextmanager
def sample_timeout(seconds: float, path: Path) -> Iterator[None]:
    """Raise TimeoutError if one pickle takes too long to load/decode/parse."""
    if seconds <= 0:
        yield
        return
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    def handler(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Timed out after {seconds:g}s while processing {path}")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def rows(
    input_dir: Path,
    tokenizer: Any,
    response_format: str,
    sample_index: int,
    source_run: str,
    source_model: str,
    allow_missing_sample: bool,
    timeout_seconds: float,
    counts: Counter[str],
    log_every: int,
    show_progress: bool,
    progress_position: int,
) -> Iterator[dict[str, Any]]:
    """Yield the requested completion from every result pickle.

    A missing requested completion is a configuration/data error, rather than
    a silently malformed response.  This prevents a source run that silently
    contains completion zero instead of the requested one.
    """
    paths = sorted(input_dir.glob("result_*.pkl"))
    iterator = paths
    progress_bar = None
    if show_progress and tqdm is not None:
        progress_bar = tqdm(
            paths,
            desc=source_run,
            unit="pkl",
            dynamic_ncols=True,
            position=progress_position,
        )
        iterator = progress_bar

    for index, path in enumerate(iterator, start=1):
        if progress_bar is not None:
            progress_bar.set_postfix_str(path.name[-72:], refresh=False)
        if log_every > 0 and (index == 1 or index % log_every == 0):
            LOGGER.info(
                "%s: processing %s/%s: %s",
                source_run,
                index,
                len(paths),
                path.name,
            )

        try:
            timeout_context = sample_timeout(timeout_seconds, path)
            with timeout_context:
                with path.open("rb") as handle:
                    request = pickle.load(handle)
                outputs = list(request.outputs)
                if sample_index >= len(outputs):
                    if allow_missing_sample:
                        counts["missing_sample"] += 1
                        continue
                    raise ValueError(
                        f"Requested sample_index={sample_index} from {path.name}, "
                        f"but the result contains only {len(outputs)} completion(s)."
                    )
                decoded = tokenizer.decode(
                    outputs[sample_index].token_ids,
                    skip_special_tokens=False,
                )
                yield {
                    "prompt_id": prompt_id(path),
                    "source_run": source_run,
                    "source_model": source_model,
                    "sample_index": sample_index,
                    **unravel_text(decoded, response_format),
                    "source_file": str(path),
                }
        except TimeoutError as exc:
            counts["timeout"] += 1
            LOGGER.error("%s: skipping timed-out pickle %s: %s", source_run, path.name, exc)
            continue


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO if args.log_every >= 0 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_path.with_suffix(args.output_path.suffix + ".tmp")
    counts: Counter[str] = Counter()
    source_run = args.source_run or args.output_path.stem
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows(
            args.input_dir,
            tokenizer,
            args.response_format,
            args.sample_index,
            source_run,
            args.model_name,
            args.allow_missing_sample,
            args.timeout_seconds,
            counts,
            args.log_every,
            not args.no_progress,
            args.progress_position,
        ):
            counts[row["parse_status"]] += 1
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
    temporary.replace(args.output_path)
    print(
        json.dumps(
            {
                "output": str(args.output_path),
                "sample_index": args.sample_index,
                "allow_missing_sample": args.allow_missing_sample,
                "timeout_seconds": args.timeout_seconds,
                "counts": counts,
            },
            ensure_ascii=True,
            default=dict,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--source-run", help="Logical source-run name recorded in output.")
    parser.add_argument(
        "--response-format", required=True, choices=("gpt_oss", "qwen")
    )
    parser.add_argument(
        "--sample-index", type=int, default=0, metavar="N",
        help="Zero-based vLLM completion index (default: 0).",
    )
    parser.add_argument(
        "--allow-missing-sample",
        action="store_true",
        help="Skip pickles that do not contain --sample-index instead of failing.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help=(
            "Maximum seconds allowed for one pickle load/decode/parse. "
            "Timed-out pickles are skipped. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log progress every N pickles. Use 0 to disable periodic logs.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress display.",
    )
    parser.add_argument(
        "--progress-position",
        type=int,
        default=0,
        help="Terminal row used by tqdm. Useful when several extractors run in parallel.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
