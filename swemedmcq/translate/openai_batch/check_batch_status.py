"""
Check the processing status of an OpenAI Batch job.

Usage examples:

    python -m swemedmcq.translate.openai_batch.check_batch_status --batch-id BATCH_ID
    python -m swemedmcq.translate.openai_batch.check_batch_status --metadata-file batch_meta.json
"""

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for a batch status lookup."""
    parser = argparse.ArgumentParser(
        description="Retrieve the current status of an OpenAI Batch job."
    )
    parser.add_argument(
        "--batch-id",
        help="Batch identifier returned by the submission step.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        help="Path to the JSON metadata file produced during submission.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key. If omitted, OPENAI_API_KEY environment variable is used.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional custom API base URL.",
    )
    return parser.parse_args()


def resolve_batch_id(args: argparse.Namespace) -> str:
    """Resolve a batch ID from direct input or a metadata file."""
    if args.batch_id:
        return args.batch_id

    if args.metadata_file:
        if not args.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {args.metadata_file}")
        with args.metadata_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        batch_id = payload.get("batch_id")
        if not batch_id:
            raise KeyError(
                f"batch_id missing from metadata file: {args.metadata_file}"
            )
        return str(batch_id)

    raise ValueError("Provide either --batch-id or --metadata-file.")


def build_client(api_key: str | None, base_url: str | None) -> OpenAI:
    """Create an OpenAI client from CLI arguments or environment variables."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY."
        )
    return OpenAI(api_key=key, base_url=base_url)


def main() -> None:
    """Fetch and print the current batch status."""
    load_dotenv()
    args = parse_args()
    batch_id = resolve_batch_id(args)
    client = build_client(args.api_key, args.base_url)
    batch = client.batches.retrieve(batch_id)

    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Created at: {batch.created_at}")
    if batch.in_progress_at:
        print(f"In progress at: {batch.in_progress_at}")
    if batch.completed_at:
        print(f"Completed at: {batch.completed_at}")
    if batch.failed_at:
        print(f"Failed at: {batch.failed_at}")
    if batch.expired_at:
        print(f"Expired at: {batch.expired_at}")
    if batch.output_file_id:
        print(f"Output file id: {batch.output_file_id}")
    if batch.error_file_id:
        print(f"Error file id: {batch.error_file_id}")


if __name__ == "__main__":
    main()
