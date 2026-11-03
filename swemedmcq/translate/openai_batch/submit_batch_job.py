"""
Submit an OpenAI Batch job for a previously prepared JSONL request file.

The script uploads the JSONL file, creates a batch targeting the Responses API,
and saves the resulting batch metadata (including the job ID) to disk for later
use in status checks or result downloads.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv


from openai import OpenAI


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for submitting a prepared batch request file."""
    parser = argparse.ArgumentParser(
        description="Submit a prepared OpenAI Batch job and persist the batch ID."
    )
    parser.add_argument(
        "--requests-file",
        required=True,
        type=Path,
        help="Path to the JSONL file produced by prepare_batch_requests.py.",
    )
    parser.add_argument(
        "--save-metadata",
        required=True,
        type=Path,
        help="File path where the batch metadata (including batch_id) will be stored.",
    )
    parser.add_argument(
        "--completion-window",
        default="24h",
        help="Requested completion window for the batch job (e.g., 24h).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key. If omitted, OPENAI_API_KEY environment variable is used.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional custom API base URL (useful for Azure or compatible endpoints).",
    )
    return parser.parse_args()


def build_client(api_key: str | None, base_url: str | None) -> OpenAI:
    """Create an OpenAI client from CLI arguments or environment variables."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OpenAI API key not provided. Use --api-key or set OPENAI_API_KEY."
        )
    return OpenAI(api_key=key, base_url=base_url)


def submit_batch(
    client: OpenAI,
    requests_file: Path,
    completion_window: str,
) -> dict[str, Any]:
    """Upload the request file and create the batch job."""
    with requests_file.open("rb") as file_handle:
        uploaded = client.files.create(file=file_handle, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
    )

    return {
        "batch_id": batch.id,
        "status": batch.status,
        "input_file_id": uploaded.id,
        "created_at": batch.created_at,
        "endpoint": batch.endpoint,
        "completion_window": batch.completion_window,
    }


def save_metadata(metadata: dict[str, Any], destination: Path) -> None:
    """Append one metadata record as JSONL."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, sort_keys=True))
        handle.write("\n")


def main() -> None:
    """Submit the batch job and persist the returned metadata."""
    load_dotenv()
    args = parse_args()
    client = build_client(args.api_key, args.base_url)
    metadata = submit_batch(client, args.requests_file, args.completion_window)
    save_metadata(metadata, args.save_metadata)
    print(
        f"Batch submitted. ID saved to {args.save_metadata} (batch_id={metadata['batch_id']})."
    )


if __name__ == "__main__":
    main()
