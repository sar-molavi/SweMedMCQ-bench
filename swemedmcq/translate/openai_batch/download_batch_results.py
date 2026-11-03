"""
Download the output (and optional error log) of a completed OpenAI Batch job.

Example usage:

    python -m swemedmcq.translate.openai_batch.download_batch_results \
        --metadata-file batch_meta.json \
        --output results.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for downloading batch outputs."""
    parser = argparse.ArgumentParser(
        description="Download the results produced by an OpenAI Batch job."
    )
    parser.add_argument(
        "--batch-id",
        help="Batch identifier returned when the job was submitted.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        help="Path to the submission metadata JSON containing the batch_id.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination file path for the batch output JSONL.",
    )
    parser.add_argument(
        "--errors-output",
        type=Path,
        help="Optional path to store the batch error JSONL (if available).",
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


def fetch_file_bytes(client: OpenAI, file_id: str) -> bytes:
    """Read file content bytes from the OpenAI Files API response."""
    response = client.files.content(file_id)
    if hasattr(response, "read"):
        return response.read()
    if hasattr(response, "content"):
        return response.content  # type: ignore[no-any-return]
    if isinstance(response, bytes):
        return response
    raise TypeError(f"Unexpected response type for file content: {type(response)}")


def save_bytes(data: bytes, destination: Path) -> None:
    """Write binary content to disk."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        handle.write(data)


def main() -> None:
    """Download the batch output file and optional error file."""
    args = parse_args()
    load_dotenv()
    batch_id = resolve_batch_id(args)
    client = build_client(args.api_key, args.base_url)
    batch = client.batches.retrieve(batch_id)

    if not batch.output_file_id:
        raise RuntimeError(
            f"Batch {batch_id} does not have an output file yet (status={batch.status})."
        )

    output_bytes = fetch_file_bytes(client, batch.output_file_id)
    save_bytes(output_bytes, args.output)
    print(f"Batch results saved to {args.output}")

    if args.errors_output and batch.error_file_id:
        error_bytes = fetch_file_bytes(client, batch.error_file_id)
        save_bytes(error_bytes, args.errors_output)
        print(f"Batch errors saved to {args.errors_output}")
    elif args.errors_output:
        print("No error file available for this batch; skipping error download.")


if __name__ == "__main__":
    main()
