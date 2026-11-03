#!/usr/bin/env python
"""Download Wikidata ATC labels (one SPARQL query).

For every Wikidata item with an ATC code (P267) we fetch: the ATC code, the
English label, English aliases, the English WHO INN (P2275, if present) and the
Swedish label.  Output: ``raw/wikidata_atc/wikidata_atc.json`` (the raw SPARQL
JSON result) plus ``wikidata_atc.meta.json`` (URL, retrieval date, sha256,
record count).  Licence of Wikidata data: CC0.

The script is idempotent: an existing download is kept unless ``--force``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

from swemedmcq.paths import path as repo_path

ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "SweMedMCQ/1.0 (academic research on Swedish medical benchmark translation) python-requests"

QUERY = """
SELECT ?item ?atc ?enLabel ?svLabel ?innEn
       (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases)
WHERE {
  ?item wdt:P267 ?atc .
  OPTIONAL { ?item rdfs:label ?enLabel . FILTER(LANG(?enLabel) = "en") }
  OPTIONAL { ?item rdfs:label ?svLabel . FILTER(LANG(?svLabel) = "sv") }
  OPTIONAL { ?item skos:altLabel ?alias . FILTER(LANG(?alias) = "en") }
  OPTIONAL { ?item wdt:P2275 ?innEn . FILTER(LANG(?innEn) = "en") }
}
GROUP BY ?item ?atc ?enLabel ?svLabel ?innEn
"""

DEFAULT_OUT = repo_path("termbase_raw") / "wikidata_atc"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fetch(out_dir: Path, force: bool = False, retries: int = 3) -> Path:
    out = out_dir / "wikidata_atc.json"
    meta = out_dir / "wikidata_atc.meta.json"
    if out.exists() and meta.exists() and not force:
        print(f"[wikidata] exists: {out} (use --force to refetch)")
        return out
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            t0 = time.time()
            r = requests.post(
                ENDPOINT,
                data={"query": QUERY, "format": "json"},
                headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
                timeout=300,
            )
            r.raise_for_status()
            data = r.content
            parsed = json.loads(data)
            n = len(parsed["results"]["bindings"])
            _atomic_write_bytes(out, data)
            info = {
                "url": ENDPOINT,
                "query": QUERY.strip(),
                "retrieved": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "licence": "CC0 1.0 (Wikidata structured data)",
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "records": n,
                "seconds": round(time.time() - t0, 1),
            }
            _atomic_write_bytes(meta, json.dumps(info, indent=2).encode())
            print(f"[wikidata] {n} rows, {len(data)/1e6:.1f} MB -> {out}")
            return out
        except Exception as e:  # network / timeout / JSON error
            last_err = e
            print(f"[wikidata] attempt {attempt} failed: {e}", file=sys.stderr)
            time.sleep(10 * attempt)
    raise RuntimeError(f"Wikidata download failed: {last_err}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    fetch(a.out_dir, a.force)


if __name__ == "__main__":
    main()
