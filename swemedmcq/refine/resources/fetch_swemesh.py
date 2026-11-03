#!/usr/bin/env python
"""Download Svensk MeSH (Karolinska Institutet University Library).

KIB publishes a bulk export of the Swedish MeSH translation as tab-separated
text files (linked from https://mesh.kib.ki.se/info/om-webbplatsen).  The export
also contains the English descriptor names (``terms.csv``), English entry terms
(``entryterms.csv``) and tree numbers (``mesh_numbers.csv``), all taken from the
NLM MeSH release of the same year, so no per-descriptor crawling of mesh.kib.ki.se is needed for the vocabulary itself.

Licence (as stated by KIB): the translation is distributed under CC0; attribute
Karolinska Institutets Bibliotek as the author of the translation.  The MeSH
vocabulary itself is produced by the U.S. National Library of Medicine and its
use is subject to the NLM MeSH terms and conditions
(https://www.nlm.nih.gov/mesh/termscon.html); NLM must be attributed.

Output: ``raw/swemesh/<zip>`` + extracted ``raw/swemesh/<year>/*.csv`` +
``swemesh.meta.json``.  Idempotent unless ``--force``.

In addition the NLM descriptor XML (``desc<year>.xml``, ~310 MB) is downloaded into
``raw/nlm_mesh/``: the KIB export flattens all entry terms of a descriptor, but only the terms of
the descriptor's *preferred concept* are synonyms; the others are narrower concepts (e.g. the
brand ``Zyderm`` under ``Collagen``).  The NLM concept structure is used to keep those at level B.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import requests

from swemedmcq.paths import path as repo_path

INFO_PAGE = "https://mesh.kib.ki.se/info/om-webbplatsen"
DEFAULT_INDEX = "https://stingray.kib.ki.se/mesh/2026_export/text_files/"
USER_AGENT = "SweMedMCQ/1.0 (academic research) python-requests"

DEFAULT_OUT = repo_path("termbase_raw") / "swemesh"


def _get(url: str, **kw) -> requests.Response:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=300, **kw)
    r.raise_for_status()
    return r


def discover_zip_url() -> str:
    """Find the text-file export index on the KIB info page, then the zip in it."""
    index = DEFAULT_INDEX
    try:
        html = _get(INFO_PAGE).text
        m = re.search(r'href="([^"]+text_files/?)"', html)
        if m:
            index = m.group(1)
    except Exception as e:  # fall back to the known index URL
        print(f"[swemesh] could not read info page ({e}); using {index}", file=sys.stderr)
    if not index.endswith("/"):
        index += "/"
    html = _get(index).text
    m = re.search(r'href="\.?/?([^"]+\.zip)"', html)
    if not m:
        raise RuntimeError(f"no zip link on {index}")
    return index + m.group(1)


def fetch(out_dir: Path, force: bool = False) -> Path:
    meta_path = out_dir / "swemesh.meta.json"
    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if (out_dir / meta["extracted_dir"]).is_dir():
            print(f"[swemesh] exists: {out_dir / meta['extracted_dir']} (use --force to refetch)")
            return out_dir / meta["extracted_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    url = discover_zip_url()
    t0 = time.time()
    data = _get(url).content
    zpath = out_dir / url.rsplit("/", 1)[-1]
    tmp = zpath.with_suffix(".zip.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, zpath)
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        for n in names:
            if n.startswith("/") or ".." in Path(n).parts:
                raise ValueError(f"unsafe path in zip: {n}")
        z.extractall(out_dir)
    top = sorted({n.split("/", 1)[0] for n in names if "/" in n})
    extracted = top[0] if top else "."
    meta = {
        "url": url,
        "info_page": INFO_PAGE,
        "retrieved": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "licence": (
            "Svensk MeSH translation: CC0, attribute Karolinska Institutets Bibliotek (KIB); "
            "MeSH vocabulary: U.S. National Library of Medicine, NLM MeSH terms and conditions "
            "(https://www.nlm.nih.gov/mesh/termscon.html)"
        ),
        "file": zpath.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "members": names,
        "extracted_dir": extracted,
        "seconds": round(time.time() - t0, 1),
    }
    tmpm = meta_path.with_suffix(".tmp")
    tmpm.write_text(json.dumps(meta, indent=2))
    os.replace(tmpm, meta_path)
    print(f"[swemesh] {len(data)/1e6:.1f} MB from {url} -> {out_dir / extracted}")
    return out_dir / extracted


NLM_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc{year}.xml"
DEFAULT_NLM_OUT = repo_path("termbase_raw") / "nlm_mesh"


def fetch_nlm(out_dir: Path, year: int = 2026, force: bool = False) -> Path:
    """NLM MeSH descriptor XML (~310 MB).  Needed for the concept structure: entry terms
    of the descriptor's *preferred concept* are synonyms (-> level A with the Swedish preferred
    term); terms of other concepts are narrower/related (-> level B only)."""
    url = NLM_URL.format(year=year)
    path = out_dir / f"desc{year}.xml"
    meta_path = out_dir / "nlm_mesh.meta.json"
    if path.exists() and meta_path.exists() and not force:
        print(f"[nlm_mesh] exists: {path}")
        return path
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    h = hashlib.sha256()
    tmp = path.with_suffix(".xml.tmp")
    n = 0
    with requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=600, stream=True) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                h.update(chunk)
                n += len(chunk)
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, path)
    meta = {
        "url": url,
        "retrieved": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "licence": "NLM MeSH: free of charge, subject to the NLM terms and conditions "
        "(https://www.nlm.nih.gov/databases/download/terms_and_conditions_mesh.html)",
        "file": path.name,
        "sha256": h.hexdigest(),
        "bytes": n,
        "seconds": round(time.time() - t0, 1),
    }
    tmpm = meta_path.with_suffix(".tmp")
    tmpm.write_text(json.dumps(meta, indent=2))
    os.replace(tmpm, meta_path)
    print(f"[nlm_mesh] {n/1e6:.1f} MB from {url} -> {path}")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--nlm-out-dir", type=Path, default=DEFAULT_NLM_OUT)
    p.add_argument("--nlm-year", type=int, default=2026)
    p.add_argument("--no-nlm", action="store_true", help="skip the NLM descriptor XML")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    fetch(a.out_dir, a.force)
    if not a.no_nlm:
        fetch_nlm(a.nlm_out_dir, a.nlm_year, a.force)


if __name__ == "__main__":
    main()
