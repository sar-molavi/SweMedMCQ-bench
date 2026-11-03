"""Paired statistics of the evaluated configurations: bootstrap intervals and McNemar tests.

For every configuration with complete English and Swedish results it reports, on the
paired core items, the accuracy in each language with a 95% percentile bootstrap
interval (10,000 resamples of items, seed 13), the English-Swedish gap with a paired
bootstrap interval, the number of items solved in only one language, and an exact
McNemar test. An answer that is missing or invalid counts as wrong.

    python -m swemedmcq.evaluate.stats                          # runs/eval/items.parquet, core items
    python -m swemedmcq.evaluate.stats --subset hard --out runs/eval/paired_hard.csv
    python -m swemedmcq.evaluate.stats --layer L4               # Swedish layer 4 instead of 4a

Reads the per-item results of ``swemedmcq.evaluate.collect`` and the ``core`` and ``hard``
flags of the benchmark file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from swemedmcq import paths

RNG_SEED = 13
N_BOOT = 10_000


def bootstrap_ci(correct: np.ndarray, n_boot: int = N_BOOT, seed: int = RNG_SEED) -> tuple[float, float]:
    """95% percentile bootstrap interval for accuracy (in %)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(correct, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1) * 100
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT,
                        seed: int = RNG_SEED) -> tuple[float, float]:
    """95% interval for mean(a) - mean(b) on the same items (in % points)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1) * 100
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar test on paired binary outcomes a, b."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return {"both": int((a & b).sum()), "only_a": only_a, "only_b": only_b,
            "neither": int((~a & ~b).sum()), "p_value": float(p)}


def item_flags(benchmark: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in benchmark.open(encoding="utf-8") if line.strip()]
    return pd.DataFrame({"prompt_id": [r["prompt_id"] for r in rows],
                         "core_flag": [bool(r["core"]) for r in rows],
                         "hard_flag": [bool(r.get("hard", False)) for r in rows]})


def paired_table(items: pd.DataFrame, flags: pd.DataFrame, layer: str, subset: str) -> pd.DataFrame:
    df = items.merge(flags, on="prompt_id", how="inner")
    keep = df.core_flag & (df.hard_flag if subset == "hard" else df.core_flag)
    if subset == "rest":
        keep = df.core_flag & ~df.hard_flag
    df = df[keep]
    df["is_correct"] = df.is_correct.fillna(False).astype(bool)
    out = []
    for config, sub in df.groupby("config", sort=False):
        en = sub[sub.layer == "en"].set_index("prompt_id").is_correct
        sv = sub[sub.layer == layer].set_index("prompt_id").is_correct
        common = en.index.intersection(sv.index)
        if len(common) == 0:
            continue
        a, b = en.loc[common].to_numpy(), sv.loc[common].to_numpy()
        test = mcnemar(a, b)
        first = sub.iloc[0]
        out.append({
            "config": config, "part": first.get("part"), "model": first.get("model"),
            "reasoning": first.get("reasoning"), "n_items": len(common),
            "en_acc": 100 * a.mean(), "en_ci_low": bootstrap_ci(a)[0], "en_ci_high": bootstrap_ci(a)[1],
            "sv_acc": 100 * b.mean(), "sv_ci_low": bootstrap_ci(b)[0], "sv_ci_high": bootstrap_ci(b)[1],
            "gap": 100 * (a.mean() - b.mean()),
            **dict(zip(("gap_ci_low", "gap_ci_high"), paired_bootstrap_ci(a, b))),
            "only_en": test["only_a"], "only_sv": test["only_b"], "mcnemar_p": test["p_value"],
        })
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, default=paths.path("results_dir") / "items.parquet")
    ap.add_argument("--benchmark", type=Path, default=paths.path("benchmark"))
    ap.add_argument("--layer", default="L4a", choices=("L4a", "L4"), help="Swedish layer to pair with English.")
    ap.add_argument("--subset", default="core", choices=("core", "hard", "rest"),
                    help="core items, their hard subset, or the core items outside it.")
    ap.add_argument("--out", type=Path, help="CSV output (default: print only).")
    args = ap.parse_args()
    items = pd.read_parquet(args.items, columns=["part", "config", "model", "reasoning", "layer",
                                                 "prompt_id", "is_correct"])
    table = paired_table(items, item_flags(args.benchmark), args.layer, args.subset)
    shown = table.drop(columns=["part"]).copy()
    shown["mcnemar_p"] = shown.mcnemar_p.map(lambda p: f"{p:.1e}" if p < 0.001 else f"{p:.3f}")
    with pd.option_context("display.max_rows", None, "display.width", 250):
        print(shown.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
