from __future__ import annotations

import numpy as np
import pandas as pd

from swemedmcq.evaluate import stats


def test_mcnemar_counts_and_p() -> None:
    a = np.array([1, 1, 1, 0, 0, 1], dtype=bool)
    b = np.array([1, 0, 0, 0, 1, 0], dtype=bool)
    res = stats.mcnemar(a, b)
    assert (res["both"], res["only_a"], res["only_b"], res["neither"]) == (1, 3, 1, 1)
    assert abs(res["p_value"] - 0.625) < 1e-9  # exact two-sided binomial, 3 vs 1


def test_bootstrap_intervals_cover_the_mean() -> None:
    rng = np.random.default_rng(0)
    a = rng.random(500) < 0.7
    b = rng.random(500) < 0.6
    low, high = stats.bootstrap_ci(a)
    assert low < 100 * a.mean() < high
    low, high = stats.paired_bootstrap_ci(a, b)
    assert low < 100 * (a.mean() - b.mean()) < high


def test_paired_table_uses_core_items_and_layer() -> None:
    items = pd.DataFrame({
        "part": "benchmark", "config": "m", "model": "M", "reasoning": "on",
        "layer": ["en", "en", "en", "L4a", "L4a", "L4a"],
        "prompt_id": ["a", "b", "c", "a", "b", "c"],
        "is_correct": [True, True, None, True, False, True],
    })
    flags = pd.DataFrame({"prompt_id": ["a", "b", "c"], "core_flag": [True, True, False],
                          "hard_flag": [False, True, False]})
    table = stats.paired_table(items, flags, "L4a", "core")
    row = table.iloc[0]
    assert row.n_items == 2 and row.en_acc == 100 and row.sv_acc == 50 and row.only_en == 1
    assert stats.paired_table(items, flags, "L4a", "hard").iloc[0].n_items == 1
