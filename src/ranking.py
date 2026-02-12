"""Excel-like ranking for OAM columns.

We replicate Excel RANK.EQ semantics:
- Best item gets rank 1.
- Ties share the same rank.
- Next rank number is skipped (method='min' in pandas).

Direction:
- direction=0 => higher is better => rank descending
- direction=1 => lower is better  => rank ascending
"""

from __future__ import annotations

from typing import List
import pandas as pd


def rank_oam_columns(x_raw: pd.DataFrame, directions: List[int]) -> pd.DataFrame:
    """Return a ranked DataFrame with same shape as x_raw."""
    if x_raw.shape[1] != len(directions):
        raise ValueError("directions length must match number of attribute columns")

    ranked = pd.DataFrame(index=x_raw.index)
    for j, col in enumerate(x_raw.columns):
        ascending = True if directions[j] == 1 else False
        ranked[col] = (
            x_raw[col]
            .rank(method="min", ascending=ascending, na_option="bottom")
            .astype(int)
        )
    return ranked
