"""OAM (Object–Attribute Matrix) CSV ingestion utilities.

This project assumes a "wide" OAM CSV similar to the one you provided:

Row 0:  Direction ID, d1, d2, ... dM, Y
Row 1:  Type,         X,  X,  ... X,   (blank)
Row 2:  Attribute ID, A1, A2, ... AM,  (blank)
Row 3:  Attribute,    name1, ...
Row 4:  Attribute Unit, ...
Row 5+: <Object name>, x1, x2, ... xM, y

Where Direction ID uses:
- 0 = higher raw value is better (rank descending)
- 1 = lower raw value is better (rank ascending)

The COCO Y0 engine expects ranked integer inputs, so we rank BEFORE submitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import io
import re

import numpy as np
import pandas as pd


HEADER_LABELS = {
    "direction id": "direction",
    "type": "type",
    "attribute id": "attr_id",
    "attribute": "attr_name",
    "attribute unit": "attr_unit",
}


@dataclass
class OAM:
    directions: List[int]          # length = M
    attr_ids: List[str]            # length = M
    attr_names: List[str]          # length = M
    objects: List[str]             # length = N
    x_raw: pd.DataFrame            # shape (N, M) numeric
    y: pd.Series                   # length = N numeric

def _make_unique_labels(labels: List[object], default_prefix: str = "Col") -> List[str]:
    """Return unique, non-empty labels while preserving order."""
    out: List[str] = []
    seen: dict[str, int] = {}
    for i, raw in enumerate(labels, start=1):
        base = str(raw).strip()
        if not base:
            base = f"{default_prefix}{i}"
        count = seen.get(base, 0) + 1
        seen[base] = count
        out.append(base if count == 1 else f"{base}_{count}")
    return out


def _parse_numeric(cell: object) -> float:
    """Parse a number from messy CSV cells (e.g., '91.20%', ' 85 ', '0')."""
    if cell is None:
        return np.nan
    if isinstance(cell, float) and np.isnan(cell):
        return np.nan
    s = str(cell).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan

    s = s.replace(" ", "")
    if s.endswith("%"):
        s = s[:-1]

    # European decimal comma
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")

    # Remove thousand separators like 1,234
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)

    try:
        return float(s)
    except ValueError:
        return np.nan


def read_oam_csv(file_bytes: bytes) -> OAM:
    """Read an OAM CSV from uploaded bytes."""
    df = pd.read_csv(io.BytesIO(file_bytes), header=None, dtype=str)

    # Find header rows
    row_index = {}
    for i, v in enumerate(df[0].fillna("")):
        key = str(v).strip().lower()
        if key in HEADER_LABELS:
            row_index[HEADER_LABELS[key]] = i

    missing = {"direction", "attr_id", "attr_name"} - set(row_index)
    if missing:
        raise ValueError(f"Missing required header row(s): {sorted(missing)}. Found: {row_index}")

    rid_dir = row_index["direction"]
    rid_attr_id = row_index["attr_id"]
    rid_attr_name = row_index["attr_name"]

    # Determine the Y column robustly.
    # Some CSVs contain trailing separators, which create an extra blank last column.
    ncols = df.shape[1]
    dir_row = df.iloc[rid_dir].fillna("").astype(str).str.strip().str.lower()
    y_candidates = [idx for idx in range(1, ncols) if dir_row.iloc[idx] == "y"]
    if y_candidates:
        y_col = y_candidates[-1]
    else:
        # Fallback to the right-most non-empty column among key header rows.
        y_col = ncols - 1
        for idx in range(ncols - 1, 0, -1):
            cells = [
                str(df.iloc[rid_dir, idx]).strip(),
                str(df.iloc[rid_attr_id, idx]).strip(),
                str(df.iloc[rid_attr_name, idx]).strip(),
            ]
            if any(c and c.lower() not in {"nan", "none"} for c in cells):
                y_col = idx
                break

    # Attribute columns are 1..(y_col-1)
    attr_cols = list(range(1, y_col))

    # Directions and attribute metadata
    dir_vals = df.iloc[rid_dir, attr_cols].tolist()
    directions: List[int] = []
    for d in dir_vals:
        try:
            directions.append(int(str(d).strip()))
        except Exception:
            directions.append(0)

    attr_ids = [str(x).strip() for x in df.iloc[rid_attr_id, attr_cols].tolist()]
    attr_ids = _make_unique_labels(attr_ids, default_prefix="A")
    attr_names = [str(x).strip() for x in df.iloc[rid_attr_name, attr_cols].tolist()]

    # Object rows: everything not a recognized header label
    header_rows = set(row_index.values())
    obj_df = df[~df.index.isin(header_rows)].copy()
    obj_df = obj_df[obj_df[0].notna() & (obj_df[0].astype(str).str.strip() != "")]

    objects = [str(x).strip() for x in obj_df[0].tolist()]

    x_raw = obj_df.loc[:, attr_cols].applymap(_parse_numeric)
    x_raw.columns = attr_ids

    y = obj_df.iloc[:, y_col].apply(_parse_numeric)
    y.name = "Y"

    # Basic sanity checks
    if x_raw.shape[1] != len(directions):
        raise ValueError("Attribute column count mismatch while parsing.")

    return OAM(
        directions=directions,
        attr_ids=attr_ids,
        attr_names=attr_names,
        objects=objects,
        x_raw=x_raw.reset_index(drop=True),
        y=y.reset_index(drop=True),
    )
