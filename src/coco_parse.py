"""Parse COCO Y0 HTML output with robust table and estimation extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import re
import unicodedata

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


@dataclass
class CocoParseResult:
    excluded_attr_ids: List[str]
    stairs2_first_row: List[Optional[float]]
    estimations: pd.Series


def _to_num(s: str) -> float:
    s = (s or "").strip()
    if s == "":
        return np.nan
    s = s.replace(" ", "")
    if s.endswith("%"):
        s = s[:-1]
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    try:
        return float(s)
    except ValueError:
        return np.nan


def _table_to_df(tbl) -> pd.DataFrame:
    rows = []
    for tr in tbl.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    if not rows:
        return pd.DataFrame()

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    return pd.DataFrame(rows)


def _find_tables(soup: BeautifulSoup) -> List:
    return soup.find_all("table")


def _norm_text(s: str) -> str:
    text = unicodedata.normalize("NFKD", s or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _contains_steps_marker(text: str, step_no: int) -> bool:
    return bool(
        re.search(rf"\blepcs\w*\s*\(\s*{step_no}\s*\)", text)
        or re.search(rf"\blepcs\w*\s*{step_no}\b", text)
        or re.search(rf"\bstairs?\s*\(\s*{step_no}\s*\)", text)
        or re.search(rf"\bstairs?\s*{step_no}\b", text)
    )


def _map_overview_key(raw: str) -> Optional[str]:
    key = _norm_text(str(raw).rstrip(":"))
    aliases = {
        "identifier": ["azonosito", "identifier", "id"],
        "objects": ["objektum", "object"],
        "attributes": ["attrib", "attribute"],
        "steps": ["lepcso", "lepes", "stairs", "steps"],
        "offset": ["eltolas", "offset"],
        "description": ["leiras", "description"],
    }
    for target, words in aliases.items():
        if any(w in key for w in words):
            return target
    return None


def extract_coco_overview(html: str) -> tuple[dict[str, str], List[str]]:
    """Extract top COCO metadata and section labels from the result page."""
    soup = BeautifulSoup(html, "html.parser")
    meta: dict[str, str] = {}

    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if len(cells) < 2:
            continue
        for i, cell in enumerate(cells):
            cell = str(cell).strip()
            if not cell:
                continue

            key_part = cell
            inline_value = ""
            if ":" in cell:
                left, right = cell.split(":", 1)
                key_part = left.strip()
                inline_value = right.strip()

            mapped = _map_overview_key(key_part)
            if not mapped or mapped in meta:
                continue

            value = inline_value
            if not value and i + 1 < len(cells):
                nxt = str(cells[i + 1]).strip()
                if nxt and _map_overview_key(nxt) is None:
                    value = nxt
            if value:
                meta[mapped] = value

    blob = _norm_text(soup.get_text(" ", strip=True))
    sections: List[str] = []
    if re.search(r"\brangsor\b", blob):
        sections.append("Rangsor")
    if _contains_steps_marker(blob, 1):
        sections.append("Lépcsők(1)")
    if _contains_steps_marker(blob, 2):
        sections.append("Lépcsők(2)")

    return meta, sections


def _map_total_key(raw: str) -> Optional[str]:
    key = _norm_text(raw)
    key = key.rstrip(":").strip()
    if key.startswith("s1") and "osszeg" in key:
        return "s1_sum"
    if key.startswith("s20") and "osszeg" in key:
        return "s20_sum"
    if "becsles" in key and "osszeg" in key:
        return "estimation_sum"
    if "teny" in key and "osszeg" in key and "negyzet" not in key and "elteres" not in key:
        return "actual_sum"
    if "teny" in key and "becsles" in key and "elteres" in key:
        return "actual_estimation_delta"
    if "teny" in key and "negyzetosszeg" in key:
        return "actual_square_sum"
    if "becsles" in key and "negyzetosszeg" in key:
        return "estimation_square_sum"
    if "negyzetosszeg" in key and "hiba" in key:
        return "square_sum_error"
    return None


def extract_coco_totals(html: str) -> dict[str, str]:
    """Extract footer totals like S1 sum / estimation sum from COCO output."""
    soup = BeautifulSoup(html, "html.parser")
    out: dict[str, str] = {}

    lines = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]
    for idx, line in enumerate(lines):
        if ":" not in line:
            continue
        left, right = line.split(":", 1)
        mapped = _map_total_key(left)
        if not mapped:
            continue

        value = right.strip()
        if not value and idx + 1 < len(lines):
            nxt = lines[idx + 1].strip()
            if nxt and _map_total_key(nxt) is None:
                value = nxt

        if mapped not in out or (not out[mapped] and value):
            out[mapped] = value

    if len(out) < 8:
        for tr in soup.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            if len(cells) < 2:
                continue
            for i, cell in enumerate(cells):
                mapped = _map_total_key(cell)
                if not mapped:
                    continue

                value = ""
                if ":" in cell:
                    _, inline = str(cell).split(":", 1)
                    value = inline.strip()

                if not value:
                    for j in range(i + 1, len(cells)):
                        nxt = str(cells[j]).strip()
                        if not nxt:
                            continue
                        if _map_total_key(nxt) is not None:
                            continue
                        value = nxt
                        break

                if mapped not in out or (not out[mapped] and value):
                    out[mapped] = value

    return out


def extract_coco_section_tables(html: str, object_names: List[str]) -> dict[str, pd.DataFrame]:
    """Extract Rangsor / Lépcsők(1) / Lépcsők(2) tables from COCO output."""
    soup = BeautifulSoup(html, "html.parser")
    obj_set = {_norm_text(o) for o in object_names}
    out: dict[str, pd.DataFrame] = {}

    for tbl in _find_tables(soup):
        raw = _table_to_df(tbl)
        if raw.empty or raw.shape[0] < 2 or raw.shape[1] < 3:
            continue

        top_blob = _norm_text(" ".join(str(x) for x in raw.iloc[:2].to_numpy().flatten()))
        context_blob = _norm_text(
            " ".join(
                [
                    str(tbl.find_previous(string=True) or ""),
                    str(tbl.parent.get_text(" ", strip=True) if tbl.parent else ""),
                    top_blob,
                ]
            )
        )
        body_first_col = [_norm_text(str(x)) for x in raw.iloc[1:, 0].tolist()]
        step_row_hits = sum(bool(re.fullmatch(r"s\d+", v)) for v in body_first_col if v)
        obj_hits = sum(v in obj_set for v in body_first_col if v)

        if "Rangsor" not in out and "rangsor" in context_blob and "coco:y0" not in context_blob and obj_hits > 0:
            out["Rangsor"] = _as_display_table(raw)
            continue

        if "Lépcsők(1)" not in out and _contains_steps_marker(context_blob, 1) and step_row_hits >= 1:
            out["Lépcsők(1)"] = _as_display_table(raw)
            continue

        if "Lépcsők(2)" not in out and _contains_steps_marker(context_blob, 2) and step_row_hits >= 1:
            out["Lépcsők(2)"] = _as_display_table(raw)
            continue

    return out


def _dedupe_columns(cols: List[str]) -> List[str]:
    out: List[str] = []
    seen: dict[str, int] = {}
    for i, raw in enumerate(cols):
        base = str(raw).strip() or f"Col{i+1}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        out.append(base if count == 0 else f"{base}_{count+1}")
    return out


def _as_display_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    header = _dedupe_columns([str(x).strip() for x in df.iloc[0].tolist()])
    body = df.iloc[1:].reset_index(drop=True).copy()
    body.columns = header
    body = body.replace(r"^\s*$", np.nan, regex=True).dropna(how="all").fillna("")
    return body


def extract_coco_y0_table(html: str, object_names: List[str]) -> pd.DataFrame:
    """Extract the primary COCO:Y0 output table from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    obj_set = {o.strip().lower() for o in object_names}

    best_score = -1
    best_df = pd.DataFrame()

    for tbl in _find_tables(soup):
        raw = _table_to_df(tbl)
        if raw.empty or raw.shape[0] < 2 or raw.shape[1] < 3:
            continue

        blob = " ".join(str(x).lower() for x in raw.iloc[:2].to_numpy().flatten())
        header = [str(x).lower() for x in raw.iloc[0].tolist()]
        body = raw.iloc[1:].reset_index(drop=True)

        score = 0
        if "coco:y0" in blob or "coco y0" in blob:
            score += 10
        if any(("becsl" in h) or ("estim" in h) for h in header):
            score += 8
        if any("delta" in h for h in header):
            score += 2
        if raw.shape[1] >= 10:
            score += 1

        first_col = [str(x).strip().lower() for x in body.iloc[:, 0].tolist()]
        obj_hits = sum(v in obj_set for v in first_col if v)
        score += min(obj_hits, 20)

        if score > best_score:
            best_score = score
            best_df = raw

    if best_score < 0:
        return pd.DataFrame()

    return _as_display_table(best_df)


def _is_mostly_numeric(row: List[str], min_numeric_ratio: float = 0.6) -> bool:
    if not row:
        return False
    nums = sum(np.isfinite(_to_num(x)) for x in row)
    return nums / max(len(row), 1) >= min_numeric_ratio


def find_stairs2_table(soup: BeautifulSoup) -> Optional[object]:
    for tbl in _find_tables(soup):
        context = " ".join((tbl.find_previous(string=True) or "").split()).lower()
        parent_text = (tbl.parent.get_text(" ", strip=True) or "").lower()
        blob = context + " " + parent_text
        if ("lepcs" in blob or "stairs" in blob) and "2" in blob:
            return tbl

    candidates = []
    for tbl in _find_tables(soup):
        df = _table_to_df(tbl)
        if df.empty or df.shape[1] < 4:
            continue
        numeric_rows = sum(_is_mostly_numeric(df.iloc[i].tolist()) for i in range(min(len(df), 10)))
        candidates.append((numeric_rows, df.shape[1], tbl))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def extract_stairs2_first_row(html: str, attr_ids: List[str]) -> List[Optional[float]]:
    soup = BeautifulSoup(html, "html.parser")
    tbl = find_stairs2_table(soup)
    if not tbl:
        return [None] * len(attr_ids)

    df = _table_to_df(tbl)
    if df.empty:
        return [None] * len(attr_ids)

    data_row = None
    for i in range(len(df)):
        row = df.iloc[i].tolist()
        if _is_mostly_numeric(row):
            data_row = row
            break

    if data_row is None:
        return [None] * len(attr_ids)

    nums = [_to_num(x) for x in data_row]
    if len(nums) > 1 and not np.isfinite(nums[0]) and np.isfinite(nums[1]):
        nums = nums[1:]

    nums = nums[: len(attr_ids)]
    if len(nums) < len(attr_ids):
        nums = nums + [np.nan] * (len(attr_ids) - len(nums))

    return [float(x) if np.isfinite(x) else None for x in nums]


def _build_series_from_df(df: pd.DataFrame, object_names: List[str], header_mode: bool) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    body = df.iloc[1:].reset_index(drop=True) if header_mode else df.reset_index(drop=True)
    if body.empty or body.shape[1] < 2:
        return pd.Series(dtype=float)

    header = df.iloc[0].tolist() if header_mode else []
    obj_set = {o.strip().lower() for o in object_names}

    obj_candidates = []
    for j in range(body.shape[1]):
        col_vals = [str(x).strip() for x in body.iloc[:, j].tolist()]
        exact_hits = sum(v.lower() in obj_set for v in col_vals if v)
        o_hits = sum(bool(re.fullmatch(r"O\d+", v)) for v in col_vals if v)
        obj_candidates.append((exact_hits * 2 + o_hits, j))

    obj_candidates.sort(reverse=True, key=lambda x: x[0])
    obj_col = obj_candidates[0][1] if obj_candidates else 0
    obj_match_score = obj_candidates[0][0] if obj_candidates else 0

    best_col = None
    best_col_score = -1.0
    for j in range(body.shape[1]):
        if j == obj_col:
            continue
        nums = [_to_num(str(x)) for x in body.iloc[:, j].tolist()]
        finite = sum(np.isfinite(v) for v in nums)
        if finite < max(2, int(0.4 * len(nums))):
            continue

        score = float(finite) + float(obj_match_score) * 0.5
        if header_mode and j < len(header):
            h = str(header[j]).lower()
            if "estim" in h or "becsl" in h:
                score += 10.0
        if score > best_col_score:
            best_col_score = score
            best_col = j

    if best_col is None:
        return pd.Series(dtype=float)

    rows = []
    for i in range(len(body)):
        row = body.iloc[i].tolist()
        nm = str(row[obj_col]).strip() if obj_col < len(row) else ""
        est = _to_num(str(row[best_col])) if best_col < len(row) else np.nan
        if nm and np.isfinite(est):
            rows.append((nm, est))

    if not rows:
        for i in range(min(len(object_names), len(body))):
            row = body.iloc[i].tolist()
            nm = str(row[obj_col]).strip() if obj_col < len(row) else f"O{i+1}"
            est = _to_num(str(row[best_col])) if best_col < len(row) else np.nan
            if np.isfinite(est):
                rows.append((nm if nm else f"O{i+1}", est))

    seen = set()
    names = []
    values = []
    for nm, est in rows:
        if nm in seen:
            continue
        seen.add(nm)
        names.append(nm)
        values.append(est)

    s = pd.Series(values, index=names, name="Estimation", dtype=float)
    if all(re.fullmatch(r"O\d+", str(n)) for n in s.index) and len(s) == len(object_names):
        s.index = object_names
    return s


def _extract_from_text_lines(soup: BeautifulSoup, object_names: List[str]) -> pd.Series:
    full_text = soup.get_text("\n", strip=True)
    text_l = full_text.lower()
    anchor_idx = max(text_l.find("estim"), text_l.find("becsl"))
    segment = full_text[anchor_idx:] if anchor_idx >= 0 else full_text
    lines = [ln.strip() for ln in segment.splitlines() if ln.strip()]
    if not lines:
        return pd.Series(dtype=float)

    line_rows = []
    obj_lookup = {o.lower(): o for o in object_names}
    for line in lines:
        ll = line.lower()
        match_obj = None
        for ol, o in obj_lookup.items():
            if ol and ol in ll:
                match_obj = o
                break
        if match_obj is None:
            mobj = re.search(r"\bO(\d+)\b", line, flags=re.IGNORECASE)
            if mobj:
                idx = int(mobj.group(1)) - 1
                if 0 <= idx < len(object_names):
                    match_obj = object_names[idx]
                else:
                    match_obj = f"O{mobj.group(1)}"
        if match_obj is None:
            continue
        nums = re.findall(r"[-+]?\d+(?:[.,]\d+)?", line)
        if not nums:
            continue
        est = _to_num(nums[-1])
        if np.isfinite(est):
            line_rows.append((match_obj, est))

    if not line_rows:
        return pd.Series(dtype=float)

    seen = set()
    names = []
    values = []
    for nm, est in line_rows:
        if nm in seen:
            continue
        seen.add(nm)
        names.append(nm)
        values.append(est)
    return pd.Series(values, index=names, name="Estimation", dtype=float)


def extract_estimations(html: str, object_names: List[str]) -> pd.Series:
    coco_y0_tbl = extract_coco_y0_table(html, object_names)
    if not coco_y0_tbl.empty:
        obj_col = coco_y0_tbl.columns[0]
        est_col = None
        for c in coco_y0_tbl.columns:
            cl = str(c).lower()
            if "becsl" in cl or "estim" in cl:
                est_col = c
                break
        if est_col is not None:
            rows = []
            for _, row in coco_y0_tbl.iterrows():
                nm = str(row.get(obj_col, "")).strip()
                est = _to_num(str(row.get(est_col, "")))
                if nm and np.isfinite(est):
                    rows.append((nm, est))
            if rows:
                seen = set()
                names: List[str] = []
                values: List[float] = []
                for nm, est in rows:
                    if nm in seen:
                        continue
                    seen.add(nm)
                    names.append(nm)
                    values.append(float(est))
                return pd.Series(values, index=names, name="Estimation", dtype=float)

    soup = BeautifulSoup(html, "html.parser")
    tables = _find_tables(soup)

    best_df = None
    best_score = -1
    for tbl in tables:
        df = _table_to_df(tbl)
        if df.empty or df.shape[0] < 2 or df.shape[1] < 2:
            continue

        header = [str(x).lower() for x in df.iloc[0].tolist()]
        score = 0
        if any("estim" in h for h in header) or any("becsl" in h for h in header):
            score += 10
        if any("delta" in h for h in header):
            score += 2
        if df.shape[0] >= len(object_names):
            score += 1

        if score > best_score:
            best_score = score
            best_df = df

    if best_df is not None:
        preferred = _build_series_from_df(best_df.copy(), object_names, header_mode=True)
        if not preferred.dropna().empty:
            return preferred

    fallback_best = pd.Series(dtype=float)
    fallback_score = -1.0
    for tbl in tables:
        df = _table_to_df(tbl)
        if df.empty or df.shape[0] < 2 or df.shape[1] < 2:
            continue
        for header_mode in (True, False):
            s = _build_series_from_df(df, object_names, header_mode=header_mode)
            if s.empty:
                continue
            score = float(s.notna().sum()) + (0.25 if header_mode else 0.0)
            if score > fallback_score:
                fallback_score = score
                fallback_best = s

    if not fallback_best.dropna().empty:
        return fallback_best

    text_fallback = _extract_from_text_lines(soup, object_names)
    return text_fallback


def parse_coco_y0(html: str, n_objects: int, attr_ids: List[str], object_names: List[str]) -> CocoParseResult:
    stairs2 = extract_stairs2_first_row(html, attr_ids)
    excluded = [aid for aid, v in zip(attr_ids, stairs2) if v is not None and int(round(v)) == (n_objects - 1)]
    estim = extract_estimations(html, object_names)
    return CocoParseResult(excluded_attr_ids=excluded, stairs2_first_row=stairs2, estimations=estim)
