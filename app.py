import hashlib
import io
import json
import re

import pandas as pd
import streamlit as st

from src.coco_client import run_coco_y0
from src.coco_parse import (
    extract_coco_overview,
    extract_coco_section_tables,
    extract_coco_totals,
    extract_coco_y0_table,
    extract_stairs2_first_row,
)
from src.db import connect as db_connect
from src.db import init_db as db_init
from src.db import delete_runs_by_filename, get_run_results, insert_object_results, insert_run, list_runs
from src.oam_io import read_oam_csv
from src.ranking import rank_oam_columns
from src.ui_display import compact_number, format_dataframe_for_display, inject_app_theme


st.set_page_config(page_title="OAM -> COCO Y0 Automator", layout="wide")
inject_app_theme("OAM -> COCO Y0 Automator")


def _direction_label(direction: int) -> str:
    return "Lower is better" if int(direction) == 1 else "Higher is better"


def _build_attribute_sheet(oam) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Attribute ID": oam.attr_ids,
            "Attribute Name": oam.attr_names,
            "Direction ID": oam.directions,
            "Direction Rule": [_direction_label(v) for v in oam.directions],
        }
    )


def _build_object_sheet(oam) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Object": oam.objects,
            "Y": oam.y.values,
        }
    )


def _build_input_sheet(oam) -> pd.DataFrame:
    df = oam.x_raw.copy()
    df.columns = oam.attr_ids
    df.insert(0, "Object", oam.objects)
    df["Y"] = oam.y.values
    return df


def _build_ranked_matrix(oam) -> pd.DataFrame:
    return rank_oam_columns(oam.x_raw, oam.directions)


def _with_last_y_scaled(y: pd.Series) -> pd.Series:
    y2 = y.copy()
    if len(y2) > 0 and pd.notna(y2.iloc[-1]):
        y2.iloc[-1] = float(y2.iloc[-1]) * 100.0
    return y2


def _build_ranked_sheet(oam) -> pd.DataFrame:
    ranked = _build_ranked_matrix(oam).copy()
    ranked.insert(0, "Object", oam.objects)
    ranked["Y"] = _with_last_y_scaled(oam.y).values
    return ranked


def _to_float_loose(value) -> float | None:
    txt = str(value).strip()
    if not txt:
        return None
    nums = re.findall(r"[-+]?\d+(?:[.,]\d+)?", txt)
    if not nums:
        return None
    token = nums[-1].replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


def _extract_excluded_attr_ids(oam, coco_html: str) -> list[str]:
    threshold = len(oam.objects) - 1

    section_tables = extract_coco_section_tables(coco_html, oam.objects)
    stairs2_df = section_tables.get("Lépcsők(2)", pd.DataFrame())
    if not stairs2_df.empty:
        row0 = stairs2_df.iloc[0].to_dict()
        normalized = {str(k).strip().upper(): v for k, v in row0.items()}
        excluded = []
        for aid in oam.attr_ids:
            raw = normalized.get(str(aid).strip().upper(), "")
            v = _to_float_loose(raw)
            if v is not None and abs(v - float(threshold)) < 1e-9:
                excluded.append(aid)
        if excluded:
            return excluded

    # Fallback path if section table extraction is incomplete.
    stairs2_first_row = extract_stairs2_first_row(coco_html, oam.attr_ids)
    return [
        aid
        for aid, v in zip(oam.attr_ids, stairs2_first_row)
        if v is not None and abs(float(v) - float(threshold)) < 1e-9
    ]


def _build_excluded_oam_sheet(oam, excluded_attr_ids: list[str]) -> pd.DataFrame:
    raw = oam.x_raw.copy()
    raw.columns = oam.attr_ids
    name_by_id = dict(zip(oam.attr_ids, oam.attr_names))

    out = pd.DataFrame({"Object": oam.objects})
    for aid in excluded_attr_ids:
        header = f"{aid} - {name_by_id.get(aid, aid)}"
        if aid in raw.columns:
            out[header] = raw[aid].values
    out["Y"] = oam.y.values
    return out


def _to_coco_matrix_tsv(ranked_x: pd.DataFrame, y: pd.Series, col_sep: str = "\t", row_sep: str = "\r\n") -> str:
    df = ranked_x.copy()
    df["Y"] = y.values
    lines = []
    for row in df.to_numpy():
        vals = []
        for value in row:
            fv = float(value)
            vals.append(str(int(fv)) if fv.is_integer() else str(value))
        lines.append(col_sep.join(vals))
    return row_sep.join(lines)


def _run_coco_estimation(oam, steps: int, identifier: str) -> tuple[pd.DataFrame, str]:
    ranked_matrix = _build_ranked_matrix(oam)
    return _run_coco_estimation_for_ranked(
        oam=oam,
        ranked_matrix=ranked_matrix,
        attr_ids=list(oam.attr_ids),
        steps=steps,
        identifier=identifier,
    )


def _run_coco_estimation_for_ranked(
    oam,
    ranked_matrix: pd.DataFrame,
    attr_ids: list[str],
    steps: int,
    identifier: str,
) -> tuple[pd.DataFrame, str]:
    y_for_coco = _with_last_y_scaled(oam.y)
    attr_with_y = list(attr_ids) + ["Y"]

    object_block = "\n".join(oam.objects)
    attribute_block = "\t".join(attr_with_y)

    auto_steps = len(oam.objects)
    steps_val = auto_steps if steps == 0 else int(steps)

    max_rank = int(ranked_matrix.max().max()) if not ranked_matrix.empty else 1
    if steps_val < max_rank:
        steps_val = max_rank

    last_html = ""

    # COCO parser is sensitive to matrix row separators.
    for row_sep in ("\r\n", "\r"):
        matrix_tsv = _to_coco_matrix_tsv(ranked_matrix, y_for_coco, col_sep="\t", row_sep=row_sep)

        html, _ = run_coco_y0(
            matrix_tsv=matrix_tsv,
            object_names=object_block,
            attribute_names=attribute_block,
            steps=steps_val,
            identifier=identifier,
        )
        last_html = html

        output_df = extract_coco_y0_table(html, oam.objects)
        if not output_df.empty:
            return output_df, html

    return pd.DataFrame(), last_html


def _run_coco_estimation_excluded(oam, excluded_attr_ids: list[str], steps: int, identifier: str) -> tuple[pd.DataFrame, str]:
    if not excluded_attr_ids:
        return pd.DataFrame(), ""

    raw = oam.x_raw.copy()
    raw.columns = oam.attr_ids
    dir_by_id = dict(zip(oam.attr_ids, oam.directions))
    ranked_excluded = rank_oam_columns(raw[excluded_attr_ids], [int(dir_by_id[aid]) for aid in excluded_attr_ids])
    ranked_excluded.columns = excluded_attr_ids

    return _run_coco_estimation_for_ranked(
        oam=oam,
        ranked_matrix=ranked_excluded,
        attr_ids=excluded_attr_ids,
        steps=steps,
        identifier=identifier,
    )


def _build_result_ranking(oam, coco_y0_df: pd.DataFrame) -> pd.DataFrame:
    if coco_y0_df is None or coco_y0_df.empty:
        return pd.DataFrame(columns=["Object", "Estimation", "Rank"])

    obj_col = str(coco_y0_df.columns[0])
    est_col = None
    for c in coco_y0_df.columns:
        cl = str(c).lower()
        if "becsl" in cl or "estim" in cl:
            est_col = str(c)
            break

    if est_col is None:
        candidates = []
        for c in coco_y0_df.columns[1:]:
            vals = [_to_float_loose(v) for v in coco_y0_df[c].tolist()]
            score = sum(v is not None for v in vals)
            candidates.append((score, str(c)))
        if candidates:
            candidates.sort(reverse=True, key=lambda x: x[0])
            est_col = candidates[0][1]

    if est_col is None:
        return pd.DataFrame(columns=["Object", "Estimation", "Rank"])

    obj_set = set(oam.objects)
    rows = []
    for _, r in coco_y0_df.iterrows():
        obj = str(r.get(obj_col, "")).strip()
        est = _to_float_loose(r.get(est_col, ""))
        if not obj or est is None:
            continue
        if obj not in obj_set:
            continue
        rows.append((obj, float(est)))

    if not rows:
        return pd.DataFrame(columns=["Object", "Estimation", "Rank"])

    out = pd.DataFrame(rows, columns=["Object", "Estimation"]).drop_duplicates(subset=["Object"], keep="first")
    out = out.sort_values("Estimation", ascending=False).reset_index(drop=True)
    out["Rank"] = range(1, len(out) + 1)
    return out


def _save_result_to_db(
    filename: str,
    oam,
    excluded_attr_ids: list[str],
    coco_html_run1: str,
    coco_html_run2: str,
    result_df: pd.DataFrame,
) -> int:
    conn = db_connect("coco_runs.sqlite")
    try:
        db_init(conn)
        run_id = insert_run(
            conn=conn,
            filename=filename,
            n_objects=len(oam.objects),
            n_attributes=len(excluded_attr_ids),
            excluded_attr_ids=excluded_attr_ids,
            coco_html_run1=coco_html_run1 or "",
            coco_html_run2=coco_html_run2 or "",
        )
        rows = [(str(r["Object"]), float(r["Estimation"]), int(r["Rank"])) for _, r in result_df.iterrows()]
        insert_object_results(conn, run_id, rows)
        return run_id
    finally:
        conn.close()


def _read_db_runs(limit: int = 200) -> pd.DataFrame:
    conn = db_connect("coco_runs.sqlite")
    try:
        db_init(conn)
        rows = list_runs(conn, limit=limit)
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(
            columns=[
                "Run ID",
                "Created At",
                "Filename",
                "Objects",
                "Attributes",
                "Excluded Count",
            ]
        )

    out_rows = []
    for run_id, created_at, filename, n_objects, n_attributes, excluded_json in rows:
        try:
            excluded_count = len(json.loads(excluded_json or "[]"))
        except Exception:
            excluded_count = 0
        out_rows.append(
            {
                "Run ID": int(run_id),
                "Created At": str(created_at),
                "Filename": str(filename or ""),
                "Objects": int(n_objects),
                "Attributes": int(n_attributes),
                "Excluded Count": int(excluded_count),
            }
        )
    return pd.DataFrame(out_rows)


def _read_db_run_result(run_id: int) -> pd.DataFrame:
    conn = db_connect("coco_runs.sqlite")
    try:
        db_init(conn)
        rows = get_run_results(conn, int(run_id))
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=["Object", "Estimation", "Rank"])
    return pd.DataFrame(rows, columns=["Object", "Estimation", "Rank"])


def _delete_db_file_history(filename: str) -> None:
    conn = db_connect("coco_runs.sqlite")
    try:
        db_init(conn)
        delete_runs_by_filename(conn, filename)
    finally:
        conn.close()


def _result_to_excel_bytes(df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Result")
    return out.getvalue()


def _result_chart_png(df: pd.DataFrame) -> bytes | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig_w = max(8, min(24, 0.5 * max(len(df), 1)))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.bar(df["Object"], df["Estimation"], color="#1f6bff")
    ax.set_xlabel("Object")
    ax.set_ylabel("Estimation")
    ax.set_title("Object Estimation Ranking")
    ax.tick_params(axis="x", rotation=75, labelsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    plt.close(fig)
    return buf.getvalue()


def _render_history_section() -> None:
    st.markdown('<div class="block-title">History</div>', unsafe_allow_html=True)
    runs_df = _read_db_runs(limit=1000)
    if runs_df.empty:
        st.info("No saved runs in database yet.")
        return

    files_df = (
        runs_df[["Filename", "Created At"]]
        .dropna(subset=["Filename"])
        .loc[lambda x: x["Filename"].astype(str).str.strip() != ""]
        .drop_duplicates(subset=["Filename"], keep="first")
        .reset_index(drop=True)
    )

    st.markdown('<div class="block-title">Previously Uploaded Files</div>', unsafe_allow_html=True)
    for _, row in files_df.iterrows():
        fname = str(row["Filename"])
        created_at = str(row["Created At"])
        cols = st.columns([5, 2, 3], gap="small")
        with cols[0]:
            if st.button(fname, key=f"hist_open_{hash(fname)}", use_container_width=True, type="secondary"):
                st.session_state.history_filename = fname
                st.session_state.current_page = "history"
                st.rerun()
        with cols[1]:
            st.caption(created_at.replace("T", " ")[:16])
        with cols[2]:
            if st.button("Delete", key=f"hist_del_{hash(fname)}", use_container_width=True, type="secondary"):
                _delete_db_file_history(fname)
                if st.session_state.history_filename == fname:
                    st.session_state.history_filename = None
                    if st.session_state.current_page == "history":
                        st.session_state.current_page = "input"
                st.rerun()


if "upload_sig" not in st.session_state:
    st.session_state.upload_sig = None
if "parsed_oam" not in st.session_state:
    st.session_state.parsed_oam = None
if "parsed_error" not in st.session_state:
    st.session_state.parsed_error = None
if "access_ranked" not in st.session_state:
    st.session_state.access_ranked = False
if "access_estimation" not in st.session_state:
    st.session_state.access_estimation = False
if "access_excluded" not in st.session_state:
    st.session_state.access_excluded = False
if "access_estimation2" not in st.session_state:
    st.session_state.access_estimation2 = False
if "access_result" not in st.session_state:
    st.session_state.access_result = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "input"
if "coco_estimation_df" not in st.session_state:
    st.session_state.coco_estimation_df = None
if "coco_estimation_html" not in st.session_state:
    st.session_state.coco_estimation_html = None
if "excluded_attr_ids" not in st.session_state:
    st.session_state.excluded_attr_ids = []
if "excluded_oam_df" not in st.session_state:
    st.session_state.excluded_oam_df = None
if "coco_estimation2_df" not in st.session_state:
    st.session_state.coco_estimation2_df = None
if "coco_estimation2_html" not in st.session_state:
    st.session_state.coco_estimation2_html = None
if "result_df" not in st.session_state:
    st.session_state.result_df = None
if "result_run_id" not in st.session_state:
    st.session_state.result_run_id = None
if "upload_filename" not in st.session_state:
    st.session_state.upload_filename = None
if "history_filename" not in st.session_state:
    st.session_state.history_filename = None


with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload OAM CSV", type=["csv"])
    _render_history_section()
steps = 0
identifier = "Teszt"


if uploaded is not None:
    uploaded_bytes = uploaded.getvalue()
    upload_sig = hashlib.md5(uploaded_bytes).hexdigest()
    if upload_sig != st.session_state.upload_sig:
        st.session_state.upload_sig = upload_sig
        st.session_state.upload_filename = uploaded.name
        st.session_state.access_ranked = False
        st.session_state.access_estimation = False
        st.session_state.access_excluded = False
        st.session_state.access_estimation2 = False
        st.session_state.access_result = False
        st.session_state.current_page = "input"
        st.session_state.coco_estimation_df = None
        st.session_state.coco_estimation_html = None
        st.session_state.excluded_attr_ids = []
        st.session_state.excluded_oam_df = None
        st.session_state.coco_estimation2_df = None
        st.session_state.coco_estimation2_html = None
        st.session_state.result_df = None
        st.session_state.result_run_id = None
        try:
            st.session_state.parsed_oam = read_oam_csv(uploaded_bytes)
            st.session_state.parsed_error = None
        except Exception as exc:
            st.session_state.parsed_oam = None
            st.session_state.parsed_error = str(exc)
else:
    st.session_state.upload_sig = None
    st.session_state.upload_filename = None
    st.session_state.parsed_oam = None
    st.session_state.parsed_error = None
    st.session_state.access_ranked = False
    st.session_state.access_estimation = False
    st.session_state.access_excluded = False
    st.session_state.access_estimation2 = False
    st.session_state.access_result = False
    if st.session_state.current_page != "history":
        st.session_state.current_page = "input"
    st.session_state.coco_estimation_df = None
    st.session_state.coco_estimation_html = None
    st.session_state.excluded_attr_ids = []
    st.session_state.excluded_oam_df = None
    st.session_state.coco_estimation2_df = None
    st.session_state.coco_estimation2_html = None
    st.session_state.result_df = None
    st.session_state.result_run_id = None


available_pages = {
    "input": True,
    "ranked": st.session_state.access_ranked,
    "estimation": st.session_state.access_estimation,
    "excluded": st.session_state.access_excluded,
    "estimation2": st.session_state.access_estimation2,
    "result": st.session_state.access_result,
    "history": True,
}
if not available_pages.get(st.session_state.current_page, False):
    st.session_state.current_page = "input"

nav_items = [
    ("input", "1) Input Data"),
    ("ranked", "2) Ranked Data"),
    ("estimation", "3) COCO Y0 Estimation"),
    ("excluded", "4) Excluded OAM"),
    ("estimation2", "5) Estimation 2"),
    ("result", "6) Result"),
    ("history", "7) History"),
]
st.markdown('<div id="page-nav-anchor"></div>', unsafe_allow_html=True)
nav_cols = st.columns(len(nav_items), gap="small")
for col, (page_id, label) in zip(nav_cols, nav_items):
    with col:
        nav_label = f"> {label}" if st.session_state.current_page == page_id else label
        if st.button(
            nav_label,
            key=f"nav_{page_id}",
            use_container_width=True,
            type="secondary",
            disabled=not available_pages[page_id],
        ):
            st.session_state.current_page = page_id
            st.rerun()

if st.session_state.current_page == "input":
    st.markdown('<div class="section-title">CSV Intake</div>', unsafe_allow_html=True)
    if uploaded is None:
        st.info("Upload a CSV in the sidebar to view objects, attributes, and input sheets.")
    elif st.session_state.parsed_error:
        st.error(f"CSV parsing failed: {st.session_state.parsed_error}")
    else:
        oam = st.session_state.parsed_oam
        object_count = len(oam.objects)
        attr_count = len(oam.attr_ids)
        y_value = oam.y.dropna().iloc[0] if oam.y.notna().any() else None
        y_text = compact_number(y_value) if y_value is not None else "-"

        st.markdown(
            (
                '<div class="intake-summary-grid">'
                '<div class="intake-item">'
                '<div class="intake-label">Objects</div>'
                f'<div class="intake-value">{object_count}</div>'
                "</div>"
                '<div class="intake-item">'
                '<div class="intake-label">Attributes</div>'
                f'<div class="intake-value">{attr_count}</div>'
                "</div>"
                '<div class="intake-item">'
                '<div class="intake-label">Y</div>'
                f'<div class="intake-value">{y_text}</div>'
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="block-title">Objects + Y</div>', unsafe_allow_html=True)
        st.dataframe(
            format_dataframe_for_display(_build_object_sheet(oam)),
            use_container_width=True,
            height=280,
            hide_index=True,
        )

        st.markdown('<div class="block-title">Attributes</div>', unsafe_allow_html=True)
        st.dataframe(
            format_dataframe_for_display(_build_attribute_sheet(oam)),
            use_container_width=True,
            height=300,
            hide_index=True,
        )

        st.markdown('<div class="block-title">Input Sheet (Objects + Attributes + Y)</div>', unsafe_allow_html=True)
        st.dataframe(
            format_dataframe_for_display(_build_input_sheet(oam)),
            use_container_width=True,
            height=420,
            hide_index=True,
        )

        if st.button("Rank", type="primary", use_container_width=True, key="rank_btn"):
            st.session_state.access_ranked = True
            st.session_state.current_page = "ranked"
            st.rerun()

elif st.session_state.current_page == "ranked":
    oam = st.session_state.parsed_oam
    st.markdown('<div class="section-title">Ranked Data (Excel RANK.EQ style)</div>', unsafe_allow_html=True)
    ranked_df = _build_ranked_sheet(oam)
    st.dataframe(
        format_dataframe_for_display(ranked_df),
        use_container_width=True,
        height=620,
        hide_index=True,
    )
    st.download_button(
        "Download Ranked Data (CSV)",
        data=ranked_df.to_csv(index=False, float_format="%.15g").encode("utf-8"),
        file_name="ranked_oam_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if st.button("Run COCO Y0", type="primary", use_container_width=True, key="run_coco_btn"):
        with st.spinner("Submitting ranked data to COCO Y0 and loading output table..."):
            try:
                estimation_df, estimation_html = _run_coco_estimation(oam, int(steps), identifier)
            except Exception as exc:
                st.error("COCO Y0 run failed. Common causes: no internet access or COCO form changes.")
                st.exception(exc)
                st.stop()

        st.session_state.coco_estimation_df = estimation_df
        st.session_state.coco_estimation_html = estimation_html
        st.session_state.access_estimation = True
        st.session_state.current_page = "estimation"
        st.rerun()

elif st.session_state.current_page == "estimation":
    st.markdown('<div class="section-title">COCO Y0 Estimation</div>', unsafe_allow_html=True)

    if st.session_state.coco_estimation_html:
        overview, sections = extract_coco_overview(st.session_state.coco_estimation_html)
        keys = ["identifier", "objects", "attributes", "steps", "offset", "description"]
        labels = {
            "identifier": "Identifier",
            "objects": "Objects",
            "attributes": "Attributes",
            "steps": "Steps",
            "offset": "Offset",
            "description": "Description",
        }
        if overview:
            overview_df = pd.DataFrame([{labels[k]: overview.get(k, "-") for k in keys}])
            st.dataframe(
                format_dataframe_for_display(overview_df),
                use_container_width=True,
                hide_index=True,
                height=88,
            )
        if sections:
            section_text_map = {
                "Rangsor": "Ranking",
                "Lépcsők(1)": "Stairs(1)",
                "Lépcsők(2)": "Stairs(2)",
            }
            st.markdown(" | ".join(f"**{section_text_map.get(label, label)}**" for label in sections))

        section_tables = extract_coco_section_tables(
            st.session_state.coco_estimation_html,
            st.session_state.parsed_oam.objects if st.session_state.parsed_oam is not None else [],
        )
        section_title_map = {
            "Rangsor": "Ranking",
            "Lépcsők(1)": "Stairs(1)",
            "Lépcsők(2)": "Stairs(2)",
        }
        for section_name in ["Rangsor", "Lépcsők(1)", "Lépcsők(2)"]:
            section_df = section_tables.get(section_name, pd.DataFrame())
            if section_df.empty:
                continue
            st.markdown(f'<div class="block-title">{section_title_map.get(section_name, section_name)}</div>', unsafe_allow_html=True)
            st.dataframe(
                format_dataframe_for_display(section_df),
                use_container_width=True,
                hide_index=True,
                height=380,
            )

        totals = extract_coco_totals(st.session_state.coco_estimation_html)
        if totals:
            totals_rows = [
                ("S1 Sum", totals.get("s1_sum", "")),
                ("S20 Sum", totals.get("s20_sum", "")),
                ("Estimation Sum", totals.get("estimation_sum", "")),
                ("Actual Sum", totals.get("actual_sum", "")),
                ("Actual - Estimation Delta", totals.get("actual_estimation_delta", "")),
                ("Actual Square Sum", totals.get("actual_square_sum", "")),
                ("Estimation Square Sum", totals.get("estimation_square_sum", "")),
                ("Square Sum Error", totals.get("square_sum_error", "")),
            ]
            totals_df = pd.DataFrame(totals_rows, columns=["Metric", "Value"])
            st.markdown('<div class="block-title">COCO Totals</div>', unsafe_allow_html=True)
            st.dataframe(
                format_dataframe_for_display(totals_df),
                use_container_width=True,
                hide_index=True,
                height=280,
            )

    if st.session_state.coco_estimation_df is None:
        st.info("No estimation found yet. Use Run COCO Y0 on page 2.")
    elif st.session_state.coco_estimation_df.empty:
        st.warning("COCO response was received, but COCO:Y0 output table could not be detected.")
        st.dataframe(pd.DataFrame(), use_container_width=True, hide_index=True)
    else:
        st.dataframe(
            format_dataframe_for_display(st.session_state.coco_estimation_df),
            use_container_width=True,
            height=680,
            hide_index=True,
        )

    if st.session_state.coco_estimation_html:
        if st.button("Exclude", type="primary", use_container_width=True, key="exclude_btn"):
            try:
                oam = st.session_state.parsed_oam
                excluded_attr_ids = _extract_excluded_attr_ids(oam, st.session_state.coco_estimation_html)
                st.session_state.excluded_attr_ids = excluded_attr_ids
                st.session_state.excluded_oam_df = _build_excluded_oam_sheet(oam, excluded_attr_ids)
                st.session_state.access_excluded = True
                st.session_state.current_page = "excluded"
                st.rerun()
            except Exception as exc:
                st.error("Could not classify excluded attributes from Stairs(2).")
                st.exception(exc)

elif st.session_state.current_page == "excluded":
    st.markdown('<div class="section-title">Excluded OAM</div>', unsafe_allow_html=True)
    oam = st.session_state.parsed_oam
    excluded_df = st.session_state.excluded_oam_df
    excluded_ids = st.session_state.excluded_attr_ids or []

    if excluded_df is None:
        st.info("Run COCO Y0 first, then click Exclude on the Estimation page.")
    else:
        name_by_id = dict(zip(oam.attr_ids, oam.attr_names)) if oam is not None else {}
        excluded_meta_df = pd.DataFrame(
            {
                "Excluded Attribute ID": excluded_ids,
                "Excluded Attribute Name": [name_by_id.get(aid, aid) for aid in excluded_ids],
            }
        )
        st.markdown('<div class="block-title">Excluded Attributes</div>', unsafe_allow_html=True)
        st.dataframe(
            format_dataframe_for_display(excluded_meta_df),
            use_container_width=True,
            hide_index=True,
            height=240,
        )

        st.markdown('<div class="block-title">Excluded OAM Table (Object + Excluded Attributes + Y)</div>', unsafe_allow_html=True)
        st.dataframe(
            format_dataframe_for_display(excluded_df),
            use_container_width=True,
            hide_index=True,
            height=620,
        )

        if st.button("Estimate", type="primary", use_container_width=True, key="estimate2_btn"):
            if not excluded_ids:
                st.warning("No excluded attributes were detected from Stairs(2).")
            else:
                with st.spinner("Re-ranking excluded attributes and running COCO Y0 again..."):
                    try:
                        est2_df, est2_html = _run_coco_estimation_excluded(
                            oam=oam,
                            excluded_attr_ids=excluded_ids,
                            steps=int(steps),
                            identifier=identifier,
                        )
                    except Exception as exc:
                        st.error("Second COCO Y0 run failed. Common causes: no internet access or COCO form changes.")
                        st.exception(exc)
                        st.stop()

                st.session_state.coco_estimation2_df = est2_df
                st.session_state.coco_estimation2_html = est2_html
                st.session_state.access_estimation2 = True
                st.session_state.current_page = "estimation2"
                st.rerun()

elif st.session_state.current_page == "estimation2":
    st.markdown('<div class="section-title">Estimation 2</div>', unsafe_allow_html=True)

    if st.session_state.coco_estimation2_html:
        overview, sections = extract_coco_overview(st.session_state.coco_estimation2_html)
        keys = ["identifier", "objects", "attributes", "steps", "offset", "description"]
        labels = {
            "identifier": "Identifier",
            "objects": "Objects",
            "attributes": "Attributes",
            "steps": "Steps",
            "offset": "Offset",
            "description": "Description",
        }
        if overview:
            overview_df = pd.DataFrame([{labels[k]: overview.get(k, "-") for k in keys}])
            st.dataframe(
                format_dataframe_for_display(overview_df),
                use_container_width=True,
                hide_index=True,
                height=88,
            )
        if sections:
            section_text_map = {
                "Rangsor": "Ranking",
                "Lépcsők(1)": "Stairs(1)",
                "Lépcsők(2)": "Stairs(2)",
            }
            st.markdown(" | ".join(f"**{section_text_map.get(label, label)}**" for label in sections))

        section_tables = extract_coco_section_tables(
            st.session_state.coco_estimation2_html,
            st.session_state.parsed_oam.objects if st.session_state.parsed_oam is not None else [],
        )
        section_title_map = {
            "Rangsor": "Ranking",
            "Lépcsők(1)": "Stairs(1)",
            "Lépcsők(2)": "Stairs(2)",
        }
        for section_name in ["Rangsor", "Lépcsők(1)", "Lépcsők(2)"]:
            section_df = section_tables.get(section_name, pd.DataFrame())
            if section_df.empty:
                continue
            st.markdown(f'<div class="block-title">{section_title_map.get(section_name, section_name)}</div>', unsafe_allow_html=True)
            st.dataframe(
                format_dataframe_for_display(section_df),
                use_container_width=True,
                hide_index=True,
                height=380,
            )

        totals = extract_coco_totals(st.session_state.coco_estimation2_html)
        if totals:
            totals_rows = [
                ("S1 Sum", totals.get("s1_sum", "")),
                ("S20 Sum", totals.get("s20_sum", "")),
                ("Estimation Sum", totals.get("estimation_sum", "")),
                ("Actual Sum", totals.get("actual_sum", "")),
                ("Actual - Estimation Delta", totals.get("actual_estimation_delta", "")),
                ("Actual Square Sum", totals.get("actual_square_sum", "")),
                ("Estimation Square Sum", totals.get("estimation_square_sum", "")),
                ("Square Sum Error", totals.get("square_sum_error", "")),
            ]
            totals_df = pd.DataFrame(totals_rows, columns=["Metric", "Value"])
            st.markdown('<div class="block-title">COCO Totals</div>', unsafe_allow_html=True)
            st.dataframe(
                format_dataframe_for_display(totals_df),
                use_container_width=True,
                hide_index=True,
                height=280,
            )

    if st.session_state.coco_estimation2_df is None:
        st.info("No second estimation found yet. Use Estimate on the Excluded OAM page.")
    elif st.session_state.coco_estimation2_df.empty:
        st.warning("COCO response was received, but COCO:Y0 output table could not be detected for Estimation 2.")
        st.dataframe(pd.DataFrame(), use_container_width=True, hide_index=True)
    else:
        st.dataframe(
            format_dataframe_for_display(st.session_state.coco_estimation2_df),
            use_container_width=True,
            height=680,
            hide_index=True,
        )

    if st.session_state.coco_estimation2_df is not None and not st.session_state.coco_estimation2_df.empty:
        if st.button("Get a Result", type="primary", use_container_width=True, key="get_result_btn"):
            try:
                oam = st.session_state.parsed_oam
                result_df = _build_result_ranking(oam, st.session_state.coco_estimation2_df)
                if result_df.empty:
                    st.warning("Could not build ranked result from Estimation 2 output.")
                    st.stop()

                run_id = _save_result_to_db(
                    filename=st.session_state.upload_filename or "uploaded.csv",
                    oam=oam,
                    excluded_attr_ids=st.session_state.excluded_attr_ids or [],
                    coco_html_run1=st.session_state.coco_estimation_html or "",
                    coco_html_run2=st.session_state.coco_estimation2_html or "",
                    result_df=result_df,
                )

                st.session_state.result_df = result_df
                st.session_state.result_run_id = run_id
                st.session_state.access_result = True
                st.session_state.current_page = "result"
                st.rerun()
            except Exception as exc:
                st.error("Failed to generate/save final result.")
                st.exception(exc)

elif st.session_state.current_page == "result":
    st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)
    result_df = st.session_state.result_df
    run_id = st.session_state.result_run_id
    base_name = (st.session_state.upload_filename or "result").rsplit(".", 1)[0]

    if result_df is None or result_df.empty:
        st.info("No result yet. Use Get a Result on Estimation 2 page.")
    else:
        st.markdown('<div class="block-title">Ranked Objects (Best to Least)</div>', unsafe_allow_html=True)
        st.dataframe(
            format_dataframe_for_display(result_df),
            use_container_width=True,
            hide_index=True,
            height=620,
        )
        if run_id is not None:
            st.caption(f"Saved to database. Run ID: {run_id}")

        csv_bytes = result_df.to_csv(index=False, float_format="%.15g").encode("utf-8")
        xlsx_bytes = _result_to_excel_bytes(result_df)
        chart_png = _result_chart_png(result_df)

        dl_cols = st.columns(3, gap="small")
        with dl_cols[0]:
            st.download_button(
                "Download Result (CSV)",
                data=csv_bytes,
                file_name=f"{base_name}_result.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_cols[1]:
            st.download_button(
                "Download Result (Excel)",
                data=xlsx_bytes,
                file_name=f"{base_name}_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with dl_cols[2]:
            if chart_png is not None:
                st.download_button(
                    "Download Graph (PNG)",
                    data=chart_png,
                    file_name=f"{base_name}_result_graph.png",
                    mime="image/png",
                    use_container_width=True,
                )
            else:
                st.write("")

        st.markdown('<div class="block-title">Estimation Graph</div>', unsafe_allow_html=True)
        chart_df = result_df.set_index("Object")[["Estimation"]]
        st.bar_chart(chart_df, use_container_width=True)

elif st.session_state.current_page == "history":
    selected_fname = st.session_state.history_filename
    st.markdown('<div class="section-title">History Details</div>', unsafe_allow_html=True)
    if not selected_fname:
        st.info("No file selected from sidebar history.")
    else:
        runs_df = _read_db_runs(limit=1000)
        file_runs_df = runs_df[runs_df["Filename"] == selected_fname].copy()
        if file_runs_df.empty:
            st.warning("Selected file is no longer available in history.")
        else:
            st.markdown(f'<div class="block-title">File: {selected_fname}</div>', unsafe_allow_html=True)
            st.dataframe(
                format_dataframe_for_display(file_runs_df),
                use_container_width=True,
                hide_index=True,
                height=220,
            )

            run_options = file_runs_df["Run ID"].astype(int).tolist()
            selected_run_id = st.selectbox(
                "Select Saved Run",
                options=run_options,
                index=0,
                key=f"hist_run_sel_{selected_fname}",
            )
            selected_result_df = _read_db_run_result(int(selected_run_id))
            st.markdown('<div class="block-title">Saved Result</div>', unsafe_allow_html=True)
            st.dataframe(
                format_dataframe_for_display(selected_result_df),
                use_container_width=True,
                hide_index=True,
                height=420,
            )
