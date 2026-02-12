# OAM to COCO Y0 Automator

A Streamlit application for OAM (Object-Attribute Matrix) analysis that:
- reads a structured OAM CSV,
- ranks attributes with Excel-style `RANK.EQ` logic,
- runs COCO Y0 estimation,
- excludes attributes using `Stairs(2)` rules,
- runs COCO Y0 again on excluded attributes,
- builds final ranked results,
- and stores results in SQLite history.

## Main Features

- End-to-end 2-pass COCO Y0 workflow in one UI.
- 7-page guided flow:
  1. `Input Data`
  2. `Ranked Data`
  3. `COCO Y0 Estimation`
  4. `Excluded OAM`
  5. `Estimation 2`
  6. `Result`
  7. `History`
- Robust COCO parser:
  - overview metadata extraction,
  - section table extraction (`Rangsor`, `Stairs(1)`, `Stairs(2)`),
  - totals extraction,
  - COCO:Y0 estimation table extraction.
- SQLite history with file-based lookup and delete from sidebar.
- Result exports:
  - CSV
  - Excel
  - PNG chart (when `matplotlib` is available)

## How the App Works

1. Upload CSV in the sidebar.
2. App parses objects, attributes, directions, and `Y`.
3. Attributes are ranked using direction rules:
   - `0` = higher is better (descending rank)
   - `1` = lower is better (ascending rank)
4. Ranked matrix is sent to COCO Y0 (run #1).
5. From `Stairs(2)`, attributes are excluded when first-row value equals:
   - `number_of_objects - 1`
6. Excluded attributes are re-ranked and sent to COCO Y0 again (run #2).
7. Final result ranks objects by COCO estimation (`Becsles/Estimation`), highest first.
8. Final result is saved to SQLite and available in `History`.

## CSV Format Requirements

Your CSV should follow the OAM layout used by this project:

- Required header labels in column 0:
  - `Direction ID`
  - `Attribute ID`
  - `Attribute`
- Last column is treated as `Y`.
- Object rows are data rows below header rows.

Direction semantics:
- `0` -> higher raw value is better
- `1` -> lower raw value is better

Notes:
- Numeric parsing supports plain numbers, percent-like strings, and decimal comma variants.
- Before COCO submission, the app scales only the last `Y` value by `x100` (current project logic).

## Tech Stack

- Python
- Streamlit
- Pandas / NumPy
- Requests + BeautifulSoup4
- SQLite
- OpenPyXL
- Matplotlib (chart export)

## Project Structure

```text
coco_oam_app/
  app.py                 # Main Streamlit app and page flow
  requirements.txt
  README.md
  src/
    coco_client.py       # COCO form discovery + submit
    coco_parse.py        # COCO HTML parsing helpers
    db.py                # SQLite schema and CRUD
    oam_io.py            # CSV ingestion and OAM model
    ranking.py           # Ranking engine (RANK.EQ style)
    ui_display.py        # UI theme + display formatting
```

## Local Setup

### 1) Create and activate virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the app

```bash
streamlit run app.py
```

## Database and History

- SQLite file: `coco_runs.sqlite`
- Stored data:
  - run metadata (filename, object/attribute counts, excluded attributes, COCO HTML snapshots)
  - final per-object results (estimation + rank)
- Sidebar `History`:
  - lists previously uploaded filenames,
  - click filename to open `History` page details,
  - delete removes all runs for that filename.

## External Dependency

COCO Y0 endpoint used by the app:

- `https://miau.my-x.hu/myx-free/coco/beker_y0.php`

The app requires outbound internet access. If the COCO form or output structure changes, parsing/submission may require updates.

## Publishing Notes

Before publishing to GitHub, it is recommended to add/update `.gitignore` for local artifacts, for example:

```gitignore
.venv/
__pycache__/
*.pyc
coco_runs.sqlite
```

Optional: keep sample CSVs (`Data1.csv`, `Data2.csv`) if you want a reproducible demo.

## Disclaimer

This tool depends on a third-party public COCO service. Validate outputs before decision-making in production environments.

