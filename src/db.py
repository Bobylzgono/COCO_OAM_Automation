"""SQLite storage for analysis runs.

This DB is intentionally simple. It stores:
- runs: metadata + excluded attribute list
- object_results: final estimation + rank for each object

You can later extend with full matrices per run if needed.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Optional


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            filename TEXT,
            n_objects INTEGER NOT NULL,
            n_attributes INTEGER NOT NULL,
            excluded_attributes_json TEXT,
            coco_html_run1 TEXT,
            coco_html_run2 TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS object_results (
            run_id INTEGER NOT NULL,
            object_name TEXT NOT NULL,
            estimation REAL,
            final_rank INTEGER,
            PRIMARY KEY (run_id, object_name),
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()


def insert_run(
    conn: sqlite3.Connection,
    filename: str,
    n_objects: int,
    n_attributes: int,
    excluded_attr_ids: list[str],
    coco_html_run1: str,
    coco_html_run2: str,
) -> int:
    created_at = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO runs (created_at, filename, n_objects, n_attributes, excluded_attributes_json, coco_html_run1, coco_html_run2)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            created_at,
            filename,
            int(n_objects),
            int(n_attributes),
            json.dumps(excluded_attr_ids),
            coco_html_run1,
            coco_html_run2,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def insert_object_results(
    conn: sqlite3.Connection,
    run_id: int,
    rows: Iterable[tuple[str, Optional[float], Optional[int]]],
) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO object_results (run_id, object_name, estimation, final_rank)
        VALUES (?, ?, ?, ?)
        """,
        [(run_id, obj, est, rnk) for (obj, est, rnk) in rows],
    )
    conn.commit()


def list_runs(conn: sqlite3.Connection, limit: int = 50):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT run_id, created_at, filename, n_objects, n_attributes, excluded_attributes_json
        FROM runs
        ORDER BY run_id DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    return cur.fetchall()


def get_run_results(conn: sqlite3.Connection, run_id: int):
    cur = conn.cursor()
    cur.execute(
        "SELECT object_name, estimation, final_rank FROM object_results WHERE run_id=? ORDER BY final_rank ASC",
        (int(run_id),),
    )
    return cur.fetchall()


def delete_runs_by_filename(conn: sqlite3.Connection, filename: str) -> None:
    conn.execute("DELETE FROM runs WHERE filename = ?", (filename,))
    conn.commit()
