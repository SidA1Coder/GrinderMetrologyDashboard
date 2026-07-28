"""SQLite history store for inspection results.

Every processed corner image becomes one row here. This local log is what the
KPIs, trend charts, gallery, heatmap and alert rules all read from, so the
dashboard works even if the SQL ODS is offline.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    image_name      TEXT NOT NULL,
    annotated_path  TEXT,
    location        TEXT,
    part_id         TEXT,
    corner          INTEGER,
    grinder         TEXT,
    grind_time      TEXT,
    decision        TEXT NOT NULL,
    defect_count    INTEGER NOT NULL DEFAULT 0,
    max_conf        REAL NOT NULL DEFAULT 0,
    classes         TEXT
);
CREATE INDEX IF NOT EXISTS idx_inspections_ts ON inspections(ts);
CREATE INDEX IF NOT EXISTS idx_inspections_loc ON inspections(location);


CREATE TABLE IF NOT EXISTS alert_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    rule_name TEXT NOT NULL,
    location  TEXT,
    message   TEXT NOT NULL,
    delivered INTEGER NOT NULL DEFAULT 0
);
"""


def _connect() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        # Migrate pre-existing databases that lack the grinder columns.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(inspections)")}
        if "grinder" not in cols:
            conn.execute("ALTER TABLE inspections ADD COLUMN grinder TEXT")
        if "grind_time" not in cols:
            conn.execute("ALTER TABLE inspections ADD COLUMN grind_time TEXT")
        # Index the grinder column only after it is guaranteed to exist.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_inspections_grinder ON inspections(grinder)"
        )


def add_inspection(
    *,
    ts: datetime,
    image_name: str,
    annotated_path: str | None,
    location: str | None,
    part_id: str | None,
    corner: int | None,
    decision: str,
    defect_count: int,
    max_conf: float,
    classes: list[str],
    grinder: str | None = None,
    grind_time: datetime | None = None,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO inspections
                (ts, image_name, annotated_path, location, part_id, corner,
                 grinder, grind_time, decision, defect_count, max_conf, classes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts.isoformat(),
                image_name,
                annotated_path,
                location,
                part_id,
                corner,
                grinder,
                grind_time.isoformat() if grind_time is not None else None,
                decision,
                defect_count,
                max_conf,
                json.dumps(classes),
            ),
        )
        return int(cur.lastrowid)


def already_processed(image_name: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM inspections WHERE image_name = ? LIMIT 1", (image_name,)
        ).fetchone()
        return row is not None


def processed_keys() -> set[str]:
    """Return the set of all already-processed image identifiers (one query)."""
    with _connect() as conn:
        rows = conn.execute("SELECT image_name FROM inspections").fetchall()
    return {r["image_name"] for r in rows}


def load_inspections() -> pd.DataFrame:
    with _connect() as conn:
        df = pd.read_sql_query("SELECT * FROM inspections ORDER BY ts DESC", conn)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
        if "grind_time" in df.columns:
            df["grind_time"] = pd.to_datetime(df["grind_time"], errors="coerce")
    return df


def recent_rejects(location: str, within_minutes: int) -> int:
    """Count REJECT inspections for a location within the last N minutes."""
    cutoff = (datetime.now()).timestamp() - within_minutes * 60
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ts FROM inspections WHERE location = ? AND decision = 'REJECT'",
            (location,),
        ).fetchall()
    return sum(1 for r in rows if datetime.fromisoformat(r["ts"]).timestamp() >= cutoff)


def window_stats(location: str, within_minutes: int) -> tuple[int, int]:
    """Return (rejects, total) for a location within the last N minutes."""
    cutoff = datetime.now().timestamp() - within_minutes * 60
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ts, decision FROM inspections WHERE location = ?", (location,)
        ).fetchall()
    rejects = total = 0
    for r in rows:
        if datetime.fromisoformat(r["ts"]).timestamp() >= cutoff:
            total += 1
            if r["decision"] == "REJECT":
                rejects += 1
    return rejects, total


def log_alert(
    *, ts: datetime, rule_name: str, location: str | None, message: str, delivered: bool
) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO alert_log (ts, rule_name, location, message, delivered) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts.isoformat(), rule_name, location, message, int(delivered)),
        )


def load_alerts() -> pd.DataFrame:
    with _connect() as conn:
        df = pd.read_sql_query("SELECT * FROM alert_log ORDER BY ts DESC", conn)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


def clear_all() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM inspections")
        conn.execute("DELETE FROM alert_log")
