"""Live Foundry data source for the metrology dashboard.

When ``FS50_DATA_SOURCE=foundry`` the dashboard reads its live data from
Palantir Foundry datasets instead of the SQL Server ODS. The rest of the
codebase keeps its existing T-SQL query builders untouched; this module is the
single seam that:

  1. rewrites those T-SQL strings into the Spark SQL dialect Foundry speaks
     (``ctx.foundry_sql_server.query_foundry_sql``), and
  2. swaps fully-qualified SQL Server table names for Foundry dataset RIDs.

The Foundry datasets carry the **same column names** as the SQL tables, so no
column mapping is needed — only table names, identifier quoting, a few function
names, and inlined parameters differ.

Auth is read from the standard ``foundry-dev-tools`` config
(``%LOCALAPPDATA%/foundry-dev-tools/foundry-dev-tools/config.toml``); this
module never handles tokens directly.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from functools import lru_cache

import pandas as pd

# --- SQL table name -> Foundry dataset RID ---------------------------------
# Keyed by the *leaf* table name (schema/database prefixes are ignored so both
# ``ProcessData.ProcessHistory.EGPData`` and ``[...].[EGPData]`` resolve here).
_RID: dict[str, str] = {
    "EGPData": "ri.foundry.main.dataset.0e8c0c3d-5fa6-4529-9627-8d94fce60873",
    "ProcessHistoryEGPSummary": "ri.foundry.main.dataset.e1a1711d-560f-4cbf-b2dd-ddaa2ea625f1",
    "PartProduced": "ri.foundry.main.dataset.8848e862-ca48-4e4b-a3d9-e446787b4771",
    "ProcessHistoryGrinder": "ri.foundry.main.dataset.437d475e-7678-447e-b33d-f82d0aec5601",
    "ProcessHistoryMarker": "ri.foundry.main.dataset.930fad26-1838-49be-a3ac-97d3f172a47a",
    "ProcessHistoryCoaterRunData": "ri.foundry.main.dataset.e3049f64-5912-4778-b7d3-51d1cdc97f74",
    "PdrScrapEvent": "ri.foundry.main.dataset.81b5f3ef-098c-454d-afcd-ba8bc5acba47",
}

# Matches a fully-qualified table reference: optional bracketed/plain schema
# parts separated by dots, ending in one of the known leaf table names.
_TABLE_RE = re.compile(
    r"(?:\[?\w+\]?\.){0,3}\[?(" + "|".join(map(re.escape, _RID)) + r")\]?",
    re.IGNORECASE,
)

# Case-insensitive lookup for the leaf name captured above.
_RID_CI = {k.lower(): v for k, v in _RID.items()}


@lru_cache(maxsize=1)
def _context():
    """Return a cached FoundryContext (auth from foundry-dev-tools config)."""
    from foundry_dev_tools import FoundryContext

    return FoundryContext()


def _lit(value) -> str:
    """Render a Python value as a Spark SQL literal."""
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return "TIMESTAMP '" + pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S") + "'"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    s = str(value)
    # Strings that are really datetimes (ISO isoformat) become timestamps so
    # comparisons against timestamp columns work in Spark.
    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt) and re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", s):
        return "TIMESTAMP '" + dt.strftime("%Y-%m-%d %H:%M:%S") + "'"
    return "'" + s.replace("'", "''") + "'"


def _inline_params(sql: str, params: dict) -> str:
    """Replace ``:name`` bind markers with inlined Spark literals.

    Foundry's SQL endpoint does not take named parameters, so every value is
    rendered as a literal. Datetimes become ``TIMESTAMP`` literals and strings
    are single-quote escaped, so this is safe against injection for the
    engine-controlled callers here.
    """
    if not params:
        return sql

    def repl(m: re.Match) -> str:
        key = m.group(1)
        if key in params:
            return _lit(params[key])
        return m.group(0)

    return re.sub(r":(\w+)", repl, sql)


def _translate(sql: str) -> str:
    """Rewrite a T-SQL string into Foundry-compatible Spark SQL."""
    out = sql

    # 1. Table names -> backticked dataset RIDs.
    def _table(m: re.Match) -> str:
        leaf = m.group(1).lower()
        return "`" + _RID_CI[leaf] + "`"

    out = _TABLE_RE.sub(_table, out)

    # 2. Drop SQL Server locking hints.
    out = re.sub(r"\bWITH\s*\(\s*NOLOCK\s*\)", "", out, flags=re.IGNORECASE)

    # 3. PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY <expr>) OVER (...)
    #    -> Spark: percentile(<expr>, 0.5) OVER (...)
    out = re.sub(
        r"PERCENTILE_CONT\s*\(\s*([0-9.]+)\s*\)\s*"
        r"WITHIN\s+GROUP\s*\(\s*ORDER\s+BY\s+(.+?)\s*\)\s*OVER",
        lambda m: f"percentile({m.group(2)}, {m.group(1)}) OVER",
        out,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # 4. Sample standard deviation function name.
    out = re.sub(r"\bSTDEV\s*\(", "STDDEV_SAMP(", out, flags=re.IGNORECASE)

    # 5. Bracket-quoted identifiers [Col] -> backtick-quoted `Col`.
    out = re.sub(r"\[([^\]]+)\]", r"`\1`", out)

    return out


def query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run a (T-SQL) query against Foundry and return a DataFrame.

    ``sql`` is the same T-SQL the SQL-Server path uses; it is translated to
    Spark SQL and its ``:name`` parameters are inlined before execution.
    """
    spark_sql = _inline_params(_translate(sql), params or {})
    ctx = _context()
    return ctx.foundry_sql_server.query_foundry_sql(spark_sql)
