"""Edge Grind Profile (metrology) detection engine.

This is the numeric counterpart to the image inspection pipeline. It reads the
per-position edge-grind measurements from the SQL ODS
(``ProcessData.ProcessHistory.EGPData``), evaluates each plate (``SubID``)
against spec limits, and produces a PASS/REJECT verdict that names the specific
out-of-spec parameter(s).

Two-stage design (per the project plan):

* **Stage 1 - rules:** hard spec limits (USL/LSL) per parameter. Deterministic
  and explainable -- it reports exactly which parameter breached which limit.
* **Stage 2 - ML:** catches the subtle "within limits but still bad" plates.
  Uses a supervised model when historical good/bad labels are available,
  otherwise falls back to unsupervised anomaly detection.

The legacy MES ``*_result_code`` is intentionally ignored as ground truth --
it is the inaccurate verdict this engine is meant to replace.

Column-agnostic: the parameters and limits come from ``config.METROLOGY_SPECS``
so new measurements can be added without touching this file.

Run a self-contained demo (no database required)::

    python -m dashboard.metrology --demo
"""

from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config

# Base measurement names in EGPData are suffixed per panel side.
_SIDES = ("Left", "Right")

# Cached pooled SQLAlchemy engine (see _make_engine); created lazily on first
# live query so repeated queries reuse one connection pool.
_ENGINE = None


def _run_parallel(tasks: dict) -> dict:
    """Run independent no-arg callables concurrently, returning name->result.

    Used to fire the independent ODS queries at once (pandas/pyodbc releases the
    GIL during the DB round-trip, so this gives real wall-clock speed-up). Each
    task is isolated: an exception is returned in place of that task's result so
    one slow/failed query never blocks the others.
    """
    results: dict = {}
    with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as ex:
        futures = {name: ex.submit(fn) for name, fn in tasks.items()}
        for name, fut in futures.items():
            try:
                results[name] = fut.result()
            except Exception as exc:  # noqa: BLE001 - surfaced to caller
                results[name] = exc
    return results


# Image filenames on the share are "<SubID>_L.bmp" / "<SubID>_R.bmp" (older
# scans used "_T"); the SubID is the plate id used to join against EGPData.
_IMG_SUFFIX_RE = re.compile(r"_(?:L|R|T)$", re.IGNORECASE)

# Grinder equipment names look like "PGT31A-GRINDER" -> friendly "Grinder A".
_GRINDER_RE = re.compile(r"1([A-E])[- ]?GRIND", re.IGNORECASE)


def grinder_label(equipment_name: str | None) -> str | None:
    """Map a raw grinder equipment name to a friendly ``Grinder A`` label.

    ``PGT31A-GRINDER`` -> ``Grinder A``. Unrecognised names are returned as-is.
    """
    if not equipment_name:
        return None
    m = _GRINDER_RE.search(str(equipment_name))
    if m:
        return f"Grinder {m.group(1).upper()}"
    return str(equipment_name)


def sub_id_from_image(image_name: str | Path) -> str:
    """Derive the metrology SubID from a corner image filename.

    Strips the extension and the trailing ``_L`` / ``_R`` / ``_T`` edge marker,
    e.g. ``260627793126_L.bmp`` -> ``260627793126``. Also accepts a
    nested/relative path and uses only the final filename.
    """
    stem = Path(str(image_name)).name
    # Drop extension(s) -- filenames on the share use a single .bmp.
    stem = Path(stem).stem
    return _IMG_SUFFIX_RE.sub("", stem)


# --- Result types ----------------------------------------------------------
@dataclass
class SpecViolation:
    """A single parameter breaching a spec limit on a plate."""

    parameter: str  # e.g. "EdgeGrind_PeakHeight_Right"
    value: float  # the extreme (worst) measured value
    limit: float  # the limit that was breached
    bound: str  # "USL" or "LSL"
    position: float | None = None  # Position along the edge, if known

    def describe(self) -> str:
        return (
            f"{self.parameter} = {self.value:.3f} "
            f"({'above' if self.bound == 'USL' else 'below'} "
            f"{self.bound} {self.limit:g})"
        )


@dataclass
class MetrologyResult:
    """Per-plate metrology verdict."""

    sub_id: str
    verdict: str  # "PASS" or "REJECT"
    equipment: str | None = None  # friendly "Grinder A".. lineage label
    grinder_loc: str | None = None  # GRINDER1/GRINDER2 = long/short edge
    grind_time: datetime | None = None  # when the plate was ground
    vtd: str | None = None  # friendly "VTD A".. coater lineage label
    vtd_time: datetime | None = None  # when the plate was VTD-coated
    profiler: str | None = None
    read_time: datetime | None = None
    n_positions: int = 0
    violations: list[SpecViolation] = field(default_factory=list)
    anomaly_score: float | None = None  # Stage 2 (higher = more anomalous)
    ml_flag: bool = False  # Stage 2 flagged it despite passing rules

    @property
    def out_of_spec_params(self) -> list[str]:
        return [v.parameter for v in self.violations]

    @property
    def reason(self) -> str:
        if self.violations:
            return "; ".join(v.describe() for v in self.violations)
        if self.ml_flag:
            return f"ML anomaly (score={self.anomaly_score:.3f})"
        return "within spec"


# --- Spec helpers ----------------------------------------------------------
def spec_columns() -> dict[str, tuple[float | None, float | None]]:
    """Expand base spec names to the concrete _Left/_Right columns."""
    cols: dict[str, tuple[float | None, float | None]] = {}
    for base, (lsl, usl) in config.METROLOGY_SPECS.items():
        for side in _SIDES:
            cols[f"{base}_{side}"] = (lsl, usl)
    return cols


# --- Data access -----------------------------------------------------------
def _egp_select() -> str:
    return """
        SELECT E.[DataID], E.[EquipmentID], E.[ReadTime], E.[SubID], E.[Position],
               E.[SerialNumber], E.[PanelPosX], E.[PanelPosY],
               E.[Dropouts_Left], E.[Dropouts_Right],
               E.[EdgeGrind_Delta_Left], E.[EdgeGrind_Delta_Right],
               E.[EdgeGrind_PeakHeight_Left], E.[EdgeGrind_PeakHeight_Right],
               E.[GlassThickness_Left], E.[GlassThickness_Right],
               E.[GlassThickness_pix_Left], E.[GlassThickness_pix_Right],
               E.[PanelWidth], E.[MaximumProfileHt_Left], E.[MaximumProfileHt_Right],
               E.[TotalProfileHt], E.[Radius_Left], E.[Radius_Right],
               E.[Radius_StdDev_Left], E.[Radius_StdDev_Right], E.[CollectedTimeUtc]
        FROM ProcessData.ProcessHistory.EGPData AS E
    """.strip()


def _make_engine():
    """Create (once) a pooled SQLAlchemy engine for the metrology ODS.

    The engine is cached as a module singleton so repeated queries reuse the
    same connection pool instead of paying ODBC/Windows-auth connection setup
    on every call (which was a major source of dashboard latency). Accepts
    either a raw ODBC connection string (assembled from the site, like the JMP
    script) or a full SQLAlchemy URL supplied via FS50_MET_CONN.
    """
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    from sqlalchemy import create_engine
    from sqlalchemy.engine import URL

    conn = config.metrology_odbc_str()
    kwargs = dict(pool_pre_ping=True, pool_recycle=1800, pool_size=8, max_overflow=8)
    # A SQLAlchemy URL contains "://"; a raw ODBC string has DRIVER=/SERVER=.
    if "://" in conn:
        _ENGINE = create_engine(conn, **kwargs)
    else:
        url = URL.create("mssql+pyodbc", query={"odbc_connect": conn})
        _ENGINE = create_engine(url, **kwargs)
    return _ENGINE


def _read_sql(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Run a read-only query against the configured live source.

    Routes to Palantir Foundry when ``FS50_DATA_SOURCE=foundry`` (translating
    the T-SQL to Spark SQL and swapping table names for dataset RIDs), otherwise
    to the pooled SQL Server engine. Callers keep writing plain T-SQL.
    """
    params = params or {}
    if config.metrology_data_source() == "foundry":
        import foundry_source

        return foundry_source.query(sql, params)
    from sqlalchemy import text

    engine = _make_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def load_egp_data(
    sub_ids: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    mock: bool | None = None,
) -> pd.DataFrame:
    """Load edge-grind profile rows from the ODS (or mock data offline).

    Returns one row per (SubID, Position, SerialNumber) -- i.e. the raw profile
    scan, many rows per plate. Set ``mock=True`` to force synthetic data.
    """
    if mock if mock is not None else config.use_mock_metrology():
        return _mock_egp_data(sub_ids)

    # Real database path.
    sql = _egp_select()
    clauses = []
    params: dict = {}
    if sub_ids:
        # Named-parameter IN list.
        keys = [f"sid{i}" for i in range(len(sub_ids))]
        clauses.append("E.[SubID] IN (" + ", ".join(f":{k}" for k in keys) + ")")
        params.update(dict(zip(keys, sub_ids)))
    if start is not None:
        clauses.append("E.[ReadTime] >= :start")
        params["start"] = start
    if end is not None:
        clauses.append("E.[ReadTime] <= :end")
        params["end"] = end
    if clauses:
        sql += "\nWHERE " + " AND ".join(clauses)
    sql += "\nORDER BY E.[SubID], E.[Position]"
    return _read_sql(sql, params)


def _egp_agg_select(cols: list[str]) -> str:
    """Build a per-plate aggregation SELECT over EGPData.

    EGPData carries ~3,300 profile rows per panel, so pulling raw rows for even
    an hour is millions of rows. Instead we aggregate server-side (GROUP BY
    SubID) into one row per plate. The aliases are named ``{col}_mean/_min/
    _max/_std`` so they line up exactly with :func:`plate_features` (and thus
    the trained ML model's feature order). ``STDEV`` is SQL Server's sample
    standard deviation, matching pandas ``std(ddof=1)``.
    """
    parts = [
        "E.[SubID]",
        "MAX(E.[ReadTime]) AS ReadTime",
        "MAX(E.[EquipmentID]) AS EquipmentID",
        "MAX(E.[SerialNumber]) AS SerialNumber",
        "COUNT(*) AS n_rows",
        "AVG(CAST(E.[PanelWidth] AS FLOAT)) AS PanelWidth",
    ]
    for c in cols:
        parts.append(f"AVG(CAST(E.[{c}] AS FLOAT)) AS [{c}_mean]")
        parts.append(f"MIN(E.[{c}]) AS [{c}_min]")
        parts.append(f"MAX(E.[{c}]) AS [{c}_max]")
        parts.append(f"STDEV(CAST(E.[{c}] AS FLOAT)) AS [{c}_std]")
    return (
        "SELECT "
        + ",\n       ".join(parts)
        + "\nFROM ProcessData.ProcessHistory.EGPData AS E"
    )


def load_egp_agg(
    start: datetime | None = None,
    end: datetime | None = None,
    mock: bool | None = None,
) -> pd.DataFrame:
    """Load one aggregated feature row per plate (SubID) from the ODS.

    This is the fast, production path: rather than streaming millions of raw
    position rows into pandas, it lets SQL Server compute the per-plate
    mean/min/max/std of every spec column. Returns a DataFrame indexed by
    position with a ``SubID`` column plus ``{col}_mean/_min/_max/_std`` columns.
    In mock mode it derives the same aggregate frame from the synthetic raw
    profiles so the two paths stay consistent.
    """
    cols = list(spec_columns().keys())
    if mock if mock is not None else config.use_mock_metrology():
        return _agg_from_raw(_mock_egp_data(), cols)

    sql = _egp_agg_select(cols)
    clauses: list[str] = []
    params: dict = {}
    if start is not None:
        clauses.append("E.[ReadTime] >= :start")
        params["start"] = start
    if end is not None:
        clauses.append("E.[ReadTime] <= :end")
        params["end"] = end
    if clauses:
        sql += "\nWHERE " + " AND ".join(clauses)
    sql += "\nGROUP BY E.[SubID]"
    df = _read_sql(sql, params)
    if not df.empty:
        df["SubID"] = df["SubID"].astype(str)
        df["ReadTime"] = pd.to_datetime(df["ReadTime"], errors="coerce")
    return df


def _agg_from_raw(raw: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Collapse raw per-position mock rows into the per-plate aggregate frame.

    Mirrors :func:`load_egp_agg` so the mock path yields the same columns.
    """
    if raw.empty or "SubID" not in raw.columns:
        return pd.DataFrame()
    present = [c for c in cols if c in raw.columns]
    agg = raw.groupby("SubID")[present].agg(["mean", "min", "max", "std"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    agg = agg.fillna(0.0).reset_index()
    meta = raw.groupby("SubID").agg(
        ReadTime=("ReadTime", "max"),
        EquipmentID=("EquipmentID", "max"),
        SerialNumber=("SerialNumber", "max"),
        n_rows=("SubID", "size"),
    )
    if "PanelWidth" in raw.columns:
        meta["PanelWidth"] = raw.groupby("SubID")["PanelWidth"].mean()
    # Carry grinder/VTD lineage through when the mock frame supplies it.
    for extra in ("EquipmentName", "GrinderLoc", "GrindTime", "VtdName", "VtdTime"):
        if extra in raw.columns:
            meta[extra] = raw.groupby("SubID")[extra].first()
    return agg.merge(meta.reset_index(), on="SubID", how="left")


# --------------------------------------------------------------------------
# Foundry wide grinder decode
# --------------------------------------------------------------------------
# The Foundry ``ProcessHistoryGrinder`` dataset is WIDE: one row per grind
# reading where ``EquipmentName`` is the grinder line and 12 columns
# ``G{1,2}S{R,L}W{C,M,F}`` carry the *active groove* (1-4) of each wheel, each
# with a matching ``{col}_MW`` giving that wheel's meters-worked. Decode:
#   * ``G1`` = long edge (GRINDER1), ``G2`` = short edge (GRINDER2)
#   * ``R``/``L`` = right/left side
#   * ``C``/``M``/``F`` = Coarse/Medium/Fine profile
# The plate id that joins to the marker (and thus EGPData) is ``GrinderSubid``
# (the numeric ``Subid`` is grinder-internal and does not link to the marker).
_GR_GRINDER = {"1": "GRINDER1", "2": "GRINDER2"}  # long / short edge
_GR_SIDE = {"R": "Right", "L": "Left"}
_GR_PROFILE = {"C": "Coarse", "M": "Medium", "F": "Fine"}
_GR_WHEELS = [
    (f"G{g}S{s}W{p}", g, s, p)
    for g in ("1", "2")
    for s in ("R", "L")
    for p in ("C", "M", "F")
]
_GR_LONG_COLS = [
    "SubID",
    "EquipmentName",
    "GrindTime",
    "GrinderLoc",
    "Side",
    "Profile",
    "Groove",
    "Meters",
]


def _foundry_grinder_long(
    start: datetime | None,
    end: datetime | None,
    sub_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Decode the wide Foundry grinder table into one row per wheel.

    Joins ``ProcessHistoryGrinder`` to ``ProcessHistoryMarker`` on
    ``phg.GrinderSubid = phm.VirtualSubId`` (so an EGP ``SubID`` resolves), then
    melts the 12 wide ``G{1,2}S{R,L}W{C,M,F}`` groove columns (and their ``_MW``
    meters-worked partners) into long form. Returns the columns in
    :data:`_GR_LONG_COLS`: ``SubID``, ``EquipmentName``, ``GrindTime``,
    ``GrinderLoc`` (GRINDER1=long / GRINDER2=short), ``Side`` (Left/Right),
    ``Profile`` (Coarse/Medium/Fine), ``Groove`` ("1".."4") and ``Meters``.
    """
    empty = pd.DataFrame(columns=_GR_LONG_COLS)
    base_sel = ", ".join(
        f"phg.[{c}] AS {c}, phg.[{c}_MW] AS {c}_MW" for c, *_ in _GR_WHEELS
    )
    select = (
        "SELECT phm.SubId AS SubID, phg.EquipmentName, "
        "phg.ReadTime AS GrindTime, " + base_sel + " "
        "FROM mfg.ProcessHistoryGrinder AS phg "
        "INNER JOIN mfg.ProcessHistoryMarker AS phm "
        "  ON phg.GrinderSubid = phm.VirtualSubId "
    )
    params: dict = {}
    if start is not None:
        # Grinding precedes the EGP read; widen the lower bound.
        clauses = ["phg.ReadTime >= :gstart"]
        params["gstart"] = start - timedelta(hours=6)
        if end is not None:
            clauses.append("phg.ReadTime <= :gend")
            params["gend"] = end
        sql = select + "WHERE " + " AND ".join(clauses)
    elif sub_ids:
        keys = [f"sid{i}" for i in range(len(sub_ids))]
        sql = select + "WHERE phm.SubId IN (" + ", ".join(f":{k}" for k in keys) + ")"
        params = dict(zip(keys, sub_ids))
    else:
        return empty

    wide = _read_sql(sql, params)
    if wide.empty:
        return empty
    wide["SubID"] = wide["SubID"].astype(str)
    wide["GrindTime"] = pd.to_datetime(wide["GrindTime"], errors="coerce")

    frames: list[pd.DataFrame] = []
    for col, g, s, p in _GR_WHEELS:
        if col not in wide.columns:
            continue
        part = wide[["SubID", "EquipmentName", "GrindTime"]].copy()
        part["GrinderLoc"] = _GR_GRINDER[g]
        part["Side"] = _GR_SIDE[s]
        part["Profile"] = _GR_PROFILE[p]
        part["Groove"] = (
            pd.to_numeric(wide[col], errors="coerce").astype("Int64").astype("string")
        )
        mw = wide.get(f"{col}_MW")
        part["Meters"] = pd.to_numeric(mw, errors="coerce") if mw is not None else pd.NA
        frames.append(part)
    if not frames:
        return empty
    long = pd.concat(frames, ignore_index=True)
    long = long[long["Groove"].notna() & (long["Groove"] != "<NA>")]
    if sub_ids:
        long = long[long["SubID"].isin({str(x) for x in sub_ids})]
    return long[_GR_LONG_COLS]


def _foundry_grinder_info(
    sub_ids: list[str],
    start: datetime | None,
    end: datetime | None,
) -> pd.DataFrame:
    """Foundry equivalent of :func:`load_grinder_info` — one row per plate.

    The wide grinder row already carries the grinder line (``EquipmentName``)
    and grind time per plate, so we only need the marker join to resolve the
    plate ``SubID``. ``GrinderLoc`` (long/short) is not a single value here —
    one physical grinder line grinds both edges — so it is left null; the
    per-edge / per-groove detail is available via :func:`load_grinder_grooves`.
    """
    empty = pd.DataFrame(columns=["SubID", "EquipmentName", "GrinderLoc", "GrindTime"])
    select = (
        "SELECT phm.SubId AS SubID, phg.EquipmentName, phg.ReadTime AS GrindTime "
        "FROM mfg.ProcessHistoryGrinder AS phg "
        "INNER JOIN mfg.ProcessHistoryMarker AS phm "
        "  ON phg.GrinderSubid = phm.VirtualSubId "
    )
    params: dict = {}
    if start is not None:
        clauses = ["phg.ReadTime >= :gstart"]
        params["gstart"] = start - timedelta(hours=6)
        if end is not None:
            clauses.append("phg.ReadTime <= :gend")
            params["gend"] = end
        sql = select + "WHERE " + " AND ".join(clauses)
    elif sub_ids:
        keys = [f"sid{i}" for i in range(len(sub_ids))]
        sql = select + "WHERE phm.SubId IN (" + ", ".join(f":{k}" for k in keys) + ")"
        params = dict(zip(keys, sub_ids))
    else:
        return empty

    try:
        gdf = _read_sql(sql, params)
    except Exception:
        return empty
    if gdf.empty:
        return empty
    gdf["SubID"] = gdf["SubID"].astype(str)
    if start is not None and sub_ids:
        gdf = gdf[gdf["SubID"].isin({str(s) for s in sub_ids})]
    gdf["GrindTime"] = pd.to_datetime(gdf["GrindTime"], errors="coerce")
    gdf = gdf.dropna(subset=["EquipmentName"])
    # One grind record per plate: keep the most recent grind pass.
    gdf = gdf.sort_values("GrindTime").drop_duplicates("SubID", keep="last")
    gdf["GrinderLoc"] = None
    return gdf[["SubID", "EquipmentName", "GrinderLoc", "GrindTime"]]


def load_grinder_info(
    sub_ids: list[str],
    mock: bool | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Return grinder lineage + location + grind time per plate.

    Joins ``mfg.ProcessHistoryGrinder`` to ``mfg.ProcessHistoryMarker`` (the
    same link the JMP script uses: ``phg.SubId = phm.VirtualSubId``) so an
    EGPData SubID resolves to its grinder line (e.g. ``PGT31A-GRINDER``), its
    grinder location (``GRINDER1``/``GRINDER2`` = long/short edge) and the time
    it was ground (``phg.ReadTime``).

    Two query strategies:

    * **Windowed scan** (fast, used for live dashboards): when ``start``/``end``
      are given, scan ``phg.ReadTime`` over the window once (grinding precedes
      the EGP read, so ``start`` is widened by a few hours) and map locally.
      This avoids a giant ``IN (...)`` list of hundreds of SubIDs, which is the
      production bottleneck.
    * **Targeted IN-list** (bounded lookups): used when only ``sub_ids`` is
      supplied and no window, e.g. resolving a handful of image SubIDs.

    Returns an empty frame in mock mode or on any failure so callers can
    degrade gracefully. Columns: ``SubID``, ``EquipmentName``, ``GrinderLoc``,
    ``GrindTime``.
    """
    empty = pd.DataFrame(columns=["SubID", "EquipmentName", "GrinderLoc", "GrindTime"])
    if mock if mock is not None else config.use_mock_metrology():
        return empty
    if not sub_ids and start is None:
        return empty
    if config.metrology_data_source() == "foundry":
        return _foundry_grinder_info(sub_ids, start, end)
    try:
        select = (
            "SELECT phm.SubId AS SubID, phg.EquipmentName, "
            "       phg.Location AS GrinderLoc, phg.ReadTime AS GrindTime "
            "FROM mfg.ProcessHistoryGrinder AS phg "
            "INNER JOIN mfg.ProcessHistoryMarker AS phm "
            "  ON phg.SubId = phm.VirtualSubId "
        )
        params: dict = {}
        if start is not None:
            # Grinding happens upstream of (before) the EGP read; widen the
            # lower bound so plates ground shortly before the window resolve.
            clauses = ["phg.ReadTime >= :gstart"]
            params["gstart"] = start - timedelta(hours=6)
            if end is not None:
                clauses.append("phg.ReadTime <= :gend")
                params["gend"] = end
            sql = select + "WHERE " + " AND ".join(clauses)
        else:
            keys = [f"sid{i}" for i in range(len(sub_ids))]
            sql = (
                select + "WHERE phm.SubId IN (" + ", ".join(f":{k}" for k in keys) + ")"
            )
            params = dict(zip(keys, sub_ids))
        gdf = _read_sql(sql, params)
        if gdf.empty:
            return empty
        gdf["SubID"] = gdf["SubID"].astype(str)
        # When we scanned a window, keep only the plates we actually need.
        if start is not None and sub_ids:
            gdf = gdf[gdf["SubID"].isin(set(str(s) for s in sub_ids))]
        # One grind record per plate: keep the most recent grind pass.
        gdf["GrindTime"] = pd.to_datetime(gdf["GrindTime"], errors="coerce")
        gdf = gdf.dropna(subset=["EquipmentName"])
        gdf = gdf.sort_values("GrindTime").drop_duplicates("SubID", keep="last")
        return gdf[["SubID", "EquipmentName", "GrinderLoc", "GrindTime"]]
    except Exception:
        return empty


def load_grinder_map(sub_ids: list[str], mock: bool | None = None) -> dict[str, str]:
    """Backward-compatible ``{SubID: grinder equipment name}`` map."""
    gdf = load_grinder_info(sub_ids, mock=mock)
    if gdf.empty:
        return {}
    return dict(zip(gdf["SubID"], gdf["EquipmentName"].astype(str)))


# VTD (coater) lineage names look like "PGT31A-VTD_COATER" -> "VTD A".
_VTD_RE = re.compile(r"1([A-E])[- ]?", re.IGNORECASE)


def vtd_label(equipment_name: str | None) -> str | None:
    """Map a raw VTD/coater equipment name to a friendly ``VTD A`` label.

    Falls back to the first six characters (the line lineage) when the name
    does not match the expected pattern.
    """
    if not equipment_name:
        return None
    m = _VTD_RE.search(str(equipment_name))
    if m:
        return f"VTD {m.group(1).upper()}"
    return str(equipment_name)[:6]


def load_vtd_info(
    sub_ids: list[str],
    mock: bool | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Return VTD (coater) lineage + coat time per plate.

    Reads ``mfg.ProcessHistoryCoaterRunData`` (the VTD coater run history) so an
    EGPData SubID resolves to the VTD line it was coated on and when. Like
    :func:`load_grinder_info` it supports a fast windowed scan (VTD coating is
    *downstream* of the EGP read, so the upper bound is widened) or a targeted
    IN-list. Returns an empty frame in mock mode or on any failure.

    Columns: ``SubID``, ``VtdName``, ``VtdTime``.
    """
    empty = pd.DataFrame(columns=["SubID", "VtdName", "VtdTime"])
    if mock if mock is not None else config.use_mock_metrology():
        return empty
    if not sub_ids and start is None:
        return empty
    try:
        select = (
            "SELECT SubId AS SubID, EquipmentName AS VtdName, "
            "       ReadTime AS VtdTime "
            "FROM mfg.ProcessHistoryCoaterRunData "
        )
        params: dict = {}
        if start is not None:
            clauses = ["ReadTime >= :vstart"]
            params["vstart"] = start
            if end is not None:
                # VTD coating trails the EGP read; widen the upper bound.
                clauses.append("ReadTime <= :vend")
                params["vend"] = end + timedelta(hours=12)
            sql = select + "WHERE " + " AND ".join(clauses)
        else:
            keys = [f"sid{i}" for i in range(len(sub_ids))]
            sql = select + "WHERE SubId IN (" + ", ".join(f":{k}" for k in keys) + ")"
            params = dict(zip(keys, sub_ids))
        vdf = _read_sql(sql, params)
        if vdf.empty:
            return empty
        vdf["SubID"] = vdf["SubID"].astype(str)
        if start is not None and sub_ids:
            vdf = vdf[vdf["SubID"].isin(set(str(s) for s in sub_ids))]
        vdf["VtdTime"] = pd.to_datetime(vdf["VtdTime"], errors="coerce")
        vdf = vdf.dropna(subset=["VtdName"])
        # One coat record per plate: keep the most recent coat pass.
        vdf = vdf.sort_values("VtdTime").drop_duplicates("SubID", keep="last")
        return vdf[["SubID", "VtdName", "VtdTime"]]
    except Exception:
        return empty


def load_broken_plates(
    start: datetime | None = None,
    end: datetime | None = None,
    mock: bool | None = None,
) -> pd.DataFrame:
    """Return plates scrapped as **Broken** at FS100 (post-VTD).

    Reads ``ModuleAssembly.ProcessHistory.PdrScrapEvent`` filtered to
    ``ScrapReasonText = 'Broken'`` at a VTD coater. The scrap event's ``ID``
    column is the plate SubID, so it joins straight back to EGPData and the
    grinder tables to explain *why* a plate broke. ``SourceLocation`` is the
    VTD coater line (``PGT31A-VTD_COATER`` -> line A..E).

    Columns: ``SubID``, ``VtdName`` (friendly ``VTD A``), ``VtdLine`` (raw
    equipment), ``BrokenTime`` (local), ``ScrapLocation``.
    Returns an empty frame in mock mode-less callers or on any failure.
    """
    empty = pd.DataFrame(
        columns=["SubID", "VtdName", "VtdLine", "BrokenTime", "ScrapLocation"]
    )
    if mock if mock is not None else config.use_mock_metrology():
        return _mock_broken_plates(start, end)
    try:
        sql = (
            "SELECT [ID] AS SubID, [SourceLocation] AS VtdLine, "
            "       [ScrapLocation] AS ScrapLocation, [TimeStamp] AS BrokenTime "
            "FROM [ModuleAssembly].[ProcessHistory].[PdrScrapEvent] "
            "WHERE [ScrapReasonText] = 'Broken' "
            "  AND [SourceLocation] LIKE '%VTD_COATER'"
        )
        clauses: list[str] = []
        params: dict = {}
        if start is not None:
            clauses.append("[TimeStamp] >= :bstart")
            params["bstart"] = start
        if end is not None:
            # Breakage trails the grind/EGP read; widen the upper bound so a
            # plate ground near the window's end still surfaces its break.
            clauses.append("[TimeStamp] <= :bend")
            params["bend"] = end + timedelta(hours=24)
        if clauses:
            sql += " AND " + " AND ".join(clauses)
        bdf = _read_sql(sql, params)
        if bdf.empty:
            return empty
        bdf["SubID"] = bdf["SubID"].astype(str)
        bdf["BrokenTime"] = pd.to_datetime(bdf["BrokenTime"], errors="coerce")
        bdf["VtdName"] = bdf["VtdLine"].map(vtd_label)
        # One break record per plate: keep the most recent scrap event.
        bdf = bdf.sort_values("BrokenTime").drop_duplicates("SubID", keep="last")
        return bdf[["SubID", "VtdName", "VtdLine", "BrokenTime", "ScrapLocation"]]
    except Exception:
        return empty


# --------------------------------------------------------------------------
# Defect alerts (authoritative run-length engine + groove attribution)
# --------------------------------------------------------------------------
@dataclass
class DefectAlert:
    """A single attributed edge-grind defect, ready to display as an alert."""

    sub_id: str
    read_time: datetime | None
    defect: str  # "Chip" | "Dropout" | "Shiner"
    metric: str  # "GlassThickness" | "Dropouts" | "Radius"
    side: str  # "Left" | "Right"
    edge: str  # "Long" | "Short"
    grinder: str | None = None  # friendly "Grinder A".. line
    grinder_loc: str | None = None  # GRINDER1 (long) / GRINDER2 (short)
    groove: str | None = None  # "1".."4"
    length_mm: float = 0.0
    pos_x: float | None = None
    pos_y: float | None = None
    worst_value: float | None = None
    suspect_focus: bool = False  # edge outside +/-tol of the 60 mm sensor target
    max_offset_mm: float | None = None  # worst |MaximumProfileHt| over the run

    @property
    def message(self) -> str:
        g = self.grinder or "?"
        gr = f"Groove {self.groove}" if self.groove else "Groove ?"
        msg = (
            f"{self.metric} out of spec — {self.edge} Edge, {self.side} Side, {g}, {gr}"
        )
        if self.suspect_focus:
            msg += " ⚠️ suspected skewed panel (out of focus)"
        return msg


def _defect_scan_sql() -> str:
    """Filtered raw scan for fixed-band defects (chips + dropouts).

    Returns the ~handful of candidate profile rows (fast, COUNT-like) that could
    be part of a defect run, carrying Position + physical PanelPos so runs can be
    grouped and attributed to a long/short edge locally. Median-relative metrics
    (Radius/shiner) are handled separately in :func:`_radius_scan_sql`.
    """
    conds = []
    cols = []
    for col, rule in config.DEFECT_RULES.items():
        if rule.get("mode", "band") != "band":
            continue
        cols += [f"E.[{col}_Left]", f"E.[{col}_Right]"]
        for side in _SIDES:
            c = f"E.[{col}_{side}]"
            if rule.get("low") is not None:
                conds.append(f"{c} < {rule['low']}")
            if rule.get("high") is not None:
                conds.append(f"{c} > {rule['high']}")
    ph = config.PROFILE_HT_METRIC
    cols += [f"E.[{ph}_Left]", f"E.[{ph}_Right]"]
    return (
        "SELECT E.[SubID], E.[EquipmentID], E.[SerialNumber], E.[ReadTime], "
        "E.[Position], E.[PanelPosX], E.[PanelPosY], "
        + ", ".join(cols)
        + " FROM ProcessData.ProcessHistory.EGPData AS E "
        "WHERE E.[ReadTime] BETWEEN :start AND :end AND (" + " OR ".join(conds) + ")"
    )


def _radius_scan_sql() -> str:
    """Median-relative scan for shiner defects (Radius).

    The edge-grind Radius baseline drifts, so the EGP defect-analysis tool flags
    a shiner when the Radius deviates from the *calculated median* by more than a
    tolerance. The median is computed per ``(SubID, SerialNumber)`` — i.e. per
    profiler pass — because the SEL and LEL profilers sit at different Radius
    baselines (~9% apart); pooling them into one panel median makes an entire
    profiler's samples look off-target and flags false shiners. Only deviating
    rows are returned, so the client never pulls the full per-position profile.
    """
    rule = config.DEFECT_RULES["Radius"]
    f = rule["tol_pct"] / 100.0
    lo, hi = 1.0 - f, 1.0 + f
    ph = config.PROFILE_HT_METRIC
    return (
        "WITH R AS ("
        " SELECT E.[SubID], E.[SerialNumber], E.[ReadTime], E.[Position],"
        " E.[EquipmentID], E.[PanelPosX], E.[PanelPosY],"
        " E.[Radius_Left], E.[Radius_Right],"
        f" E.[{ph}_Left], E.[{ph}_Right],"
        " PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY E.[Radius_Left])"
        " OVER (PARTITION BY E.[SubID], E.[SerialNumber]) AS medL,"
        " PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY E.[Radius_Right])"
        " OVER (PARTITION BY E.[SubID], E.[SerialNumber]) AS medR"
        " FROM ProcessData.ProcessHistory.EGPData AS E"
        " WHERE E.[ReadTime] BETWEEN :start AND :end)"
        " SELECT SubID, SerialNumber, ReadTime, Position, EquipmentID,"
        " PanelPosX, PanelPosY,"
        f" Radius_Left, Radius_Right, [{ph}_Left], [{ph}_Right], medL, medR FROM R"
        f" WHERE Radius_Left < medL * {lo} OR Radius_Left > medL * {hi}"
        f" OR Radius_Right < medR * {lo} OR Radius_Right > medR * {hi}"
    )


def _bad_mask(values: pd.Series, rule: dict) -> pd.Series:
    """Boolean mask of samples that breach a defect rule's band."""
    mask = pd.Series(False, index=values.index)
    if rule.get("low") is not None:
        mask |= values < rule["low"]
    if rule.get("high") is not None:
        mask |= values > rule["high"]
    return mask


def _runs_from_positions(sub: pd.DataFrame, min_mm: float) -> list[dict]:
    """Group consecutive breaching samples into runs meeting the length rule.

    ``sub`` holds the breaching rows for one (SubID, metric, side), with columns
    ``Position``, ``PanelPosX``, ``PanelPosY``, ``val`` and (optionally) ``ph``
    (``MaximumProfileHt`` offset from the 60 mm target). Samples ~1.25 mm apart;
    a gap of a single missing sample (Position delta <= 2) still counts as one
    run. A run qualifies when its physical span reaches ``min_mm``. Runs whose
    edge sits beyond +/- ``config.PROFILE_HT_TOL_MM`` of the sensor target are
    kept but flagged ``suspect_focus`` (possible skewed / mis-positioned panel).
    """
    if sub.empty:
        return []
    sub = sub.sort_values("Position")
    pos = sub["Position"].to_numpy()
    px = sub["PanelPosX"].to_numpy()
    py = sub["PanelPosY"].to_numpy()
    val = sub["val"].to_numpy()
    ph = (
        pd.to_numeric(sub["ph"], errors="coerce").to_numpy()
        if "ph" in sub.columns
        else None
    )
    runs: list[dict] = []
    s = 0
    for i in range(1, len(pos) + 1):
        if i == len(pos) or pos[i] - pos[i - 1] > 2:
            span_mm = (pos[i - 1] - pos[s]) * config.EGP_SAMPLE_MM
            if span_mm >= min_mm:
                dx = abs(px[i - 1] - px[s])
                dy = abs(py[i - 1] - py[s])
                edge = "Long" if dx >= dy else "Short"
                seg = val[s:i]
                lo, hi = float(seg.min()), float(seg.max())
                # Worst = furthest excursion (min for chips, max for dropouts).
                worst = lo if abs(lo) >= abs(hi) else hi
                # Focus: is the plate edge outside +/-tol of the 60 mm target?
                suspect = False
                max_off = 0.0
                if ph is not None:
                    seg_ph = ph[s:i]
                    seg_ph = seg_ph[~np.isnan(seg_ph)]
                    if seg_ph.size:
                        max_off = float(np.abs(seg_ph).max())
                        suspect = max_off > config.PROFILE_HT_TOL_MM
                runs.append(
                    {
                        "edge": edge,
                        "grinder_loc": "GRINDER1" if edge == "Long" else "GRINDER2",
                        "length_mm": round(span_mm, 1),
                        "pos_x": round(float((px[s] + px[i - 1]) / 2), 1),
                        "pos_y": round(float((py[s] + py[i - 1]) / 2), 1),
                        "worst_value": round(worst, 3),
                        "suspect_focus": suspect,
                        "max_offset_mm": round(max_off, 1),
                    }
                )
            s = i
    return runs


def detect_defects(
    start: datetime,
    end: datetime,
    mock: bool | None = None,
) -> pd.DataFrame:
    """Detect chip/dropout/shiner defect runs over an EGP time window.

    Uses the authoritative run-length definitions in ``config.DEFECT_RULES`` on a
    *filtered* raw scan (only breaching rows), then groups consecutive samples
    into qualifying runs and classifies each to a long/short edge. Returns one
    row per detected defect (no grinder/groove yet — see :func:`build_alerts`).
    """
    if mock if mock is not None else config.use_mock_metrology():
        return _mock_defects()

    out: list[dict] = []

    radius_rule = config.DEFECT_RULES.get("Radius")
    want_radius = bool(radius_rule and radius_rule.get("mode") == "median_tol")

    def _fetch_band():
        return _read_sql(_defect_scan_sql(), {"start": start, "end": end})

    def _fetch_radius():
        return _read_sql(_radius_scan_sql(), {"start": start, "end": end})

    # The chip/dropout band scan and the shiner radius scan are independent
    # reads of EGPData; run them concurrently instead of back-to-back.
    tasks = {"band": _fetch_band}
    if want_radius:
        tasks["radius"] = _fetch_radius
    fetched = _run_parallel(tasks)
    raw = fetched.get("band")
    if not isinstance(raw, pd.DataFrame):
        raw = pd.DataFrame()

    # --- Fixed-band defects (chips + dropouts) from the filtered scan ---
    if not raw.empty:
        raw["SubID"] = raw["SubID"].astype(str)
        raw["ReadTime"] = pd.to_datetime(raw["ReadTime"], errors="coerce")
        for metric, rule in config.DEFECT_RULES.items():
            if rule.get("mode", "band") != "band":
                continue
            for side in _SIDES:
                col = f"{metric}_{side}"
                if col not in raw.columns:
                    continue
                vals = pd.to_numeric(raw[col], errors="coerce")
                mask = _bad_mask(vals, rule)
                if not mask.any():
                    continue
                hit = raw.loc[
                    mask,
                    [
                        "SubID",
                        "SerialNumber",
                        "EquipmentID",
                        "ReadTime",
                        "Position",
                        "PanelPosX",
                        "PanelPosY",
                    ],
                ].copy()
                hit["val"] = vals[mask].to_numpy()
                phcol = f"{config.PROFILE_HT_METRIC}_{side}"
                if phcol in raw.columns:
                    hit["ph"] = pd.to_numeric(
                        raw.loc[mask, phcol], errors="coerce"
                    ).to_numpy()
                # Group by (panel, profiler): the SEL and LEL passes share the
                # same Position values, so merging them would fabricate runs.
                for (sub_id, _serial), sub in hit.groupby(["SubID", "SerialNumber"]):
                    rt = sub["ReadTime"].max()
                    equip = sub["EquipmentID"].iloc[0]
                    for run in _runs_from_positions(sub, rule["min_mm"]):
                        out.append(
                            {
                                "sub_id": sub_id,
                                "read_time": rt,
                                "equip_id": equip,
                                "defect": rule["defect"],
                                "metric": metric,
                                "side": side,
                                **run,
                            }
                        )

    # --- Shiner (Radius): per-panel median +/- tolerance, computed in SQL ---
    rule = radius_rule
    if want_radius:
        rad = fetched.get("radius")
        if not isinstance(rad, pd.DataFrame):
            rad = pd.DataFrame()
        if not rad.empty:
            rad["SubID"] = rad["SubID"].astype(str)
            rad["ReadTime"] = pd.to_datetime(rad["ReadTime"], errors="coerce")
            tol = rule["tol_pct"] / 100.0
            for side, medcol in (("Left", "medL"), ("Right", "medR")):
                vals = pd.to_numeric(rad[f"Radius_{side}"], errors="coerce")
                med = pd.to_numeric(rad[medcol], errors="coerce")
                mask = (vals < med * (1.0 - tol)) | (vals > med * (1.0 + tol))
                if not mask.any():
                    continue
                hit = rad.loc[
                    mask,
                    [
                        "SubID",
                        "SerialNumber",
                        "EquipmentID",
                        "ReadTime",
                        "Position",
                        "PanelPosX",
                        "PanelPosY",
                    ],
                ].copy()
                hit["val"] = vals[mask].to_numpy()
                phcol = f"{config.PROFILE_HT_METRIC}_{side}"
                if phcol in rad.columns:
                    hit["ph"] = pd.to_numeric(
                        rad.loc[mask, phcol], errors="coerce"
                    ).to_numpy()
                # Group by (panel, profiler) so the two profiler passes — which
                # interleave the same Position values — don't merge into one run.
                for (sub_id, _serial), sub in hit.groupby(["SubID", "SerialNumber"]):
                    rt = sub["ReadTime"].max()
                    equip = sub["EquipmentID"].iloc[0]
                    for run in _runs_from_positions(sub, rule["min_mm"]):
                        out.append(
                            {
                                "sub_id": sub_id,
                                "read_time": rt,
                                "equip_id": equip,
                                "defect": rule["defect"],
                                "metric": "Radius",
                                "side": side,
                                **run,
                            }
                        )

    return pd.DataFrame(out)


def load_grinder_grooves(
    start: datetime,
    end: datetime,
    mock: bool | None = None,
) -> pd.DataFrame:
    """Resolve the active grinding groove per plate per grinder location.

    Mirrors the user's grinder-groove JMP: scan ``mfg.ProcessHistoryGrinder``
    (joined to the marker table on ``phg.SubId = phm.VirtualSubId``) over the
    grind window, order by (EquipmentName, Location, Name, ReadTime) and infer
    the active groove for each record as the ``GrooveN`` whose *cumulative*
    meters-worked increases at the next reading of the same ``Name``.

    Returns columns ``SubID``, ``GrinderLoc`` (GRINDER1/2), ``EquipmentName`` and
    ``Groove`` ("1".."4"). One row per (SubID, GrinderLoc).
    """
    empty = pd.DataFrame(columns=["SubID", "GrinderLoc", "EquipmentName", "Groove"])
    if mock if mock is not None else config.use_mock_metrology():
        return empty
    if config.metrology_data_source() == "foundry":
        long = _foundry_grinder_long(start, end)
        if long.empty:
            return empty
        # The Fine (finishing) wheel cuts the final edge surface, so its active
        # groove is the one attributable to an edge defect. Collapse the wide
        # decode to one row per (SubID, GrinderLoc) using the Fine profile and
        # the most recent grind reading, carrying Profile/Meters for display.
        fine = long[long["Profile"] == "Fine"]
        if fine.empty:
            fine = long
        fine = fine.sort_values("GrindTime").drop_duplicates(
            ["SubID", "GrinderLoc"], keep="last"
        )
        return fine[
            ["SubID", "GrinderLoc", "EquipmentName", "Groove", "Profile", "Meters"]
        ]
    try:
        gcols = ", ".join(f"phg.[Groove{i}MetersWorked] AS G{i}" for i in range(1, 6))
        sql = (
            "SELECT phm.SubId AS SubID, phg.EquipmentName, phg.Location AS GrinderLoc, "
            "phg.Name AS GrName, phg.ReadTime AS GrindTime, " + gcols + " "
            "FROM mfg.ProcessHistoryGrinder AS phg "
            "INNER JOIN mfg.ProcessHistoryMarker AS phm "
            "  ON phg.SubId = phm.VirtualSubId "
            "WHERE phg.ReadTime >= :gstart AND phg.ReadTime <= :gend "
            "ORDER BY phg.EquipmentName, phg.Location, phg.Name, phg.ReadTime"
        )
        g = _read_sql(sql, {"gstart": start - timedelta(hours=6), "gend": end})
        if g.empty:
            return empty
        g["SubID"] = g["SubID"].astype(str)
        g = g.sort_values(["EquipmentName", "GrinderLoc", "GrName", "GrindTime"])
        # Active groove = the GrooveN whose cumulative meters grow at the *next*
        # reading of the same (EquipmentName, Location, Name) run.
        grp = g.groupby(["EquipmentName", "GrinderLoc", "GrName"], sort=False)
        groove = pd.Series(pd.NA, index=g.index, dtype="object")
        for i in range(1, 5):
            cur = pd.to_numeric(g[f"G{i}"], errors="coerce")
            nxt = pd.to_numeric(grp[f"G{i}"].shift(-1), errors="coerce")
            grew = nxt > cur
            groove = groove.mask(groove.isna() & grew, str(i))
        g["Groove"] = groove
        g = g.dropna(subset=["Groove"])
        # One groove per (SubID, GrinderLoc): keep the latest grind record.
        g = g.sort_values("GrindTime").drop_duplicates(
            ["SubID", "GrinderLoc"], keep="last"
        )
        return g[["SubID", "GrinderLoc", "EquipmentName", "Groove"]]
    except Exception:
        return empty


def build_alerts(
    start: datetime,
    end: datetime,
    mock: bool | None = None,
) -> pd.DataFrame:
    """Produce fully-attributed defect alerts for an EGP window.

    Combines :func:`detect_defects` (chip/dropout/shiner runs, each already
    classified to a long/short edge) with :func:`load_grinder_grooves` so every
    alert names the responsible ``Grinder`` line and ``Groove``. Returns a
    DataFrame ordered newest-first with a ready-to-show ``message`` column.
    """
    use_mock = mock if mock is not None else config.use_mock_metrology()
    # The groove/grinder-lineage scan is independent of the defect scan, so run
    # them concurrently: while EGPData is scanned for defect runs, the grinder
    # table is scanned for grooves in parallel.
    if use_mock:
        defects = detect_defects(start, end, mock=True)
        grooves = load_grinder_grooves(start, end, mock=True)
    else:
        res = _run_parallel(
            {
                "defects": lambda: detect_defects(start, end, mock=False),
                "grooves": lambda: load_grinder_grooves(start, end, mock=False),
            }
        )
        defects = (
            res["defects"]
            if isinstance(res["defects"], pd.DataFrame)
            else pd.DataFrame()
        )
        grooves = (
            res["grooves"]
            if isinstance(res["grooves"], pd.DataFrame)
            else pd.DataFrame()
        )
    if defects.empty:
        return pd.DataFrame()

    if "groove" not in defects.columns:
        if not grooves.empty:
            gmap = grooves.set_index(["SubID", "GrinderLoc"])

            def _lookup(row, field):
                try:
                    return gmap.loc[(row["sub_id"], row["grinder_loc"]), field]
                except KeyError:
                    return None

            defects["EquipmentName"] = defects.apply(
                lambda r: _lookup(r, "EquipmentName"), axis=1
            )
            defects["groove"] = defects.apply(lambda r: _lookup(r, "Groove"), axis=1)
        else:
            defects["EquipmentName"] = None
            defects["groove"] = None

    defects["grinder"] = defects["EquipmentName"].map(grinder_label)
    defects["message"] = [
        DefectAlert(
            sub_id=r.sub_id,
            read_time=r.read_time,
            defect=r.defect,
            metric=r.metric,
            side=r.side,
            edge=r.edge,
            grinder=r.grinder,
            grinder_loc=r.grinder_loc,
            groove=r.groove,
            length_mm=r.length_mm,
            suspect_focus=bool(getattr(r, "suspect_focus", False)),
            max_offset_mm=getattr(r, "max_offset_mm", None),
        ).message
        for r in defects.itertuples()
    ]
    return defects.sort_values("read_time", ascending=False).reset_index(drop=True)


def broken_with_attribution(
    start: datetime,
    end: datetime,
    mock: bool | None = None,
) -> pd.DataFrame:
    """Backtrack every FS100-broken plate to its grinder / edge / groove.

    Combines :func:`load_broken_plates` (the actual scrap-at-VTD feed) with the
    grinder lineage (``load_grinder_info``) and, when available, the same
    run-length defect attribution used for live alerts (:func:`build_alerts`).
    The result is the "why did it break" view: for each broken SubID it names
    the grinder line, the edge (Long/Short via GRINDER1/2), the groove and any
    edge-grind defect that was already detected on that plate.

    Columns: ``SubID``, ``BrokenTime``, ``VtdName``, ``grinder``,
    ``grinder_loc``, ``edge``, ``groove``, ``had_defect``, ``defect``,
    ``side``, ``length_mm``.
    """
    use_mock = mock if mock is not None else config.use_mock_metrology()
    cols = [
        "SubID",
        "BrokenTime",
        "VtdName",
        "grinder",
        "grinder_loc",
        "edge",
        "groove",
        "had_defect",
        "defect",
        "side",
        "length_mm",
    ]
    broken = load_broken_plates(start, end, mock=use_mock)
    if broken.empty:
        return pd.DataFrame(columns=cols)

    sub_ids = [str(s) for s in broken["SubID"].dropna().unique()]
    # Grinder lineage for exactly the broken plates (targeted IN-list — breaks
    # can occur long after the grind, so a windowed scan would miss them).
    gdf = load_grinder_info(sub_ids, mock=use_mock)
    gmap_name = dict(zip(gdf["SubID"], gdf["EquipmentName"])) if not gdf.empty else {}
    gmap_loc = dict(zip(gdf["SubID"], gdf["GrinderLoc"])) if not gdf.empty else {}

    # Any edge-grind defects already attributed in this window (best-effort:
    # gives groove/side/edge/length when the plate also tripped the detector).
    alerts = build_alerts(start, end, mock=use_mock)
    adef: dict[str, dict] = {}
    if not alerts.empty:
        for r in alerts.itertuples():
            adef.setdefault(
                str(r.sub_id),
                {
                    "defect": r.defect,
                    "side": r.side,
                    "edge": r.edge,
                    "groove": getattr(r, "groove", None),
                    "length_mm": getattr(r, "length_mm", None),
                    "grinder": getattr(r, "grinder", None),
                },
            )

    out: list[dict] = []
    for r in broken.itertuples():
        sid = str(r.SubID)
        loc = gmap_loc.get(sid)
        edge = "Long" if loc == "GRINDER1" else ("Short" if loc == "GRINDER2" else None)
        d = adef.get(sid)
        out.append(
            {
                "SubID": sid,
                "BrokenTime": r.BrokenTime,
                "VtdName": r.VtdName,
                "grinder": grinder_label(gmap_name.get(sid))
                or (d or {}).get("grinder"),
                "grinder_loc": loc,
                "edge": (d or {}).get("edge") or edge,
                "groove": (d or {}).get("groove"),
                "had_defect": d is not None,
                "defect": (d or {}).get("defect"),
                "side": (d or {}).get("side"),
                "length_mm": (d or {}).get("length_mm"),
            }
        )
    return (
        pd.DataFrame(out, columns=cols)
        .sort_values("BrokenTime", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def _mock_defects() -> pd.DataFrame:
    """A few synthetic defect runs so the alerts UI works offline."""
    now = datetime.now()
    rows = [
        (
            "260710800001",
            now - timedelta(minutes=4),
            "Chip",
            "GlassThickness",
            "Right",
            "Long",
            "GRINDER1",
            "PGT31A-GRINDER",
            "2",
            8.7,
            1180.0,
            3.0,
            2.11,
            False,
            1.4,
        ),
        (
            "260710800002",
            now - timedelta(minutes=11),
            "Shiner",
            "Radius",
            "Left",
            "Short",
            "GRINDER2",
            "PGT31C-GRINDER",
            "1",
            62.4,
            2300.0,
            640.0,
            1.29,
            True,
            8.3,
        ),
        (
            "260710800003",
            now - timedelta(minutes=19),
            "Dropout",
            "Dropouts",
            "Right",
            "Long",
            "GRINDER1",
            "PGT31A-GRINDER",
            "4",
            6.2,
            410.0,
            1215.0,
            14.0,
            False,
            2.1,
        ),
        (
            "260710800004",
            now - timedelta(minutes=27),
            "Chip",
            "GlassThickness",
            "Left",
            "Short",
            "GRINDER2",
            "PGT31D-GRINDER",
            "3",
            7.5,
            0.0,
            300.0,
            2.08,
            False,
            0.9,
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "sub_id",
            "read_time",
            "defect",
            "metric",
            "side",
            "edge",
            "grinder_loc",
            "EquipmentName",
            "groove",
            "length_mm",
            "pos_x",
            "pos_y",
            "worst_value",
            "suspect_focus",
            "max_offset_mm",
        ],
    ).assign(
        equip_id=lambda d: [
            f"EGP {'LE' if str(e).startswith('L') else 'SE'}-{str(n)[5:6] or '?'}"
            for e, n in zip(d["edge"], d["EquipmentName"])
        ]
    )


def _mock_broken_plates(
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """A couple of synthetic FS100-broken plates so the Broken tab works offline."""
    now = datetime.now()
    site = config.METROLOGY_SITE
    rows = [
        (
            "260710800002",
            "VTD C",
            f"{site}1C-VTD_COATER",
            now - timedelta(minutes=45),
            f"{site}1C-POST_VTD_COATER_AICV_BGD",
        ),
        (
            "260710800009",
            "VTD A",
            f"{site}1A-VTD_COATER",
            now - timedelta(hours=2),
            f"{site}1A-POST_VTD_COATER_AICV_BGD",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["SubID", "VtdName", "VtdLine", "BrokenTime", "ScrapLocation"],
    )


def _mock_egp_data(
    sub_ids: list[str] | None = None, n_plates: int = 12
) -> pd.DataFrame:
    """Synthesise realistic edge-grind profiles for offline demos.

    Most plates sit comfortably in spec; a deterministic minority are pushed
    out of spec on a specific parameter so Stage 1 has something to catch.
    """
    import random

    rng = random.Random(42)
    ids = sub_ids or [f"2606{rng.randint(10000000, 99999999)}" for _ in range(n_plates)]
    grinders = [f"{config.METROLOGY_SITE}1{g}-GRINDER" for g in "ABCDE"]
    vtds = [f"{config.METROLOGY_SITE}1{v}-VTD_COATER" for v in "ABCD"]
    positions = list(range(0, 300, 10))  # 30 positions along the edge
    base_time = datetime.now() - timedelta(hours=6)

    # Nominal centre for each base parameter (roughly mid-spec).
    centres = {
        "Dropouts": 3.0,
        "EdgeGrind_Delta": 0.0,
        "EdgeGrind_PeakHeight": 1.2,
        "Radius": 1.25,
        "GlassThickness": 2.0,
    }
    spreads = {
        "Dropouts": 2.0,
        "EdgeGrind_Delta": 0.25,
        "EdgeGrind_PeakHeight": 0.1,
        "Radius": 0.08,
        "GlassThickness": 0.15,
    }

    rows: list[dict] = []
    for pi, sid in enumerate(ids):
        grinder = grinders[pi % len(grinders)]
        grinder_loc = "GRINDER1" if pi % 2 == 0 else "GRINDER2"
        vtd = vtds[pi % len(vtds)]
        rt = base_time + timedelta(minutes=pi * 3)
        # Grinding happens shortly before the edge-profile scan.
        gt = rt - timedelta(minutes=7)
        # VTD coating happens downstream, after the edge-profile scan.
        vt = rt + timedelta(minutes=25)
        # Every 4th plate gets an injected out-of-spec excursion.
        bad_param = None
        bad_side = None
        if pi % 4 == 3:
            bad_param = rng.choice(list(centres))
            bad_side = rng.choice(_SIDES)
        for pos in positions:
            row: dict = {
                "DataID": pi * 1000 + pos,
                "EquipmentID": pi % len(grinders),
                "ReadTime": rt,
                "SubID": sid,
                "Position": float(pos),
                "SerialNumber": config.METROLOGY_PROFILER,
                "PanelPosX": float(pos),
                "PanelPosY": 0.0,
                "EquipmentName": grinder,
                "GrindTime": gt,
                "VtdName": vtd,
                "VtdTime": vt,
                "PanelWidth": 300.0,
                "GrinderLoc": grinder_loc,
                "TotalProfileHt": rng.uniform(2.5, 3.0),
                "CollectedTimeUtc": rt,
            }
            for base, centre in centres.items():
                for side in _SIDES:
                    lsl, usl = config.METROLOGY_SPECS[base]
                    val = rng.gauss(centre, spreads[base])
                    # Keep nominal ("good") plates comfortably inside spec so
                    # only the injected excursions below actually fail.
                    if lsl is not None:
                        val = max(val, lsl + spreads[base])
                    if usl is not None:
                        val = min(val, usl - spreads[base])
                    if (
                        base == bad_param
                        and side == bad_side
                        and pos == positions[len(positions) // 2]
                    ):
                        # Push one position clearly out of spec.
                        if usl is not None:
                            val = usl + spreads[base] * 3
                        elif lsl is not None:
                            val = lsl - spreads[base] * 3
                    row[f"{base}_{side}"] = round(val, 3)
                    if base == "GlassThickness":
                        row[f"{base}_pix_{side}"] = round(val * 100, 1)
                    if base == "Radius":
                        row[f"Radius_StdDev_{side}"] = round(abs(rng.gauss(0, 0.02)), 4)
                    if base in ("EdgeGrind_PeakHeight",):
                        row[f"MaximumProfileHt_{side}"] = round(val + 0.5, 3)
            rows.append(row)
    return pd.DataFrame(rows)


# --- Stage 1: spec-limit rules ---------------------------------------------
def evaluate_plate(sub_id: str, plate_df: pd.DataFrame) -> MetrologyResult:
    """Evaluate a single plate's profile against spec limits."""
    equipment = None
    if "EquipmentName" in plate_df.columns and len(plate_df):
        raw = plate_df["EquipmentName"].iloc[0]
        if pd.notna(raw):
            # Prefer the friendly "Grinder A".."Grinder E" label.
            equipment = grinder_label(str(raw))
    # NOTE: EGPData's EquipmentID identifies the *edge profiler* that captured
    # the data (e.g. "LE - B"/"SE B"), not the grinder that ground the panel.
    # We deliberately do NOT fall back to it: the grinder is the tool we want to
    # attribute defects to, so leave equipment unresolved (None) when the
    # ProcessHistoryGrinder join finds no match.
    profiler = None
    if "SerialNumber" in plate_df.columns and len(plate_df):
        profiler = str(plate_df["SerialNumber"].iloc[0])
    read_time = None
    if "ReadTime" in plate_df.columns and len(plate_df):
        read_time = pd.to_datetime(plate_df["ReadTime"].iloc[0], errors="coerce")
        read_time = None if pd.isna(read_time) else read_time.to_pydatetime()
    grind_time = None
    if "GrindTime" in plate_df.columns and len(plate_df):
        gt = pd.to_datetime(plate_df["GrindTime"].iloc[0], errors="coerce")
        grind_time = None if pd.isna(gt) else gt.to_pydatetime()
    vtd = None
    if "VtdName" in plate_df.columns and len(plate_df):
        raw_vtd = plate_df["VtdName"].iloc[0]
        if pd.notna(raw_vtd):
            vtd = vtd_label(str(raw_vtd))
    vtd_time = None
    if "VtdTime" in plate_df.columns and len(plate_df):
        vt = pd.to_datetime(plate_df["VtdTime"].iloc[0], errors="coerce")
        vtd_time = None if pd.isna(vt) else vt.to_pydatetime()

    violations: list[SpecViolation] = []
    for col, (lsl, usl) in spec_columns().items():
        if col not in plate_df.columns:
            continue
        series = pd.to_numeric(plate_df[col], errors="coerce").dropna()
        if series.empty:
            continue
        if usl is not None and series.max() > usl:
            idx = series.idxmax()
            pos = _pos_at(plate_df, idx)
            violations.append(SpecViolation(col, float(series.max()), usl, "USL", pos))
        if lsl is not None and series.min() < lsl:
            idx = series.idxmin()
            pos = _pos_at(plate_df, idx)
            violations.append(SpecViolation(col, float(series.min()), lsl, "LSL", pos))

    return MetrologyResult(
        sub_id=sub_id,
        verdict="REJECT" if violations else "PASS",
        equipment=equipment,
        grind_time=grind_time,
        vtd=vtd,
        vtd_time=vtd_time,
        profiler=profiler,
        read_time=read_time,
        n_positions=int(plate_df["Position"].nunique())
        if "Position" in plate_df
        else len(plate_df),
        violations=violations,
    )


def _pos_at(df: pd.DataFrame, idx) -> float | None:
    if "Position" in df.columns and idx in df.index:
        try:
            return float(df.loc[idx, "Position"])
        except (TypeError, ValueError):
            return None
    return None


def evaluate(df: pd.DataFrame) -> list[MetrologyResult]:
    """Evaluate every plate (SubID) in a profile DataFrame (Stage 1 only)."""
    if df.empty or "SubID" not in df.columns:
        return []
    # Keep only the selected profiler side if that column is present.
    if "SerialNumber" in df.columns and config.METROLOGY_PROFILER:
        mask = df["SerialNumber"] == config.METROLOGY_PROFILER
        if mask.any():
            df = df[mask]
    return [evaluate_plate(sid, g) for sid, g in df.groupby("SubID", sort=False)]


def evaluate_agg(agg: pd.DataFrame) -> list[MetrologyResult]:
    """Stage-1 evaluation from the per-plate aggregate frame.

    Each row already holds ``{col}_min`` / ``{col}_max`` per spec column, so a
    plate is out of spec when its max exceeds the USL or its min falls below the
    LSL. Position is unavailable at this granularity (the raw rows were
    collapsed server-side), so violations report the extreme value only.
    """
    if agg.empty or "SubID" not in agg.columns:
        return []
    specs = spec_columns()
    results: list[MetrologyResult] = []
    for _, row in agg.iterrows():
        sub_id = str(row["SubID"])
        equipment = None
        if pd.notna(row.get("EquipmentName")):
            equipment = grinder_label(str(row["EquipmentName"]))
        # Do NOT fall back to EquipmentID: that is the edge profiler ("LE - B"/
        # "SE B"), i.e. the data source, not the grinder tool we attribute to.
        grinder_loc = (
            str(row["GrinderLoc"]) if pd.notna(row.get("GrinderLoc")) else None
        )
        vtd = vtd_label(str(row["VtdName"])) if pd.notna(row.get("VtdName")) else None
        grind_time = _to_dt(row.get("GrindTime"))
        vtd_time = _to_dt(row.get("VtdTime"))
        read_time = _to_dt(row.get("ReadTime"))

        violations: list[SpecViolation] = []
        for col, (lsl, usl) in specs.items():
            cmax, cmin = row.get(f"{col}_max"), row.get(f"{col}_min")
            if usl is not None and pd.notna(cmax) and cmax > usl:
                violations.append(SpecViolation(col, float(cmax), usl, "USL", None))
            if lsl is not None and pd.notna(cmin) and cmin < lsl:
                violations.append(SpecViolation(col, float(cmin), lsl, "LSL", None))

        results.append(
            MetrologyResult(
                sub_id=sub_id,
                verdict="REJECT" if violations else "PASS",
                equipment=equipment,
                grinder_loc=grinder_loc,
                grind_time=grind_time,
                vtd=vtd,
                vtd_time=vtd_time,
                profiler=str(row["SerialNumber"])
                if pd.notna(row.get("SerialNumber"))
                else None,
                read_time=read_time,
                n_positions=int(row["n_rows"]) if pd.notna(row.get("n_rows")) else 0,
                violations=violations,
            )
        )
    return results


def _to_dt(val) -> datetime | None:
    ts = pd.to_datetime(val, errors="coerce")
    return None if pd.isna(ts) else ts.to_pydatetime()


# --- Stage 2: ML anomaly detection -----------------------------------------
def plate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-position profile into one feature row per plate."""
    cols = [c for c in spec_columns() if c in df.columns]
    if not cols or "SubID" not in df.columns:
        return pd.DataFrame()
    agg = df.groupby("SubID")[cols].agg(["mean", "min", "max", "std"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    return agg.fillna(0.0)


def plate_features_agg(agg: pd.DataFrame) -> pd.DataFrame:
    """Select the ML feature columns straight from the per-plate aggregate.

    ``load_egp_agg`` already names its columns ``{col}_mean/_min/_max/_std``,
    which is exactly what :func:`plate_features` produces, so the trained model
    consumes them without any recomputation.
    """
    if agg.empty or "SubID" not in agg.columns:
        return pd.DataFrame()
    feat_cols = []
    for c in spec_columns():
        for stat in ("mean", "min", "max", "std"):
            name = f"{c}_{stat}"
            if name in agg.columns:
                feat_cols.append(name)
    if not feat_cols:
        return pd.DataFrame()
    out = agg.set_index("SubID")[feat_cols]
    return out.fillna(0.0)


def apply_ml_agg(
    results: list[MetrologyResult], agg: pd.DataFrame
) -> list[MetrologyResult]:
    """Score plates from the aggregate frame (production/live path)."""
    feats = plate_features_agg(agg)
    if feats.empty:
        return results
    try:
        scores = _score(feats)
    except Exception:
        return results
    by_id = {r.sub_id: r for r in results}
    for sid, score in scores.items():
        r = by_id.get(str(sid))
        if r is None:
            continue
        r.anomaly_score = float(score)
        if not r.violations and score >= _ANOMALY_THRESHOLD:
            r.ml_flag = True
            r.verdict = "REJECT"
    return results


def apply_ml(results: list[MetrologyResult], df: pd.DataFrame) -> list[MetrologyResult]:
    """Score plates with an anomaly model and flag subtle within-spec failures.

    Uses a trained supervised model at ``config.METROLOGY_MODEL_PATH`` if it
    exists; otherwise fits an unsupervised IsolationForest on the current batch.
    Silently no-ops if scikit-learn is unavailable.
    """
    feats = plate_features(df)
    if feats.empty:
        return results
    try:
        scores = _score(feats)
    except Exception:
        return results
    by_id = {r.sub_id: r for r in results}
    for sid, score in scores.items():
        r = by_id.get(str(sid))
        if r is None:
            continue
        r.anomaly_score = float(score)
        # Flag anomalies that Stage 1 rules did not already reject.
        if not r.violations and score >= _ANOMALY_THRESHOLD:
            r.ml_flag = True
            r.verdict = "REJECT"
    return results


# Break-risk decision threshold for the Stage-2 model. Lowered from 0.65 to 0.50
# after retraining on 541 labelled panels (2026-07): at 0.50 the model catches
# ~74% of downstream breaks (vs ~59% at 0.65) for a small precision cost
# (~0.88 vs ~0.96). Missing a break is costlier than a review, so favour recall.
_ANOMALY_THRESHOLD = 0.50


def _score(feats: pd.DataFrame) -> pd.Series:
    """Return a 0..1 anomaly score per plate (higher = more anomalous)."""
    import joblib  # noqa: F401  (only needed for the trained-model path)
    from sklearn.ensemble import IsolationForest

    if config.METROLOGY_MODEL_PATH.exists():
        artifact = joblib.load(config.METROLOGY_MODEL_PATH)
        # The trainer saves a dict {"model", "features"} so inference can align
        # columns exactly; older/plain estimators are still accepted.
        if isinstance(artifact, dict):
            model = artifact.get("model")
            feat_names = artifact.get("features")
        else:
            model, feat_names = artifact, None
        X = (
            feats
            if feat_names is None
            else feats.reindex(columns=feat_names, fill_value=0.0)
        )
        # Supervised model: probability of the "bad" (will-break) class.
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X.values)[:, 1]
            return pd.Series(proba, index=feats.index)
        raw = -model.score_samples(X.values)
    else:
        iso = IsolationForest(random_state=0, contamination="auto")
        iso.fit(feats.values)
        raw = -iso.score_samples(feats.values)
    # Normalise to 0..1.
    lo, hi = raw.min(), raw.max()
    norm = (raw - lo) / (hi - lo) if hi > lo else raw * 0.0
    return pd.Series(norm, index=feats.index)


def retrain_break_model(
    start: datetime,
    end: datetime,
    mock: bool | None = None,
    min_history_days: int = 30,
) -> str:
    """Retrain the Stage-2 break-risk model on real FS100-broken labels.

    Positives are plates scrapped as **Broken** at a VTD (``load_broken_plates``);
    the rest of the scanned plates are treated as negatives. Features are the
    per-plate EGP aggregates (``plate_features_agg``) — the *exact* columns the
    live scorer consumes. The fitted classifier is saved as
    ``{"model", "features"}`` to :data:`config.METROLOGY_MODEL_PATH`, so
    :func:`_score` immediately picks it up.

    Called by the dashboard's "Retrain now" button and by a nightly job.
    Returns a human-readable summary of what was trained.
    """
    use_mock = mock if mock is not None else config.use_mock_metrology()
    # Widen the label window: breaks are rare, so always look back at least
    # ``min_history_days`` to gather enough positive examples.
    hist_start = min(start, end - timedelta(days=min_history_days))
    broken = load_broken_plates(hist_start, end, mock=use_mock)
    agg = load_egp_agg(start=hist_start, end=end, mock=use_mock)
    feats = plate_features_agg(agg)
    if feats.empty:
        return "No EGP feature data available for the training window."
    broken_ids = set(str(s) for s in broken["SubID"]) if not broken.empty else set()
    y = pd.Series(
        [1 if str(sid) in broken_ids else 0 for sid in feats.index], index=feats.index
    )
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    if n_pos < 3 or n_neg < 3:
        return (
            f"Not enough labelled data to train: {n_pos} broken vs {n_neg} good "
            f"plates in the last {min_history_days} days (need \u22653 of each). "
            "The model was left unchanged."
        )
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=0,
            n_jobs=-1,
        )
        X = feats.values
        auc_txt = ""
        try:
            cv = min(5, n_pos, n_neg)
            if cv >= 2:
                scores = cross_val_score(model, X, y.values, cv=cv, scoring="roc_auc")
                auc_txt = f" CV ROC-AUC {scores.mean():.2f}."
        except Exception:
            pass
        model.fit(X, y.values)
        config.METROLOGY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": model, "features": list(feats.columns)},
            config.METROLOGY_MODEL_PATH,
        )
        return (
            f"Retrained on {n_pos} broken + {n_neg} good plates "
            f"({len(feats.columns)} features).{auc_txt} "
            f"Saved to {config.METROLOGY_MODEL_PATH.name}."
        )
    except Exception as e:
        return f"Training failed: {e}"


def analyze(
    sub_ids: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    use_ml: bool = True,
    mock: bool | None = None,
) -> list[MetrologyResult]:
    """End-to-end: load profiles, apply spec rules, then ML. One call.

    Uses the fast per-plate SQL aggregate path (``load_egp_agg``) so it scales
    to production volume (~3,300 raw rows per panel). When ``sub_ids`` is given
    (targeted lookups, e.g. for a handful of images) the raw path is used since
    the volume is bounded.
    """
    if sub_ids:
        return _analyze_raw(sub_ids, start, end, use_ml, mock)

    use_mock = mock if mock is not None else config.use_mock_metrology()
    # Fire the three independent ODS queries (per-plate aggregate + grinder and
    # VTD lineage windows) concurrently instead of one-after-another. The
    # lineage scans are windowed and don't depend on the aggregate result, so
    # running them in parallel roughly collapses three round-trips into one.
    if use_mock:
        agg = load_egp_agg(start=start, end=end, mock=True)
        gdf = load_grinder_info([], mock=True, start=start, end=end)
        vdf = load_vtd_info([], mock=True, start=start, end=end)
    else:
        res = _run_parallel(
            {
                "agg": lambda: load_egp_agg(start=start, end=end, mock=False),
                "grinder": lambda: load_grinder_info(
                    [], mock=False, start=start, end=end
                ),
                "vtd": lambda: load_vtd_info([], mock=False, start=start, end=end),
            }
        )
        agg = res["agg"] if isinstance(res["agg"], pd.DataFrame) else pd.DataFrame()
        gdf = res["grinder"] if isinstance(res["grinder"], pd.DataFrame) else None
        vdf = res["vtd"] if isinstance(res["vtd"], pd.DataFrame) else None

    if agg.empty or "SubID" not in agg.columns:
        return []
    # Attach Grinder + VTD lineage per plate when not already carried through.
    if "EquipmentName" not in agg.columns and gdf is not None and not gdf.empty:
        sid = agg["SubID"].astype(str)
        agg["EquipmentName"] = sid.map(dict(zip(gdf["SubID"], gdf["EquipmentName"])))
        agg["GrinderLoc"] = sid.map(dict(zip(gdf["SubID"], gdf["GrinderLoc"])))
        agg["GrindTime"] = sid.map(dict(zip(gdf["SubID"], gdf["GrindTime"])))
    if "VtdName" not in agg.columns and vdf is not None and not vdf.empty:
        sid = agg["SubID"].astype(str)
        agg["VtdName"] = sid.map(dict(zip(vdf["SubID"], vdf["VtdName"])))
        agg["VtdTime"] = sid.map(dict(zip(vdf["SubID"], vdf["VtdTime"])))
    results = evaluate_agg(agg)
    if use_ml:
        results = apply_ml_agg(results, agg)
    return results


def _analyze_raw(
    sub_ids: list[str] | None,
    start: datetime | None,
    end: datetime | None,
    use_ml: bool,
    mock: bool | None,
) -> list[MetrologyResult]:
    """Raw per-position analysis for bounded SubID lookups."""
    df = load_egp_data(sub_ids=sub_ids, start=start, end=end, mock=mock)
    # Resolve real Grinder A-E names + grind time via the marker/grinder tables
    # when the profile rows don't already carry them (EGPData has neither).
    if not df.empty and "SubID" in df.columns and "EquipmentName" not in df.columns:
        present = [str(s) for s in df["SubID"].dropna().unique()]
        gdf = load_grinder_info(present, mock=mock)
        if not gdf.empty:
            sid = df["SubID"].astype(str)
            name_map = dict(zip(gdf["SubID"], gdf["EquipmentName"]))
            time_map = dict(zip(gdf["SubID"], gdf["GrindTime"]))
            df["EquipmentName"] = sid.map(name_map)
            df["GrindTime"] = sid.map(time_map)
    # Resolve VTD (coater) lineage + coat time so the Broken monitor can group
    # by the VTD line where a plate was coated.
    if not df.empty and "SubID" in df.columns and "VtdName" not in df.columns:
        present = [str(s) for s in df["SubID"].dropna().unique()]
        vdf = load_vtd_info(present, mock=mock)
        if not vdf.empty:
            sid = df["SubID"].astype(str)
            df["VtdName"] = sid.map(dict(zip(vdf["SubID"], vdf["VtdName"])))
            df["VtdTime"] = sid.map(dict(zip(vdf["SubID"], vdf["VtdTime"])))
    results = evaluate(df)
    if use_ml:
        results = apply_ml(results, df)
    return results


def analyze_images(
    image_names: list[str],
    use_ml: bool = True,
    mock: bool | None = None,
) -> list[MetrologyResult]:
    """Evaluate metrology for a set of corner image filenames.

    Maps each ``<SubID>_L.bmp`` / ``_T.bmp`` filename to its plate SubID and
    runs the standard analysis, so image and metrology verdicts share a key.
    """
    sub_ids = sorted({sub_id_from_image(n) for n in image_names})
    return analyze(sub_ids=sub_ids, use_ml=use_ml, mock=mock)


# --- CLI / demo ------------------------------------------------------------
def _demo() -> None:
    results = analyze(use_ml=True, mock=True)
    print(f"Evaluated {len(results)} plates (mock=True)\n")
    for r in results:
        line = f"  {r.sub_id}  [{r.verdict:6}]  {r.equipment or '?':16}"
        if r.anomaly_score is not None:
            line += f"  anomaly={r.anomaly_score:.2f}"
        print(line)
        if r.verdict == "REJECT":
            print(f"        -> {r.reason}")
    n_bad = sum(1 for r in results if r.verdict == "REJECT")
    print(f"\n{n_bad}/{len(results)} rejected.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Edge grind profile metrology engine")
    ap.add_argument("--demo", action="store_true", help="run offline mock demo")
    ap.add_argument("--sub-ids", nargs="*", help="specific SubIDs to evaluate")
    ap.add_argument("--no-ml", action="store_true", help="skip Stage 2 ML")
    args = ap.parse_args()

    if args.demo:
        _demo()
        return

    results = analyze(sub_ids=args.sub_ids, use_ml=not args.no_ml)
    for r in results:
        print(f"{r.sub_id}\t{r.verdict}\t{r.reason}")


if __name__ == "__main__":
    main()
