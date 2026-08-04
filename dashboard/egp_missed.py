"""EGP Missed-Detection queries.

Pulls bad plates (MES_ResultCode 998 / 99) from the EGP summary table and
cross-references them against downstream barcode reads at FS100 (VTD_COATER)
and FS350 (BUSSING) to surface plates that EGP marked as bad but that
continued to downstream processes anyway.

Tables used (read-only):
  ODS.mfg.ProcessHistoryEGPSummary — one row per (SubID, EGP edge read)
  ProcessData.Events.PartProduced  — barcode reads at every process station

Lane extraction: EquipmentID format is ``"EGP LE-A"`` / ``"EGP SE-B"`` etc.
  lane = last character of EquipmentID (A–E).
  type = "LE" or "SE" (Long-Edge / Short-Edge).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import text

import config

# Bad MES result codes — 998 used by EGP A/B, 99 used by EGP C.
BAD_CODES: list[int] = [998, 99]

# FS100 = VTD Coater lanes A–E.
FS100_LOCATIONS = [f"PGT31{lane}-VTD_COATER" for lane in "ABCDE"]

# FS350 = Bussing lanes A–D (C added per process owner).
FS350_LOCATIONS = [f"PGT31{lane}-BUSSING" for lane in "ABCD"]

# SQL Server error codes that indicate a server-side session kill (Resource Governor).
_KILL_CODES = ("596", "08S01")

_LANE_RE = re.compile(r"-([A-E])$", re.IGNORECASE)


def _run_query(sql, params: dict) -> pd.DataFrame:
    """Execute a read-only SQL query; retry once if the server killed the session.

    SQL Server error 596 ("session in kill state") means the Resource Governor
    terminated the connection.  We dispose the pool to flush dead connections
    and open a fresh one for the retry.  When ``FS50_DATA_SOURCE=foundry`` the
    query is routed to Foundry instead (no retry needed there).
    """
    if config.metrology_data_source() == "foundry":
        import foundry_source

        # sql may be a SQLAlchemy TextClause; foundry_source wants a plain str.
        return foundry_source.query(str(getattr(sql, "text", sql)), params)

    from metrology import _make_engine  # local import — keeps dep optional

    def _attempt():
        eng = _make_engine()
        with eng.connect() as conn:
            return pd.read_sql(sql, conn, params=params)

    try:
        return _attempt()
    except Exception as exc:
        # Check for known session-kill codes; if found, flush pool and retry.
        msg = str(exc)
        if any(code in msg for code in _KILL_CODES):
            try:
                from metrology import _make_engine as _mke

                _mke().dispose()
            except Exception:
                pass
            return _attempt()
        raise


def _extract_lane(equip_id: str) -> str:
    """Return the lane letter (A–E) from an EquipmentID like 'EGP LE-A'."""
    m = _LANE_RE.search(str(equip_id))
    return m.group(1).upper() if m else "?"


def _extract_type(equip_id: str) -> str:
    """Return 'LE' or 'SE' from an EquipmentID like 'EGP LE-A'."""
    s = str(equip_id).upper()
    if " LE" in s:
        return "LE"
    if " SE" in s:
        return "SE"
    return "?"


def load_bad_egp(
    start: datetime | str,
    end: datetime | str,
) -> pd.DataFrame:
    """Return one row per (SubID, EGP read) marked bad (MES_ResultCode 998/99).

    Columns: SubID, EquipmentID, Lane, EdgeType, MES_ResultCode, ReadTime.
    Sorted by ReadTime descending.  Returns empty DataFrame on any error.
    """
    try:
        codes = ",".join(str(c) for c in BAD_CODES)
        sql = text(
            f"""
            SELECT [SubID], [EquipmentID], [MES_ResultCode], [ReadTime]
            FROM ODS.mfg.ProcessHistoryEGPSummary WITH (NOLOCK)
            WHERE [ReadTime] >= :start
              AND [ReadTime] <  :end
              AND [MES_ResultCode] IN ({codes})
            ORDER BY [ReadTime] DESC
            """
        )
        df = _run_query(sql, {"start": start, "end": end})
        if df.empty:
            return df
        df["Lane"] = df["EquipmentID"].apply(_extract_lane)
        df["EdgeType"] = df["EquipmentID"].apply(_extract_type)
        df["ReadTime"] = pd.to_datetime(df["ReadTime"], errors="coerce")
        return df
    except Exception:  # noqa: BLE001
        return pd.DataFrame(
            columns=[
                "SubID",
                "EquipmentID",
                "Lane",
                "EdgeType",
                "MES_ResultCode",
                "ReadTime",
            ]
        )


def load_downstream_reads(
    sub_ids: list[str],
    start: datetime | str,
    end: datetime | str,
) -> pd.DataFrame:
    """Return PartProduced rows for the given SubIDs at FS100 and FS350.

    Columns: SubId, Location, Process, TimestampUtc.
    """
    if not sub_ids:
        return pd.DataFrame(columns=["SubId", "Location", "Process", "TimestampUtc"])
    try:
        # Inline SubIDs and locations directly — SubIDs are all-numeric (safe),
        # locations are known constants.  Avoids generating thousands of :sid0…
        # bind params which causes SQL Server to build a terrible query plan.
        all_locs = FS100_LOCATIONS
        loc_in = ", ".join(f"'{loc}'" for loc in all_locs)
        # Validate: only allow numeric SubIDs to prevent any injection risk.
        clean_ids = [s for s in sub_ids if str(s).isdigit()]
        if not clean_ids:
            return pd.DataFrame(
                columns=["SubId", "Location", "Process", "TimestampUtc"]
            )
        sub_in = ", ".join(clean_ids)

        sql = text(
            f"""
            SELECT [SubId], [Location], [ProcessApplied] AS [Process], [TimestampUtc]
            FROM ProcessData.Events.PartProduced WITH (NOLOCK)
            WHERE [TimestampUtc] >= :start
              AND [TimestampUtc] <  :end
              AND [Location] IN ({loc_in})
              AND [SubId]   IN ({sub_in})
            ORDER BY [TimestampUtc] DESC
            """
        )
        df = _run_query(sql, {"start": start, "end": end})
        df["TimestampUtc"] = pd.to_datetime(df["TimestampUtc"], errors="coerce")
        return df
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["SubId", "Location", "Process", "TimestampUtc"])


def build_missed_summary(
    start: datetime | str,
    end: datetime | str,
) -> dict:
    """Return a summary dict ready for the dashboard section.

    Keys:
      bad_df       : raw bad-plate rows (SubID, Lane, EdgeType, …)
      bad_subids   : unique set of bad SubIDs
      bad_count    : int — distinct bad SubIDs
      fs100_df     : PartProduced rows for bad plates at FS100
      fs100_count  : int — distinct bad SubIDs seen at FS100
      fs350_df     : PartProduced rows for bad plates at FS350
      fs350_count  : int — distinct bad SubIDs seen at FS350
      lane_counts  : Series — bad reads per lane (for the bar)
    """
    bad_df = load_bad_egp(start, end)
    bad_subids: list[str] = (
        bad_df["SubID"].dropna().unique().tolist() if not bad_df.empty else []
    )
    bad_count = len(bad_subids)

    if bad_subids:
        ds = load_downstream_reads(bad_subids, start, end)
        fs100_df = ds[ds["Location"].isin(FS100_LOCATIONS)].copy()
        fs350_df = ds[ds["Location"].isin(FS350_LOCATIONS)].copy()
    else:
        fs100_df = pd.DataFrame(
            columns=["SubId", "Location", "Process", "TimestampUtc"]
        )
        fs350_df = pd.DataFrame(
            columns=["SubId", "Location", "Process", "TimestampUtc"]
        )

    lane_counts = (
        bad_df["Lane"].value_counts().sort_index()
        if not bad_df.empty
        else pd.Series(dtype=int)
    )

    return {
        "bad_df": bad_df,
        "bad_subids": bad_subids,
        "bad_count": bad_count,
        "fs100_df": fs100_df,
        "fs100_count": int(fs100_df["SubId"].nunique()) if not fs100_df.empty else 0,
        "fs350_df": fs350_df,
        "fs350_count": int(fs350_df["SubId"].nunique()) if not fs350_df.empty else 0,
        "lane_counts": lane_counts,
    }
