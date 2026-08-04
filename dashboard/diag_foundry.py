"""Ad-hoc Foundry diagnostics: surface the REAL errors that the metrology
loaders normally swallow (they return empty frames on any exception) and probe
the wide grinder schema so the decoder can be built correctly.

Run on the remote laptop:
    $env:FS50_DATA_SOURCE="foundry"
    & "C:\\New folder\\envs\\fs50defect\\python.exe" diag_foundry.py
"""

from __future__ import annotations

import traceback
from datetime import datetime, timedelta

import pandas as pd

import foundry_source

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

end = datetime.now()
start = end - timedelta(hours=6)


def run(label: str, sql: str, params: dict) -> "pd.DataFrame | None":
    print(f"\n=== {label} ===")
    try:
        df = foundry_source.query(sql, params)
        print("OK", df.shape)
        print(list(df.columns))
        return df
    except Exception:  # noqa: BLE001 - diagnostic
        traceback.print_exc()
        return None


# 1) Which grinder key joins to marker.VirtualSubId? Try both candidates.
for key in ("Subid", "GrinderSubid"):
    run(
        f"join grinder.{key} = marker.VirtualSubId",
        "SELECT phm.SubId AS SubID, phg.EquipmentName, phg.ReadTime AS GrindTime "
        "FROM mfg.ProcessHistoryGrinder AS phg "
        "INNER JOIN mfg.ProcessHistoryMarker AS phm "
        f"  ON phg.{key} = phm.VirtualSubId "
        "WHERE phg.ReadTime >= :gstart AND phg.ReadTime <= :gend",
        {"gstart": start, "gend": end},
    )

# 2) Sample the wide groove/meters columns to confirm the row values.
#    Base col (e.g. G1SRWC) should be the active groove 1..4; _MW = meters.
sample = run(
    "grinder wide sample",
    "SELECT ReadTime, EquipmentName, Subid, GrinderSubid, "
    "G1SRWC, G1SRWM, G1SRWF, G1SLWC, G1SLWM, G1SLWF, "
    "G2SRWC, G2SRWM, G2SRWF, G2SLWC, G2SLWM, G2SLWF, "
    "G1SRWC_MW, G1SRWM_MW, G1SRWF_MW, G2SLWF_MW "
    "FROM mfg.ProcessHistoryGrinder "
    "WHERE ReadTime >= :gstart AND ReadTime <= :gend",
    {"gstart": start, "gend": end},
)
if sample is not None and not sample.empty:
    print("\n--- first 8 rows ---")
    print(sample.head(8).to_string())
    print("\n--- distinct values in base groove columns (expect 1..4) ---")
    for c in ("G1SRWC", "G1SLWF", "G2SRWC", "G2SLWF"):
        vals = sorted(pd.Series(sample[c]).dropna().unique().tolist())[:12]
        print(f"{c}:", vals)

# 3) What does EGP.SubID look like, and which grinder id matches it?
#    Goal: find out if we can join grinder DIRECTLY to EGP (skip the marker).
egp = run(
    "EGP SubID sample",
    "SELECT E.SubID FROM ProcessData.ProcessHistory.EGPData AS E "
    "WHERE E.ReadTime >= :gstart AND E.ReadTime <= :gend",
    {"gstart": start, "gend": end},
)
if egp is not None and not egp.empty:
    egp_ids = pd.Series(egp["SubID"]).astype(str)
    print("\n--- sample EGP.SubID values ---")
    print(egp_ids.dropna().unique()[:8].tolist())

    if sample is not None and not sample.empty:
        egp_set = set(egp_ids.dropna().unique())
        sub_set = set(pd.Series(sample["Subid"]).astype(str).dropna().unique())
        gsub_set = set(pd.Series(sample["GrinderSubid"]).astype(str).dropna().unique())
        print("\n--- overlap of EGP.SubID with grinder ids ---")
        print("EGP.SubID matches grinder.Subid       :", len(egp_set & sub_set))
        print("EGP.SubID matches grinder.GrinderSubid:", len(egp_set & gsub_set))

# 4) Does grinder.Subid join to EGP directly (no marker)?
run(
    "direct join grinder.Subid = EGP.SubID",
    "SELECT E.SubID, phg.EquipmentName, phg.ReadTime AS GrindTime "
    "FROM mfg.ProcessHistoryGrinder AS phg "
    "INNER JOIN ProcessData.ProcessHistory.EGPData AS E "
    "  ON phg.Subid = E.SubID "
    "WHERE phg.ReadTime >= :gstart AND phg.ReadTime <= :gend",
    {"gstart": start, "gend": end},
)

# 5) End-to-end: the actual metrology loaders under Foundry.
import metrology as m  # noqa: E402

print("\n=== load_grinder_info (windowed) ===")
try:
    gi = m.load_grinder_info([], start=start, end=end)
    print("rows:", gi.shape)
    print(gi.head(8).to_string())
except Exception:
    traceback.print_exc()

print("\n=== load_grinder_grooves (windowed) ===")
try:
    gg = m.load_grinder_grooves(start, end)
    print("rows:", gg.shape)
    print(list(gg.columns))
    print(gg.head(10).to_string())
except Exception:
    traceback.print_exc()

print("\n=== _foundry_grinder_long (full detail incl Profile) ===")
try:
    gl = m._foundry_grinder_long(start, end)
    print("rows:", gl.shape)
    print(gl.head(14).to_string())
except Exception:
    traceback.print_exc()
