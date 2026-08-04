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
