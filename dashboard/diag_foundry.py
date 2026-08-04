"""Ad-hoc Foundry diagnostics: surface the REAL errors that the metrology
loaders normally swallow (they return empty frames on any exception).

Run on the remote laptop:
    $env:FS50_DATA_SOURCE="foundry"
    & "C:\\New folder\\envs\\fs50defect\\python.exe" diag_foundry.py
"""
from __future__ import annotations

import traceback
from datetime import datetime, timedelta

import foundry_source

end = datetime.now()
start = end - timedelta(hours=1)


def run(label: str, sql: str, params: dict) -> None:
    print(f"\n=== {label} ===")
    try:
        df = foundry_source.query(sql, params)
        print("OK", df.shape)
        print(list(df.columns))
    except Exception:  # noqa: BLE001 - diagnostic
        traceback.print_exc()


# 1) Grinder + Marker join (load_grinder_info / load_grinder_grooves core)
run(
    "grinder+marker join",
    "SELECT phm.SubId AS SubID, phg.EquipmentName, phg.Location AS GrinderLoc, "
    "phg.ReadTime AS GrindTime "
    "FROM mfg.ProcessHistoryGrinder AS phg "
    "INNER JOIN mfg.ProcessHistoryMarker AS phm "
    "  ON phg.SubId = phm.VirtualSubId "
    "WHERE phg.ReadTime >= :gstart AND phg.ReadTime <= :gend",
    {"gstart": start - timedelta(hours=6), "gend": end},
)

# 2) Grinder table alone - dump columns
run(
    "grinder columns",
    "SELECT * FROM mfg.ProcessHistoryGrinder LIMIT 5",
    {},
)

# 3) Marker table alone - dump columns
run(
    "marker columns",
    "SELECT * FROM mfg.ProcessHistoryMarker LIMIT 5",
    {},
)
