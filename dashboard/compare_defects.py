"""Standalone defect-count comparison probe.

Run the SAME command on BOTH machines (laptop and remote) and compare the
printed counts. It bypasses Streamlit and all caching, calling the metrology
functions directly against ODS for one fixed time window.

Usage (from the dashboard/ folder, in the fs50defect env):

    python compare_defects.py                 # last 30 minutes (default)
    python compare_defects.py --minutes 120   # last 2 hours
    python compare_defects.py --start "2026-07-28 14:00" --end "2026-07-28 14:30"

Compare the OUTPUT between the two machines. Same numbers -> the code/query
behave identically (difference is cache/window). Different numbers -> the
environment makes the query behave differently, and the printed error/empty
reasons will show why.
"""

from __future__ import annotations

import argparse
import platform
import socket
import sys
import traceback
from datetime import datetime, timedelta

# Force UTF-8 output so redirecting to a file on Windows (cp1252 default) does
# not crash when alert messages contain non-ASCII characters (e.g. the warning
# emoji). Fall back silently if reconfigure is unavailable.
try:  # pragma: no cover - environment dependent
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass

import config
import metrology


def _ascii(text: str) -> str:
    """Drop non-ASCII so printing never fails regardless of console codepage."""
    return str(text).encode("ascii", "ignore").decode("ascii")


def _parse_ts(text: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise SystemExit(f"Could not parse timestamp: {text!r} (use 'YYYY-MM-DD HH:MM')")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare defect counts across machines.")
    ap.add_argument(
        "--minutes", type=int, default=30, help="window size back from now (default 30)"
    )
    ap.add_argument(
        "--start", type=str, default=None, help="explicit start 'YYYY-MM-DD HH:MM'"
    )
    ap.add_argument(
        "--end", type=str, default=None, help="explicit end 'YYYY-MM-DD HH:MM'"
    )
    args = ap.parse_args()

    if args.start or args.end:
        if not (args.start and args.end):
            raise SystemExit("Provide BOTH --start and --end, or neither.")
        start = _parse_ts(args.start)
        end = _parse_ts(args.end)
    else:
        end = datetime.now()
        start = end - timedelta(minutes=args.minutes)

    print("=" * 70)
    print(f"HOST            : {socket.gethostname()}")
    print(f"PYTHON          : {platform.python_version()}  ({sys.executable})")
    print(f"WINDOW (local)  : {start:%Y-%m-%d %H:%M:%S}  ->  {end:%Y-%m-%d %H:%M:%S}")
    print(f"use_mock        : {config.use_mock_metrology()}")
    try:
        print(f"ODBC string     : {config.metrology_odbc_str()}")
    except Exception as exc:  # noqa: BLE001
        print(f"ODBC string     : <error building: {exc}>")
    print("=" * 70)

    # --- analyze() : fast per-plate aggregate path -----------------------
    try:
        results = metrology.analyze(start=start, end=end, use_ml=False)
        n_plates = len(results)
        n_fail = sum(
            1
            for r in results
            if getattr(r, "status", None) not in (None, "pass", "PASS", "ok")
        )
        print(f"analyze()       : {n_plates} plates  (non-pass status: {n_fail})")
    except Exception:  # noqa: BLE001
        print("analyze()       : ERROR")
        traceback.print_exc()

    # --- detect_defects() : chip/dropout/shiner run scan -----------------
    try:
        ddf = metrology.detect_defects(start, end)
        print(f"detect_defects(): {len(ddf)} defect rows")
        if not ddf.empty and "type" in ddf.columns:
            counts = ddf["type"].value_counts().to_dict()
            print(f"                  by type: {counts}")
    except Exception:  # noqa: BLE001
        print("detect_defects(): ERROR")
        traceback.print_exc()

    # --- build_alerts() : fully-attributed alerts ------------------------
    try:
        adf = metrology.build_alerts(start, end)
        print(f"build_alerts()  : {len(adf)} alert rows")
        if not adf.empty:
            cols = [
                c
                for c in ("time", "grinder", "groove", "type", "message")
                if c in adf.columns
            ]
            sample = adf[cols].head(10) if cols else adf.head(10)
            print(_ascii(sample.to_string(index=False)))
    except Exception:  # noqa: BLE001
        print("build_alerts()  : ERROR")
        traceback.print_exc()

    print("=" * 70)
    print("Compare these numbers against the other machine for the SAME window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
