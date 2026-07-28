"""Diagnose WHY the run-length defect engine and the ML model miss FS100-broken
panels, using the labelled CSVs as ground truth.

Answers three questions the process owner raised:
  1. The run-length defect engine "explains" only a fraction of broken panels
     even though the EGP visibly shows dropouts / profile variation. Why?
  2. Spec-based (Edge Grind Profile) rejects are huge but ML rejects are ~63.
     What is the difference?
  3. Are the limits / parameters right?

For every panel we compute, per side (Left/Right):
  - point signals: max dropouts, # positions with Dropouts>10, min GlassThickness,
    Radius spread vs median, MaximumProfileHt offset.
  - RUN-LENGTH defects with the CURRENT config thresholds (chip / dropout /
    shiner), exactly as metrology.detect_defects would.
  - the SAME run-length defects with RELAXED thresholds (isolated points, no
    6 mm minimum) to see how much the "6 mm consecutive" rule is costing recall.

Then it reports catch-rate on broken vs false-alarm rate on good for each rule,
so we can see which parameter actually separates the two classes.

Run:
    python scripts/diagnose_defects.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DASHBOARD = _REPO_ROOT / "dashboard"
if str(_DASHBOARD) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD))

import config  # noqa: E402
import metrology  # noqa: E402

SIDES = ("Left", "Right")


def _run_length_hits(
    sub_rows: pd.DataFrame, col: str, mask: pd.Series, min_mm: float
) -> int:
    """Number of qualifying consecutive runs (>= min_mm) for a boolean mask,
    grouped per profiler pass (SerialNumber) like the live engine."""
    if not mask.any():
        return 0
    hit = sub_rows.loc[
        mask, ["SerialNumber", "Position", "PanelPosX", "PanelPosY"]
    ].copy()
    hit["val"] = sub_rows.loc[mask, col].to_numpy()
    n = 0
    for _serial, g in hit.groupby("SerialNumber"):
        n += len(metrology._runs_from_positions(g, min_mm))
    return n


def analyse_panel(rows: pd.DataFrame) -> dict:
    """Compute all defect signals for one panel's raw EGP rows."""
    rec: dict = {}
    # --- Dropouts (chip/edge light-loss) ---
    drop_rule = config.DEFECT_RULES["Dropouts"]
    chip_rule = config.DEFECT_RULES["GlassThickness"]
    rad_rule = config.DEFECT_RULES["Radius"]

    any_drop_run = any_drop_run_relaxed = any_drop_point = 0
    any_chip_run = any_chip_point = 0
    any_shiner_run = 0
    max_drop = 0.0
    min_thk = np.inf
    max_ph_off = 0.0

    for side in SIDES:
        dcol = f"Dropouts_{side}"
        if dcol in rows.columns:
            dvals = pd.to_numeric(rows[dcol], errors="coerce")
            max_drop = max(
                max_drop, float(np.nanmax(dvals.values)) if len(dvals) else 0.0
            )
            dmask = dvals > drop_rule["high"]
            any_drop_point += int(dmask.sum())
            any_drop_run += _run_length_hits(
                rows.assign(**{dcol: dvals}), dcol, dmask, drop_rule["min_mm"]
            )
            # relaxed: any single breaching point counts (min_mm ~ 0)
            any_drop_run_relaxed += _run_length_hits(
                rows.assign(**{dcol: dvals}), dcol, dmask, 1.0
            )

        tcol = f"GlassThickness_{side}"
        if tcol in rows.columns:
            tvals = pd.to_numeric(rows[tcol], errors="coerce")
            min_thk = min(
                min_thk, float(np.nanmin(tvals.values)) if len(tvals) else np.inf
            )
            tmask = tvals < chip_rule["low"]
            any_chip_point += int(tmask.sum())
            any_chip_run += _run_length_hits(
                rows.assign(**{tcol: tvals}), tcol, tmask, chip_rule["min_mm"]
            )

        rcol = f"Radius_{side}"
        if rcol in rows.columns:
            rvals = pd.to_numeric(rows[rcol], errors="coerce")
            # per profiler pass median (matches live per-(SubID,SerialNumber))
            for _serial, g in rows.assign(**{rcol: rvals}).groupby("SerialNumber"):
                gv = pd.to_numeric(g[rcol], errors="coerce")
                med = np.nanmedian(gv.values)
                if not np.isfinite(med) or med == 0:
                    continue
                tol = rad_rule["tol_pct"] / 100.0
                rmask = (gv < med * (1 - tol)) | (gv > med * (1 + tol))
                any_shiner_run += _run_length_hits(g, rcol, rmask, rad_rule["min_mm"])

        phcol = f"MaximumProfileHt_{side}"
        if phcol in rows.columns:
            phv = pd.to_numeric(rows[phcol], errors="coerce")
            if len(phv):
                max_ph_off = max(max_ph_off, float(np.nanmax(np.abs(phv.values))))

    rec["max_dropouts"] = max_drop
    rec["min_thickness"] = None if not np.isfinite(min_thk) else min_thk
    rec["max_ph_offset"] = max_ph_off
    rec["drop_points"] = any_drop_point
    rec["chip_run"] = any_chip_run
    rec["dropout_run"] = any_drop_run
    rec["dropout_run_relaxed"] = any_drop_run_relaxed
    rec["shiner_run"] = any_shiner_run
    rec["any_defect_run"] = int((any_chip_run + any_drop_run + any_shiner_run) > 0)
    rec["any_defect_relaxed"] = int(
        (any_chip_point + any_drop_run_relaxed + any_shiner_run) > 0
    )
    return rec


def build_table(csv_path: Path, label: int) -> pd.DataFrame:
    print(f"Loading {csv_path.name} ...")
    raw = pd.read_csv(csv_path)
    raw["SubID"] = raw["SubID"].astype(str)
    recs = []
    for sub_id, rows in raw.groupby("SubID"):
        rec = analyse_panel(rows)
        rec["SubID"] = sub_id
        rec["label"] = label
        recs.append(rec)
    df = pd.DataFrame(recs)
    print(f"  {csv_path.name}: {len(df)} panels analysed")
    return df


def rate(df: pd.DataFrame, col: str) -> float:
    return (
        100.0 * (df[col] > 0).mean()
        if df[col].dtype != bool
        else 100.0 * df[col].mean()
    )


def main() -> None:
    broken = build_table(config.METROLOGY_LABELS_BAD, 1)
    good = build_table(config.METROLOGY_LABELS_GOOD, 0)

    print("\n" + "=" * 68)
    print("RUN-LENGTH DEFECT ENGINE — catch rate on BROKEN vs false-alarm on GOOD")
    print("(current config thresholds: chip>=6mm, dropout>=6mm, shiner>=50mm)")
    print("=" * 68)
    rows = [
        ("Chip run (GlassThk<2.25, >=6mm)", "chip_run"),
        ("Dropout run (Dropouts>10, >=6mm)", "dropout_run"),
        ("Shiner run (Radius median+-5%, >=50mm)", "shiner_run"),
        ("ANY run-length defect", "any_defect_run"),
        ("--- relaxed (isolated points, no 6mm) ---", None),
        ("Dropout ANY point (Dropouts>10)", "drop_points"),
        ("ANY defect, relaxed", "any_defect_relaxed"),
    ]
    print(f"\n{'signal':<42}{'BROKEN caught':>15}{'GOOD false':>13}")
    for label, col in rows:
        if col is None:
            print(label)
            continue
        b = rate(broken, col)
        g = rate(good, col)
        print(f"{label:<42}{b:>13.1f}%{g:>12.1f}%")

    print("\n" + "=" * 68)
    print("POINT-SIGNAL separation (median value per class)")
    print("=" * 68)
    for col in ["max_dropouts", "min_thickness", "max_ph_offset", "drop_points"]:
        b = broken[col].median()
        g = good[col].median()
        print(f"  {col:<18} broken median={b}   good median={g}")

    # How many broken panels have NO signal at all under current rules?
    missed = broken[broken["any_defect_run"] == 0]
    print(f"\nBROKEN panels the run-length engine MISSES: {len(missed)}/{len(broken)}")
    missed_relaxed = broken[broken["any_defect_relaxed"] == 0]
    print(
        f"BROKEN panels missed even RELAXED (isolated points): {len(missed_relaxed)}/{len(broken)}"
    )

    out = _REPO_ROOT / "reports" / "defect_diagnosis.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([broken, good]).to_csv(out, index=False)
    print(f"\nPer-panel diagnosis written -> {out}")


if __name__ == "__main__":
    main()
