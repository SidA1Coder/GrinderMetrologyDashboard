"""Train the Stage-2 "will this panel break downstream?" metrology model.

Some panels PASS the Edge Grind Profile (EGP) spec rules yet still break later
(micro-chips, fractures) because they carry sub-threshold defects. This script
learns that sub-threshold signature from labelled history so the dashboard can
flag risky panels and monitor which grinder produces them.

Inputs (CSV exports of raw ``EGPData`` rows -- many rows per panel/SubID):
    dashboard/data/training/broken_panels.csv   -> label 1 (broke downstream)
    dashboard/data/training/good_panels.csv     -> label 0 (stayed good)

Each panel is aggregated into one feature row using the SAME function the live
engine uses at inference (``metrology.plate_features``), so train and serve
features are guaranteed identical. The trained model is saved to
``config.METROLOGY_MODEL_PATH`` as ``{"model", "features", ...}`` and picked up
automatically by ``metrology.apply_ml`` (Stage 2).

Usage:
    python scripts/train_metrology.py
    python scripts/train_metrology.py --bad path/to/bad.csv --good path/to/good.csv
    python scripts/train_metrology.py --test-size 0.25 --n-estimators 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the dashboard package importable when run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DASHBOARD = _REPO_ROOT / "dashboard"
if str(_DASHBOARD) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD))

import config  # noqa: E402  (path set above)
import metrology  # noqa: E402


def _load_features(csv_path: Path, label: int) -> pd.DataFrame:
    """Read a raw-EGPData CSV and aggregate to one labelled row per panel."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")
    raw = pd.read_csv(csv_path)
    if "SubID" not in raw.columns:
        # Accept a few common spellings for the panel id.
        for alt in ("sub_id", "subid", "SubId", "panel_id", "serial"):
            if alt in raw.columns:
                raw = raw.rename(columns={alt: "SubID"})
                break
    feats = metrology.plate_features(raw)
    if feats.empty:
        raise ValueError(
            f"No usable feature columns found in {csv_path.name}. "
            "Expected raw EGPData columns (EdgeGrind_Delta_Left, Radius_Left, ...)."
        )
    feats = feats.copy()
    feats["label"] = label
    print(
        f"  {csv_path.name}: {len(raw):,} rows -> {len(feats):,} panels "
        f"({feats.shape[1] - 1} features)"
    )
    return feats


def _build_model(n_estimators: int, seed: int):
    from sklearn.ensemble import RandomForestClassifier

    # class_weight balanced: broken panels are the rare, costly class.
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def _evaluate(model, X, y, seed: int) -> None:
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_splits = max(2, min(5, n_pos, n_neg))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)

    print(f"\nCross-validated performance ({n_splits}-fold):")
    try:
        print(f"  ROC-AUC : {roc_auc_score(y, proba):.3f}")
    except ValueError:
        print("  ROC-AUC : n/a (single class in a fold)")
    print("  Confusion matrix [rows=true 0/1, cols=pred 0/1]:")
    print("   ", confusion_matrix(y, pred).tolist())
    print(
        classification_report(y, pred, target_names=["good", "broken"], zero_division=0)
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bad", type=Path, default=config.METROLOGY_LABELS_BAD)
    ap.add_argument("--good", type=Path, default=config.METROLOGY_LABELS_GOOD)
    ap.add_argument("--out", type=Path, default=config.METROLOGY_MODEL_PATH)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("Loading labelled panels...")
    frames = [_load_features(args.bad, 1)]
    if args.good.exists():
        frames.append(_load_features(args.good, 0))
    else:
        print(
            f"\n[!] Good-panel file not found: {args.good}\n"
            "    Supervised training needs BOTH classes. Add the good-panel CSV\n"
            "    (same raw-EGPData columns) and re-run. Nothing was saved."
        )
        sys.exit(1)

    data = pd.concat(frames, axis=0)
    # Align columns across both files (union), fill gaps with 0.
    feature_cols = [c for c in data.columns if c != "label"]
    data[feature_cols] = data[feature_cols].fillna(0.0)
    X = data[feature_cols].to_numpy()
    y = data["label"].to_numpy().astype(int)

    print(
        f"\nDataset: {len(data):,} panels "
        f"({int((y == 1).sum()):,} broken / {int((y == 0).sum()):,} good), "
        f"{len(feature_cols)} features."
    )

    model = _build_model(args.n_estimators, args.seed)
    _evaluate(model, X, y, args.seed)

    # Refit on all data and persist alongside the exact feature order.
    model.fit(X, y)
    import joblib

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": feature_cols,
            "n_broken": int((y == 1).sum()),
            "n_good": int((y == 0).sum()),
            "trained_at": pd.Timestamp.now().isoformat(),
        },
        args.out,
    )
    print(f"\nSaved model -> {args.out}")
    print("The Metrology tab will use it automatically (Stage 2 = supervised).")

    # Show which measurements drive the break prediction.
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        top = (
            pd.Series(importances, index=feature_cols)
            .sort_values(ascending=False)
            .head(10)
        )
        print("\nTop break-risk drivers:")
        for name, imp in top.items():
            print(f"  {imp:6.3f}  {name}")


if __name__ == "__main__":
    main()
