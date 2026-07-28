"""Retrain the Stage-2 break-risk model on the enlarged EGP dataset and compare
against the currently deployed model.

Steps
-----
1. Load broken/good raw-EGPData CSVs and aggregate to one feature row per panel
   using ``metrology.plate_features`` (identical to the live serving features).
2. Benchmark the CURRENT saved model on this labelled set (baseline).
3. Train an improved model with an honest stratified cross-validation and report
   accuracy / precision / recall / F1 / ROC-AUC out-of-fold.
4. Refit on all data and save to ``config.METROLOGY_MODEL_PATH`` (old model is
   backed up first).

Run:
    python scripts/retrain_and_compare.py
"""

from __future__ import annotations

import shutil
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

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

THRESHOLD = 0.65  # production decision threshold (metrology._ANOMALY_THRESHOLD)


def load_features(csv_path: Path, label: int) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    if "SubID" not in raw.columns:
        for alt in ("sub_id", "subid", "SubId", "panel_id", "serial"):
            if alt in raw.columns:
                raw = raw.rename(columns={alt: "SubID"})
                break
    feats = metrology.plate_features(raw)
    feats = feats.copy()
    feats["label"] = label
    print(
        f"  {csv_path.name}: {len(raw):,} rows -> {len(feats):,} panels "
        f"({feats.shape[1] - 1} features)"
    )
    return feats


def report(name: str, y, proba, thr: float) -> dict:
    pred = (proba >= thr).astype(int)
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    try:
        auc = roc_auc_score(y, proba)
    except ValueError:
        auc = float("nan")
    print(f"\n=== {name} (threshold={thr:.2f}) ===")
    print(f"  Accuracy : {acc:6.3f}")
    print(
        f"  Precision: {prec:6.3f}   (of panels flagged broken, how many truly broke)"
    )
    print(f"  Recall   : {rec:6.3f}   (of truly-broken panels, how many we caught)")
    print(f"  F1       : {f1:6.3f}")
    print(f"  ROC-AUC  : {auc:6.3f}")
    print("  Confusion [rows=true good/broken, cols=pred good/broken]:")
    print("   ", confusion_matrix(y, pred).tolist())
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}


def main() -> None:
    print("Loading labelled panels (this reads ~380 MB of CSV)...")
    bad = load_features(config.METROLOGY_LABELS_BAD, 1)
    good = load_features(config.METROLOGY_LABELS_GOOD, 0)
    data = pd.concat([bad, good], axis=0)

    feature_cols = [c for c in data.columns if c != "label"]
    data[feature_cols] = data[feature_cols].fillna(0.0)
    X = data[feature_cols].to_numpy()
    y = data["label"].to_numpy().astype(int)
    print(
        f"\nDataset: {len(data):,} panels "
        f"({int((y == 1).sum())} broken / {int((y == 0).sum())} good), "
        f"{len(feature_cols)} features."
    )

    # ---- 1) Baseline: current deployed model on this labelled set -----------
    baseline = None
    if config.METROLOGY_MODEL_PATH.exists():
        import joblib

        art = joblib.load(config.METROLOGY_MODEL_PATH)
        model = art["model"] if isinstance(art, dict) else art
        feat_names = art.get("features") if isinstance(art, dict) else None
        meta = (
            f"trained on {art.get('n_broken', '?')} broken / "
            f"{art.get('n_good', '?')} good @ {art.get('trained_at', '?')}"
            if isinstance(art, dict)
            else "legacy estimator"
        )
        print(f"\nCurrent deployed model: {meta}")
        Xc = (
            data[feature_cols]
            if feat_names is None
            else data[feature_cols].reindex(columns=feat_names, fill_value=0.0)
        )
        proba_cur = model.predict_proba(Xc.to_numpy())[:, 1]
        baseline = report(
            "CURRENT model on new data (in-sample-ish, optimistic)",
            y,
            proba_cur,
            THRESHOLD,
        )
    else:
        print("\nNo current model found to benchmark.")

    # ---- 2) New model: honest out-of-fold cross-validation ------------------
    def make_model():
        return RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=0,
        )

    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    n_splits = max(2, min(5, n_pos, n_neg))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    proba_cv = cross_val_predict(
        make_model(), X, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    new_065 = report(
        f"NEW model ({n_splits}-fold cross-validated, out-of-fold)",
        y,
        proba_cv,
        THRESHOLD,
    )
    # Also show the balanced 0.50 operating point for reference.
    new_050 = report(
        f"NEW model ({n_splits}-fold cross-validated, out-of-fold)", y, proba_cv, 0.50
    )
    print("\nFull classification report (NEW, threshold 0.50):")
    print(
        classification_report(
            y,
            (proba_cv >= 0.5).astype(int),
            target_names=["good", "broken"],
            zero_division=0,
        )
    )

    # ---- 3) Refit on ALL data and persist -----------------------------------
    final = make_model()
    final.fit(X, y)

    out = config.METROLOGY_MODEL_PATH
    if out.exists():
        backup = out.with_suffix(".joblib.bak")
        shutil.copy2(out, backup)
        print(f"\nBacked up previous model -> {backup}")

    import joblib

    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final,
            "features": feature_cols,
            "n_broken": n_pos,
            "n_good": n_neg,
            "trained_at": pd.Timestamp.now().isoformat(),
            "cv_metrics": new_065,
        },
        out,
    )
    print(f"Saved new model -> {out}")

    importances = pd.Series(final.feature_importances_, index=feature_cols)
    print("\nTop break-risk drivers:")
    for name, imp in importances.sort_values(ascending=False).head(12).items():
        print(f"  {imp:6.3f}  {name}")

    # ---- 4) Summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY (production threshold 0.65)")
    print("=" * 60)
    if baseline:
        print(
            f"  CURRENT : acc {baseline['acc']:.3f} | prec {baseline['prec']:.3f} | "
            f"rec {baseline['rec']:.3f} | AUC {baseline['auc']:.3f}"
        )
    print(
        f"  NEW(CV) : acc {new_065['acc']:.3f} | prec {new_065['prec']:.3f} | "
        f"rec {new_065['rec']:.3f} | AUC {new_065['auc']:.3f}"
    )
    print(
        f"  NEW@0.50: acc {new_050['acc']:.3f} | prec {new_050['prec']:.3f} | "
        f"rec {new_050['rec']:.3f} | AUC {new_050['auc']:.3f}"
    )


if __name__ == "__main__":
    main()
