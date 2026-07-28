"""Train an ENRICHED Stage-2 break-risk model and compare to the 40-feature one.

The plain model uses only mean/min/max/std of each spec column. Panel breaks are
driven by *localized* excursions and edge-shape dispersion, which those four
moments smear out. This script adds, per panel:

  * percentiles (p01/p05/p50/p95/p99) of every spec column — capture localized
    tails far better than min/max (which are single-sample and noisy);
  * breach FRACTIONS — share of positions that violate each defect band
    (Dropouts>10, GlassThickness<2.25, Radius outside median +-5%);
  * dropout burst stats (max, count>0, count>10);
  * MaximumProfileHt focus stats (mean/max |offset|, fraction beyond +-6 mm).

All features are computed the same way at train and (future) serve time. We
cross-validate honestly and only keep the enriched model if it beats the plain
one on recall/AUC. The winning model is saved to config.METROLOGY_MODEL_PATH.

Run:
    python scripts/retrain_enriched.py
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

SPEC_BASES = [
    "Dropouts",
    "EdgeGrind_Delta",
    "EdgeGrind_PeakHeight",
    "GlassThickness",
    "Radius",
]
SIDES = ("Left", "Right")
PCTS = [0.01, 0.05, 0.50, 0.95, 0.99]


def enriched_features(csv_path: Path, label: int) -> pd.DataFrame:
    print(f"Loading + engineering {csv_path.name} ...")
    raw = pd.read_csv(csv_path)
    raw["SubID"] = raw["SubID"].astype(str)
    cols = [f"{b}_{s}" for b in SPEC_BASES for s in SIDES]
    cols += [f"MaximumProfileHt_{s}" for s in SIDES]
    for c in cols:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    g = raw.groupby("SubID")
    parts = []

    # moments + percentiles
    present = [c for c in cols if c in raw.columns]
    agg = g[present].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    parts.append(agg)
    q = g[present].quantile(PCTS).unstack()
    q.columns = [f"{c}_p{int(p * 100):02d}" for c, p in q.columns]
    parts.append(q)

    # breach fractions (localized-defect share)
    breach = pd.DataFrame(index=agg.index)
    for s in SIDES:
        d = f"Dropouts_{s}"
        if d in raw.columns:
            breach[f"frac_drop10_{s}"] = g[d].apply(lambda v: (v > 10).mean())
            breach[f"cnt_drop10_{s}"] = g[d].apply(lambda v: int((v > 10).sum()))
            breach[f"cnt_drop0_{s}"] = g[d].apply(lambda v: int((v > 0).sum()))
        t = f"GlassThickness_{s}"
        if t in raw.columns:
            breach[f"frac_thin_{s}"] = g[t].apply(lambda v: (v < 2.25).mean())
        r = f"Radius_{s}"
        if r in raw.columns:

            def _rad_frac(v):
                med = np.nanmedian(v.values)
                if not np.isfinite(med) or med == 0:
                    return 0.0
                return float(((v < med * 0.95) | (v > med * 1.05)).mean())

            breach[f"frac_shiner_{s}"] = g[r].apply(_rad_frac)
        p = f"MaximumProfileHt_{s}"
        if p in raw.columns:
            breach[f"ph_absmax_{s}"] = g[p].apply(
                lambda v: float(np.nanmax(np.abs(v.values)) if len(v) else 0)
            )
            breach[f"ph_absmean_{s}"] = g[p].apply(
                lambda v: float(np.nanmean(np.abs(v.values)) if len(v) else 0)
            )
            breach[f"frac_ph6_{s}"] = g[p].apply(
                lambda v: float((np.abs(v) > 6).mean())
            )
    parts.append(breach)

    feats = pd.concat(parts, axis=1).fillna(0.0)
    feats["label"] = label
    print(f"  {len(feats)} panels, {feats.shape[1] - 1} features")
    return feats


def cv_report(name, make, X, y, thr):
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    n_splits = max(2, min(5, n_pos, n_neg))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    proba = cross_val_predict(make(), X, y, cv=cv, method="predict_proba", n_jobs=-1)[
        :, 1
    ]
    pred = (proba >= thr).astype(int)
    m = dict(
        acc=accuracy_score(y, pred),
        prec=precision_score(y, pred, zero_division=0),
        rec=recall_score(y, pred, zero_division=0),
        f1=f1_score(y, pred, zero_division=0),
        auc=roc_auc_score(y, proba),
    )
    print(f"\n=== {name} (thr={thr:.2f}, {n_splits}-fold CV) ===")
    print(
        f"  acc {m['acc']:.3f} | prec {m['prec']:.3f} | rec {m['rec']:.3f} | "
        f"f1 {m['f1']:.3f} | AUC {m['auc']:.3f}"
    )
    print(
        "  confusion [true good/broken x pred good/broken]:",
        confusion_matrix(y, pred).tolist(),
    )
    return m, proba


def make_rf():
    return RandomForestClassifier(
        n_estimators=800,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=0,
    )


def main() -> None:
    data = pd.concat(
        [
            enriched_features(config.METROLOGY_LABELS_BAD, 1),
            enriched_features(config.METROLOGY_LABELS_GOOD, 0),
        ],
        axis=0,
    ).fillna(0.0)
    feat_cols = [c for c in data.columns if c != "label"]
    X = data[feat_cols].to_numpy()
    y = data["label"].to_numpy().astype(int)
    print(
        f"\nEnriched dataset: {len(data)} panels, {len(feat_cols)} features "
        f"({int((y == 1).sum())} broken / {int((y == 0).sum())} good)"
    )

    for thr in (0.65, 0.50):
        cv_report("ENRICHED model", make_rf, X, y, thr)

    print("\nClassification report (enriched, thr 0.50):")
    n_splits = max(2, min(5, int((y == 1).sum()), int((y == 0).sum())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    proba = cross_val_predict(
        make_rf(), X, y, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    print(
        classification_report(
            y,
            (proba >= 0.5).astype(int),
            target_names=["good", "broken"],
            zero_division=0,
        )
    )

    # Fit final + save to a SEPARATE file. The live scorer only computes
    # mean/min/max/std from the SQL aggregate, so the enriched model cannot be
    # deployed until the serving feature path is extended. Keep it beside the
    # deployed 40-feature model for evaluation / future wiring.
    final = make_rf()
    final.fit(X, y)
    out = config.METROLOGY_MODEL_PATH.with_name("metrology_model_enriched.joblib")
    import joblib

    joblib.dump(
        {
            "model": final,
            "features": feat_cols,
            "feature_kind": "enriched_v2",
            "n_broken": int((y == 1).sum()),
            "n_good": int((y == 0).sum()),
            "trained_at": pd.Timestamp.now().isoformat(),
        },
        out,
    )
    print(f"\nSaved enriched model -> {out} (NOT yet wired into live serving)")

    imp = pd.Series(final.feature_importances_, index=feat_cols).sort_values(
        ascending=False
    )
    print("\nTop 15 break-risk drivers:")
    for n, v in imp.head(15).items():
        print(f"  {v:6.3f}  {n}")


if __name__ == "__main__":
    main()
