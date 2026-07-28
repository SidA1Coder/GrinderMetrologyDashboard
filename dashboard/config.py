"""Central configuration for the FS50 Corner Metrology dashboard.

All values are environment-overridable so the same code runs locally (demo /
mock mode) and against the real camera folder + SQL ODS database. Copy
``.env.example`` to ``.env`` and adjust, or set real environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

# Optionally load a local .env file if python-dotenv is installed.
try:  # pragma: no cover - convenience only
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


# --- Paths -----------------------------------------------------------------
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent

# Trained model produced by scripts/train.py.
WEIGHTS = Path(
    os.getenv("FS50_WEIGHTS", PROJECT_ROOT / "runs/detect/corner/weights/best.pt")
)

# Folder the camera/tool drops new corner images into (live source).
# Images are nested in per-timestamp / per-head subfolders, so ingestion scans
# this tree RECURSIVELY and READ-ONLY (the source folder is never modified).
WATCH_DIR = Path(
    os.getenv(
        "FS50_WATCH_DIR",
        r"\\fs.local\fsglobal\Global\Shared\Manufacturing\PGT3\PGT3FS50CornerInspectImages",
    )
)

# Scan the watched folder recursively (images live in nested subfolders).
WATCH_RECURSIVE = os.getenv("FS50_WATCH_RECURSIVE", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Cap how many new images to process per scan so an initial run against a large
# network share does not block for a long time on CPU. 0 = no limit.
MAX_PER_SCAN = int(os.getenv("FS50_MAX_PER_SCAN", "100"))

# Where annotated (boxed) images are written for the gallery.
ANNOTATED_DIR = Path(os.getenv("FS50_ANNOTATED_DIR", DASHBOARD_DIR / "annotated"))

# Local SQLite history log (inspection results power the KPIs / charts).
DB_PATH = Path(os.getenv("FS50_DB_PATH", DASHBOARD_DIR / "data" / "inspections.db"))

# Persisted, UI-editable alert rules.
RULES_FILE = Path(os.getenv("FS50_RULES_FILE", DASHBOARD_DIR / "data" / "rules.json"))


# --- Inference -------------------------------------------------------------
CONF_THRESHOLD = float(os.getenv("FS50_CONF", "0.25"))
IMG_SIZE = int(os.getenv("FS50_IMGSZ", "640"))
DEVICE = os.getenv("FS50_DEVICE", "cpu")

# Class semantics (must match data.yaml).
CLASS_NAMES = {0: "WaterDrops", 1: "BrokenChips"}
# A panel is REJECTED only if one of these defect classes is detected.
REJECT_CLASSES = {"BrokenChips"}


# --- Product info database (SQL ODS) ---------------------------------------
# When ODS connection details are absent, the dashboard runs in MOCK mode and
# generates deterministic product info from the image filename so the whole
# app is demonstrable without the real database.
ODS_CONN_STR = os.getenv("FS50_ODS_CONN", "").strip()
USE_MOCK_DB = os.getenv("FS50_USE_MOCK_DB", "auto").lower()
# SQL query used to fetch product info; must return the named columns. The
# ``:key`` bind parameter is the identifier parsed from the image filename.
ODS_QUERY = os.getenv(
    "FS50_ODS_QUERY",
    "SELECT location, part_id, corner FROM product_ods WHERE serial = :key",
)


def use_mock_db() -> bool:
    """Decide whether to use mock product info."""
    if USE_MOCK_DB in ("1", "true", "yes", "on"):
        return True
    if USE_MOCK_DB in ("0", "false", "no", "off"):
        return False
    # auto: mock unless a real connection string is provided.
    return not bool(ODS_CONN_STR)


# --- Metrology (Edge Grind Profile) ----------------------------------------
# The edge-grind profile measurements live in the SQL ODS. Two source tables:
#   * ProcessData.ProcessHistory.EGPData  -> per-position continuous measurements
#   * ODS.mfg.ProcessHistoryGrinder       -> grinder machine parameters
# Connection is a SQL Server / Trusted-connection string, per site (mirrors the
# JMP grinder-profile script). By default the engine connects to the live ODS;
# set FS50_MET_USE_MOCK=1 to synthesise profiles for offline development.
#
# The current MES "result code" is the *legacy* verdict we are trying to
# improve, so it is deliberately NOT used as ground truth here.

METROLOGY_SITE = os.getenv("FS50_MET_SITE", "PGT3")

# Connection parts (same layout the JMP script builds: {site}messqlods.fs.local,
# database ODS, Windows/Trusted authentication).
METROLOGY_SERVER = os.getenv("FS50_MET_SERVER", f"{METROLOGY_SITE}messqlods.fs.local")
METROLOGY_DATABASE = os.getenv("FS50_MET_DB", "ODS")
METROLOGY_DRIVER = os.getenv("FS50_MET_DRIVER", "SQL Server")

# Optional full override: a raw ODBC string or a SQLAlchemy URL. When empty the
# string is assembled from the parts above.
METROLOGY_CONN_STR = os.getenv("FS50_MET_CONN", "").strip()
METROLOGY_USE_MOCK = os.getenv("FS50_MET_USE_MOCK", "auto").lower()

# Which edge profiler side(s) to evaluate. The EGPData.SerialNumber column
# labels rows e.g. "Edge Profiler 1 (LEL)" / "... (SEL)".
METROLOGY_PROFILER = os.getenv("FS50_MET_PROFILER", "Edge Profiler 1 (LEL)")

# Per-parameter spec limits (LSL, USL). None = unbounded on that side. Each base
# name is applied to both the _Left and _Right measurement columns. Defaults are
# taken from the JMP grinder-profile script's graph constants.
METROLOGY_SPECS: dict[str, tuple[float | None, float | None]] = {
    "Dropouts": (0.0, 30.0),
    "EdgeGrind_Delta": (-1.0, 1.0),
    "EdgeGrind_PeakHeight": (0.8, 1.6),
    "Radius": (1.0, 1.5),
    "GlassThickness": (0.0, None),
}

# Authoritative EGP defect definitions (from FS50 process owner, 2026-07). These
# replace the provisional min/max specs above for *alerting*: a defect is a RUN
# of consecutive out-of-band data along the ground edge, not a single point.
#   * Chip     = GlassThickness < 2.25 mm for >= 6 mm consecutively.
#   * Dropouts = Dropouts       > 10     for >= 6 mm consecutively.
#   * Shiner   = Radius deviates from the panel's calculated MEDIAN by more than
#               +/- 5% for >= 50 mm consecutively (incomplete edge grind).
# Each rule is applied independently to the _Left and _Right measurement columns.
#
# Two "mode"s:
#   "band"       — fixed limits: a sample is "bad" below `low` or above `high`
#                  (None = unbounded on that side).
#   "median_tol" — a sample is "bad" when it falls outside the panel's median
#                  +/- `tol_pct` percent. This matches the EGP defect-analysis
#                  tool's "Use calculated median" option (the Radius baseline
#                  drifts, so a fixed band is not meaningful).
# min_mm is the consecutive run length required to qualify as a defect.
DEFECT_RULES: dict[str, dict] = {
    "GlassThickness": {
        "defect": "Chip",
        "mode": "band",
        "low": 2.25,
        "high": None,
        "min_mm": 6.0,
    },
    "Dropouts": {
        "defect": "Dropout",
        "mode": "band",
        "low": None,
        "high": 10.0,
        "min_mm": 6.0,
    },
    "Radius": {
        "defect": "Shiner",
        "mode": "median_tol",
        "tol_pct": 5.0,
        "min_mm": 50.0,
    },
}

# Panel geometry (mm): the FS50 panel perimeter is traced by the edge profiler.
# The LONG edges run along X (0..2300) and the SHORT edges along Y (0..1215).
# A defect run whose extent is greater along X is on a LONG edge (ground by the
# GRINDER1 station); greater along Y is a SHORT edge (GRINDER2). Consecutive
# profile samples sit ~1.25 mm apart.
PANEL_LONG_MM = 2300.0
PANEL_SHORT_MM = 1215.0
EGP_SAMPLE_MM = 1.247  # median physical spacing between consecutive positions

# Sensor-distance validity flag (false-detection guard). MaximumProfileHt_{side}
# is the edge's offset from the 60 mm sensor target: 0 = exactly 60 mm, +X = X mm
# CLOSER (e.g. +6 -> 54 mm), -X = X mm FURTHER (e.g. -6 -> 66 mm). Beyond +/- 6 mm
# the profile data gets unreliable. Such detections are NOT dropped — they are
# still reported but FLAGGED as a suspected skewed / mis-positioned panel so a
# reviewer can tell a real defect from a focus artifact.
PROFILE_HT_METRIC = "MaximumProfileHt"
PROFILE_HT_TARGET_MM = 60.0
PROFILE_HT_TOL_MM = 6.0


# Trained metrology ML model (Stage 2). Populated once historical good/bad
# labels are supplied; absent => Stage 2 uses unsupervised anomaly detection.
METROLOGY_MODEL_PATH = Path(
    os.getenv("FS50_MET_MODEL", DASHBOARD_DIR / "data" / "metrology_model.joblib")
)

# Labelled training data for the Stage-2 "will this panel break downstream?"
# model. These are panels that PASSED the EGP spec rules but whose real
# downstream outcome is known. Drop the CSVs in dashboard/data/training/.
METROLOGY_TRAIN_DIR = Path(
    os.getenv("FS50_MET_TRAIN_DIR", DASHBOARD_DIR / "data" / "training")
)
METROLOGY_LABELS_BAD = Path(
    os.getenv("FS50_MET_LABELS_BAD", METROLOGY_TRAIN_DIR / "broken_panels.csv")
)
METROLOGY_LABELS_GOOD = Path(
    os.getenv("FS50_MET_LABELS_GOOD", METROLOGY_TRAIN_DIR / "good_panels.csv")
)


def use_mock_metrology() -> bool:
    """Decide whether the metrology engine should synthesise mock data."""
    if METROLOGY_USE_MOCK in ("1", "true", "yes", "on"):
        return True
    if METROLOGY_USE_MOCK in ("0", "false", "no", "off"):
        return False
    # auto: a connection string is always available (assembled from the site),
    # so default to the live ODS. Force mock with FS50_MET_USE_MOCK=1.
    return False


def metrology_odbc_str() -> str:
    """Return the ODBC/connection string for the metrology ODS.

    Uses the explicit override if provided, otherwise assembles the same
    trusted-connection string the JMP grinder-profile script builds.
    """
    if METROLOGY_CONN_STR:
        return METROLOGY_CONN_STR
    return (
        f"DRIVER={{{METROLOGY_DRIVER}}};SERVER={METROLOGY_SERVER};"
        f"DATABASE={METROLOGY_DATABASE};Trusted_Connection=Yes;APP=FS50Dashboard"
    )


# --- Alerting --------------------------------------------------------------
TEAMS_WEBHOOK_URL = os.getenv("FS50_TEAMS_WEBHOOK", "").strip()

# Known inspection locations / stations (used for the heatmap + mock data).
LOCATIONS = [
    s.strip()
    for s in os.getenv("FS50_LOCATIONS", "Line-A,Line-B,Line-C,Line-D").split(",")
    if s.strip()
]


def ensure_dirs() -> None:
    """Create the dashboard's OWN runtime folders.

    Note: the watched source folder is intentionally NOT created here -- it is
    treated as read-only external input and must never be modified.
    """
    for d in (ANNOTATED_DIR, DB_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
