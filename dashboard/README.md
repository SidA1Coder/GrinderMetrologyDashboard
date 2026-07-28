# FS50 Corner Metrology — Live Inspection Dashboard

A Streamlit dashboard that runs the trained YOLO model on a **live stream of
corner images**, enriches each with product context from the **SQL ODS**, and
raises **Microsoft Teams alerts** when configurable defect thresholds are hit.

It lives in its own `dashboard/` folder but loads the model
(`runs/detect/corner/weights/best.pt`) produced by the training scripts.

## Features

- **Live KPI tiles** — inspected count, pass rate, rejects, defects today, alerts.
- **Trend charts** — inspections & defects over time, pass/reject split, defect classes.
- **Live feed gallery** — most recent inspections with bounding boxes drawn.
- **Per-location breakdown + heatmap** — defects by line/station over time.
- **Filterable history table** — filter by location/decision/part id, export CSV.
- **Configurable alert rules (in the UI)**:
  - `count` — N rejects from a location within a time window.
  - `rate` — reject rate exceeds a % over a rolling window.
  - `immediate` — any single broken chip.
  - Each with cooldown to avoid alert spam.
- **Teams delivery** — fires an Incoming Webhook message when a rule triggers.

## How data flows

```
camera/tool ──> watched folder ──> ingestion.py
                                      │  (1) inference.py  -> run best.pt, draw boxes, PASS/REJECT
                                      │  (2) database.py   -> product info from SQL ODS (or mock)
                                      │  (3) store.py      -> append to SQLite history
                                      │  (4) alerts.py     -> evaluate rules -> Teams
                                      ▼
                                   app.py (Streamlit reads the history store)
```

Decision rule: a panel is **REJECT** only if a `BrokenChips` defect is
detected; `WaterDrops` and clean corners **PASS** (matches `data.yaml`).

## Quick start (demo / mock mode — no DB or camera needed)

```bash
conda activate fs50defect
pip install -r dashboard/requirements.txt

# 1. Seed some demo inspections from existing dataset images
python dashboard/demo_seed.py --n 40

# 2. Launch the dashboard
streamlit run dashboard/app.py
```

Then click **Scan now** in the sidebar to process any new images, or tick
**Auto-refresh**.

## Connecting real data

Copy `.env.example` to `.env` and set:

- `FS50_WATCH_DIR` — the folder the camera/tool drops images into.
- `FS50_ODS_CONN` + `FS50_ODS_QUERY` — SQL ODS connection + lookup query
  (returns `location, part_id, corner` for a given serial). Leave blank to stay
  in mock mode.
- `FS50_TEAMS_WEBHOOK` — Incoming Webhook URL for the Teams channel.

**Image filename convention** used to parse context and key the ODS lookup:

```
<LINE>_<SERIAL>_C<CORNER>.jpg      e.g.  Line-A_SN12345_C3.jpg
```

Filenames that don't match still work — missing fields are filled from the ODS
query or mocked.

## Notes

- The SQLite history store (`data/inspections.db`) is the single source of
  truth for KPIs/charts/alerts, so the dashboard keeps working if the ODS is
  temporarily unreachable.
- For Palantir Foundry deployment, this same pipeline (`inference` +
  `database` + `alerts`) can be wrapped in a Foundry transform/function; the
  Streamlit UI is for local/on-prem monitoring.
