# FS50 Grinder Metrology Dashboard — Quick User Guide (OPL)

**Audience:** Shift Managers · MET Techs · Semicon Managers   |   **Purpose:** Read live edge‑grind defects, find the responsible grinder, and act.
**Access:** On the wall display / browser go to **`http://<dashboard-pc>:8502`** (kiosk view — auto‑refreshes every 15 min).

---

## 1. What this dashboard tells you
It reads live **Edge Grind Profile (EGP)** data from the ODS and flags panels whose ground edge is out of spec, then **pins each defect to the exact grinder, groove, edge and side** — so you know precisely where to intervene.

**Three authoritative defect types:**

| Defect | Means | Triggered when |
|--------|-------|----------------|
| **Chip** (red) | Glass too thin / chipped edge | `GlassThickness < 2.25 mm` for ≥ 6 mm |
| **Dropout** (yellow) | Material missing along edge | `Dropouts > 10` for ≥ 6 mm |
| **Shiner** (blue) | Incomplete / uneven grind | `Radius` off panel median by > ±5 % for ≥ 50 mm |

> The **Edge Grind Profile** tab shows extra spec violations — treat those as *advisory only*. Chip / Dropout / Shiner are the trusted signal.

---

## 2. Reading the landing page (Grinder Health)

1. **KPI tiles (top):** Plates scanned · Pass rate · Defect plates · FS100 broken · Alerts logged · Latest scan time. *Quick health glance.*
2. **Active defect alerts (table):** every current defect. Key columns — **Grinder (A–E)**, **Groove (1–4)**, **Edge (Long/Short)**, **Side (Left/Right)**, **Defect**, **Run (mm)**, **Skewed?**
   - Alert wording:
     `GlassThickness out of spec — Long Edge, Right Side, Grinder B, Groove 1`
3. **Panel defect map:** a module diagram per active grinder with corner labels **LL / TL / TR / LR**; markers colour‑coded by defect type. *Hollow marker = ⚠️ suspected skewed / out‑of‑focus reading — verify before acting.*
4. **Grinder health bar chart:** worst grinder on top (by reject %). **Fix top‑of‑list first.**
5. **Spec‑defect heatmap:** rows = grinders A–E, columns = parameters → shows *which* measurement each grinder struggles with.

**Time range** (default *Last 30 minutes*) and other windows are set on the dashboard PC; the wall view is locked to keep it clean.

---

## 3. How to use it to FIX a grinder issue

```
See defect  ──►  Identify Grinder + Groove + Edge + Side  ──►  Inspect that wheel  ──►  Adjust  ──►  Recheck in 15 min
```

1. **Pinpoint the source** from the Active defects row — e.g. *Grinder C, Groove 2, Short Edge, Left Side*. That narrows it to **one grinding wheel**.
2. **Read the pattern** on the panel map:
   - Defects **clustered at one corner/groove** → wheel **wear or calibration drift** on that grinder.
   - Defects **scattered across the panel** → **focus / panel‑positioning** issue, not the wheel.
3. **Match the defect type to the action:**
   - **Chip** → check wheel wear / dressing, feed rate too aggressive, reduce infeed.
   - **Dropout** → check coolant flow, chipped/glazed wheel, glass edge quality.
   - **Shiner** → wheel not fully engaging: check wheel height/offset, alignment, dressing.
4. **Prioritise** using the bar chart (worst grinder first) and heatmap (which parameter).
5. **Correct** the identified grinder (wheel dress/replace, feed/speed, coolant, wear offset).
6. **Verify:** wait one refresh (15 min) — the defect should drop off the Active list and panel map. If it persists, escalate to MET / maintenance.

---

## 4. Teams alerts (automatic)

| Alert | Fires when | Cooldown |
|-------|-----------|----------|
| **Burst of defects** | ≥ 3 defects in 30 min on one grinder | 15 min |
| **High defect rate** | ≥ 25 % reject in 60 min (≥ 8 plates) | 30 min |
| **FS100 broken** | Plate scrapped post‑VTD — traced back to its grinder/groove | 15 min |

Alerts post to the team Teams channel and are logged in the **Alerts / Log / Rules** tab (with delivery status). Header shows **Teams ✅** when the webhook is connected.

---

## 5. If the dashboard shows NO data / “no defects”
1. Confirm the grinders are actually running (no data = no plates).
2. Widen the time range on the dashboard PC (e.g. Last 8 hours) to confirm it’s just a quiet window.
3. If the whole page is blank or stale for a long time, **restart the dashboard PC** and relaunch `run_dashboard.bat` — a wedged SQL connection is the usual cause.
4. Still empty → contact MET / the dashboard owner.

---
*One‑page OPL — keep at the FS50 grinder station. Questions: MET / dashboard owner.*
