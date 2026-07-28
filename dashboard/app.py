"""FS50 Corner Metrology — live inspection dashboard (Streamlit).

Run with:
    streamlit run dashboard/app.py

Features
--------
- Live KPI tiles (inspected, pass rate, defects today, active alerts)
- Trend charts (defects over time, by location)
- Recent-defect image gallery with bounding boxes
- Per-location breakdown + heatmap
- Filterable history table with CSV export
- UI-configurable alert rules + Teams alert log
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

# Ensure sibling modules import whether run via "streamlit run dashboard/app.py"
# or from within the dashboard folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.express as px
import streamlit as st

import alerts
import branding
import config
import ingestion
import metrology
import panel_diagram
import store

st.set_page_config(
    page_title="FS50 Corner & Edge Metrology",
    page_icon=str(branding.LOGO_PATH) if branding.LOGO_PATH.exists() else "🔎",
    layout="wide",
)

branding.inject_theme()
config.ensure_dirs()
store.init_db()


# --------------------------------------------------------------------------
# Sidebar — controls
# --------------------------------------------------------------------------
branding.sidebar_logo()
st.sidebar.title("FS50 Corner & Edge Metrology")
st.sidebar.caption("Live corner-defect inspection")

mode = "MOCK product data" if config.use_mock_db() else "SQL ODS connected"
teams = "Teams ✅" if config.TEAMS_WEBHOOK_URL else "Teams ⚠️ not set"
st.sidebar.info(f"**Mode:** {mode}\n\n**Alerts:** {teams}")

conf = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.05,
    max_value=0.9,
    value=float(config.CONF_THRESHOLD),
    step=0.05,
    help="Lower = catch more defects (higher recall) but more false alarms.",
)

st.sidebar.write(f"**Watched folder:**\n`{config.WATCH_DIR}`")

col_a, col_b = st.sidebar.columns(2)
if col_a.button("🔄 Scan now", use_container_width=True):
    with st.spinner("Processing new images…"):
        summaries = ingestion.scan_and_process(conf=conf)
    new = [s for s in summaries if "error" not in s]
    st.sidebar.success(f"Processed {len(new)} new image(s).")
    for s in summaries:
        if s.get("alerts"):
            for a in s["alerts"]:
                st.toast(f"🚨 {a['rule']}: {a['message']}", icon="🚨")

auto = st.sidebar.checkbox(
    "Auto-refresh (30s)",
    value=False,
    help="Periodically reload live data. Heavy live EGP queries mean each "
    "refresh can take a while to finish before the next tick.",
)

st.sidebar.divider()
time_window = st.sidebar.selectbox(
    "Dashboard time range",
    [
        "Last 1 hour",
        "Last 8 hours",
        "Last 24 hours",
        "Last 7 days",
        "All time",
        "Custom range…",
    ],
    index=0,
    help="Live EGP scans are heavy — start with 1 hour for a fast landing, "
    "widen the window only when you need more history. Pick 'Custom range…' "
    "for an exact calendar window.",
)
_preset_delta = {
    "Last 1 hour": timedelta(hours=1),
    "Last 8 hours": timedelta(hours=8),
    "Last 24 hours": timedelta(hours=24),
    "Last 7 days": timedelta(days=7),
    "All time": timedelta(days=30),
}

if time_window == "Custom range…":
    _now = datetime.now()
    _default_start = (_now - timedelta(hours=24)).date()
    date_range = st.sidebar.date_input(
        "Calendar range",
        value=(_default_start, _now.date()),
        max_value=_now.date(),
        help="Pick the start and end day of the window.",
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        d_start, d_end = date_range
    else:  # user is mid-selection (single date returned)
        d_start = d_end = (
            date_range[0] if isinstance(date_range, (list, tuple)) else date_range
        )
    c_t1, c_t2 = st.sidebar.columns(2)
    t_start = c_t1.time_input("From", value=dtime(0, 0))
    t_end = c_t2.time_input("To", value=dtime(23, 59))
    win_start = datetime.combine(d_start, t_start)
    win_end = datetime.combine(d_end, t_end)
    if win_end <= win_start:
        st.sidebar.warning("End must be after start — using a 1-hour window.")
        win_end = win_start + timedelta(hours=1)
    st.sidebar.caption(f"Window: {win_start:%Y-%m-%d %H:%M} → {win_end:%Y-%m-%d %H:%M}")
else:
    win_end = datetime.now()
    win_start = win_end - _preset_delta[time_window]

# Hours of history the ODS range query must cover for this window.
lookback = max(1, int(round((win_end - win_start).total_seconds() / 3600)))


# --------------------------------------------------------------------------
# Load + filter data
# --------------------------------------------------------------------------
df = store.load_inspections()
if not df.empty:
    df = df[(df["ts"] >= win_start) & (df["ts"] <= win_end)]


@st.cache_data(ttl=300, show_spinner=False)
def load_metrology(start_iso: str, end_iso: str, use_ml: bool, force_mock: bool):
    """Load + evaluate edge-grind profiles and return per-plate DataFrames.

    Cached for 5 minutes (keyed by the arguments) so the heavy ODS query does
    not re-run on every Streamlit rerun.
    """
    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso)
    results = metrology.analyze(
        start=start, end=end, use_ml=use_ml, mock=force_mock or None
    )
    rows = [
        {
            "sub_id": r.sub_id,
            "verdict": r.verdict,
            "grinder": r.equipment or "?",
            "grinder_loc": r.grinder_loc or "?",
            "grind_time": r.grind_time,
            "vtd": r.vtd or "?",
            "vtd_time": r.vtd_time,
            "profiler": r.profiler or "",
            "read_time": r.read_time,
            "n_positions": r.n_positions,
            "n_violations": len(r.violations),
            "out_of_spec": ", ".join(sorted(set(r.out_of_spec_params))),
            "anomaly_score": r.anomaly_score,
            "ml_flag": r.ml_flag,
            "reason": r.reason,
        }
        for r in results
    ]
    mdf = pd.DataFrame(rows)
    # Flatten every individual spec violation with full attribution so the
    # grinder-health views can group by grinder / parameter / side / value.
    viol_rows = []
    for r in results:
        for v in r.violations:
            param = v.parameter
            side = (
                "Left"
                if param.endswith("_Left")
                else ("Right" if param.endswith("_Right") else "—")
            )
            base = param.rsplit("_", 1)[0] if side != "—" else param
            viol_rows.append(
                {
                    "sub_id": r.sub_id,
                    "grinder": r.equipment or "?",
                    "grinder_loc": r.grinder_loc or "?",
                    "grind_time": r.grind_time,
                    "read_time": r.read_time,
                    "parameter": param,
                    "base": base,
                    "side": side,
                    "value": v.value,
                    "limit": v.limit,
                    "bound": v.bound,
                    "position": v.position,
                }
            )
    vdf = pd.DataFrame(viol_rows)
    return mdf, vdf


@st.cache_data(ttl=300, show_spinner=False)
def load_alerts(start_iso: str, end_iso: str, force_mock: bool):
    """Detect fully-attributed edge-grind defect alerts for the window.

    Uses the authoritative run-length engine (chip/dropout/shiner) plus groove
    attribution, so each alert names the defect, edge (Long/Short), side
    (Left/Right), grinder line and groove. Cached 5 min like the monitor frame.
    """
    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso)
    return metrology.build_alerts(start=start, end=end, mock=force_mock or None)


@st.cache_data(ttl=300, show_spinner=False)
def load_broken(start_iso: str, end_iso: str, force_mock: bool):
    """FS100-broken plates (post-VTD scrap) backtracked to their grinder.

    Wraps :func:`metrology.broken_with_attribution`: for each plate scrapped as
    'Broken' at a VTD coater it resolves the grinder line, edge, groove and any
    edge-grind defect already detected, so the FS100 Broken view can explain the
    likely root cause. Cached 5 min.
    """
    start = datetime.fromisoformat(start_iso)
    end = datetime.fromisoformat(end_iso)
    return metrology.broken_with_attribution(
        start=start, end=end, mock=force_mock or None
    )


# --------------------------------------------------------------------------
# Monitor settings + load the metrology (Broken Monitor) data once, up front
# so the Overview, Metrology and drill-down logs all share the same frame.
# --------------------------------------------------------------------------
with st.sidebar.expander("Monitor settings", expanded=False):
    use_ml = st.checkbox("Stage 2 ML break-risk", value=True)
    force_mock = st.checkbox(
        "Force mock data",
        value=config.use_mock_metrology(),
        help="Use synthetic profiles instead of the live ODS query.",
    )
    if st.button("🔄 Refresh monitor data", use_container_width=True):
        load_metrology.clear()
        load_alerts.clear()
        load_broken.clear()
        st.rerun()
    auto_teams = st.checkbox(
        "Auto-send Teams alerts",
        value=not config.use_mock_metrology(),
        help="On each refresh, post to Teams when a grinder logs \u22653 defects "
        "within 15 min, or any plate breaks at FS100.",
    )

# Auto-refresh: schedule a periodic rerun and clear the cached live loaders so
# each tick pulls fresh EGP data (defined here, after the loaders exist).
if auto:
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=30_000, key="auto")
        load_metrology.clear()
        load_alerts.clear()
        load_broken.clear()
        if not config.use_mock_db():
            ingestion.scan_and_process(conf=conf)
    except Exception:
        st.sidebar.caption("Install streamlit-autorefresh for auto mode.")

with st.spinner("Loading Broken Monitor data…"):
    mdf, vdf = load_metrology(
        win_start.isoformat(), win_end.isoformat(), bool(use_ml), bool(force_mock)
    )

with st.spinner("Detecting defect alerts…"):
    adf = load_alerts(win_start.isoformat(), win_end.isoformat(), bool(force_mock))

# FS100 broken feed is only needed by the FS100 Broken tab (and its KPI tile).
# Load it lazily so the main Grinder-Health landing page stays fast: the query
# runs the first time the FS100 tab is opened (or when the tile is expanded),
# then is cached for the rest of the session window.
if "load_fs100" not in st.session_state:
    st.session_state["load_fs100"] = False


def get_broken_df():
    """Return the FS100 broken frame, loading it on first demand only."""
    if not st.session_state.get("load_fs100"):
        return None
    with st.spinner("Checking FS100 breaks…"):
        return load_broken(win_start.isoformat(), win_end.isoformat(), bool(force_mock))


broken_df = get_broken_df()


def _dispatch_teams_alerts() -> list[dict]:
    """Fire the FS50 burst (\u22653 defects/15 min) and FS100-broken Teams alerts."""
    fired: list[dict] = []
    fired += alerts.send_defect_burst_alerts(
        adf, window_minutes=15, min_count=3, cooldown_minutes=15
    )
    fired += alerts.send_broken_alerts(broken_df, cooldown_minutes=15)
    return fired


# Auto-send on refresh (respects per-alert cooldown so it won't spam).
if auto_teams:
    try:
        _sent = _dispatch_teams_alerts()
        for _f in _sent:
            icon = "\u2705" if _f.get("delivered") else "\u26a0\ufe0f"
            st.toast(f"{icon} Teams: {_f['rule']}", icon="\U0001f4e8")
    except Exception as _e:  # never let alerting break the dashboard
        st.sidebar.caption(f"Teams auto-send skipped: {_e}")

# Consolidated lineage filters (sidebar) applied to every monitor view.
# Always offer the full roster (Grinder A–E / VTD A–E) so an engineer can
# select a line even when it happens to have no plates in the current window —
# not just the lines that appear in the loaded data.
_ALL_GRINDERS = [
    metrology.grinder_label(f"{config.METROLOGY_SITE}1{g}-GRINDER") for g in "ABCDE"
]
_ALL_VTDS = [
    metrology.vtd_label(f"{config.METROLOGY_SITE}1{v}-VTD_COATER") for v in "ABCDE"
]

if not mdf.empty and "grinder" in mdf.columns:
    _seen = [g for g in mdf["grinder"].dropna().unique() if g not in (None, "?")]
    _grinders = [g for g in _ALL_GRINDERS if g] + [
        g for g in _seen if g not in _ALL_GRINDERS
    ]
    if _grinders:
        pick_grinder = st.sidebar.multiselect(
            "Grinder lineage",
            _grinders,
            default=_grinders,
            help="Filter every monitor view by the grinder that ground the plate. "
            "All lines A–E are always listed.",
        )
        if pick_grinder:
            mdf = mdf[mdf["grinder"].isin(pick_grinder) | (mdf["grinder"] == "?")]
            if not df.empty and "grinder" in df.columns:
                df = df[df["grinder"].isin(pick_grinder) | df["grinder"].isna()]
            if adf is not None and not adf.empty and "grinder" in adf.columns:
                adf = adf[adf["grinder"].isin(pick_grinder) | adf["grinder"].isna()]

if not mdf.empty and "vtd" in mdf.columns:
    _seen_v = [v for v in mdf["vtd"].dropna().unique() if v not in (None, "?")]
    _vtds = [v for v in _ALL_VTDS if v] + [v for v in _seen_v if v not in _ALL_VTDS]
    if _vtds:
        pick_vtd = st.sidebar.multiselect(
            "VTD lineage",
            _vtds,
            default=_vtds,
            help="Filter the Broken monitor by the VTD line that coated the plate. "
            "All lines are always listed.",
        )
        if pick_vtd:
            mdf = mdf[mdf["vtd"].isin(pick_vtd) | (mdf["vtd"] == "?")]


def _resample_freq(hours: int) -> tuple[str, str]:
    """Pick an hour/day bucket + label based on the lookback window."""
    if hours <= 24:
        return "h", "hour"
    return "D", "day"


branding.header()

# Landing summary tiles. Per the agreed model, FS50 pass/fail is the
# authoritative run-length defect engine (chip/dropout/shiner), NOT the
# provisional spec-limit verdict (which flags nearly every plate and is only
# kept as an experimental view). "FS100 broken" is the real post-VTD scrap feed.
_total = len(mdf)
_defect_plates = (
    int(adf["sub_id"].nunique()) if adf is not None and not adf.empty else 0
)
_pass_rate = (100.0 * (_total - _defect_plates) / _total) if _total else 0.0
_broken = (
    int(broken_df["SubID"].nunique())
    if broken_df is not None and not broken_df.empty
    else None
)
_alerts = len(store.load_alerts())
_last = (
    pd.to_datetime(mdf["read_time"], errors="coerce").max()
    if _total and "read_time" in mdf
    else None
)
_when = _last.strftime("%b %d %H:%M") if _last is not None and pd.notna(_last) else "—"
s1, s2, s3, s4, s5, s6 = st.columns(6)
s1.metric("Plates scanned", f"{_total:,}")
s2.metric("FS50 pass rate", f"{_pass_rate:.1f}%")
s3.metric(
    "Defect plates",
    f"{_defect_plates:,}",
    help="Plates with a run-length edge-grind defect",
)
s4.metric(
    "FS100 broken",
    f"{_broken:,}" if _broken is not None else "—",
    help="Open the FS100 Broken tab to load",
)
s5.metric("Alerts logged", f"{_alerts:,}")
s6.metric("Latest scan", _when)
st.divider()

(
    tab_overview,
    tab_broken,
    tab_live,
    tab_locations,
    tab_metrology,
    tab_history,
    tab_alerts,
) = st.tabs(
    [
        "Grinder Health",
        "FS100 Broken",
        "Corner Images",
        "Locations",
        "Edge Grind Profile",
        "History",
        "Alerts / Log / Rules",
    ]
)


# --------------------------------------------------------------------------
# Overview — trends
# --------------------------------------------------------------------------
with tab_overview:
    # ----------------------------------------------------------------------
    # ACTIVE ALERTS — attributed edge-grind defects (chip / dropout / shiner)
    # with the exact grinder + groove + edge + side so MET can act now.
    # ----------------------------------------------------------------------
    branding.section(
        "Active defect alerts",
        "Attributed edge-grind defects with grinder · groove · edge · side",
    )
    if adf is None or adf.empty:
        st.success("No edge-grind defects detected in this window.")
    else:
        a = adf.copy()
        counts = a["defect"].value_counts().to_dict()
        n_suspect = (
            int(a["suspect_focus"].fillna(False).astype(bool).sum())
            if "suspect_focus" in a.columns
            else 0
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total alerts", len(a))
        c2.metric("Chips", int(counts.get("Chip", 0)))
        c3.metric("Dropouts", int(counts.get("Dropout", 0)))
        c4.metric("Shiners", int(counts.get("Shiner", 0)))
        c5.metric("Skewed?", n_suspect, help="Suspected mis-positioned panels")
        st.caption(
            "Shiner = Radius deviates from the panel's calculated median by "
            "more than ±5% for ≥ 50 mm (matches the EGP defect-analysis tool). "
            "Rows flagged **⚠️ skewed** have an edge beyond ±6 mm of the 60 mm "
            "sensor target (MaximumProfileHt) — the plate was likely too far / "
            "too close, so those detections may be false. They are still listed."
        )
        show = a.copy()
        show["when"] = pd.to_datetime(show["read_time"], errors="coerce").dt.strftime(
            "%m-%d %H:%M"
        )
        show["run (mm)"] = show["length_mm"]
        if "suspect_focus" in show.columns:
            off = show["max_offset_mm"] if "max_offset_mm" in show.columns else None
            show["Skewed?"] = [
                (f"⚠️ {o:g} mm" if (o is not None and pd.notna(o)) else "⚠️")
                if bool(s)
                else ""
                for s, o in zip(
                    show["suspect_focus"].fillna(False),
                    off if off is not None else [None] * len(show),
                )
            ]
        else:
            show["Skewed?"] = ""
        disp = show.rename(
            columns={
                "defect": "Defect",
                "metric": "Metric",
                "edge": "Edge",
                "side": "Side",
                "grinder": "Grinder",
                "groove": "Groove",
                "sub_id": "Plate",
            }
        )[
            [
                "when",
                "Defect",
                "Metric",
                "Edge",
                "Side",
                "Grinder",
                "Groove",
                "run (mm)",
                "Skewed?",
                "Plate",
            ]
        ]
        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
            height=min(420, 60 + 35 * min(len(disp), 10)),
        )
        st.caption(
            "Each row: defect type, which edge (Long/Short), which side "
            "(Left/Right), the grinder line and groove responsible, the "
            "consecutive run length, and whether the panel is a suspected "
            "skewed / out-of-focus reading. Newest first."
        )
    st.divider()

    # ----------------------------------------------------------------------
    # PANEL DEFECT MAP — one solar-module diagram per grinder (A–E) with the
    # corner labels (LL/TL/TR/LR) and every detected defect plotted at its
    # position along the panel perimeter, so ops can see WHERE on the panel a
    # given grinder is chipping.
    # ----------------------------------------------------------------------
    branding.section(
        "Panel defect map by grinder",
        "Where on the module each grinder's defects land · LL/TL/TR/LR corners",
    )
    if adf is None or adf.empty:
        st.info("No defects to map in this window.")
    else:
        _grinders_present = [g for g in _ALL_GRINDERS if g] or sorted(
            adf["grinder"].dropna().unique()
        )
        # Show the full A–E roster so an empty grinder still reads "clean".
        cols_per_row = 3
        for i in range(0, len(_grinders_present), cols_per_row):
            row_grinders = _grinders_present[i : i + cols_per_row]
            cols = st.columns(len(row_grinders))
            for col, g in zip(cols, row_grinders):
                gdf = (
                    adf[adf["grinder"] == g]
                    if "grinder" in adf.columns
                    else adf.iloc[0:0]
                )
                n = len(gdf)
                fig = panel_diagram.panel_figure(
                    gdf, title=f"{g}  ·  {n} defect{'s' if n != 1 else ''}"
                )
                col.plotly_chart(fig, use_container_width=True, key=f"panel_{g}")
        st.caption(
            "Marker colour = defect type (Chip · Dropout · Shiner). Hollow marker "
            "= suspected skewed / out-of-focus reading. Long edges run along the "
            "top & bottom rails (Grinder 1); short edges along the left & right "
            "rails (Grinder 2). Corner map is provisional pending the TR/LL/LR/TL "
            "training material."
        )

        # Defect-type × grinder and defect-type × edge/side summary charts.
        cc1, cc2 = st.columns(2)
        by_g = adf.groupby(["grinder", "defect"]).size().reset_index(name="count")
        if not by_g.empty:
            fig_g = px.bar(
                by_g,
                x="grinder",
                y="count",
                color="defect",
                color_discrete_map=panel_diagram.DEFECT_COLORS,
                title="Defects by grinder & type",
            )
            fig_g.update_layout(height=320, margin=dict(l=6, r=6, t=40, b=6))
            cc1.plotly_chart(fig_g, use_container_width=True, key="def_by_grinder")
        by_e = (
            adf.assign(loc=adf["edge"].astype(str) + " · " + adf["side"].astype(str))
            .groupby(["loc", "defect"])
            .size()
            .reset_index(name="count")
        )
        if not by_e.empty:
            fig_e = px.bar(
                by_e,
                x="loc",
                y="count",
                color="defect",
                color_discrete_map=panel_diagram.DEFECT_COLORS,
                title="Defects by edge · side & type",
            )
            fig_e.update_layout(
                height=320,
                margin=dict(l=6, r=6, t=40, b=6),
                xaxis_title="",
                yaxis_title="count",
            )
            cc2.plotly_chart(fig_e, use_container_width=True, key="def_by_edge")
    st.divider()

    if mdf.empty:
        st.info(
            "No metrology data for this window. Widen the **Dashboard time range** "
            "or enable **Force mock data** under Monitor settings."
        )
    else:
        freq, freq_label = _resample_freq(int(lookback))
        m = mdf.copy()
        m["bucket"] = pd.to_datetime(m["read_time"], errors="coerce").dt.floor(freq)
        m["rejected"] = (m["verdict"] == "REJECT").astype(int)
        m["broken"] = m["ml_flag"].astype(int) if "ml_flag" in m else 0

        # ------------------------------------------------------------------
        # GRINDER HEALTH — the core view: which grinder produces which spec
        # defects, so MET knows which grinder to troubleshoot.
        # ------------------------------------------------------------------
        branding.section("Grinder health", "Spec defects by grinder")
        st.caption(
            "Which grinder is throwing which edge-profile spec problems. "
            "⚠️ Spec limits are **provisional placeholders** — set the real OCAP "
            "limits in `config.METROLOGY_SPECS` for accurate defect attribution."
        )

        per_g = (
            m.groupby("grinder")
            .agg(plates=("sub_id", "count"), rejects=("rejected", "sum"))
            .reset_index()
        )
        per_g = per_g[per_g["grinder"] != "?"]
        per_g["defect_rate"] = (per_g["rejects"] / per_g["plates"]).fillna(0.0)

        h1, h2 = st.columns([2, 3])

        # Health scoreboard: per-grinder defect rate, sorted worst-first.
        if not per_g.empty:
            board = per_g.sort_values("defect_rate", ascending=False)
            h1.plotly_chart(
                px.bar(
                    board,
                    x="defect_rate",
                    y="grinder",
                    orientation="h",
                    title="Defect rate by grinder (worst on top)",
                    text=board["defect_rate"].map(lambda x: f"{x:.0%}"),
                    labels={"defect_rate": "out-of-spec rate", "grinder": ""},
                    color="defect_rate",
                    color_continuous_scale="Reds",
                ).update_layout(
                    coloraxis_showscale=False,
                    yaxis={"categoryorder": "total ascending"},
                ),
                use_container_width=True,
            )

        # Defect heatmap: grinder × parameter+side violation counts.
        if not vdf.empty:
            vv = vdf[vdf["grinder"] != "?"].copy()
            if not vv.empty:
                heat = (
                    vv.groupby(["grinder", "parameter"])
                    .size()
                    .reset_index(name="count")
                    .pivot(index="grinder", columns="parameter", values="count")
                    .fillna(0)
                )
                h2.plotly_chart(
                    px.imshow(
                        heat,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="Reds",
                        title="Spec-defect counts: grinder × parameter (side)",
                        labels={"x": "parameter", "y": "grinder", "color": "panels"},
                    ),
                    use_container_width=True,
                )
            else:
                h2.info("No spec defects attributed to a grinder in this window.")
        else:
            h2.info("No spec violations in this window.")

        # Defect breakdown by parameter and edge side, plus grinder location.
        if not vdf.empty and (vdf["grinder"] != "?").any():
            vv = vdf[vdf["grinder"] != "?"].copy()
            b1, b2 = st.columns(2)
            by_param = vv.groupby(["grinder", "base"]).size().reset_index(name="count")
            b1.plotly_chart(
                px.bar(
                    by_param,
                    x="grinder",
                    y="count",
                    color="base",
                    title="Defects by grinder & parameter",
                    labels={"count": "panels", "base": "parameter"},
                ),
                use_container_width=True,
            )
            by_side = vv.groupby(["grinder", "side"]).size().reset_index(name="count")
            b2.plotly_chart(
                px.bar(
                    by_side,
                    x="grinder",
                    y="count",
                    color="side",
                    title="Defects by grinder & edge side (Left/Right)",
                    labels={"count": "panels", "side": "edge"},
                    color_discrete_map={"Left": "#1f77b4", "Right": "#ff7f0e"},
                ),
                use_container_width=True,
            )

        st.divider()

        # -- Grinder defects: throughput + rejected, stacked by grinder --------
        st.markdown("#### Grinder throughput & rejects")
        g_left, g_right = st.columns(2)

        g_tp = m.groupby(["bucket", "grinder"]).size().reset_index(name="plates")
        g_left.plotly_chart(
            px.bar(
                g_tp,
                x="bucket",
                y="plates",
                color="grinder",
                title=f"Total throughput by grinder (per {freq_label})",
                labels={"bucket": freq_label, "plates": "plates"},
            ),
            use_container_width=True,
        )
        g_rej = (
            m[m["rejected"] == 1]
            .groupby(["bucket", "grinder"])
            .size()
            .reset_index(name="rejected")
        )
        if g_rej.empty:
            g_right.success("No rejected plates by grinder in this window.")
        else:
            g_right.plotly_chart(
                px.bar(
                    g_rej,
                    x="bucket",
                    y="rejected",
                    color="grinder",
                    title=f"Rejected panels by grinder (per {freq_label})",
                    labels={"bucket": freq_label, "rejected": "rejected"},
                ),
                use_container_width=True,
            )

        # -- Broken monitor: throughput + broken, by VTD line ------------------
        st.markdown("#### VTD throughput & broken panels")
        has_vtd = "vtd" in m and (m["vtd"] != "?").any()
        if not has_vtd:
            st.caption(
                "VTD lineage unavailable for these plates (no coater run records "
                "resolved). Broken-by-VTD charts appear once VTD data is present."
            )
        else:
            v = m[m["vtd"] != "?"].copy()
            v_left, v_right = st.columns(2)
            v_tp = v.groupby(["bucket", "vtd"]).size().reset_index(name="plates")
            v_left.plotly_chart(
                px.bar(
                    v_tp,
                    x="bucket",
                    y="plates",
                    color="vtd",
                    title=f"Total throughput by VTD (per {freq_label})",
                    labels={"bucket": freq_label, "plates": "plates"},
                ),
                use_container_width=True,
            )
            v_brk = (
                v[v["broken"] == 1]
                .groupby(["bucket", "vtd"])
                .size()
                .reset_index(name="broken")
            )
            if v_brk.empty:
                v_right.success("No break-risk panels by VTD in this window.")
            else:
                v_right.plotly_chart(
                    px.bar(
                        v_brk,
                        x="bucket",
                        y="broken",
                        color="vtd",
                        title=f"Break-risk panels by VTD (per {freq_label})",
                        labels={"bucket": freq_label, "broken": "panels"},
                    ),
                    use_container_width=True,
                )


# --------------------------------------------------------------------------
# FS100 Broken — post-VTD scrap, backtracked to the grinder that likely caused it
# --------------------------------------------------------------------------
with tab_broken:
    branding.section(
        "FS100 broken plates (post-VTD scrap)",
        "Traced back to grinder · edge · groove · detected defect",
    )
    if not st.session_state.get("load_fs100"):
        st.info("The FS100 broken feed is loaded on demand to keep the main view fast.")
        if st.button("Load FS100 broken data", type="primary"):
            st.session_state["load_fs100"] = True
            st.rerun()
    elif broken_df is None or broken_df.empty:
        st.success("No FS100-broken plates in this window.")
    else:
        st.caption(
            "Every plate scrapped as **Broken** at a VTD coater, traced back to "
            "the grinder line, edge (Long/Short), groove and any edge-grind "
            "defect that was already detected — so you can see *why* it broke."
        )
        b = broken_df.copy()
        n_broken = int(b["SubID"].nunique())
        n_explained = int(b["had_defect"].fillna(False).astype(bool).sum())
        n_new = n_broken - n_explained
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Broken plates", f"{n_broken:,}")
        k2.metric(
            "Explained by FS50 defect",
            f"{n_explained:,}",
            help="A run-length edge-grind defect was already flagged on the plate.",
        )
        k3.metric(
            "New failure modes",
            f"{n_new:,}",
            help="Broke with no prior detected defect — the model should learn these.",
        )
        top_g = (
            b["grinder"].dropna().value_counts().idxmax()
            if b["grinder"].notna().any()
            else "—"
        )
        k4.metric("Top grinder", top_g)

        # Commonality — where do breaks concentrate?
        st.markdown("#### Commonality")
        cc1, cc2, cc3 = st.columns(3)
        for col, title, container in (
            ("grinder", "By grinder", cc1),
            ("edge", "By edge (Long/Short)", cc2),
            ("side", "By side (Left/Right)", cc3),
        ):
            with container:
                st.caption(title)
                vc = b[col].dropna().value_counts()
                if vc.empty:
                    st.write("—")
                else:
                    st.bar_chart(vc, use_container_width=True)

        # Detail table
        st.markdown("#### Broken plate detail")
        disp = b.copy()
        disp["Broke at"] = pd.to_datetime(
            disp["BrokenTime"], errors="coerce"
        ).dt.strftime("%m-%d %H:%M")
        disp["Root cause"] = [
            (f"{d} ({s})" if pd.notna(d) else "— new failure mode —")
            for d, s in zip(disp["defect"], disp["side"])
        ]
        disp = disp.rename(
            columns={
                "SubID": "Plate",
                "VtdName": "VTD",
                "grinder": "Grinder",
                "edge": "Edge",
                "groove": "Groove",
            }
        )[["Broke at", "Plate", "VTD", "Grinder", "Edge", "Groove", "Root cause"]]
        st.dataframe(disp, use_container_width=True, hide_index=True)

        if n_new > 0:
            st.info(
                f"**{n_new}** plate(s) broke without a prior edge-grind defect. "
                "Retrain the break-risk model (Alerts tab → *Retrain now*, or the "
                "nightly job) so it learns these grinder signatures."
            )


# --------------------------------------------------------------------------
# Live Feed — recent defect gallery with boxes
# --------------------------------------------------------------------------
with tab_live:
    st.subheader("Most recent inspections")
    only_defects = st.checkbox("Show defects only", value=True)
    feed = df.copy()
    if only_defects:
        feed = feed[feed["decision"] == "REJECT"]
    feed = feed.head(12)

    if feed.empty:
        st.info("Nothing to show for the current filter.")
    else:
        cols = st.columns(4)
        for i, (_, row) in enumerate(feed.iterrows()):
            with cols[i % 4]:
                img = row["annotated_path"]
                if img and Path(img).exists():
                    st.image(img, use_container_width=True)
                badge = "🟥 REJECT" if row["decision"] == "REJECT" else "🟩 PASS"
                st.markdown(
                    f"**{badge}** · conf {row['max_conf']:.0%}\n\n"
                    f"📍 {row['location']} · Part `{row['part_id']}` · "
                    f"Corner {row['corner']}\n\n"
                    f"🕒 {row['ts']:%H:%M:%S}"
                )


# --------------------------------------------------------------------------
# Locations — breakdown + heatmap
# --------------------------------------------------------------------------
with tab_locations:
    if df.empty:
        st.info("No data yet.")
    else:
        grp = (
            df.assign(reject=(df["decision"] == "REJECT").astype(int))
            .groupby("location")
            .agg(inspected=("id", "count"), defects=("reject", "sum"))
            .reset_index()
        )
        grp["defect_rate_%"] = (100.0 * grp["defects"] / grp["inspected"]).round(1)
        st.dataframe(grp, use_container_width=True, hide_index=True)

        heat = df.copy()
        heat["hour"] = heat["ts"].dt.floor("h")
        heat["reject"] = (heat["decision"] == "REJECT").astype(int)
        pivot = heat.pivot_table(
            index="location",
            columns="hour",
            values="reject",
            aggfunc="sum",
            fill_value=0,
        )
        if not pivot.empty:
            fig_heat = px.imshow(
                pivot,
                aspect="auto",
                color_continuous_scale="Reds",
                title="Defects by location over time (heatmap)",
                labels={"color": "defects"},
            )
            st.plotly_chart(fig_heat, use_container_width=True)


# --------------------------------------------------------------------------
# Metrology — edge grind profile (numeric) inspection
# --------------------------------------------------------------------------
with tab_metrology:
    st.subheader("Edge Grind Profile — metrology inspection")
    src = "MOCK profiles" if config.use_mock_metrology() else "Live SQL ODS"
    st.caption(
        f"Source: **{src}** · table `ProcessData.ProcessHistory.EGPData` · "
        f"profiler `{config.METROLOGY_PROFILER}`. Each plate (SubID) is scored "
        "against spec limits (Stage 1) plus an ML anomaly check (Stage 2)."
    )
    st.caption(
        "Window, lineage filters and data source are set in the sidebar "
        "(**Dashboard time range** + ⚙️ **Monitor settings**)."
    )

    if mdf.empty:
        st.info("No plates found in the selected window.")
    else:
        # --- Filter + sort by grinder lineage and/or grind time ---
        fc1, fc2, fc3 = st.columns([2, 2, 2])
        grinders = sorted(mdf["grinder"].dropna().unique())
        pick = fc1.multiselect(
            "Grinder lineage", grinders, default=grinders, key="metro_grinder"
        )

        has_gt = "grind_time" in mdf and mdf["grind_time"].notna().any()
        gt_range = None
        if has_gt:
            gmin = pd.to_datetime(mdf["grind_time"]).min().to_pydatetime()
            gmax = pd.to_datetime(mdf["grind_time"]).max().to_pydatetime()
            if gmin < gmax:
                gt_range = fc2.slider(
                    "Grind time range",
                    min_value=gmin,
                    max_value=gmax,
                    value=(gmin, gmax),
                    format="MM/DD HH:mm",
                )
            else:
                fc2.caption(f"Grind time: {gmin:%Y-%m-%d %H:%M}")
        else:
            fc2.caption("Grind time: not available")

        sort_by = fc3.selectbox(
            "Sort by",
            [
                "Grind time (newest)",
                "Grind time (oldest)",
                "Grinder lineage",
                "Scan time (newest)",
            ],
        )

        view = mdf[mdf["grinder"].isin(pick)] if pick else mdf
        if gt_range is not None:
            lo, hi = pd.Timestamp(gt_range[0]), pd.Timestamp(gt_range[1])
            gtv = pd.to_datetime(view["grind_time"])
            view = view[(gtv >= lo) & (gtv <= hi) | gtv.isna()]
        vview = vdf[vdf["grinder"].isin(pick)] if (pick and not vdf.empty) else vdf

        _sort_map = {
            "Grind time (newest)": ("grind_time", False),
            "Grind time (oldest)": ("grind_time", True),
            "Grinder lineage": ("grinder", True),
            "Scan time (newest)": ("read_time", False),
        }
        _col, _asc = _sort_map[sort_by]
        if _col in view.columns:
            view = view.sort_values(_col, ascending=_asc, na_position="last")

        # KPI tiles.
        total = len(view)
        rej = int((view["verdict"] == "REJECT").sum())
        passes = total - rej
        pass_rate = (100.0 * passes / total) if total else 0.0
        ml_only = int(view["ml_flag"].sum()) if "ml_flag" in view else 0
        top_param = (
            vview["parameter"].value_counts().idxmax() if not vview.empty else "—"
        )
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Plates", f"{total:,}")
        k2.metric("Pass rate", f"{pass_rate:.1f}%")
        k3.metric("Rejected", f"{rej:,}")
        k4.metric("ML-only flags", f"{ml_only:,}")
        k5.metric("Top OOS param", top_param)

        left, right = st.columns(2)

        by_v = view["verdict"].value_counts().reset_index()
        by_v.columns = ["verdict", "count"]
        left.plotly_chart(
            px.pie(
                by_v,
                names="verdict",
                values="count",
                color="verdict",
                color_discrete_map={"PASS": "#2e7d32", "REJECT": "#d93025"},
                title="Plate verdicts",
                hole=0.5,
            ),
            use_container_width=True,
        )

        if not vview.empty:
            pc = vview["parameter"].value_counts().reset_index()
            pc.columns = ["parameter", "count"]
            right.plotly_chart(
                px.bar(
                    pc,
                    x="count",
                    y="parameter",
                    orientation="h",
                    title="Out-of-spec parameters (frequency)",
                    color="count",
                    color_continuous_scale="Reds",
                ),
                use_container_width=True,
            )
        else:
            right.info("No spec violations in this window.")

        # Per-grinder breakdown.
        grp = (
            view.assign(reject=(view["verdict"] == "REJECT").astype(int))
            .groupby("grinder")
            .agg(plates=("sub_id", "count"), rejected=("reject", "sum"))
            .reset_index()
        )
        grp["reject_rate_%"] = (100.0 * grp["rejected"] / grp["plates"]).round(1)
        st.markdown("**Per-grinder breakdown**")
        gb1, gb2 = st.columns([1, 1])
        gb1.dataframe(grp, use_container_width=True, hide_index=True)
        gb2.plotly_chart(
            px.bar(
                grp,
                x="grinder",
                y="reject_rate_%",
                title="Reject rate by grinder",
                color="reject_rate_%",
                color_continuous_scale="Reds",
            ),
            use_container_width=True,
        )

        # Grinder break-risk monitor (Stage 2 supervised model).
        branding.section("Grinder break-risk monitor", "ML model scoring")
        _model_live = config.METROLOGY_MODEL_PATH.exists()
        if not _model_live:
            st.info(
                "Supervised break-risk model not found — scores below are "
                "unsupervised anomaly scores. Train the model with "
                "`python scripts/train_metrology.py` to predict downstream breaks."
            )
        else:
            st.caption(
                "Predicts panels that pass the EGP spec rules but are likely to "
                "break downstream (sub-threshold defects), and flags grinders "
                "shipping too many of them."
            )

        risk = view.copy()
        risk["break_risk"] = pd.to_numeric(risk.get("anomaly_score"), errors="coerce")
        risk = risk.dropna(subset=["break_risk"])
        if risk.empty:
            st.caption("No break-risk scores in this window (enable Stage 2 ML).")
        else:
            risk_thresh = st.slider(
                "Break-risk flag threshold",
                min_value=0.05,
                max_value=0.95,
                value=0.65,
                step=0.05,
                help="Panels scoring at/above this are counted as high break-risk.",
            )
            risk["high_risk"] = (risk["break_risk"] >= risk_thresh).astype(int)

            gr = (
                risk.groupby("grinder")
                .agg(
                    panels=("sub_id", "count"),
                    avg_risk=("break_risk", "mean"),
                    high_risk=("high_risk", "sum"),
                )
                .reset_index()
            )
            gr["high_risk_%"] = (100.0 * gr["high_risk"] / gr["panels"]).round(1)
            gr["avg_risk"] = gr["avg_risk"].round(3)
            gr = gr.sort_values("high_risk_%", ascending=False)

            # Alert on grinders whose high-risk share exceeds the alert line.
            alert_pct = st.slider(
                "Alert when a grinder's high-risk share exceeds (%)",
                min_value=5,
                max_value=80,
                value=30,
                step=5,
            )
            offenders = gr[gr["high_risk_%"] >= alert_pct]
            if not offenders.empty:
                for _, o in offenders.iterrows():
                    st.error(
                        f"🚨 **{o['grinder']}** — {o['high_risk_%']:.0f}% of "
                        f"{int(o['panels'])} panels are high break-risk "
                        f"(avg {o['avg_risk']:.2f}). Investigate this grinder.",
                    )
            else:
                st.success("No grinder exceeds the high break-risk alert line. ✅")

            mon1, mon2 = st.columns([1, 1])
            mon1.dataframe(gr, use_container_width=True, hide_index=True)
            mon2.plotly_chart(
                px.bar(
                    gr,
                    x="grinder",
                    y="high_risk_%",
                    title="High break-risk share by grinder",
                    color="high_risk_%",
                    color_continuous_scale="Oranges",
                ),
                use_container_width=True,
            )

            # Break-risk trend over grind time (spot a grinder drifting).
            if "grind_time" in risk and risk["grind_time"].notna().any():
                trend = risk.dropna(subset=["grind_time"]).sort_values("grind_time")
                st.plotly_chart(
                    px.scatter(
                        trend,
                        x="grind_time",
                        y="break_risk",
                        color="grinder",
                        hover_data=["sub_id"],
                        title="Break-risk over grind time",
                    ).add_hline(
                        y=risk_thresh,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="flag threshold",
                    ),
                    use_container_width=True,
                )

        st.markdown("**Rejected plates**")
        rejects = view[view["verdict"] == "REJECT"].copy()
        if rejects.empty:
            st.success("No rejected plates in this window. ✅")
        else:
            cols = [
                "grind_time",
                "read_time",
                "sub_id",
                "grinder",
                "out_of_spec",
                "reason",
                "anomaly_score",
                "n_positions",
            ]
            cols = [c for c in cols if c in rejects.columns]
            st.dataframe(
                rejects[cols],
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "⬇️ Export rejected plates (CSV)",
                rejects[cols].to_csv(index=False).encode("utf-8"),
                file_name="metrology_rejects.csv",
                mime="text/csv",
            )

        with st.expander("Active spec limits (Stage 1)"):
            spec_rows = [
                {
                    "parameter": base,
                    "LSL": lsl,
                    "USL": usl,
                }
                for base, (lsl, usl) in config.METROLOGY_SPECS.items()
            ]
            st.dataframe(
                pd.DataFrame(spec_rows), use_container_width=True, hide_index=True
            )
            st.caption(
                "Limits are provisional (from the JMP graph constants). Replace "
                "with confirmed USL/LSL via config.METROLOGY_SPECS."
            )


# --------------------------------------------------------------------------
# History — filterable table + export
# --------------------------------------------------------------------------
with tab_history:
    if df.empty:
        st.info("No inspections recorded yet.")
    else:
        f1, f2, f3 = st.columns(3)
        loc_filter = f1.multiselect(
            "Location", sorted(df["location"].dropna().unique())
        )
        dec_filter = f2.multiselect("Decision", ["PASS", "REJECT"])
        search = f3.text_input("Search part id")

        view = df.copy()
        if loc_filter:
            view = view[view["location"].isin(loc_filter)]
        if dec_filter:
            view = view[view["decision"].isin(dec_filter)]
        if search:
            view = view[view["part_id"].astype(str).str.contains(search, case=False)]

        show_cols = [
            "ts",
            "location",
            "part_id",
            "corner",
            "decision",
            "defect_count",
            "max_conf",
            "classes",
            "image_name",
        ]
        st.dataframe(view[show_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Export CSV",
            view[show_cols].to_csv(index=False).encode("utf-8"),
            file_name="inspection_history.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------
# Alert Rules — configurable + alert log
# --------------------------------------------------------------------------
with tab_alerts:
    st.subheader("Alert rules")
    st.caption(
        "Rules are evaluated per location after each inspection. A fired rule "
        "posts to Teams (if a webhook is configured) and is logged below."
    )

    # --- Manual actions: push alerts to Teams + retrain the break-risk model.
    st.markdown("#### 📨 Teams + model actions")
    ma1, ma2 = st.columns(2)
    with ma1:
        st.caption(
            "Fires the FS50 burst rule (≥3 defects within 15 min on a grinder) "
            "and one alert per FS100-broken plate. Respects cooldown."
        )
        if st.button("📨 Send alerts to Teams now", use_container_width=True):
            if not config.TEAMS_WEBHOOK_URL:
                st.warning("No Teams webhook configured (set FS50_TEAMS_WEBHOOK).")
            else:
                sent = _dispatch_teams_alerts()
                if not sent:
                    st.info("Nothing to send — no burst / break, or all on cooldown.")
                else:
                    ok = sum(1 for f in sent if f.get("delivered"))
                    st.success(f"Sent {ok}/{len(sent)} alert(s) to Teams.")
                    for f in sent:
                        st.write(("✅ " if f.get("delivered") else "⚠️ ") + f["rule"])
    with ma2:
        st.caption(
            "Retrain the Stage-2 break-risk model on the latest FS100-broken "
            "plates vs good plates. A nightly job can call the same script."
        )
        if st.button("🧠 Retrain break-risk model now", use_container_width=True):
            with st.spinner("Retraining on FS100-broken labels…"):
                try:
                    result = metrology.retrain_break_model(
                        win_start, win_end, mock=bool(force_mock)
                    )
                    load_metrology.clear()
                    st.success(result)
                except Exception as e:
                    st.error(f"Retrain failed: {e}")
    st.divider()

    rules = alerts.load_rules()
    updated: list[dict] = []
    for i, rule in enumerate(rules):
        with st.expander(
            f"{'🟢' if rule.get('enabled') else '⚪'} {rule['name']} ({rule['type']})",
            expanded=False,
        ):
            enabled = st.checkbox(
                "Enabled", value=rule.get("enabled", False), key=f"en_{i}"
            )
            cooldown = st.number_input(
                "Cooldown (min)",
                min_value=0,
                value=int(rule.get("cooldown_minutes", 15)),
                key=f"cd_{i}",
            )
            new_rule = {
                "name": rule["name"],
                "type": rule["type"],
                "enabled": enabled,
                "cooldown_minutes": cooldown,
            }
            if rule["type"] == "count":
                new_rule["threshold"] = st.number_input(
                    "Reject count threshold",
                    min_value=1,
                    value=int(rule.get("threshold", 3)),
                    key=f"th_{i}",
                )
                new_rule["window_minutes"] = st.number_input(
                    "Window (min)",
                    min_value=1,
                    value=int(rule.get("window_minutes", 30)),
                    key=f"wn_{i}",
                )
            elif rule["type"] == "rate":
                new_rule["threshold_pct"] = st.number_input(
                    "Reject rate threshold (%)",
                    min_value=1.0,
                    max_value=100.0,
                    value=float(rule.get("threshold_pct", 25.0)),
                    key=f"rp_{i}",
                )
                new_rule["window_minutes"] = st.number_input(
                    "Window (min)",
                    min_value=1,
                    value=int(rule.get("window_minutes", 60)),
                    key=f"rw_{i}",
                )
                new_rule["min_samples"] = st.number_input(
                    "Min samples",
                    min_value=1,
                    value=int(rule.get("min_samples", 8)),
                    key=f"ms_{i}",
                )
            updated.append(new_rule)

    if st.button("💾 Save rules"):
        alerts.save_rules(updated)
        st.success("Rules saved.")

    if config.TEAMS_WEBHOOK_URL and st.button("Send test Teams alert"):
        ok = alerts.post_to_teams(
            "Test alert", "This is a test from the FS50 dashboard.", "Line-A"
        )
        st.success("Sent ✅") if ok else st.error("Failed to deliver.")

    st.divider()
    st.subheader("Alert log")
    alog = store.load_alerts()
    if alog.empty:
        st.info("No alerts fired yet.")
    else:
        alog["delivered"] = alog["delivered"].map({1: "✅", 0: "—"})
        st.dataframe(
            alog[["ts", "rule_name", "location", "message", "delivered"]],
            use_container_width=True,
            hide_index=True,
        )
