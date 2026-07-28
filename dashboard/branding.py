"""First Solar branding + shared UI theming for the FS50 dashboard.

Dark-mode only. All look-and-feel lives here: the logo path, an atmospheric
"engineering console" palette and a single ``inject_theme()`` that lays a
distinctive skin over Streamlit — deep IDE-dark canvas, a single razor-sharp
red accent, expressive Space Grotesk / JetBrains Mono type, glowing tactile
cards and a staggered page-load reveal.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent
LOGO_PATH = _HERE / "207-2070303_first-solar-logo-vector-first-solar-inc.png"

# --- Dark "engineering console" palette --------------------------------------
BG = "#0A0E17"  # near-black canvas
BG_2 = "#111725"  # raised surface / cards
BG_3 = "#0E1420"  # sunken panels
BORDER = "#1E2A3D"  # hairline dividers
TEXT = "#E7EEF7"  # primary text
MUTED = "#8595AB"  # secondary text
ACCENT = "#FF2E4D"  # razor-sharp red (on-dark pop of the FS brand red)
ACCENT_SOFT = "#FF6B80"
BLUE = "#38BDF8"  # electric cyan
AMBER = "#F5B301"
GREEN = "#34D399"

# Legacy aliases kept so existing imports keep working.
NAVY = TEXT
DEEP_BLUE = BLUE
GREY = MUTED

_FONTS = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Space+Grotesk:wght@400;500;600;700&"
    "family=JetBrains+Mono:wght@400;500;700&display=swap');"
)

_ROOT = f"""
:root {{
  --bg: {BG}; --bg2: {BG_2}; --bg3: {BG_3}; --border: {BORDER};
  --text: {TEXT}; --muted: {MUTED};
  --accent: {ACCENT}; --accent-soft: {ACCENT_SOFT};
  --blue: {BLUE}; --amber: {AMBER}; --green: {GREEN};
  --font-display: 'Space Grotesk', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, monospace;
}}
"""

_CSS_BODY = r"""
/* ---- Canvas: atmospheric dark with subtle radial glow ------------------ */
.stApp, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(900px 500px at 82% -8%, rgba(255,46,77,0.10), transparent 60%),
    radial-gradient(760px 520px at -6% 4%, rgba(56,189,248,0.08), transparent 55%),
    var(--bg);
  color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.1rem; padding-bottom: 2.4rem; max-width: 1520px; }

html, body, [class*="css"], .stMarkdown, p, span, label, div {
  font-family: var(--font-display);
}

/* ---- Typography: bold, tight display headers --------------------------- */
h1, h2, h3, h4 {
  font-family: var(--font-display);
  color: var(--text);
  font-weight: 700;
  letter-spacing: -0.02em;
}
h1 { font-size: 2rem; }
a { color: var(--blue); }

/* ---- Staggered page-load reveal ---------------------------------------- */
@keyframes fsRise {
  from { opacity: 0; transform: translateY(14px); }
  to   { opacity: 1; transform: translateY(0); }
}
.block-container > div > div > div[data-testid="stVerticalBlock"] > div {
  animation: fsRise 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
}
.block-container > div > div > div[data-testid="stVerticalBlock"] > div:nth-child(1){animation-delay:.02s}
.block-container > div > div > div[data-testid="stVerticalBlock"] > div:nth-child(2){animation-delay:.06s}
.block-container > div > div > div[data-testid="stVerticalBlock"] > div:nth-child(3){animation-delay:.10s}
.block-container > div > div > div[data-testid="stVerticalBlock"] > div:nth-child(4){animation-delay:.14s}
.block-container > div > div > div[data-testid="stVerticalBlock"] > div:nth-child(5){animation-delay:.18s}
.block-container > div > div > div[data-testid="stVerticalBlock"] > div:nth-child(6){animation-delay:.22s}

/* ---- Metric cards: glass, glowing accent rail -------------------------- */
[data-testid="stMetric"] {
  position: relative;
  background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0));
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 18px 14px 20px;
  overflow: hidden;
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}
[data-testid="stMetric"]::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
  background: linear-gradient(180deg, var(--accent), var(--accent-soft));
  box-shadow: 0 0 14px 1px rgba(255,46,77,0.55);
}
[data-testid="stMetric"]:hover {
  transform: translateY(-3px);
  border-color: rgba(255,46,77,0.45);
  box-shadow: 0 10px 30px -12px rgba(255,46,77,0.35);
}
[data-testid="stMetricValue"] {
  color: var(--text); font-family: var(--font-mono);
  font-weight: 700; font-size: 1.7rem; letter-spacing: -0.02em;
}
[data-testid="stMetricLabel"] {
  color: var(--muted); font-family: var(--font-mono);
  text-transform: uppercase; font-size: 0.68rem; letter-spacing: 0.14em;
}

/* ---- Tabs: mono labels, glowing active underline ----------------------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] {
  background: transparent; border-radius: 9px 9px 0 0;
  color: var(--muted); font-family: var(--font-mono);
  font-weight: 500; font-size: 0.82rem; letter-spacing: 0.02em; padding: 9px 15px;
  transition: color .15s ease, background .15s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text); background: rgba(255,255,255,0.03); }
.stTabs [aria-selected="true"] {
  color: var(--text);
  border-bottom: 2px solid var(--accent);
  text-shadow: 0 0 18px rgba(255,46,77,0.45);
}

/* ---- Section band ------------------------------------------------------ */
.fs-band {
  display: flex; align-items: baseline; gap: 12px; margin: 20px 0 12px;
  padding: 2px 0 2px 14px;
  border-left: 3px solid var(--accent);
}
.fs-band .fs-title {
  font-family: var(--font-display); font-size: 1.12rem; font-weight: 700;
  color: var(--text); letter-spacing: -0.01em;
}
.fs-band .fs-sub {
  font-family: var(--font-mono); font-size: 0.72rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.12em;
}

/* ---- Header bar -------------------------------------------------------- */
.fs-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 4px 0 14px; border-bottom: 1px solid var(--border); margin-bottom: 8px;
}
.fs-header .fs-app {
  font-family: var(--font-display); font-size: 1.5rem; font-weight: 700;
  color: var(--text); letter-spacing: -0.03em;
}
.fs-header .fs-app b { color: var(--accent); }
.fs-header .fs-tag {
  font-family: var(--font-mono); font-size: 0.74rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.16em; margin-top: 3px;
}

/* ---- Sidebar ----------------------------------------------------------- */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--bg3), var(--bg));
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { font-family: var(--font-display); }

/* ---- Buttons: tactile, glowing ----------------------------------------- */
.stButton > button {
  border-radius: 10px; font-family: var(--font-mono); font-weight: 500;
  letter-spacing: 0.03em; border: 1px solid var(--border);
  background: var(--bg2); color: var(--text);
  transition: transform .12s ease, box-shadow .18s ease, border-color .18s ease;
}
.stButton > button:hover {
  border-color: var(--accent);
  box-shadow: 0 0 0 1px rgba(255,46,77,0.4), 0 8px 22px -10px rgba(255,46,77,0.5);
  transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0) scale(0.99); }
.stButton > button[kind="primary"] {
  background: linear-gradient(180deg, var(--accent), #D91E38);
  border-color: transparent; color: #fff;
  box-shadow: 0 8px 24px -10px rgba(255,46,77,0.7);
}

/* ---- Inputs / tables / misc ------------------------------------------- */
[data-testid="stDataFrame"], [data-baseweb="input"], [data-baseweb="select"] {
  border-radius: 10px;
}
[data-testid="stExpander"] {
  border: 1px solid var(--border); border-radius: 12px; background: var(--bg2);
}
hr { border-color: var(--border); }
.stAlert { border-radius: 12px; }
::selection { background: rgba(255,46,77,0.3); }
"""

_CSS = f"<style>{_FONTS}{_ROOT}{_CSS_BODY}</style>"


def inject_theme() -> None:
    """Apply the shared dark CSS skin. Call once, right after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def header(
    app_name: str = "FS50 Corner & Edge Metrology",
    tagline: str = "Live grinder health · defect attribution · break-risk",
) -> None:
    """Render the branded top bar with the First Solar logo."""
    c_logo, c_title = st.columns([1, 6], vertical_alignment="center")
    with c_logo:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=150)
    with c_title:
        st.markdown(
            f'<div class="fs-header"><div>'
            f'<div class="fs-app">{app_name}</div>'
            f'<div class="fs-tag">{tagline}</div>'
            f"</div></div>",
            unsafe_allow_html=True,
        )


def section(title: str, sub: str = "") -> None:
    """Render a section band (replaces emoji-heavy markdown headers)."""
    sub_html = f'<span class="fs-sub">{sub}</span>' if sub else ""
    st.markdown(
        f'<div class="fs-band"><span class="fs-title">{title}</span>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def sidebar_logo() -> None:
    """Place the logo at the top of the sidebar."""
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width=160)
