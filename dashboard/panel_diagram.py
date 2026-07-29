"""Visual panel-diagram helpers for the FS50 landing page.

Renders a solar-module rectangle (Long edge x Short edge) per grinder with the
four corners labelled and every detected edge-grind defect plotted at its
position along the panel perimeter. Colour = defect type; a suspected
skewed/out-of-focus reading is drawn hollow.

Geometry / labelling notes
--------------------------
* Process order: a plate goes through GRINDER 1 (LONG edges), rotates, then
  GRINDER 2 (SHORT edges), rotates again, then the EGP profiler.
* EGP reports every defect with an ``edge`` (Long/Short), a ``side``
  (Left/Right) and a position (``pos_x`` along the long axis, ``pos_y`` along
  the short axis). We map that to the panel outline below.
* Corners are numbered 1-4 so ops can name a defect location off the drawing.
  Each corner also carries the process reference (which Short-Edge/Long-Edge
  grinder groove position it maps to). Layout::

        x=0                                   x=2300
   y=1215  4 ─────────── top long edge ─────────── 3
           │  (Left long edge · Grinder 1)         │
     right │                                       │ left
     short │                                       │ short
      edge │                                       │ edge
      (G2) │  (Right long edge · Grinder 1)        │ (G2)
   y=0     1 ─────────── bottom long edge ──────── 2
            ▪ SubID  (500 mm from corner 1)

* Long edge is the X axis (0 → 2300, corner 4→3 across the top / 1→2 bottom).
* Short edge is the Y axis (0 → 1215, corner 1→4 left / 2→3 right).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

import config

# Numbered corners + the process reference each corner maps to. Keys are
# (x_end, y_end) where 0 = start (min) and 1 = end (max) of each axis.
#   SE = Short Edge (Grinder 2), LE = Long Edge (Grinder 1).
CORNER_LABELS: dict[tuple[int, int], dict[str, str]] = {
    (0, 0): {"num": "1", "sub": "SE→G2 LT | LE→G1 RT"},  # bottom-left
    (1, 0): {"num": "2", "sub": "SE→G2 RT | LE→G1 RL"},  # bottom-right
    (1, 1): {"num": "3", "sub": "SE→G2 RL | LE→G1 LL"},  # top-right
    (0, 1): {"num": "4", "sub": "SE→G2 LL | LE→G1 LT"},  # top-left
}

# Where the SubID is physically marked: on the long (bottom) edge, 500 mm from
# corner 1.
SUBID_POS_MM = 500.0

# Fixed column order for the corner heatmap — 4 numbered corners x 2 edges
# (Long / Short) = 8 possible defect slots per panel. Each defect is attributed
# to whichever corner it sits closest to along its own edge, then reported as
# the grinder-groove position that corner maps to (SE→Grinder 2, LE→Grinder 1).
CORNER_SLOTS = [
    "G2 LL",
    "G2 LT",
    "G2 RL",
    "G2 RT",
    "G1 LT",
    "G1 RT",
    "G1 LL",
    "G1 RL",
]

# (corner number, edge) -> grinder-groove label, per CORNER_LABELS above.
#   Short edge = Grinder 2, Long edge = Grinder 1.
_SLOT_GROOVE = {
    (1, "short"): "G2 LT",
    (1, "long"): "G1 RT",
    (2, "short"): "G2 RT",
    (2, "long"): "G1 RL",
    (3, "short"): "G2 RL",
    (3, "long"): "G1 LL",
    (4, "short"): "G2 LL",
    (4, "long"): "G1 LT",
}

# Defect palette (high-contrast on the dark canvas).
DEFECT_COLORS = {
    "Chip": "#FF2E4D",  # razor red
    "Dropout": "#F5B301",  # amber
    "Shiner": "#38BDF8",  # electric cyan
}
DEFECT_SYMBOL = {"Chip": "x", "Dropout": "square", "Shiner": "diamond"}


def _defect_xy(row: pd.Series, long_mm: float, short_mm: float) -> tuple[float, float]:
    """Map one defect (edge/side/pos) to an (x, y) point on the panel outline.

    * A LONG-edge defect lies along the top or bottom rail: x = pos_x, y pinned
      to the Left/Right rail. * A SHORT-edge defect lies along the left or right
      rail: y = pos_y, x pinned. ``side`` (Left/Right) selects which rail.
    """
    edge = str(row.get("edge") or "").lower()
    side = str(row.get("side") or "").lower()
    px = row.get("pos_x")
    py = row.get("pos_y")

    if edge.startswith("long"):
        x = float(px) if pd.notna(px) else long_mm / 2.0
        x = min(max(x, 0.0), long_mm)
        y = 0.0 if side == "right" else short_mm  # Right rail = bottom
    else:  # short edge
        y = float(py) if pd.notna(py) else short_mm / 2.0
        y = min(max(y, 0.0), short_mm)
        x = 0.0 if side == "left" else long_mm  # Left rail = x-start
    return x, y


def defect_corner_slot(
    row: pd.Series,
    long_mm: float | None = None,
    short_mm: float | None = None,
) -> str | None:
    """Attribute one defect to its nearest corner slot (one of CORNER_SLOTS).

    A defect lies on a Long or a Short edge; whichever end of that edge its
    position is closer to picks the corner number (1-4). The corner+edge is then
    reported as the grinder-groove label it maps to (e.g. ``"G1 RT"``). Returns
    ``None`` if the edge is unknown.
    """
    long_mm = float(long_mm or config.PANEL_LONG_MM)
    short_mm = float(short_mm or config.PANEL_SHORT_MM)
    edge = str(row.get("edge") or "").lower()
    side = str(row.get("side") or "").lower()

    if edge.startswith("long"):
        px = row.get("pos_x")
        px = float(px) if pd.notna(px) else long_mm / 2.0
        near_start = px < long_mm / 2.0  # closer to x = 0
        if side == "right":  # bottom rail: C1 (x0) .. C2 (xmax)
            corner = 1 if near_start else 2
        else:  # top rail: C4 (x0) .. C3 (xmax)
            corner = 4 if near_start else 3
        return _SLOT_GROOVE[(corner, "long")]
    if edge.startswith("short"):
        py = row.get("pos_y")
        py = float(py) if pd.notna(py) else short_mm / 2.0
        near_start = py < short_mm / 2.0  # closer to y = 0
        if side == "left":  # x=0 rail: C1 (y0) .. C4 (ymax)
            corner = 1 if near_start else 4
        else:  # x=max rail: C2 (y0) .. C3 (ymax)
            corner = 2 if near_start else 3
        return _SLOT_GROOVE[(corner, "short")]
    return None


def panel_figure(
    defects: pd.DataFrame | None,
    title: str,
    long_mm: float | None = None,
    short_mm: float | None = None,
    height: int = 340,
) -> go.Figure:
    """Build a module-outline figure with the corner labels and defect markers.

    ``defects`` may be ``None``/empty (draws a clean panel). Otherwise it must
    have columns edge/side/pos_x/pos_y/defect (+ optional suspect_focus).
    """
    long_mm = float(long_mm or config.PANEL_LONG_MM)
    short_mm = float(short_mm or config.PANEL_SHORT_MM)

    fig = go.Figure()
    # Panel body.
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=long_mm,
        y1=short_mm,
        line=dict(color="#38BDF8", width=2),
        fillcolor="rgba(56,189,248,0.06)",
        layer="below",
    )
    # Corner numbers (1-4) + process reference sub-label, placed just outside
    # each corner so ops can name a defect location straight off the drawing.
    x_out = long_mm * 0.06
    y_out = short_mm * 0.13
    for (xe, ye), info in CORNER_LABELS.items():
        # Big corner number, sitting inside the panel corner.
        fig.add_annotation(
            x=(long_mm - x_out) if xe else x_out,
            y=(short_mm - y_out) if ye else y_out,
            text=f"<b>{info['num']}</b>",
            showarrow=False,
            font=dict(size=22, color="#E7EEF7"),
            bgcolor="rgba(56,189,248,0.18)",
            bordercolor="#38BDF8",
            borderwidth=1,
            borderpad=4,
        )
        # Process reference, just outside the panel (above top / below bottom).
        fig.add_annotation(
            x=(long_mm - x_out) if xe else x_out,
            y=(short_mm + y_out * 1.3) if ye else (-y_out * 1.3),
            text=info["sub"],
            showarrow=False,
            font=dict(size=10, color="#FFFFFF"),
            xanchor="center",
        )

    # SubID marker: a small labelled box on the bottom long edge, ~80 mm from
    # corner 1, matching where the plate is physically marked.
    box_w = long_mm * 0.05
    box_h = short_mm * 0.09
    fig.add_shape(
        type="rect",
        x0=SUBID_POS_MM - box_w / 2,
        x1=SUBID_POS_MM + box_w / 2,
        y0=0,
        y1=box_h,
        line=dict(color="#E7EEF7", width=1.4),
        fillcolor="rgba(231,238,247,0.14)",
        layer="above",
    )
    fig.add_annotation(
        x=SUBID_POS_MM,
        y=box_h / 2,
        text="SubID",
        showarrow=False,
        font=dict(size=9, color="#E7EEF7"),
    )

    # Defect markers.
    if defects is not None and not defects.empty:
        for defect, grp in defects.groupby("defect"):
            xs, ys, texts, open_flags = [], [], [], []
            for _, r in grp.iterrows():
                x, y = _defect_xy(r, long_mm, short_mm)
                xs.append(x)
                ys.append(y)
                sus = bool(r.get("suspect_focus", False))
                open_flags.append(sus)
                texts.append(
                    f"{defect} — {r.get('edge')} edge, {r.get('side')} side<br>"
                    f"run {r.get('length_mm', 0):.0f} mm"
                    f"{' ⚠ skewed' if sus else ''}<br>plate {r.get('sub_id')}"
                )
            color = DEFECT_COLORS.get(defect, "#666")
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    name=str(defect),
                    marker=dict(
                        size=12,
                        color=["rgba(0,0,0,0)" if o else color for o in open_flags],
                        symbol=DEFECT_SYMBOL.get(defect, "circle"),
                        line=dict(color=color, width=2),
                    ),
                    text=texts,
                    hoverinfo="text",
                )
            )

    # Long edge = X axis (0 -> long), short edge = Y axis (0 -> short). Show the
    # end-point ticks so ops can read position in mm straight off the drawing.
    fig.update_xaxes(
        range=[-long_mm * 0.05, long_mm * 1.05],
        visible=True,
        title=dict(
            text="Long edge (mm)",
            font=dict(size=9, color="#8595AB"),
        ),
        tickvals=[0, long_mm / 2, long_mm],
        ticktext=["0", f"{long_mm / 2:.0f}", f"{long_mm:.0f}"],
        tickfont=dict(size=9, color="#8595AB"),
        showgrid=False,
        zeroline=False,
        linecolor="#334155",
    )
    fig.update_yaxes(
        range=[-short_mm * 0.22, short_mm * 1.22],
        visible=True,
        title=dict(
            text="Short edge (mm)",
            font=dict(size=9, color="#8595AB"),
        ),
        tickvals=[0, short_mm / 2, short_mm],
        ticktext=["0", "615", "1230"],
        tickfont=dict(size=9, color="#8595AB"),
        showgrid=False,
        zeroline=False,
        linecolor="#334155",
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#E7EEF7"), x=0.02, y=0.97),
        height=height,
        margin=dict(l=6, r=6, t=28, b=30),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E7EEF7"),
    )
    return fig
