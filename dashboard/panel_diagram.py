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
* Corner acronyms LL / TL / TR / LR (Leading Left, Trailing Left, Trailing
  Right, Leading Right) map to the four rectangle corners via
  :data:`CORNER_LABELS`. The exact placement is process-defined and easy to
  re-map once the training material is supplied — only this one dict changes.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

import config

# Corner acronyms placed at each rectangle corner. Keys are (x_end, y_end) where
# 0 = start (min) and 1 = end (max) of each axis. Provisional mapping — adjust
# once the TR/LL/LR/TL training material is supplied; nothing else needs to
# change.
CORNER_LABELS: dict[tuple[int, int], str] = {
    (0, 0): "LL",  # Leading Left   (x min, y min)
    (1, 0): "LR",  # Leading Right  (x max, y min)
    (0, 1): "TL",  # Trailing Left  (x min, y max)
    (1, 1): "TR",  # Trailing Right (x max, y max)
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


def panel_figure(
    defects: pd.DataFrame | None,
    title: str,
    long_mm: float | None = None,
    short_mm: float | None = None,
    height: int = 230,
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
    # Corner labels.
    for (xe, ye), label in CORNER_LABELS.items():
        fig.add_annotation(
            x=(long_mm - 60) if xe else 60,
            y=(short_mm - 40) if ye else 40,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=11, color="#8595AB"),
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

    fig.update_xaxes(range=[-80, long_mm + 80], visible=False)
    fig.update_yaxes(
        range=[-60, short_mm + 60],
        visible=False,
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#E7EEF7"), x=0.02, y=0.97),
        height=height,
        margin=dict(l=6, r=6, t=28, b=6),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E7EEF7"),
    )
    return fig
