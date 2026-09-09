"""Reusable display components.  No science here - only presentation.

The one idea worth stating: every volume is shown as three orthogonal
sections through a **shared cursor in metres** (spec section 123).  Sharing
a physical cursor rather than an index is what lets the geological model,
the property volumes and the migrated image - which live on different grids
at different spacings - be inspected at the same place.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..core.grid import Grid3D
from . import theme


def nearest(axis: np.ndarray, value: float) -> int:
    """Index of the closest node on a coordinate axis."""
    return int(np.argmin(np.abs(axis - value)))


def _limits(volume: np.ndarray, kind: str, robust: bool = True):
    """Colour limits: symmetric about zero for signed fields, always.

    ``robust`` clips to the 0.5-99.5 percentile so one hot cell cannot
    flatten the whole display, which on a migrated image is the difference
    between seeing the reservoir and seeing a single bright artefact.
    """
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return -1.0, 1.0
    if kind == "diverging":
        peak = np.percentile(np.abs(finite), 99.5) if robust else np.max(np.abs(finite))
        peak = float(peak) or 1.0
        return -peak, peak
    if robust:
        return float(np.percentile(finite, 0.5)), float(np.percentile(finite, 99.5))
    return float(finite.min()), float(finite.max())


def slice_figure(volume: np.ndarray, grid: Grid3D, cursor, *, title: str,
                 kind: str = "sequential", unit: str = "", scale: float = 1.0,
                 wells=None, height: int = 340, robust: bool = True) -> go.Figure:
    """Three orthogonal sections through ``cursor`` (x, y, z in metres).

    Panels are, left to right: an inline at the cursor's y, a crossline at
    its x, and a depth slice at its z.  Depth increases downwards on the
    two sections.
    """
    x, y, z = grid.axis(0), grid.axis(1), grid.axis(2)
    ix, iy, iz = nearest(x, cursor[0]), nearest(y, cursor[1]), nearest(z, cursor[2])
    data = np.asarray(volume, dtype=float) / scale
    lo, hi = _limits(data, kind, robust)
    colourscale = theme.DIVERGING if kind == "diverging" else theme.SEQUENTIAL

    fig = make_subplots(
        rows=1, cols=3, horizontal_spacing=0.07,
        subplot_titles=(f"inline · y = {y[iy]:,.0f} m",
                        f"crossline · x = {x[ix]:,.0f} m",
                        f"depth slice · z = {z[iz]:,.0f} m"),
    )
    panels = [
        (data[:, iy, :].T, x, z, "x (m)", "z (m)", True),
        (data[ix, :, :].T, y, z, "y (m)", "z (m)", True),
        (data[:, :, iz].T, x, y, "x (m)", "y (m)", False),
    ]
    for col, (plane, ax0, ax1, xlabel, ylabel, flip) in enumerate(panels, start=1):
        fig.add_trace(
            go.Heatmap(z=plane, x=ax0, y=ax1, zmin=lo, zmax=hi,
                       colorscale=colourscale, showscale=(col == 3),
                       colorbar=dict(title=unit, thickness=12, len=0.9,
                                     tickfont=dict(color=theme.INK_MUTED)),
                       hovertemplate=f"{xlabel[0]}=%{{x:,.0f}} m<br>"
                                     f"{ylabel[0]}=%{{y:,.0f}} m<br>"
                                     f"%{{z:,.4g}} {unit}<extra></extra>"),
            row=1, col=col)
        fig.update_xaxes(title_text=xlabel, row=1, col=col)
        fig.update_yaxes(title_text=ylabel, autorange="reversed" if flip else True,
                         row=1, col=col)

    # Cursor cross-hairs, so the three panels visibly agree on one point.
    for col, (h, v) in enumerate([(x[ix], z[iz]), (y[iy], z[iz]), (x[ix], y[iy])], 1):
        fig.add_vline(x=h, line=dict(color=theme.INK_MUTED, width=1, dash="dot"),
                      row=1, col=col)
        fig.add_hline(y=v, line=dict(color=theme.INK_MUTED, width=1, dash="dot"),
                      row=1, col=col)

    if wells is not None:
        for role in ("injector", "producer", "observation"):
            group = [w for w in wells if w.role == role]
            if not group:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[w.x for w in group], y=[w.y for w in group],
                    mode="markers+text", text=[w.name for w in group],
                    textposition="top center",
                    textfont=dict(color=theme.INK_SECONDARY, size=10),
                    marker=dict(size=10, color=theme.WELL_COLOUR[role],
                                symbol=theme.WELL_SYMBOL_2D[role],
                                line=dict(width=1.5, color=theme.SURFACE)),
                    hovertext=[f"{w.name} ({role})" for w in group],
                    hoverinfo="text", showlegend=False),
                row=1, col=3)

    fig.update_layout(**theme.plotly_layout(
        title=dict(text=title, font=dict(size=14, color=theme.INK_PRIMARY)),
        height=height, showlegend=False))
    for annotation in fig.layout.annotations:
        annotation.font.update(size=11, color=theme.INK_MUTED)
    return fig


def map_figure(wells=None, acquisition=None, grid: Grid3D | None = None,
               pml_nodes: int = 0, height: int = 520,
               bounds=None) -> go.Figure:
    """Plan view of sources, receivers, wells and the domain boundaries.

    Carries no in-figure title: the legend sits above the plot area, and a
    title there would collide with it. Streamlit renders the heading.

    ``bounds`` pins the axis range. Without it, adding a trace that extends
    past ``grid`` - the invisible placement lattice, say - rescales the whole
    map and the domain box shrinks to a corner.
    """
    fig = go.Figure()
    if grid is not None:
        (x0, x1), (y0, y1) = grid.bounds[0], grid.bounds[1]
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      line=dict(color=theme.AXIS, width=1.5))
        fig.add_annotation(x=x0, y=y1, text="propagation domain", showarrow=False,
                           xanchor="left", yanchor="bottom",
                           font=dict(color=theme.INK_MUTED, size=10))
        if pml_nodes:
            interior = grid.padded(-pml_nodes)
            (px0, px1), (py0, py1) = interior.bounds[0], interior.bounds[1]
            fig.add_shape(type="rect", x0=px0, x1=px1, y0=py0, y1=py1,
                          line=dict(color=theme.AXIS, width=1, dash="dash"))
            fig.add_annotation(x=px0, y=py1, text="inner edge of the absorbing layer",
                               showarrow=False, xanchor="left", yanchor="bottom",
                               font=dict(color=theme.INK_MUTED, size=10))
    if acquisition is not None:
        fig.add_trace(go.Scatter(
            x=acquisition.sources[:, 0], y=acquisition.sources[:, 1], mode="markers",
            name=f"sources ({acquisition.n_sources})",
            marker=dict(symbol="x", size=7, color=theme.SERIES[1], line=dict(width=1)),
            hovertemplate="source<br>x=%{x:,.0f} m<br>y=%{y:,.0f} m<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=acquisition.receivers[:, 0], y=acquisition.receivers[:, 1],
            mode="markers", name=f"nodes ({acquisition.n_receivers})",
            marker=dict(symbol="square", size=6, color=theme.SERIES[0]),
            hovertemplate="node<br>x=%{x:,.0f} m<br>y=%{y:,.0f} m<extra></extra>"))
    if wells is not None:
        for role in ("injector", "producer", "observation"):
            group = [w for w in wells if w.role == role]
            if not group:
                continue
            fig.add_trace(go.Scatter(
                x=[w.x for w in group], y=[w.y for w in group],
                mode="markers+text", name=f"{role}s ({len(group)})",
                text=[w.name for w in group], textposition="top center",
                textfont=dict(color=theme.INK_SECONDARY, size=11),
                marker=dict(symbol=theme.WELL_SYMBOL_2D[role], size=14,
                            color=theme.WELL_COLOUR[role],
                            line=dict(width=2, color=theme.SURFACE)),
                hovertemplate="%{text}<br>x=%{x:,.0f} m<br>y=%{y:,.0f} m<extra></extra>"))
    x_axis = dict(gridcolor=theme.GRIDLINE, linecolor=theme.AXIS, zeroline=False,
                  tickfont=dict(color=theme.INK_MUTED), title="x (m)")
    y_axis = dict(scaleanchor="x", scaleratio=1, gridcolor=theme.GRIDLINE,
                  linecolor=theme.AXIS, zeroline=False,
                  tickfont=dict(color=theme.INK_MUTED), title="y (m)")
    if bounds is not None:
        x_axis["range"] = list(bounds[0])
        y_axis["range"] = list(bounds[1])
    fig.update_layout(**theme.plotly_layout(
        height=height, margin=dict(l=56, r=16, t=52, b=44),
        xaxis=x_axis, yaxis=y_axis,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)))
    return fig


def series_figure(x, series: dict[str, np.ndarray], *, xlabel: str, ylabel: str,
                  height: int = 340, colours=None) -> go.Figure:
    """Line chart for continuous data.

    With two or more series a legend is always present and each line is also
    labelled at its end: two of the categorical slots sit below 3:1 contrast
    on this surface, so identity is never left to colour alone. A single
    series carries neither - the heading above the chart already names it.
    """
    fig = go.Figure()
    colours = colours or {}
    multiple = len(series) > 1
    for i, (name, values) in enumerate(series.items()):
        colour = colours.get(name) or theme.SERIES[i % len(theme.SERIES)]
        fig.add_trace(go.Scatter(
            x=x, y=values, mode="lines", name=name, showlegend=multiple,
            line=dict(color=colour, width=2),
            hovertemplate=f"{name}<br>%{{x:,.0f}}<br>%{{y:,.4g}}<extra></extra>"))
        finite = np.isfinite(values)
        if multiple and np.any(finite):
            last = int(np.max(np.where(finite)[0]))
            fig.add_annotation(x=x[last], y=values[last], text=f" {name}",
                               showarrow=False, xanchor="left",
                               font=dict(color=theme.INK_SECONDARY, size=10))
    fig.update_layout(**theme.plotly_layout(
        height=height, margin=dict(l=56, r=76, t=40 if multiple else 16, b=44),
        xaxis_title=xlabel, yaxis_title=ylabel,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)))
    return fig


def trace_figure(y, series: dict[str, np.ndarray], *, ylabel: str,
                 xlabel: str = "amplitude", height: int = 460,
                 colours=None, zero_line: bool = True) -> go.Figure:
    """Vertical seismic traces: amplitude across, time or depth down.

    The transpose of :func:`series_figure`, because a trace is read against
    a downward axis - the same convention as every section in this app, so
    an event at 1,200 m sits at the same height as it does on a slice.
    Identity is never colour-alone: two or more traces carry a legend and
    each is labelled at its deep end.
    """
    fig = go.Figure()
    colours = colours or {}
    multiple = len(series) > 1
    for i, (name, values) in enumerate(series.items()):
        colour = colours.get(name) or theme.SERIES[i % len(theme.SERIES)]
        fig.add_trace(go.Scatter(
            x=values, y=y, mode="lines", name=name, showlegend=multiple,
            line=dict(color=colour, width=2),
            hovertemplate=f"{name}<br>%{{y:,.4g}}<br>%{{x:,.4g}}<extra></extra>"))
        finite = np.isfinite(values)
        if multiple and np.any(finite):
            last = int(np.max(np.where(finite)[0]))
            fig.add_annotation(x=values[last], y=y[last], text=f" {name}",
                               showarrow=False, xanchor="left", yanchor="top",
                               font=dict(color=theme.INK_SECONDARY, size=10))
    if zero_line:
        fig.add_vline(x=0.0, line=dict(color=theme.GRIDLINE, width=1))
    fig.update_layout(**theme.plotly_layout(
        height=height, margin=dict(l=64, r=76, t=40 if multiple else 16, b=44),
        xaxis_title=xlabel, yaxis_title=ylabel,
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)))
    return fig


def bar_figure(centres, counts, *, xlabel: str, ylabel: str, width: float | None = None,
               height: int = 340, colour: str | None = None) -> go.Figure:
    """Histogram as bars.

    A distribution over bins is a bar chart, not a line: joining bin tops
    with a line implies the data is continuous between them, which for an
    offset or azimuth histogram it is not.
    """
    fig = go.Figure(go.Bar(
        x=centres, y=counts, width=width,
        marker=dict(color=colour or theme.SERIES[0], line=dict(width=0)),
        hovertemplate=f"%{{x:,.0f}}<br>%{{y:,.0f}} {ylabel}<extra></extra>"))
    fig.update_layout(**theme.plotly_layout(
        height=height, margin=dict(l=56, r=16, t=16, b=44), bargap=0.12,
        xaxis_title=xlabel, yaxis_title=ylabel, showlegend=False))
    return fig


def gather_figure(traces: np.ndarray, dt: float, *, title: str = "",
                  height: int = 460, robust: bool = True) -> go.Figure:
    """Variable-density display of one shot gather, trace number against time."""
    data = np.asarray(traces, dtype=float)
    lo, hi = _limits(data, "diverging", robust)
    times = np.arange(data.shape[1]) * dt
    fig = go.Figure(go.Heatmap(
        z=data.T, x=np.arange(data.shape[0]), y=times, zmin=lo, zmax=hi,
        colorscale=theme.SEISMIC, colorbar=dict(title="p", thickness=12),
        hovertemplate="trace %{x}<br>t=%{y:.3f} s<br>%{z:.4g}<extra></extra>"))
    fig.update_layout(**theme.plotly_layout(
        title=dict(text=title, font=dict(size=14)), height=height,
        xaxis_title="receiver", yaxis_title="time (s)",
        yaxis=dict(autorange="reversed", gridcolor=theme.GRIDLINE,
                   linecolor=theme.AXIS, tickfont=dict(color=theme.INK_MUTED))))
    return fig
