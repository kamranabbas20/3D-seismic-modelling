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
                text=[f" {w.name} " for w in group], textposition="top center",
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


def aerial_figure(acquisition=None, wells=None, domains=None, pml_nodes: int = 0,
                  height: int = 560, show_receivers: bool = True) -> go.Figure:
    """Plan view of the survey, drawn to scale.

    The axes share a scale, so an aerial view reads as distance rather than
    as a stretched rectangle - which matters here, because whether the
    spread is wide enough for the depth of the target is exactly the
    question this figure exists to answer.

    Layers, outermost first: the geological model, the propagation domain,
    the inner edge of the absorbing layer, the imaging target, then the
    receivers, the sources and the wells.
    """
    fig = go.Figure()

    def box(grid, colour, dash, label, width=1.5):
        (x0, x1), (y0, y1) = grid.bounds[0], grid.bounds[1]
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      line=dict(color=colour, width=width, dash=dash))
        # A plate behind the text: over a dense receiver carpet an unbacked
        # label is unreadable, and the carpet is the normal case.
        fig.add_annotation(x=x0, y=y1, text=label, showarrow=False,
                           xanchor="left", yanchor="bottom",
                           bgcolor=theme.LABEL_PLATE, borderpad=2,
                           font=dict(color=theme.INK_MUTED, size=10))

    if domains is not None:
        box(domains.geology, theme.GRIDLINE, "solid", "geological model")
        box(domains.propagation, theme.AXIS, "solid", "propagation domain")
        if pml_nodes:
            box(domains.propagation.padded(-pml_nodes), theme.AXIS, "dash",
                "inner edge of the absorbing layer", width=1.0)
        box(domains.target, theme.SERIES[2], "dot", "imaging target")

    if acquisition is not None:
        if show_receivers:
            fig.add_trace(go.Scatter(
                x=acquisition.receivers[:, 0], y=acquisition.receivers[:, 1],
                mode="markers", name=f"receivers ({acquisition.n_receivers:,})",
                marker=dict(symbol="square", size=5, color=theme.SERIES[0]),
                hovertemplate="receiver<br>x=%{x:,.0f} m<br>y=%{y:,.0f} m<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=acquisition.sources[:, 0], y=acquisition.sources[:, 1],
            mode="markers", name=f"sources ({acquisition.n_sources:,})",
            marker=dict(symbol="x", size=9, color=theme.SERIES[1],
                        line=dict(width=1)),
            hovertemplate="source<br>x=%{x:,.0f} m<br>y=%{y:,.0f} m<extra></extra>"))

    if wells is not None:
        for role in ("injector", "producer", "observation"):
            group = [w for w in wells if w.role == role]
            if not group:
                continue
            fig.add_trace(go.Scatter(
                x=[w.x for w in group], y=[w.y for w in group],
                mode="markers", name=role,
                marker=dict(size=13, color=theme.WELL_COLOUR[role],
                            symbol=theme.WELL_SYMBOL_2D[role],
                            line=dict(width=2, color=theme.SURFACE)),
                hovertext=[f"{w.name} ({role})" for w in group],
                hoverinfo="text"))
            # Annotations rather than marker text, so the name gets the same
            # plate the box labels do and stays readable over the receivers.
            for w in group:
                fig.add_annotation(x=w.x, y=w.y, text=w.name, showarrow=False,
                                   yshift=13, yanchor="bottom",
                                   bgcolor=theme.LABEL_PLATE, borderpad=2,
                                   font=dict(color=theme.INK_SECONDARY, size=11))

    fig.update_layout(**theme.plotly_layout(
        height=height, margin=dict(l=60, r=20, t=44, b=48),
        xaxis_title="x (m)", yaxis_title="y (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1, gridcolor=theme.GRIDLINE,
                   linecolor=theme.AXIS, tickfont=dict(color=theme.INK_MUTED)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)))
    return fig


def elevation_figure(acquisition=None, domains=None, pml_nodes: int = 0,
                     height: int = 620) -> go.Figure:
    """Side view of the survey: depth down, x across, drawn to scale.

    The plan view answers whether the spread is wide enough.  It cannot
    answer the other half of the question, which is how far above the target
    the acquisition sits and how far the sources sit from the receivers -
    and both decided the image on this project.  Sources and receivers 20 m
    apart put their injection near-fields in the same slab of model, where
    they reinforced each other into a band brighter than the reservoir;
    acquisition less than about two wavelengths above the target overlaps it
    with that same near-field, which no filter separates.

    Depth increases downward, as a geologist reads it, and the axes share a
    scale so a standoff reads as a distance rather than as whatever the
    aspect ratio made of it.
    """
    fig = go.Figure()

    def box(grid, colour, dash, label, width=1.5):
        (x0, x1), _, (z0, z1) = grid.bounds
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=z0, y1=z1,
                      line=dict(color=colour, width=width, dash=dash))
        fig.add_annotation(x=x0, y=z0, text=label, showarrow=False,
                           xanchor="left", yanchor="bottom",
                           bgcolor=theme.LABEL_PLATE, borderpad=2,
                           font=dict(color=theme.INK_MUTED, size=10))

    if domains is not None:
        box(domains.geology, theme.GRIDLINE, "solid", "geological model")
        box(domains.propagation, theme.AXIS, "solid", "propagation domain")
        if pml_nodes:
            box(domains.propagation.padded(-pml_nodes), theme.AXIS, "dash",
                "inner edge of the absorbing layer", width=1.0)
        box(domains.target, theme.SERIES[2], "dot", "imaging target")

    if acquisition is not None:
        fig.add_trace(go.Scatter(
            x=acquisition.receivers[:, 0], y=acquisition.receivers[:, 2],
            mode="markers", name=f"receivers ({acquisition.n_receivers:,})",
            marker=dict(symbol="square", size=5, color=theme.SERIES[0]),
            hovertemplate="receiver<br>x=%{x:,.0f} m<br>z=%{y:,.0f} m<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=acquisition.sources[:, 0], y=acquisition.sources[:, 2],
            mode="markers", name=f"sources ({acquisition.n_sources:,})",
            marker=dict(symbol="x", size=9, color=theme.SERIES[1],
                        line=dict(width=1)),
            hovertemplate="source<br>x=%{x:,.0f} m<br>z=%{y:,.0f} m<extra></extra>"))

        # The standoff, drawn rather than described: the gap between the
        # deepest instrument and the top of the target.
        if domains is not None:
            deepest = float(max(acquisition.sources[:, 2].max(),
                                acquisition.receivers[:, 2].max()))
            z_top = float(domains.target.bounds[2][0])
            if z_top > deepest:
                mid = float(np.mean(acquisition.sources[:, 0]))
                fig.add_shape(type="line", x0=mid, x1=mid, y0=deepest, y1=z_top,
                              line=dict(color=theme.INK_MUTED, width=1,
                                        dash="dot"))
                # Anchored right of the line and nudged up, clear of the
                # target box's own label, which sits at its top-left corner.
                fig.add_annotation(
                    x=mid, y=0.5 * (deepest + z_top),
                    text=f"standoff {z_top - deepest:,.0f} m", showarrow=False,
                    xanchor="right", xshift=-8, yshift=-10,
                    bgcolor=theme.LABEL_PLATE, borderpad=2,
                    font=dict(color=theme.INK_SECONDARY, size=10))

    # Pin x to the model rather than letting the equal-scale constraint choose
    # it: unpinned, a 3,000 x 2,200 m model in a wide, short plot area stretched
    # the axis out to -1,000 - 3,000 m and drew the survey in the middle third
    # of a mostly empty figure.
    x_range = None
    if domains is not None:
        (gx0, gx1), _, _ = domains.geology.bounds
        pad = 0.02 * (gx1 - gx0)
        x_range = [gx0 - pad, gx1 + pad]
    fig.update_layout(**theme.plotly_layout(
        height=height, margin=dict(l=60, r=20, t=44, b=48),
        xaxis_title="x (m)", yaxis_title="depth (m)",
        xaxis=dict(range=x_range, constrain="domain", gridcolor=theme.GRIDLINE,
                   linecolor=theme.AXIS, tickfont=dict(color=theme.INK_MUTED)),
        yaxis=dict(autorange="reversed", scaleanchor="x", scaleratio=1,
                   constrain="domain", gridcolor=theme.GRIDLINE,
                   linecolor=theme.AXIS, tickfont=dict(color=theme.INK_MUTED)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)))
    return fig


def fold_figure(x, y, fold, *, wells=None, height: int = 460,
                title: str = "") -> go.Figure:
    """Common-midpoint fold in map view.

    Fold is a count, so it takes the sequential ramp, not the diverging one:
    there is no meaningful zero to diverge about.
    """
    values = np.asarray(fold, dtype=float)
    fig = go.Figure(go.Heatmap(
        z=values.T, x=np.asarray(x, dtype=float), y=np.asarray(y, dtype=float),
        colorscale=theme.SEQUENTIAL, zmin=0.0,
        colorbar=dict(title="traces", thickness=12,
                      tickfont=dict(color=theme.INK_MUTED)),
        hovertemplate="x=%{x:,.0f} m<br>y=%{y:,.0f} m<br>%{z:,.0f} traces<extra></extra>"))
    if wells is not None:
        for role in ("injector", "producer", "observation"):
            group = [w for w in wells if w.role == role]
            if not group:
                continue
            fig.add_trace(go.Scatter(
                x=[w.x for w in group], y=[w.y for w in group],
                mode="markers+text", text=[w.name for w in group],
                textposition="top center", showlegend=False,
                textfont=dict(color=theme.INK_PRIMARY, size=10),
                marker=dict(size=11, color=theme.WELL_COLOUR[role],
                            symbol=theme.WELL_SYMBOL_2D[role],
                            line=dict(width=2, color=theme.SURFACE)),
                hovertext=[f"{w.name} ({role})" for w in group], hoverinfo="text"))
    fig.update_layout(**theme.plotly_layout(
        title=dict(text=title, font=dict(size=14)), height=height,
        margin=dict(l=60, r=20, t=44, b=48),
        xaxis_title="x (m)", yaxis_title="y (m)",
        yaxis=dict(scaleanchor="x", scaleratio=1, gridcolor=theme.GRIDLINE,
                   linecolor=theme.AXIS, tickfont=dict(color=theme.INK_MUTED)),
        showlegend=False))
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

    Direct labels are anchored at each trace's own extremum rather than at
    its end: a trace tails off to zero, so end labels would all land on the
    one point where the series are guaranteed to agree.  When the extrema
    themselves coincide - four earth models differing by a few percent peak
    on the same event - no anchor separates them, and the labels are
    dropped rather than overprinted.  The legend still carries identity, so
    it is never left to colour alone.
    """
    fig = go.Figure()
    colours = colours or {}
    multiple = len(series) > 1
    anchors = []
    for i, (name, values) in enumerate(series.items()):
        colour = colours.get(name) or theme.SERIES[i % len(theme.SERIES)]
        fig.add_trace(go.Scatter(
            x=values, y=y, mode="lines", name=name, showlegend=multiple,
            line=dict(color=colour, width=2),
            hovertemplate=f"{name}<br>%{{y:,.4g}}<br>%{{x:,.4g}}<extra></extra>"))
        finite = np.isfinite(values)
        if multiple and np.any(finite) and np.nanmax(np.abs(values)) > 0:
            peak = int(np.nanargmax(np.abs(np.where(finite, values, 0.0))))
            anchors.append((name, float(values[peak]), float(y[peak])))

    if anchors:
        span_x = max(abs(a[1]) for a in anchors) or 1.0
        span_y = (float(np.nanmax(y)) - float(np.nanmin(y))) or 1.0
        crowded = any(
            abs(a[1] - b[1]) < 0.04 * span_x and abs(a[2] - b[2]) < 0.04 * span_y
            for i, a in enumerate(anchors) for b in anchors[i + 1:])
        if not crowded:
            for name, x, at in anchors:
                side = "left" if x >= 0 else "right"
                fig.add_annotation(
                    x=x, y=at,
                    text=f" {name}" if side == "left" else f"{name} ",
                    showarrow=False, xanchor=side, yanchor="middle",
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


def stratigraphy_figure(layers, grid, *, axis: int = 0, at: float | None = None,
                        title: str = "", height: int = 420, wells=None,
                        highlight: str | None = None, picks=None,
                        placement: bool = False) -> go.Figure:
    """The layer stack as filled bands down one section line.

    Evaluates the horizon surfaces directly rather than the built property
    cube, so it costs milliseconds and can be drawn while the stratigraphy is
    still being edited.  That is the point of it: seeing where a unit sits,
    how thick it is and where it pinches out should not need a grid build.
    """
    import numpy as np

    other = 1 - axis
    along = grid.axis(axis)
    fixed = grid.axis(other)
    index = nearest(fixed, grid.origin[other] + 0.5 * grid.extent[other]
                    if at is None else at)

    tops = []
    for layer in layers:
        surface = layer.top.on_grid(grid)
        tops.append(surface[:, index] if axis == 0 else surface[index, :])
    base = grid.origin[2] + grid.extent[2]

    fig = go.Figure()
    for i, layer in enumerate(layers):
        upper = tops[i]
        lower = tops[i + 1] if i + 1 < len(tops) else np.full_like(upper, base)
        colour = theme.FACIES_COLOUR.get(layer.facies, theme.FACIES_FALLBACK)
        thickness = float(np.max(lower - upper))
        pinches = bool(np.min(lower - upper) <= 1e-6) and thickness > 1e-6
        chosen = highlight is not None and layer.name == highlight
        fig.add_trace(go.Scatter(
            x=np.concatenate([along, along[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself", mode="lines",
            line=dict(color=theme.INK_PRIMARY if chosen else theme.SURFACE,
                      width=2.2 if chosen else 1.0),
            fillcolor=colour,
            opacity=0.95 if chosen else (0.35 if highlight else 0.92),
            name=layer.name + (" (pinches out)" if pinches else ""),
            hovertemplate=(f"<b>{layer.name}</b><br>{layer.facies}"
                           f"<br>thickness up to {thickness:,.0f} m"
                           f"<extra></extra>")))
    for well in (wells or []):
        position = well.position
        fig.add_trace(go.Scatter(
            x=[position[axis]] * 2, y=[grid.origin[2], base], mode="lines",
            line=dict(color=theme.INK_PRIMARY, width=1.6, dash="dot"),
            name=well.name, showlegend=False,
            hovertemplate=f"{well.name}<extra></extra>"))

    if picks:
        fig.add_trace(go.Scatter(
            x=[float(p[0]) for p in picks], y=[float(p[1]) for p in picks],
            mode="lines+markers", name="drawn base",
            line=dict(color=theme.INK_PRIMARY, width=2.0, dash="dash"),
            marker=dict(size=10, color=theme.SURFACE,
                        line=dict(color=theme.INK_PRIMARY, width=2)),
            hovertemplate="pick %{x:,.0f} m, %{y:,.0f} m<extra></extra>"))
    if placement:
        # Plotly reports selections on traces, not on empty canvas, so a
        # transparent lattice is what makes the section clickable at all.
        step_along = max(1, len(along) // 70)
        depths = grid.axis(2)
        step_depth = max(1, len(depths) // 70)
        mesh_x, mesh_z = np.meshgrid(along[::step_along], depths[::step_depth],
                                     indexing="ij")
        fig.add_trace(go.Scatter(
            x=mesh_x.ravel(), y=mesh_z.ravel(), mode="markers",
            marker=dict(size=13, color="rgba(0,0,0,0)"), showlegend=False,
            name="draw here",
            hovertemplate=("draw the base here<br>%{x:,.0f} m, %{y:,.0f} m"
                           "<extra></extra>")))

    label = ("x", "y")[axis]
    fig.update_layout(**theme.plotly_layout(
        title=dict(text=title, font=dict(size=14)), height=height,
        xaxis_title=f"{label} (m)", yaxis_title="depth (m)",
        xaxis=dict(range=[float(along.min()), float(along.max())],
                   gridcolor=theme.GRIDLINE, linecolor=theme.AXIS),
        yaxis=dict(autorange="reversed", gridcolor=theme.GRIDLINE,
                   linecolor=theme.AXIS, tickfont=dict(color=theme.INK_MUTED)),
        legend=dict(orientation="v", x=1.01, y=1.0, font=dict(size=10))))
    return fig


def section_figure(data: np.ndarray, x, y, *, xlabel: str, ylabel: str,
                   title: str = "", height: int = 420,
                   robust: bool = True) -> go.Figure:
    """A seismic section: a physical x axis and a downward y axis.

    Distinct from :func:`gather_figure`, which numbers its traces, and from
    :func:`slice_figure`, which cuts a depth volume: this is one panel of a
    time-domain cube, so both axes carry real units.
    """
    values = np.asarray(data, dtype=float)
    lo, hi = _limits(values, "diverging", robust)
    fig = go.Figure(go.Heatmap(
        z=values, x=np.asarray(x, dtype=float), y=np.asarray(y, dtype=float),
        zmin=lo, zmax=hi, colorscale=theme.SEISMIC,
        colorbar=dict(title="amplitude", thickness=12),
        hovertemplate=(f"{xlabel} %{{x:,.0f}}<br>{ylabel} %{{y:.3f}}"
                       f"<br>%{{z:.4g}}<extra></extra>")))
    fig.update_layout(**theme.plotly_layout(
        title=dict(text=title, font=dict(size=14)), height=height,
        xaxis_title=xlabel, yaxis_title=ylabel,
        yaxis=dict(autorange="reversed", gridcolor=theme.GRIDLINE,
                   linecolor=theme.AXIS, tickfont=dict(color=theme.INK_MUTED))))
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
