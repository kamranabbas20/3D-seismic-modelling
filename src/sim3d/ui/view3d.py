"""Interactive 3D scene (requirements 5 and 6).

Plotly's WebGL scene gives rotate, pan, zoom, a camera reset and a
perspective/orthographic switch for free.  What this module adds is the
*content*: horizons, fault planes, property volumes, and wells drawn as
real vertical trajectories with their open intervals picked out.

Every component is built as a separate trace, so visibility is a matter of
which traces are added - one checkbox per layer, per fault and per well,
exactly as the requirement asks.  Nothing is hidden by setting an opacity
to zero, because a trace that is not built is also not sent to the browser.

Volume rendering is decimated.  A 3 km model at 25 m is three million
points; a browser will accept perhaps a hundred thousand before it becomes
unusable, so :func:`property_volume` subsamples to a target budget and says
by how much.  A decimated volume is for looking at, not for measuring - the
sections in :mod:`sim3d.ui.components` are the quantitative view.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..core.grid import Grid3D
from . import theme

#: Points a volume trace is decimated to before being sent to the browser.
VOLUME_BUDGET = 120_000


def _decimation(shape, budget: int) -> tuple[int, int, int]:
    """Per-axis stride that brings a grid under ``budget`` points."""
    total = int(np.prod(shape))
    if total <= budget:
        return (1, 1, 1)
    factor = (total / budget) ** (1.0 / 3.0)
    return tuple(max(1, int(np.ceil(factor))) for _ in range(3))


def built_horizons(geology) -> dict[str, np.ndarray]:
    """The top of each unit *as built*, one depth per map position.

    ``geology.horizons`` holds the surfaces the model was defined from -
    before faulting.  Drawing those would show a fault plane cutting
    horizons that do not offset across it, which is exactly backwards.
    Reading the tops back out of the layer-index volume gives the surfaces
    the model actually has, throw and all.

    A unit absent from a column - pinched out, or cut away by a fault -
    gets ``NaN`` there, and Plotly leaves that part of the surface unfilled.
    """
    grid = geology.grid
    z = grid.axis(2)
    index = geology.layer_index
    out: dict[str, np.ndarray] = {}
    for i, layer in enumerate(geology.layers):
        present = index == i
        # First True down each column, or NaN where the unit is missing.
        first = np.argmax(present, axis=2)
        depths = np.where(present.any(axis=2), z[first], np.nan)
        out[layer.name] = depths
    return out


def horizon_surface(grid: Grid3D, depths: np.ndarray, name: str, colour: str,
                    opacity: float = 0.6, visible: bool = True) -> go.Surface:
    """One structural surface, drawn at its own depths."""
    return go.Surface(
        x=grid.axis(0), y=grid.axis(1), z=np.asarray(depths, dtype=float).T,
        name=name, showscale=False, opacity=opacity, visible=visible,
        colorscale=[[0, colour], [1, colour]],
        hovertemplate=f"{name}<br>x=%{{x:,.0f}} m<br>y=%{{y:,.0f}} m"
                      f"<br>z=%{{z:,.0f}} m<extra></extra>",
        contours=dict(x=dict(highlight=False), y=dict(highlight=False),
                      z=dict(highlight=False)),
    )


def fault_mesh(fault, grid: Grid3D, opacity: float = 0.45) -> go.Mesh3d:
    """A fault plane as a quadrilateral, clipped to the model box."""
    strike = fault.strike_vector
    dip = fault.dip_vector
    lx, ly, lz = grid.extent
    along = fault.strike_extent or 0.5 * float(np.hypot(lx, ly))
    down = fault.dip_extent or 0.5 * lz
    origin = np.array(fault.origin, dtype=float)

    def quad(a: float, d: float) -> np.ndarray:
        return np.array([origin - a * strike - d * dip,
                         origin + a * strike - d * dip,
                         origin + a * strike + d * dip,
                         origin - a * strike + d * dip])

    # Shrink the whole quadrilateral until it fits the model box, rather than
    # clipping each corner coordinate: clipping per axis would pull corners
    # sideways and draw a plane that is not the fault plane.
    corners = quad(along, down)
    for _ in range(40):
        inside = all(lo - 1e-6 <= corners[:, axis].min()
                     and corners[:, axis].max() <= hi + 1e-6
                     for axis, (lo, hi) in enumerate(grid.bounds))
        if inside:
            break
        along *= 0.92
        down *= 0.92
        corners = quad(along, down)
    seal = ("sealing" if fault.transmissibility == 0 else
            "open" if fault.transmissibility == 1 else
            f"transmissibility {fault.transmissibility:g}")
    return go.Mesh3d(
        x=corners[:, 0], y=corners[:, 1], z=corners[:, 2],
        i=[0, 0], j=[1, 2], k=[2, 3], name=f"fault {fault.name}",
        color=theme.STATUS["FAIL"] if fault.transmissibility == 0 else theme.INK_MUTED,
        opacity=opacity, showlegend=True, hoverinfo="name",
        hovertext=f"{fault.name}: throw {fault.throw:g} m, {seal}")


def well_traces(well, grid: Grid3D, intervals=None, selected: bool = False
                ) -> list[go.Scatter3d]:
    """A vertical well: full trajectory, open intervals, and the wellhead.

    Three traces so the open intervals read at a glance - the trajectory as
    a thin muted line, the completions as a thick line in the well's own
    colour, and the wellhead as a filled circle above the model.
    """
    colour = theme.WELL_COLOUR.get(well.role, theme.INK_SECONDARY)
    top, base = grid.bounds[2]
    traces = [go.Scatter3d(
        x=[well.x, well.x], y=[well.y, well.y], z=[top, base],
        mode="lines", name=f"{well.name} trajectory", legendgroup=well.name,
        showlegend=False, hoverinfo="skip",
        line=dict(color=theme.INK_PRIMARY, width=5))]

    for index, (interval_top, interval_base, unit) in enumerate(intervals or []):
        traces.append(go.Scatter3d(
            x=[well.x, well.x], y=[well.y, well.y],
            z=[interval_top, interval_base], mode="lines",
            name=f"{well.name} open", legendgroup=well.name, showlegend=False,
            line=dict(color=colour, width=22),
            hovertemplate=(f"{well.name} open in {unit.name}<br>"
                           f"{interval_top:,.0f} - {interval_base:,.0f} m"
                           f"<extra></extra>")))

    traces.append(go.Scatter3d(
        x=[well.x], y=[well.y], z=[top], mode="markers+text",
        text=[well.name], textposition="top center",
        textfont=dict(color=theme.INK_PRIMARY, size=13),
        name=f"{well.name} ({well.role})", legendgroup=well.name, showlegend=True,
        marker=dict(size=13 if not selected else 20, color=colour,
                    symbol=theme.WELL_SYMBOL_3D.get(well.role, "circle"),
                    line=dict(width=3 if selected else 1,
                              color=theme.INK_PRIMARY if selected else theme.SURFACE)),
        hovertemplate=f"{well.name} ({well.role})<br>"
                      f"x=%{{x:,.0f}} m<br>y=%{{y:,.0f}} m<extra></extra>"))
    return traces


def property_volume(grid: Grid3D, values: np.ndarray, name: str, *,
                    kind: str = "sequential", opacity: float = 0.25,
                    mask: np.ndarray | None = None,
                    budget: int = VOLUME_BUDGET,
                    surface_count: int = 12) -> tuple[go.Volume, tuple[int, int, int]]:
    """A decimated volume rendering, with the stride it used."""
    data = np.asarray(values, dtype=float)
    if mask is not None:
        data = np.where(mask, data, np.nan)
    stride = _decimation(grid.shape, budget)
    sub = data[::stride[0], ::stride[1], ::stride[2]]
    x = grid.axis(0)[::stride[0]]
    y = grid.axis(1)[::stride[1]]
    z = grid.axis(2)[::stride[2]]
    gx, gy, gz = np.meshgrid(x, y, z, indexing="ij")

    finite = sub[np.isfinite(sub)]
    if finite.size == 0:
        low, high = 0.0, 1.0
    elif kind == "diverging":
        peak = float(np.percentile(np.abs(finite), 99.5)) or 1.0
        low, high = -peak, peak
    else:
        low = float(np.percentile(finite, 1.0))
        high = float(np.percentile(finite, 99.0))

    trace = go.Volume(
        x=gx.ravel(), y=gy.ravel(), z=gz.ravel(), value=sub.ravel(),
        isomin=low, isomax=high, opacity=opacity, surface_count=surface_count,
        colorscale=theme.DIVERGING if kind == "diverging" else theme.SEQUENTIAL,
        name=name, showscale=True,
        colorbar=dict(title=name, thickness=12, len=0.6,
                      tickfont=dict(color=theme.INK_MUTED)),
        caps=dict(x_show=False, y_show=False, z_show=False),
        hovertemplate="%{value:,.4g}<extra></extra>")
    return trace, stride


def scene(traces, grid: Grid3D, *, title: str = "", height: int = 720,
          orthographic: bool = False, camera: dict | None = None,
          vertical_exaggeration: float = 1.0) -> go.Figure:
    """Assemble traces into a 3D scene with depth pointing down."""
    figure = go.Figure(data=list(traces))
    lx, ly, lz = grid.extent
    longest = max(lx, ly, 1e-9)
    projection = "orthographic" if orthographic else "perspective"
    # The z axis range is already reversed so depth increases downwards.
    # Flipping "up" as well would invert it a second time and put the
    # surface at the bottom of the screen.
    default_camera = default_camera_position()
    figure.update_layout(
        **theme.plotly_layout(
            height=height, margin=dict(l=0, r=0, t=36 if title else 8, b=0),
            # Passing title=None makes Plotly render the string "undefined";
            # an empty dict leaves the slot alone.
            title=(dict(text=title, font=dict(size=14)) if title else {}),
            legend=dict(orientation="v", yanchor="top", y=0.98, x=0.01,
                        bgcolor="rgba(252,252,251,0.7)")),
        scene=dict(
            xaxis=dict(title="x (m)", backgroundcolor=theme.SURFACE,
                       gridcolor=theme.GRIDLINE, showbackground=True,
                       range=list(grid.bounds[0])),
            yaxis=dict(title="y (m)", backgroundcolor=theme.SURFACE,
                       gridcolor=theme.GRIDLINE, showbackground=True,
                       range=list(grid.bounds[1])),
            # Depth increases downwards, so the z axis is reversed.
            zaxis=dict(title="depth (m)", backgroundcolor=theme.SURFACE,
                       gridcolor=theme.GRIDLINE, showbackground=True,
                       range=[grid.bounds[2][1], grid.bounds[2][0]]),
            aspectmode="manual",
            aspectratio=dict(x=lx / longest, y=ly / longest,
                             z=vertical_exaggeration * lz / longest),
            camera={**default_camera, **(camera or {}),
                    "projection": dict(type=projection)},
        ),
    )
    # Plotly's default axis titles fight the scene; keep the chrome recessive.
    figure.update_scenes(xaxis_showspikes=False, yaxis_showspikes=False,
                         zaxis_showspikes=False)
    return figure


def default_camera_position() -> dict:
    """The reset view: from above and to the south-east, looking down."""
    return dict(eye=dict(x=1.45, y=-1.45, z=0.95), up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=-0.05))
