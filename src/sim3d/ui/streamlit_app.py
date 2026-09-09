"""The research GUI (spec sections 108-110, 119, 121-124).

Streamlit is only a frontend.  Every page calls
:class:`sim3d.experiments.pipeline.Pipeline`, the same object the CLI drives,
and no page computes anything scientific of its own.

Two behaviours matter more than the layout:

**Expensive work never happens because a slider moved** (section 108).  Moving
a front radius updates the reservoir state and the rock physics - seconds of
work - and marks the shot gathers and the migrated images stale.  It does not
re-run them.  Full-wave modelling and RTM happen only when their buttons are
pressed.

**Changing the science invalidates the science** (section 109).  The pipeline
is keyed on the configuration's content hash, which covers every scientific
input and excludes display settings.  Edit a saturation target and the
gathers are dropped; change a colour limit and nothing is.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import streamlit as st

# Streamlit executes this file as ``__main__`` rather than importing it as a
# package module, so these must be absolute imports. The package itself is
# imported normally, which is why everything below it can stay relative.
from sim3d.core.config import ExperimentConfig
from sim3d.core.errors import Sim3DError
from sim3d.experiments.pipeline import Pipeline
from sim3d.fourd.decomposition import nonlinearity_ratio
from sim3d.fourd.metrics import radial_profile
from sim3d.fourd.scenarios import SCENARIO_NAMES
from sim3d.ui import components as ui
from sim3d.ui import theme
from sim3d.validation.physics import physics_table
from sim3d.validation.qc import Status

EXAMPLES = sorted(Path("examples/configs").glob("*.yaml")) if Path("examples/configs").is_dir() else []

PAGES = ("Project", "Geology", "Wells & Reservoir", "Rock Physics",
         "Acquisition & QC", "Simulation & Imaging", "4D Analysis")


# --------------------------------------------------------------------- state
def config() -> ExperimentConfig:
    if "config" not in st.session_state:
        default = EXAMPLES[0] if EXAMPLES else None
        st.session_state.config = (ExperimentConfig.load(default) if default
                                   else ExperimentConfig())
    return st.session_state.config


def pipeline() -> Pipeline:
    """The pipeline for the current configuration, rebuilt when it changes."""
    digest = config().content_hash()
    if st.session_state.get("pipeline_hash") != digest:
        st.session_state.pipeline = Pipeline(copy.deepcopy(config()))
        st.session_state.pipeline_hash = digest
    return st.session_state.pipeline


def invalidate() -> None:
    """Drop everything downstream of a scientific edit."""
    st.session_state.pop("pipeline_hash", None)


def cursor(grid) -> tuple[float, float, float]:
    """The shared inspection point, in metres, clamped to ``grid``.

    It starts at the centre of the *target window* rather than of whichever
    grid is on screen: the target is where the reservoir is, and a cursor
    defaulting to the middle of a 1.8 km geological model opens every page
    on overburden.
    """
    if "cursor" not in st.session_state:
        target = pipeline().domains.target
        st.session_state.cursor = tuple(0.5 * (lo + hi) for lo, hi in target.bounds)
    return tuple(float(np.clip(c, lo, hi))
                 for c, (lo, hi) in zip(st.session_state.cursor, grid.bounds))


def stage(label: str, fn):
    """Run a cheap pipeline stage with a spinner, surfacing errors as errors."""
    try:
        with st.spinner(f"{label}…"):
            return fn()
    except Sim3DError as exc:
        st.error(f"**{type(exc).__name__}** — {exc}")
        st.stop()


# ------------------------------------------------------------------- sidebar
def sidebar() -> str:
    st.sidebar.title("sim3d")
    st.sidebar.caption("Reservoir-to-Seismic & 4D Laboratory")

    names = [p.name for p in EXAMPLES]
    if names:
        chosen = st.sidebar.selectbox("Experiment", names, key="example")
        if st.session_state.get("loaded") != chosen:
            st.session_state.config = ExperimentConfig.load(
                next(p for p in EXAMPLES if p.name == chosen))
            st.session_state.loaded = chosen
            st.session_state.pop("cursor", None)
            invalidate()

    uploaded = st.sidebar.file_uploader("…or upload a YAML", type=("yaml", "yml"))
    if uploaded is not None and st.session_state.get("uploaded") != uploaded.name:
        try:
            import yaml
            st.session_state.config = ExperimentConfig.from_dict(
                yaml.safe_load(uploaded.getvalue().decode("utf-8")))
            st.session_state.uploaded = uploaded.name
            st.session_state.pop("cursor", None)
            invalidate()
        except Sim3DError as exc:
            st.sidebar.error(str(exc))

    page = st.sidebar.radio("Page", PAGES, key="page")

    st.sidebar.divider()
    st.sidebar.caption(f"config hash `{config().short_hash}`")
    pipe = st.session_state.get("pipeline")
    if pipe is not None:
        done = [name for name, ok in (
            ("geology", pipe.result.geology is not None),
            ("rock physics", pipe.result.earth is not None),
            ("gathers", bool(pipe.result.gathers)),
            ("images", bool(pipe.result.images))) if ok]
        st.sidebar.caption("computed: " + (", ".join(done) if done else "nothing yet"))
    return page


def cursor_controls(grid) -> tuple[float, float, float]:
    """Linked cursor shared by every volume view (section 123)."""
    st.sidebar.divider()
    st.sidebar.caption("**Inspection point** — shared by every volume")
    current = cursor(grid)
    values = []
    for axis, label in enumerate("xyz"):
        lo, hi = grid.bounds[axis]
        values.append(st.sidebar.slider(
            f"{label} (m)", float(lo), float(hi), float(current[axis]),
            step=float(grid.spacing[axis]), key=f"cursor_{label}"))
    st.session_state.cursor = tuple(values)
    return tuple(values)


# --------------------------------------------------------------------- pages
def page_project() -> None:
    cfg = config()
    st.title(cfg.project.name)
    if cfg.project.description:
        st.caption(cfg.project.description)

    domains = pipeline().domains
    a, b, c = st.columns(3)
    for column, (name, grid) in zip((a, b, c), (
            ("Geological model", domains.geology),
            ("Wave-equation domain", domains.propagation),
            ("Target window", domains.target))):
        lx, ly, lz = grid.extent
        column.metric(name, f"{lx / 1000:.2f} × {ly / 1000:.2f} × {lz / 1000:.2f} km",
                      f"{grid.n_cells:,} cells")
    st.caption("These three are deliberately different (spec section 6): the "
               "geological context stays large while the wave-equation "
               "experiment stays as small as is scientifically reasonable.")

    st.subheader("Configuration")
    st.code(cfg.describe(), language="text")

    st.subheader("What each mode does and does not model")
    st.caption("Shown here so that no result has to be interpreted by guessing "
               "which physics produced it (spec section 83).")
    st.code(physics_table(), language="text")


def page_geology() -> None:
    pipe = pipeline()
    geology = stage("Building geology", pipe.geology)
    grid = geology.grid
    point = cursor_controls(grid)

    st.title("Geology")
    st.caption(geology.summary().splitlines()[1])

    volumes = {
        "porosity": (geology.porosity, "porosity", 1.0, "fraction"),
        "shale volume": (geology.vsh, "vsh", 1.0, "fraction"),
        "net-to-gross": (geology.ntg, "ntg", 1.0, "fraction"),
        "facies": (geology.facies_code.astype(float), "facies", 1.0, "code"),
        "layer": (geology.layer_index.astype(float), "layer", 1.0, "index"),
    }
    chosen = st.selectbox("Property", list(volumes), index=0)
    volume, _, scale, unit = volumes[chosen]
    st.plotly_chart(ui.slice_figure(volume, grid, point, title=chosen,
                                    kind="sequential", unit=unit, scale=scale,
                                    wells=pipe.wells()),
                    width="stretch")

    left, right = st.columns(2)
    left.subheader("Layers")
    left.dataframe({
        layer.name: {
            "facies": layer.facies,
            "cells (%)": round(100 * float(np.mean(geology.layer_index == i)), 1),
            "reservoir": bool(geology.reservoir_mask[geology.layer_index == i].any()),
        }
        for i, layer in enumerate(geology.layers)
    }, width="stretch")
    right.subheader("Faults")
    right.code(geology.faults.describe(), language="text")
    right.caption("Faults are kinematic: they displace the stratigraphy and "
                  "attenuate transport across the plane. They do not solve for "
                  "stress.")


def page_reservoir() -> None:
    pipe = pipeline()
    cfg = config()
    st.title("Wells & reservoir state")

    st.markdown("**Well pattern**")
    st.plotly_chart(ui.map_figure(wells=pipe.wells(), grid=pipe.domains.propagation,
                                  pml_nodes=cfg.solver.pml_nodes), width="stretch")
    warnings = pipe.wells().check_spacing()
    for message in warnings:
        st.warning(message, icon="⚠️")
    if not warnings:
        st.caption("All well pairs meet the configured minimum spacing.")

    st.subheader("Scenario")
    st.caption("Editing these updates the reservoir state and the rock physics. "
               "It does **not** re-run wave modelling or migration — those stay "
               "on their own buttons (spec section 108).")

    scenario = cfg.reservoir.scenario
    edited = False
    for label, entries, field, fmt in (
            ("Pressure", scenario.pressure, "delta_p_bar", "%.0f bar"),
            ("Water fronts", scenario.water_fronts, "target_sw", "%.2f"),
            ("Gas", scenario.gas, "target_sg", "%.2f")):
        if not entries:
            continue
        st.markdown(f"**{label}**")
        for i, entry in enumerate(entries):
            c1, c2 = st.columns(2)
            key = f"{label}_{i}"
            if field == "delta_p_bar":
                new = c1.slider(f"{entry['well']} ΔP (bar)", -120.0, 120.0,
                                float(entry[field]), 1.0, key=key + "_v")
            else:
                new = c1.slider(f"{entry['well']} {field}", 0.0, 1.0,
                                float(entry[field]), 0.01, key=key + "_v")
            radius = c2.slider(f"{entry['well']} major radius (m)", 50.0, 1200.0,
                               float(entry.get("radius", [500, 500, 100])[0]), 10.0,
                               key=key + "_r")
            if new != entry[field] or radius != entry.get("radius", [0])[0]:
                entry[field] = new
                r = list(entry.get("radius", [radius, radius, 100.0]))
                entry["radius"] = [radius, r[1], r[2]]
                edited = True

    time_state = st.select_slider("Pseudo-time state", options=list("T0 T1 T2 T3 T4".split()),
                                  value=scenario.time_state)
    if time_state != scenario.time_state:
        scenario.time_state = time_state
        edited = True
    if edited:
        invalidate()

    if st.button("Update reservoir state", type="primary"):
        invalidate()
    pipe = pipeline()
    states = stage("Building the four reservoir states", pipe.reservoir)
    grid = states.baseline.grid
    point = cursor_controls(grid)

    st.subheader("State and change")
    which = st.selectbox("Volume", ["ΔP", "ΔSw", "ΔSg", "pressure", "Sw", "Sg"])
    monitor = states.combined
    delta = monitor.difference(states.baseline)
    lookup = {
        "ΔP": (delta["dP"], "diverging", 1e5, "bar"),
        "ΔSw": (delta["dSw"], "diverging", 1.0, "fraction"),
        "ΔSg": (delta["dSg"], "diverging", 1.0, "fraction"),
        "pressure": (monitor.pressure, "sequential", 1e5, "bar"),
        "Sw": (monitor.sw, "sequential", 1.0, "fraction"),
        "Sg": (monitor.sg, "sequential", 1.0, "fraction"),
    }
    volume, kind, scale, unit = lookup[which]
    st.plotly_chart(ui.slice_figure(volume, grid, point, title=which, kind=kind,
                                    unit=unit, scale=scale, wells=pipe.wells()),
                    width="stretch")

    mask = states.baseline.reservoir_mask
    cell = grid.dx * grid.dy * grid.dz / 1e9
    a, b, c = st.columns(3)
    a.metric("Pressure-affected volume", f"{np.sum((np.abs(delta['dP']) > 1e5) & mask) * cell:.3f} km³",
             help="reservoir cells where |ΔP| exceeds 1 bar")
    b.metric("Water-affected volume", f"{np.sum((delta['dSw'] > 0.01) & mask) * cell:.3f} km³")
    c.metric("Gas-affected volume", f"{np.sum((delta['dSg'] > 0.01) & mask) * cell:.3f} km³")
    st.caption("The pressure halo is normally much the largest of the three. "
               "That asymmetry is why pressure and saturation are generated "
               "independently rather than tied together.")


def page_rockphysics() -> None:
    pipe = pipeline()
    earth = stage("Running the rock-physics chain", pipe.rockphysics)
    grid = pipe.domains.geology
    point = cursor_controls(grid)

    st.title("Rock physics")
    for warning in pipe.result.earth.rock_physics["baseline"].warnings:
        st.info(warning, icon="ℹ️")

    attribute = st.selectbox("Attribute", ["ai", "vp", "vs", "rho"], index=0)
    scale, unit, _ = theme.DISPLAY.get(attribute, (1.0, "", "sequential"))
    scenario = st.radio("Earth model", list(SCENARIO_NAMES), horizontal=True)
    st.plotly_chart(ui.slice_figure(
        getattr(earth.rock_physics[scenario], attribute), grid, point,
        title=f"{attribute.upper()} — {scenario}", kind="sequential",
        unit=unit, scale=scale, wells=pipe.wells()), width="stretch")

    st.subheader("Difference from the baseline")
    st.caption("Each of these is a separate earth model run through the whole "
               "chain, not a scaled copy of the combined answer.")
    deltas = earth.property_deltas(attribute)
    shown = st.radio("Difference", list(deltas), horizontal=True, key="delta_pick")
    st.plotly_chart(ui.slice_figure(
        deltas[shown], grid, point, title=f"Δ{attribute.upper()} — {shown}",
        kind="diverging", unit=unit, scale=scale, wells=pipe.wells()),
        width="stretch")

    st.subheader("Decomposition")
    parts = pipe.decompose()
    mask = parts["property_mask"]
    table = {}
    for name in ("d_pressure", "d_saturation", "d_sum", "d_combined", "d_interaction"):
        values = parts["property"][attribute][name][mask] / scale
        table[name] = dict(minimum=float(values.min()), maximum=float(values.max()),
                           rms=float(np.sqrt(np.mean(values**2))))
    st.dataframe(table, width="stretch")
    ratio = nonlinearity_ratio(parts["property"][attribute], mask)
    st.metric("Interaction / combined (RMS)", f"{100 * ratio:.2f} %",
              help="How far the pressure and saturation responses are from "
                   "simply adding. Rock physics alone, at this stage.")


def page_acquisition() -> None:
    pipe = pipeline()
    cfg = config()
    st.title("Acquisition & QC")

    acquisition = stage("Building the geometry", pipe.acquisition)
    st.markdown("**Sources, nodes, wells and the domain boundaries**")
    st.plotly_chart(ui.map_figure(
        wells=pipe.wells(), acquisition=acquisition,
        grid=pipe.domains.propagation, pml_nodes=cfg.solver.pml_nodes),
        width="stretch")
    a, b, c = st.columns(3)
    a.metric("Sources", f"{acquisition.n_sources:,}")
    b.metric("Nodes", f"{acquisition.n_receivers:,}")
    c.metric("Traces", f"{acquisition.n_traces:,}")

    from sim3d.acquisition.geometry import azimuth_distribution, offset_distribution
    left, right = st.columns(2)
    centres, counts = offset_distribution(acquisition, 20)
    left.markdown("**Offset distribution**")
    left.plotly_chart(ui.bar_figure(
        centres, counts, xlabel="offset (m)", ylabel="traces",
        width=float(np.diff(centres).mean()) * 0.9, height=300), width="stretch")
    az, az_counts = azimuth_distribution(acquisition, 24, min_offset=100.0)
    right.markdown("**Azimuth distribution**")
    right.plotly_chart(ui.bar_figure(
        az, az_counts, xlabel="azimuth (° clockwise from +y)", ylabel="traces",
        width=float(np.diff(az).mean()) * 0.9, height=300,
        colour=theme.SERIES[1]), width="stretch")
    st.caption("Geometric coverage only. Whether the wavefield actually reaches "
               "the target is a different question, and one the platform exists "
               "to ask.")

    st.subheader("Model QC")
    result = stage("Running QC", pipe.qc)
    failures = [c for c in result.checks if c.status is Status.FAIL]
    for check in result.checks:
        icon = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}[check.status.value]
        (st.error if check.status is Status.FAIL else
         st.warning if check.status is Status.WARNING else st.success)(
            f"{icon} {check.message}")
    if failures:
        st.error("QC failed. sim3d will not adjust the grid, the bandwidth or "
                 "the geometry for you — change the configuration.")

    st.subheader("Computational estimate")
    try:
        estimate = pipe.plan()
        st.code(estimate.describe(), language="text")
        st.success("Within the configured budget.")
    except Sim3DError as exc:
        st.error(str(exc))


def page_simulation() -> None:
    pipe = pipeline()
    st.title("Simulation & imaging")
    qc = stage("Running QC", pipe.qc)
    blocked = qc.failed
    if blocked:
        st.error("QC has failed for this configuration — see **Acquisition & QC**. "
                 "Fix it before modelling.")

    a, b = st.columns(2)
    if a.button("Run full-wave simulation", type="primary", disabled=blocked):
        progress = st.progress(0.0, text="modelling")
        total = {"n": 0}

        def report(name, done, count):
            total["n"] += 1
            progress.progress(min(total["n"] / (count * 4), 1.0),
                              text=f"{name}: shot {done} of {count}")
        try:
            pipe.simulate(progress=report)
        except Sim3DError as exc:
            st.error(str(exc))
        progress.empty()
    if b.button("Run 3D RTM", disabled=blocked or not pipe.result.gathers):
        progress = st.progress(0.0, text="migrating")
        total = {"n": 0}

        def report(name, done, count):
            total["n"] += 1
            progress.progress(min(total["n"] / (count * 4), 1.0),
                              text=f"{name}: shot {done} of {count}")
        try:
            pipe.migrate(progress=report)
        except Sim3DError as exc:
            st.error(str(exc))
        progress.empty()

    if not pipe.result.gathers:
        st.info("No shot gathers yet. Modelling four earth models is minutes of "
                "work, so it happens only when you ask for it.")
        return

    st.subheader("Shot gathers")
    scenario = st.radio("Earth model", list(pipe.result.gathers), horizontal=True)
    records = pipe.result.gathers[scenario]
    index = st.slider("Shot", 0, len(records) - 1, 0)
    record = records[index]
    st.plotly_chart(ui.gather_figure(
        record.traces, record.dt,
        title=f"{scenario} — shot {index + 1} of {len(records)}"), width="stretch")
    st.caption("Raw modelled amplitudes. Display gain is never applied to data "
               "that will be migrated or differenced.")

    if pipe.result.images:
        st.subheader("Migrated images")
        grid = pipe.domains.propagation
        point = cursor_controls(grid)
        which = st.radio("Image", list(pipe.result.images), horizontal=True,
                         key="image_pick")
        st.plotly_chart(ui.slice_figure(
            pipe.result.images[which].image, grid, point,
            title=f"RTM — {which}", kind="diverging", unit="amplitude",
            wells=pipe.wells()), width="stretch")


def page_fourd() -> None:
    pipe = pipeline()
    st.title("4D analysis")
    if not pipe.result.images:
        st.info("Run the simulation and RTM on the **Simulation & Imaging** page "
                "first. The property-space decomposition on the **Rock Physics** "
                "page needs neither and is available now.")
        return

    grid = pipe.domains.propagation
    point = cursor_controls(grid)
    parts = pipe.decompose()
    seismic = parts["seismic"]["rtm"]

    st.subheader("Seismic-space decomposition")
    which = st.radio("Volume", ["d_pressure", "d_saturation", "d_sum",
                                "d_combined", "d_interaction"],
                     horizontal=True)
    st.plotly_chart(ui.slice_figure(
        seismic[which], grid, point, title=which.replace("_", " "),
        kind="diverging", unit="amplitude", wells=pipe.wells()), width="stretch")

    a, b = st.columns(2)
    a.metric("Interaction / combined (RMS), seismic",
             f"{100 * nonlinearity_ratio(seismic):.2f} %")
    b.metric("Interaction / combined (RMS), property (AI)",
             f"{100 * nonlinearity_ratio(parts['property']['ai'], parts['property_mask']):.2f} %")
    st.caption("If the seismic figure is much the larger, most of the departure "
               "from superposition is being introduced after the rock physics — "
               "by finite-frequency interference, illumination and the migration "
               "operator, none of which a rock-physics model can anticipate.")

    st.subheader("Repeatability")
    st.dataframe({name: {"NRMS (%)": round(value, 3)}
                  for name, value in parts["seismic"]["nrms"].items()},
                 width="stretch")

    st.subheader("Response around a well")
    wells = pipe.wells()
    name = st.selectbox("Well", [w.name for w in wells])
    well = wells[name]
    profiles = {}
    for scenario in ("d_pressure", "d_saturation", "d_combined"):
        radii, means, counts = radial_profile(
            grid, well, np.abs(seismic[scenario]), bin_width=50.0, max_radius=800.0)
        profiles[scenario.replace("d_", "")] = means
    st.markdown(f"**Mean |4D amplitude| against distance from {name}**")
    st.plotly_chart(ui.series_figure(
        radii, profiles,
        xlabel="distance from the well (m)", ylabel="|amplitude|",
        colours={k: theme.SCENARIO_COLOUR.get(f"{k}_only", theme.SCENARIO_COLOUR.get(k))
                 for k in profiles}), width="stretch")
    st.caption("Plotting the seismic difference against the same radius axis as "
               "ΔP and ΔSw is the most direct way to see whether an anomaly "
               "tracks the pressure halo or the flood front.")


PAGE_FUNCTIONS = {
    "Project": page_project,
    "Geology": page_geology,
    "Wells & Reservoir": page_reservoir,
    "Rock Physics": page_rockphysics,
    "Acquisition & QC": page_acquisition,
    "Simulation & Imaging": page_simulation,
    "4D Analysis": page_fourd,
}


def main() -> None:
    st.set_page_config(page_title="sim3d", page_icon="🌊", layout="wide")
    PAGE_FUNCTIONS[sidebar()]()


if __name__ == "__main__":
    main()
