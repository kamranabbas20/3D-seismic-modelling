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
from sim3d.core.graph import explain as explain_dependencies
from sim3d.core.units import PSI, pa_to_psi, psi_to_pa, si_to_stb_per_day
from sim3d.io import ScenarioStore, ViewState
from sim3d.processing.sparse import LAYOUTS
from sim3d.ui import view3d
from sim3d.wells.completion import layer_intersections, resolve_completions
from sim3d.experiments.pipeline import Pipeline
from sim3d.fourd.decomposition import nonlinearity_ratio
from sim3d.fourd.metrics import nrms, radial_profile
from sim3d.fourd.scenarios import SCENARIO_NAMES
from sim3d.ui import components as ui
from sim3d.ui import theme
from sim3d.validation.physics import physics_table
from sim3d.validation.qc import Status

EXAMPLES = sorted(Path("examples/configs").glob("*.yaml")) if Path("examples/configs").is_dir() else []

PAGES = ("Project", "Geology", "3D Model", "Wells & Completions",
         "Flow Simulation", "Rock Physics", "Acquisition & QC",
         "Simulation & Imaging", "4D Analysis", "Scenarios")

WELL_TYPES = ("producer", "injector")


def store() -> ScenarioStore:
    if "store" not in st.session_state:
        st.session_state.store = ScenarioStore("scenarios")
    return st.session_state.store


def well_specs() -> list[dict]:
    """The editable well list, seeded from the pattern on first use.

    Once a well has been placed or edited the configuration carries the
    wells themselves and the pattern name stops being consulted, so the
    seeding happens exactly once.
    """
    cfg = config()
    if not cfg.wells.wells:
        cfg.wells.wells = [
            {"name": w.name, "role": w.role, "x": float(w.x), "y": float(w.y),
             "perforation": list(w.perforation), "completions": [],
             "control": ("water_rate" if w.role == "injector" else "liquid_rate"),
             "target": None, "bhp_limit_psi": None,
             "start_day": 0.0, "end_day": None}
            for w in pipeline().wells()]
    return cfg.wells.wells


def unique_name(prefix: str) -> str:
    taken = {w["name"] for w in config().wells.wells}
    index = 1
    while f"{prefix}{index}" in taken:
        index += 1
    return f"{prefix}{index}"


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
    for label, entries, field in (
            ("Pressure", scenario.pressure, "delta_p_bar"),
            ("Water fronts", scenario.water_fronts, "target_sw"),
            ("Gas", scenario.gas, "target_sg")):
        if not entries:
            continue
        st.markdown(f"**{label}**")
        for i, entry in enumerate(entries):
            c1, c2 = st.columns(2)
            key = f"{label}_{i}"
            if field == "delta_p_bar":
                shown = c1.slider(f"{entry['well']} ΔP (psi)", -1740.0, 1740.0,
                                  round(pa_to_psi(float(entry[field]) * 1e5)),
                                  10.0, key=key + "_v")
                new = psi_to_pa(shown) / 1e5
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
        "ΔP": (delta["dP"], "diverging", PSI, "psi"),
        "ΔSw": (delta["dSw"], "diverging", 1.0, "fraction"),
        "ΔSg": (delta["dSg"], "diverging", 1.0, "fraction"),
        "pressure": (monitor.pressure, "sequential", PSI, "psi"),
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
    threshold = psi_to_pa(10.0)
    a.metric("Pressure-affected volume",
             f"{np.sum((np.abs(delta['dP']) > threshold) & mask) * cell:.3f} km³",
             help="reservoir cells where |ΔP| exceeds 10 psi")
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

    _sparse_section(pipe)

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


def _sparse_section(pipe) -> None:
    """The second seismic mode: K vertical synthetics instead of a migration."""
    st.subheader("Sparse vertical synthetics")
    cfg = config().synthetic
    a, b, c = st.columns([1, 1, 2])
    layout = a.selectbox("Trace layout", list(LAYOUTS),
                         index=list(LAYOUTS).index(cfg.layout),
                         help="Where the K traces go. 'wells' answers the "
                              "question the wells were placed to ask.")
    count = b.number_input("K (lattice only)", 1, 400, int(cfg.count), 1,
                           disabled=(layout != "grid"))
    if layout != cfg.layout or int(count) != cfg.count:
        cfg.layout, cfg.count = layout, int(count)
        invalidate()
    c.caption("Seconds, not minutes: no wavefield is propagated. These are "
              "1D normal-incidence traces, so they carry no lateral "
              "propagation, no offset, no migration — and are never an image.")

    if st.button("Run sparse synthetic"):
        try:
            stage("Building vertical synthetics", pipe.synthetic)
        except Sim3DError as exc:
            st.error(str(exc))

    synthetics = pipe.result.synthetics
    if not synthetics:
        return
    base = synthetics["baseline"]
    st.caption(base.label)

    which = st.selectbox("Trace", list(base.names), key="sparse_trace")
    i = base.index(which)
    domain = st.radio("Axis", ["time", "depth"], horizontal=True, key="sparse_axis")
    if domain == "time":
        axis, ylabel = base.times, "two-way time (s)"
        curves = {n: synthetics[n].traces[i] for n in synthetics}
    else:
        axis, ylabel = base.depths, "depth (m)"
        curves = {n: synthetics[n].depth_traces[i] for n in synthetics}
    st.plotly_chart(ui.trace_figure(
        axis, curves, ylabel=ylabel,
        colours=theme.SCENARIO_COLOUR), width="stretch")

    differences = {n: curves[n] - curves["baseline"]
                   for n in curves if n != "baseline"}
    if differences:
        st.markdown("**Difference from baseline**")
        st.plotly_chart(ui.trace_figure(
            axis, differences, ylabel=ylabel,
            colours=theme.SCENARIO_COLOUR), width="stretch")

    rows = {}
    for name in list(synthetics)[1:]:
        rows[name] = {loc.name: round(nrms(base.traces[k],
                                           synthetics[name].traces[k]), 3)
                      for k, loc in enumerate(base.locations)}
    if rows:
        st.markdown("**NRMS against baseline, per trace (%)**")
        st.dataframe(rows, width="stretch")
        st.caption("Larger than the NRMS of a migrated volume, and not "
                   "comparable with it: this is measured on the trace at the "
                   "well, where the change is, while the volume figure is "
                   "diluted by every cell that did not change.")


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



def page_model3d() -> None:
    """Requirements 5 and 6: the interactive 3D model."""
    pipe = pipeline()
    geology = stage("Building geology", pipe.geology)
    grid = geology.grid
    view = st.session_state.setdefault("view", ViewState())
    st.title("3D model")

    controls, canvas = st.columns([1, 3])
    with controls:
        st.markdown("**Layers**")
        visible_layers = [
            layer.name for layer in geology.layers
            if st.checkbox(layer.name, value=(layer.name not in view.visible_layers
                                              or not view.visible_layers),
                           key=f"layer_{layer.name}")]
        opacity = st.slider("Layer transparency", 0.05, 1.0, 0.35, 0.05,
                            help="Overburden layers hide the reservoir and the "
                                 "wells; this is how you see through them.")

        st.markdown("**Wells**")
        which = st.radio("Show", ("all", "producers", "injectors", "none"),
                         horizontal=True, key="well_filter")
        wells = list(pipe.wells())
        if which == "producers":
            wells = [w for w in wells if w.role == "producer"]
        elif which == "injectors":
            wells = [w for w in wells if w.role == "injector"]
        elif which == "none":
            wells = []
        chosen = [w for w in wells
                  if st.checkbox(w.name, value=True, key=f"well3d_{w.name}")]

        st.markdown("**Other**")
        show_faults = st.checkbox("Faults", value=True)
        show_volume = st.checkbox("Property volume", value=False)
        property_name = st.selectbox(
            "Property", ("porosity", "permeability", "Vsh", "pressure",
                         "water saturation", "ΔP", "ΔSw"),
            disabled=not show_volume)
        volume_opacity = st.slider("Volume opacity", 0.02, 0.6, 0.15, 0.02,
                                   disabled=not show_volume)
        exaggeration = st.slider("Vertical exaggeration", 0.5, 5.0, 1.5, 0.1)
        orthographic = st.checkbox("Orthographic projection", value=False)
        if st.button("Reset camera"):
            st.session_state.pop("camera", None)

    traces = []
    shades = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
              "#184f95", "#0d366b")
    built = view3d.built_horizons(geology)
    for index, layer in enumerate(geology.layers):
        if layer.name not in visible_layers:
            continue
        reservoir = geology.reservoir_mask[geology.layer_index == index].any()
        traces.append(view3d.horizon_surface(
            grid, built[layer.name], layer.name, shades[index % len(shades)],
            opacity=min(1.0, opacity * (2.2 if reservoir else 1.0))))
    if show_faults:
        traces += [view3d.fault_mesh(f, grid, opacity=0.35) for f in geology.faults]

    completions = stage("Resolving completions", pipe.completions)
    for well in chosen:
        try:
            intervals = resolve_completions(well, geology, completions[well.name])
        except Sim3DError:
            intervals = []
        traces += view3d.well_traces(well, grid, intervals,
                                     selected=(well.name == st.session_state.get("selected_well")))

    note = ""
    if show_volume:
        volume, kind, mask = _volume_for(pipe, geology, property_name)
        if volume is not None:
            trace, stride = view3d.property_volume(
                grid, volume, property_name, kind=kind, opacity=volume_opacity,
                mask=mask)
            traces.append(trace)
            if stride != (1, 1, 1):
                note = (f"Volume decimated by {stride} for the browser. Use the "
                        f"sections on the other pages for quantitative reading.")

    with canvas:
        st.plotly_chart(view3d.scene(traces, grid, height=760,
                                     orthographic=orthographic,
                                     vertical_exaggeration=exaggeration),
                        width="stretch")
        if note:
            st.caption(note)
        st.caption("Drag to rotate, scroll to zoom, shift-drag to pan. Producers "
                   "are green and injectors blue, with the open completion "
                   "intervals drawn thick.")


def _volume_for(pipe, geology, name):
    """Resolve a property name to (values, colour job, mask)."""
    reservoir = geology.reservoir_mask
    if name == "porosity":
        return geology.porosity, "sequential", reservoir
    if name == "permeability":
        return geology.permeability_md, "sequential", reservoir
    if name == "Vsh":
        return geology.vsh, "sequential", None
    states = stage("Building reservoir states", pipe.reservoir)
    baseline, monitor = states.baseline, states.combined
    if name == "pressure":
        return pa_to_psi(monitor.pressure), "sequential", reservoir
    if name == "water saturation":
        return monitor.sw, "sequential", reservoir
    delta = monitor.difference(baseline)
    if name == "ΔP":
        return pa_to_psi(delta["dP"]), "diverging", reservoir
    if name == "ΔSw":
        return delta["dSw"], "diverging", reservoir
    return None, "sequential", None


def page_wells() -> None:
    """Requirements 1, 2, 10 and 12: place, edit and complete wells."""
    pipe = pipeline()
    geology = stage("Building geology", pipe.geology)
    grid = geology.grid
    specs = well_specs()
    st.title("Wells & completions")

    st.markdown("**Add a well**")
    a, b, c = st.columns([1, 1, 2])
    adding = a.toggle("Add well", key="adding",
                      help="Then click the map to place it.")
    well_type = b.radio("Type", WELL_TYPES, horizontal=True, key="new_role",
                        disabled=not adding)
    if adding:
        c.info(f"Click the map to drop a **{well_type}**. "
               f"{'Green' if well_type == 'producer' else 'Blue'} filled circle.",
               icon="🖱️")

    figure = ui.map_figure(wells=pipe.wells(), grid=pipe.domains.propagation,
                           pml_nodes=config().solver.pml_nodes,
                           bounds=grid.bounds)
    _add_placement_layer(figure, grid, enabled=adding)
    event = st.plotly_chart(figure, width="stretch", key="wellmap",
                            on_select="rerun" if adding else "ignore",
                            selection_mode="points")

    if adding and event and event.get("selection", {}).get("points"):
        point = event["selection"]["points"][-1]
        x, y = float(point["x"]), float(point["y"])
        if not grid.contains((x, y, 0.5 * sum(grid.bounds[2]))):
            st.error("That point is outside the geological model.")
        else:
            prefix = "P" if well_type == "producer" else "I"
            specs.append({
                "name": unique_name(prefix), "role": well_type, "x": x, "y": y,
                "perforation": [grid.bounds[2][0], grid.bounds[2][1]],
                "completions": [], "control": ("liquid_rate" if
                                               well_type == "producer" else "water_rate"),
                "target": None, "bhp_limit_psi": None,
                "start_day": 0.0, "end_day": None})
            invalidate()
            st.rerun()

    if not specs:
        st.info("No wells yet. Turn on **Add well** and click the map.")
        return

    st.markdown("**Edit a well**")
    names = [w["name"] for w in specs]
    selected = st.selectbox("Well", names, key="selected_well")
    spec = next(w for w in specs if w["name"] == selected)
    well = pipe.wells()[selected]

    edit, complete = st.columns(2)
    with edit:
        new_name = st.text_input("Name", spec["name"])
        new_role = st.radio("Type", WELL_TYPES,
                            index=WELL_TYPES.index(spec["role"]), horizontal=True)
        (x0, x1), (y0, y1) = grid.bounds[0], grid.bounds[1]
        new_x = st.number_input("x (m)", float(x0), float(x1), float(spec["x"]),
                                step=float(grid.dx))
        new_y = st.number_input("y (m)", float(y0), float(y1), float(spec["y"]),
                                step=float(grid.dy))
        changed = (new_name != spec["name"] or new_role != spec["role"]
                   or new_x != spec["x"] or new_y != spec["y"])
        if changed and st.button("Apply", type="primary"):
            if new_name != spec["name"] and new_name in names:
                st.error(f"There is already a well called {new_name!r}.")
            else:
                spec.update(name=new_name, role=new_role, x=new_x, y=new_y)
                if new_role != spec.get("role"):
                    spec["control"] = ("liquid_rate" if new_role == "producer"
                                       else "water_rate")
                invalidate()
                st.rerun()
        if st.button("Delete well", type="secondary"):
            specs.remove(spec)
            invalidate()
            st.rerun()

    with complete:
        st.caption("Completions are chosen by geological unit; the local top "
                   "and base come from the model, so they stay correct when "
                   "the well moves across a dipping or faulted structure.")
        try:
            units = layer_intersections(well, geology)
        except Sim3DError as exc:
            st.error(str(exc))
            return
        current = set(spec.get("completions") or
                      [c.layer for c in pipe.completions().get(selected, [])])
        chosen = []
        for unit in units:
            label = (f"{unit.name} — {unit.top:,.0f}–{unit.base:,.0f} m "
                     f"({unit.net_thickness:.0f} m net, {unit.permeability_md:,.0f} mD)")
            if st.checkbox(label, value=unit.name in current,
                           key=f"comp_{selected}_{unit.name}",
                           disabled=not unit.is_reservoir and unit.name not in current):
                chosen.append(unit.name)
        if chosen != list(spec.get("completions") or []):
            spec["completions"] = chosen
            invalidate()

    st.markdown("**Control and schedule**")
    controls = stage("Suggesting rates", pipe.controls)
    control = controls.get(selected)
    if control is not None:
        d, e, f, g = st.columns(4)
        modes = (("liquid_rate", "oil_rate", "bhp") if spec["role"] == "producer"
                 else ("water_rate", "bhp"))
        mode = d.selectbox("Control", modes,
                           index=modes.index(spec.get("control", modes[0]))
                           if spec.get("control") in modes else 0)
        if mode == "bhp":
            value = e.number_input("Target BHP (psi)", 0.0, 20000.0,
                                   float(spec.get("target")
                                         or pa_to_psi(control.target)), 25.0)
        else:
            value = e.number_input("Target rate (STB/day)", 0.0, 500000.0,
                                   float(spec.get("target")
                                         or si_to_stb_per_day(control.target)), 100.0)
        start = f.number_input("Start (day)", 0.0, 1e5, float(spec.get("start_day", 0.0)))
        end_value = spec.get("end_day")
        end = g.number_input("End (day, 0 = never)", 0.0, 1e5,
                             float(end_value or 0.0))
        st.caption(f"Suggested: {control.describe(spec['role'])} — "
                   f"{control.provenance}")
        if st.button("Apply control"):
            spec.update(control=mode, target=value, start_day=start,
                        end_day=(None if end == 0.0 else end))
            invalidate()
            st.rerun()

    for warning in pipe.result.rate_warnings:
        st.warning(warning, icon="⚠️")


def _add_placement_layer(figure, grid, enabled: bool) -> None:
    """A clickable lattice of candidate positions.

    Plotly reports selections on *existing* points, not on empty canvas, so
    placing a well by clicking needs something to click. This lays down an
    invisible lattice at the geological grid's own resolution; the click
    lands on the nearest node, which is where the well would be snapped to
    anyway.
    """
    if not enabled:
        return
    import numpy as np
    step = max(1, grid.nx // 60), max(1, grid.ny // 60)
    x = grid.axis(0)[::step[0]]
    y = grid.axis(1)[::step[1]]
    gx, gy = np.meshgrid(x, y, indexing="ij")
    figure.add_trace(go_scatter_lattice(gx.ravel(), gy.ravel()))


def go_scatter_lattice(x, y):
    import plotly.graph_objects as go
    return go.Scatter(x=x, y=y, mode="markers", name="click to place",
                      marker=dict(size=14, color="rgba(0,0,0,0)"),
                      hovertemplate="place here<br>x=%{x:,.0f} m<br>"
                                    "y=%{y:,.0f} m<extra></extra>",
                      showlegend=False)


def page_flow() -> None:
    """Requirements 7 and 11: run the flow model and read the well histories."""
    pipe = pipeline()
    cfg = config()
    st.title("Flow simulation")

    a, b, c, d = st.columns(4)
    duration = a.number_input("Duration (days)", 30.0, 36500.0,
                              float(cfg.simulation.duration_days), 30.0)
    report = b.number_input("Report every (days)", 1.0, 3650.0,
                            float(cfg.simulation.report_every_days), 1.0)
    step = c.number_input("Max timestep (days)", 0.1, 365.0,
                          float(cfg.simulation.max_timestep_days), 1.0)
    gravity = d.checkbox("Gravity", value=cfg.simulation.gravity)
    if (duration, report, step, gravity) != (
            cfg.simulation.duration_days, cfg.simulation.report_every_days,
            cfg.simulation.max_timestep_days, cfg.simulation.gravity):
        cfg.simulation.duration_days = duration
        cfg.simulation.report_every_days = report
        cfg.simulation.max_timestep_days = step
        cfg.simulation.gravity = gravity
        invalidate()

    if st.button("Run flow simulation", type="primary"):
        bar = st.progress(0.0, text="simulating")
        try:
            pipeline().flow(progress=lambda day, total: bar.progress(
                min(day / total, 1.0), text=f"day {day:,.0f} of {total:,.0f}"))
        except Sim3DError as exc:
            st.error(str(exc))
        bar.empty()

    flow = pipe.result.flow
    if flow is None:
        st.info("Not run yet. The flow model is seconds to minutes, not hours — "
                "but it still waits to be asked.")
        return

    st.success(f"{flow.n_timesteps:,} timesteps · material balance closes to "
               f"{flow.material_balance_error:.2e} of throughput")

    st.markdown("**Field state through time**")
    day = st.select_slider("Day", options=[float(d) for d in flow.days],
                           value=float(flow.days[-1]))
    grid = pipe.geology().grid
    point = cursor_controls(grid)
    what = st.radio("Volume", ("pressure", "water saturation", "ΔP", "ΔSw"),
                    horizontal=True)
    pressure, saturation = flow.at(day)
    p0, s0 = flow.at(flow.days[0])
    mask = pipe.geology().reservoir_mask
    lookup = {
        "pressure": (np.where(mask, pa_to_psi(pressure), np.nan), "sequential", "psi"),
        "water saturation": (np.where(mask, saturation, np.nan), "sequential", "fraction"),
        "ΔP": (np.where(mask, pa_to_psi(pressure - p0), np.nan), "diverging", "psi"),
        "ΔSw": (np.where(mask, saturation - s0, np.nan), "diverging", "fraction"),
    }
    volume, kind, unit = lookup[what]
    st.plotly_chart(ui.slice_figure(volume, grid, point,
                                    title=f"{what} — day {day:,.0f}", kind=kind,
                                    unit=unit, wells=pipe.wells()), width="stretch")

    st.markdown("**Well histories**")
    for name, history in flow.wells.items():
        arrays = history.arrays()
        if arrays["days"].size == 0:
            continue
        st.caption(history.summary())
        left, right = st.columns(2)
        left.plotly_chart(ui.series_figure(
            arrays["days"],
            {"oil": si_to_stb_per_day(np.abs(arrays["oil_rate"])),
             "water": si_to_stb_per_day(np.abs(arrays["water_rate"]))},
            xlabel="day", ylabel="rate (STB/day)", height=260,
            colours={"oil": theme.SERIES[1], "water": theme.SERIES[0]}),
            width="stretch", key=f"rate_{name}")
        right.plotly_chart(ui.series_figure(
            arrays["days"],
            {"BHP": pa_to_psi(arrays["bhp"]),
             "water cut ×1000": history.water_cut * 1000.0},
            xlabel="day", ylabel="psi  /  water cut × 1000", height=260),
            width="stretch", key=f"bhp_{name}")


def page_scenarios() -> None:
    """Requirements 13, 14 and 15: save, load, duplicate, delete, compare."""
    cfg = config()
    st.title("Scenarios")
    st.caption("A scenario stores its definition, its flow results and its "
               "seismic results in three separate files, so editing a setup "
               "never rewrites a migration and a definition stays small "
               "enough to read.")

    save, manage = st.columns(2)
    with save:
        st.markdown("**Save**")
        name = st.text_input("Name", cfg.project.name)
        keep = st.checkbox("Include results", value=True,
                           help="Flow and seismic arrays. Turn off to store "
                                "the definition only.")
        if st.button("Save scenario", type="primary"):
            pipe = st.session_state.get("pipeline")
            simulation = seismic = None
            if keep and pipe is not None:
                if pipe.result.flow is not None:
                    simulation = ScenarioStore.pack_flow(pipe.result.flow)
                if pipe.result.images:
                    seismic = ScenarioStore.pack_images(pipe.result.images)
            try:
                info = store().save(name, cfg, view=st.session_state.get("view"),
                                    simulation=simulation, seismic=seismic)
                st.success(f"Saved {info.name!r} [{info.config_hash}]")
            except Sim3DError as exc:
                st.error(str(exc))

    with manage:
        st.markdown("**Manage**")
        listing = store().list()
        if not listing:
            st.info("Nothing saved yet.")
        else:
            chosen = st.selectbox("Scenario", [s.name for s in listing])
            info = next(s for s in listing if s.name == chosen)
            st.caption(info.describe())
            load, duplicate, delete = st.columns(3)
            if load.button("Load"):
                scenario = store().load(chosen, results=False)
                st.session_state.config = scenario.config
                st.session_state.view = scenario.view
                invalidate()
                st.rerun()
            if duplicate.button("Duplicate"):
                try:
                    store().duplicate(chosen, f"{chosen} copy", results=False)
                    st.rerun()
                except Sim3DError as exc:
                    st.error(str(exc))
            if delete.button("Delete"):
                store().delete(chosen)
                st.rerun()

    listing = store().list()
    if len(listing) >= 2:
        st.divider()
        st.markdown("**Compare two scenarios**")
        left, right = st.columns(2)
        a = left.selectbox("A", [s.name for s in listing], index=0, key="cmp_a")
        b = right.selectbox("B", [s.name for s in listing], index=1, key="cmp_b")
        if a != b:
            first = store().load(a, results=False).config
            second = store().load(b, results=False).config
            st.code(explain_dependencies(first, second), language="text")
            st.caption("What differs between the two setups, and which stages "
                       "would have to be rerun to turn one into the other.")
        else:
            st.caption("Choose two different scenarios.")


PAGE_FUNCTIONS = {
    "Project": page_project,
    "Geology": page_geology,
    "3D Model": page_model3d,
    "Wells & Completions": page_wells,
    "Flow Simulation": page_flow,
    "Rock Physics": page_rockphysics,
    "Acquisition & QC": page_acquisition,
    "Simulation & Imaging": page_simulation,
    "4D Analysis": page_fourd,
    "Scenarios": page_scenarios,
}


def main() -> None:
    st.set_page_config(page_title="sim3d", page_icon="🌊", layout="wide")
    PAGE_FUNCTIONS[sidebar()]()


if __name__ == "__main__":
    main()
