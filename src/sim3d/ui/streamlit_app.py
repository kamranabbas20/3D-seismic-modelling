"""The research GUI (spec sections 108-110, 119, 121-124).

Streamlit is only a frontend.  Every page calls
:class:`sim3d.experiments.pipeline.Pipeline`, the same object the CLI drives,
and no page computes anything scientific of its own.

Three behaviours matter more than the layout:

**The app sets experiments up and models what is cheap; it does not migrate.**
Flow, rock physics, sim2seis and the sparse vertical synthetics all run here,
because none of them propagates a wavefield.  Full-wave modelling and RTM used
to run here too, behind two buttons, and that was always the wrong place for
them: a survey worth migrating is hours of work, a browser session is not a
batch queue, and a page that blocks for four hours can neither report progress
honestly nor survive a reload.  What the app is good at is deciding *what* to
run - the geometry, the sampling, the imaging choices, and the checks that
catch a survey which will alias its operator before any of it costs anything.
So **Migration Setup** writes a configuration instead, and
``examples/run_migration.py`` runs it wherever the cores are; **4D Analysis**
reads the resulting ``images.npz`` back in.

**Expensive work never happens because a slider moved** (section 108).  Moving
a front radius updates the reservoir state and the rock physics - seconds of
work - and marks everything downstream stale.  It does not re-run it.

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
from sim3d.fourd.noise import NoiseModel
from sim3d.geology.bodies import BODY_TYPES, GeoBody
from sim3d.wave.wavelets import DEFAULT_ORMSBY_CORNERS, WAVELETS
from sim3d.geology.facies import FACIES
from sim3d.geology.templates import DEFAULT_UNITS, TEMPLATES, template
from sim3d.geology.faults import build_faults
from sim3d.geology.picking import (picks_to_profile, profile_is_empty,
                                   profile_to_picks)
from sim3d.rockphysics.dryframe import DRY_FRAME_MODELS
from sim3d.core.graph import explain as explain_dependencies
from sim3d.core.units import PSI, pa_to_psi, psi_to_pa, si_to_stb_per_day
from sim3d.io import ScenarioStore, ViewState
from sim3d.processing.sim2seis import build_stacks
from sim3d.processing.sparse import LAYOUTS
from sim3d.ui import view3d
from sim3d.wells.completion import layer_intersections, resolve_completions
from sim3d.experiments.pipeline import Pipeline
from sim3d.fourd.decomposition import nonlinearity_ratio
from sim3d.acquisition.geometry import (
    azimuth_distribution, fold_map, offset_distribution, sampling_report,
)
from sim3d.imaging.rtm import IMAGING_CONDITIONS
from sim3d.fourd.metrics import nrms, radial_profile
from sim3d.fourd.scenarios import SCENARIO_NAMES
from sim3d.ui import components as ui
from sim3d.ui import theme
from sim3d.validation.physics import physics_table
from sim3d.validation.qc import Status

EXAMPLES = sorted(Path("examples/configs").glob("*.yaml")) if Path("examples/configs").is_dir() else []

#: The app sets an experiment up and models everything that is cheap; the
#: migration itself is a batch job somewhere else, so "Migration Setup"
#: builds and validates the configuration for it rather than running it.
PAGES = ("Project", "Geology", "3D Model", "Wells & Completions",
         "Flow Simulation", "Rock Physics", "Synthetic Volume",
         "Acquisition & QC", "Migration Setup", "4D Analysis",
         "Scenarios")

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


def unique_name(prefix: str, specs=None) -> str:
    """The first free ``prefix<n>``, against ``specs`` or the live config.

    Taking the list as an argument keeps the naming testable without a
    Streamlit session, and keeps it honest when the caller is working on a
    list it has not yet stored.
    """
    taken = {w["name"] for w in (config().wells.wells if specs is None else specs)}
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
    # Saving is reachable from every page, not only from the one that builds a
    # migration: an experiment is worth keeping the moment its geology is
    # right, and losing a session's setup to a browser reload is the kind of
    # loss that makes people stop using a tool.
    import yaml as _yaml
    cfg = config()
    _text = _yaml.safe_dump(cfg.to_dict(), sort_keys=False,
                            default_flow_style=False)
    st.sidebar.download_button(
        "Save configuration", _text,
        file_name=f"{cfg.project.name or 'experiment'}.yaml".replace(" ", "_"),
        mime="application/x-yaml", width="stretch",
        help="The complete experiment as YAML — every section, loadable by "
             "the CLI and by the migration runner. This is the file the "
             "Migration Setup page hands to a batch job.")
    st.sidebar.caption(f"config hash `{config().short_hash}`")
    pipe = st.session_state.get("pipeline")
    if pipe is not None:
        done = [name for name, ok in (
            ("geology", pipe.result.geology is not None),
            ("flow", pipe.result.flow is not None),
            ("rock physics", pipe.result.earth is not None),
            ("traces", bool(pipe.result.synthetics)),
            ("volume", bool(pipe.result.volumes)),
            ("gathers", bool(pipe.result.gathers)),
            ("images", bool(pipe.result.images))) if ok]
        st.sidebar.caption("computed: " + (", ".join(done) if done else "nothing yet"))
        if pipe.result.timings:
            total = sum(pipe.result.timings.values())
            slowest = max(pipe.result.timings.items(), key=lambda kv: kv[1])
            st.sidebar.caption(
                f"time spent: {_format_seconds(total)} "
                f"(most of it {slowest[0]}, {_format_seconds(slowest[1])})")
    return page


def _format_seconds(seconds: float) -> str:
    """Seconds, minutes or hours - whichever reads without counting zeros."""
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 120.0:
        return f"{seconds:.1f} s"
    if seconds < 7200.0:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


#: Stage keys in the order the pipeline runs them, so the table reads as a
#: sequence rather than as whatever order the dictionary happened to fill.
STAGE_ORDER = ("geology", "reservoir", "flow", "rockphysics",
               "preview", "synthetic", "sim2seis", "time shifts",
               "simulate", "migrate")


def _timings_table(pipe) -> dict | None:
    """What each stage of this configuration actually cost.

    The pipeline has timed every stage since it was written and nothing in
    the GUI ever showed it, so the only way to find out whether the flow or
    the rock physics was the wait was to run the CLI. Cached stages are the
    point of the table as much as slow ones: a stage that is not listed has
    not been run for this configuration.
    """
    timings = dict(pipe.result.timings)
    if not timings:
        return None
    total = sum(timings.values())
    ordered = ([k for k in STAGE_ORDER if k in timings]
               + [k for k in timings if k not in STAGE_ORDER])
    return {
        "elapsed": {k: _format_seconds(timings[k]) for k in ordered},
        "share": {k: f"{timings[k] / total:.0%}" if total else "—"
                  for k in ordered},
    }


def _timings_panel(pipe, title: str = "Time spent") -> None:
    table = _timings_table(pipe)
    st.subheader(title)
    if table is None:
        st.caption("Nothing has been computed for this configuration yet. "
                   "Every stage is timed as it runs and appears here.")
        return
    st.dataframe(table, width="stretch")
    total = sum(pipe.result.timings.values())
    st.caption(
        f"{_format_seconds(total)} in total for this configuration. Editing a "
        f"scientific input drops the stages downstream of it, so a stage that "
        f"vanishes from this table is one that has to be paid for again.")


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

    _timings_panel(pipeline())

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

    _structure_editor(pipe)

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
    with left:
        _layer_properties(geology)
    right.subheader("Faults")
    right.code(geology.faults.describe(), language="text")
    right.caption("Faults are kinematic: they displace the stratigraphy and "
                  "attenuate transport across the plane. They do not solve for "
                  "stress.")

    _fault_editor(pipe, geology)
    _geobody_editor(pipe, geology)


#: Columns of the layer-cake editor, in the order they are shown.
_UNIT_COLUMNS = ("name", "facies", "thickness", "reservoir", "pinch out",
                 "azimuth", "from", "to", "porosity", "Vsh", "NTG")

_STRUCTURES = ("flat", "dipping", "anticline", "syncline")


def _units_to_rows(units: list[dict]) -> list[dict]:
    """The configured units as editor rows, pinchouts flattened into columns."""
    rows = []
    for unit in units:
        pinch = unit.get("pinch_out") or {}
        shape = str(pinch.get("shape", "wedge")) if pinch else "none"
        rows.append({
            "name": str(unit.get("name", "")),
            "facies": str(unit.get("facies", "shale")),
            "thickness": float(unit.get("thickness", 100.0)),
            "reservoir": bool(unit.get("is_reservoir", False)),
            "pinch out": shape,
            "azimuth": float(pinch.get("azimuth", 90.0)),
            "from": float(pinch.get("start", pinch.get("radius", [0.3, 0.3])[0]
                                    if shape == "lens" else 0.3)),
            "to": float(pinch.get("end", pinch.get("radius", [0.3, 0.3])[1]
                                  if shape == "lens" else 0.8)),
            "porosity": unit.get("porosity"),
            "Vsh": unit.get("vsh"),
            "NTG": unit.get("ntg"),
        })
    return rows


def _rows_to_units(rows, extent) -> list[dict]:
    """Editor rows back into unit dictionaries, blanks left unset.

    A blank property column means *take the facies default*, which is not the
    same as zero - so an empty cell has to stay absent rather than become a
    number nobody chose.
    """
    units: list[dict] = []
    for row in rows:
        name = str(row.get("name") or "").strip()
        if not name:
            continue                      # an empty row is a row being added
        unit = {"name": name,
                "facies": str(row.get("facies") or "shale"),
                "thickness": float(row.get("thickness") or 0.0),
                "is_reservoir": bool(row.get("reservoir"))}
        for key, column in (("porosity", "porosity"), ("vsh", "Vsh"),
                            ("ntg", "NTG")):
            value = row.get(column)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue
            unit[key] = float(value)
        shape = str(row.get("pinch out") or "none")
        if shape != "none":
            azimuth = float(row.get("azimuth", 90.0))
            start = float(row.get("from", 0.3))
            end = float(row.get("to", 0.8))
            if shape == "lens":
                unit["pinch_out"] = {
                    "shape": "lens", "azimuth": azimuth,
                    "radius": [start * extent[0], end * extent[1]]}
            else:
                unit["pinch_out"] = {"shape": "wedge", "azimuth": azimuth,
                                     "start": start, "end": end}
        units.append(unit)
    return units


def _template_parameters(name: str, current: dict, extent) -> dict:
    """Number and checkbox inputs for whatever a template's signature declares.

    Read off the signature rather than written out per template, so every
    template's own parameters - dip, fold amplitude, gross thickness, the
    number of stacked units - are editable the moment they exist, and a
    template that gains one does not need this page changed.
    """
    import inspect

    builder = TEMPLATES[name]
    values = dict(current)
    editable = [(key, param.default)
                for key, param in inspect.signature(builder).parameters.items()
                if key not in ("extent", "units", "structure")
                and isinstance(param.default, (int, float, bool))]
    if not editable:
        st.caption("This template takes no numerical parameters.")
        return values

    columns = st.columns(min(len(editable), 4))
    for index, (key, default) in enumerate(editable):
        column = columns[index % len(columns)]
        shown = values.get(key, default)
        label = key.replace("_", " ").capitalize()
        widget_key = f"tmpl_{name}_{key}"
        if isinstance(default, bool):
            values[key] = column.checkbox(label, value=bool(shown), key=widget_key)
        elif isinstance(default, int):
            values[key] = int(column.number_input(
                label, 1, 60, int(shown), 1, key=widget_key))
        else:
            values[key] = float(column.number_input(
                label, 0.0, 10000.0, float(shown),
                1.0 if abs(float(default)) < 100 else 10.0, key=widget_key))
    return values


def _stratigraphy_warnings(layers, grid, frequency: float) -> list[str]:
    """What is about to be unrepresentable, said before it is built.

    Thicknesses are measured on what the *model* holds, not on what the table
    asked for: a unit whose base falls past the bottom of the geological
    domain is only as thick as the room left for it, and the last unit always
    runs to the base whatever thickness it was given.
    """
    top_of_model = grid.origin[2]
    base = top_of_model + grid.extent[2]
    tops = [layer.top.on_grid(grid) for layer in layers]

    notes = []
    for index, layer in enumerate(layers):
        upper = np.clip(tops[index], top_of_model, base)
        lower = np.clip(tops[index + 1] if index + 1 < len(tops)
                        else np.full_like(upper, base), top_of_model, base)
        thickness = lower - upper
        peak = float(np.max(thickness))
        if peak <= 1e-6:
            notes.append(
                f"**{layer.name}** has no thickness inside the model and will "
                f"not appear in it. Either the units above it fill the domain, "
                f"or it pinches out everywhere.")
            continue
        if peak < 2 * grid.dz:
            notes.append(
                f"**{layer.name}** is at most {peak:,.0f} m thick against a "
                f"{grid.dz:,.0f} m cell — fewer than two cells, so the grid "
                f"does not carry it whatever the table says.")
        if float(np.min(thickness)) <= 1e-6:
            notes.append(f"**{layer.name}** pinches out inside the model.")

    if tops and float(np.min(tops[0])) < top_of_model - 1e-6:
        notes.append(
            f"The first unit starts above the model at "
            f"{float(np.min(tops[0])):,.0f} m. Raise the datum, or extend the "
            f"geological domain upwards.")
    deepest = float(np.max(tops[-1])) if tops else 0.0
    if deepest > base + 1e-6:
        notes.append(
            f"The stack reaches {deepest:,.0f} m against a model base at "
            f"{base:,.0f} m, so the deepest units are cut off. Thin the units "
            f"above them, or deepen the geological domain.")
    return notes


def _horizon_editor(pipe, parameters: dict, layers) -> tuple[dict, dict]:
    """Draw one unit's base on a section.

    Returns the parameters with the drawing injected for the preview, and
    what the preview needs to show it. Nothing is committed here: the
    drawing is one more edit alongside the table, and the single
    "Apply stratigraphy" button below commits the lot.
    """
    grid = pipe.domains.geology
    units = list(parameters.get("units") or [])
    names = [unit.get("name", f"unit_{i + 1}") for i, unit in enumerate(units)]
    if not names:
        return parameters, {}

    st.markdown("**Draw a horizon**")
    drawing = st.toggle(
        "Draw on the section", key="draw_horizon",
        help="Click the section to say where the base of a unit should sit. "
             "What is stored is the thickness that implies, measured from "
             "the unit's own top and clamped at zero — so a base drawn above "
             "that top is a pinchout rather than a model that cannot exist.")
    if not drawing:
        st.session_state.pop("horizon_picks", None)
        return parameters, {}

    a, b = st.columns([2, 1])
    unit_name = a.selectbox("Unit", names, key="horizon_unit")
    axis = 0 if b.radio("Along", ("x", "y"), horizontal=True,
                        key="horizon_axis") == "x" else 1
    index = names.index(unit_name)

    # Switching unit or axis starts a fresh drawing, seeded from whatever
    # that unit already carries so an existing profile can be adjusted
    # rather than only replaced.
    token = (unit_name, axis)
    if st.session_state.get("horizon_token") != token:
        st.session_state.horizon_token = token
        stored = units[index].get("thickness_profile")
        st.session_state.horizon_picks = (
            profile_to_picks(stored, layers, index, grid) if stored else [])
    picks = st.session_state.setdefault("horizon_picks", [])

    st.caption(f"Clicking snaps to the geological grid. **{unit_name}** is "
               f"outlined below; everything else is faded. The thickness is "
               f"taken along {'x' if axis == 0 else 'y'} and is constant "
               f"across the other direction — one section says nothing about "
               f"the rest of the model, so nothing is invented for it.")

    trial = dict(parameters)
    preview = {"highlight": unit_name, "picks": picks, "placement": True,
               "axis": axis}
    if picks:
        profile = picks_to_profile(picks, layers, index, grid, axis)
        if not profile_is_empty(profile):
            trial_units = [dict(u) for u in units]
            trial_units[index] = {**trial_units[index],
                                  "thickness_profile": profile}
            trial_units[index].pop("pinch_out", None)
            trial = {**parameters, "units": trial_units}
        else:
            st.warning("Every pick is at or above the unit's own top, so it "
                       "would be absent everywhere. Draw at least one point "
                       "below the top.")

    undo, clear, drop = st.columns(3)
    if undo.button("Undo last pick", disabled=not picks, key="horizon_undo"):
        picks.pop()
        st.rerun()
    if clear.button("Clear drawing", disabled=not picks, key="horizon_clear"):
        st.session_state.horizon_picks = []
        st.rerun()
    if drop.button("Remove drawn profile",
                   disabled="thickness_profile" not in units[index],
                   key="horizon_drop"):
        stripped = [dict(u) for u in units]
        stripped[index].pop("thickness_profile", None)
        st.session_state.horizon_picks = []
        return {**parameters, "units": stripped}, {}
    return trial, preview


def _describe_layers(name: str, parameters: dict) -> str:
    """A canonical description of what a template and its parameters build.

    Surfaces and layers are dataclasses all the way down, so their repr is a
    faithful, comparable statement of the stratigraphy. A configuration that
    fails to build compares equal to nothing, which is the right answer: a
    broken current state is always a change worth applying away from.
    """
    try:
        layers, faults = template(name, **parameters)
    except Exception as exc:                      # noqa: BLE001 - reported, not raised
        return f"unbuildable: {name}: {exc}"
    return repr((name, layers, faults))


def _structure_editor(pipe) -> None:
    """Choose the structure and the stratigraphy without editing YAML.

    Everything about the earth's shape - which template, how many units, how
    thick, what dips, what pinches out - lived only in the configuration file
    until now. The page could show you the model and let you retouch one
    layer's petrophysics, but not change the model.
    """
    cfg = config()
    grid = pipe.domains.geology
    extent = grid.extent
    with st.expander("Structure and stratigraphy", expanded=False):
        names = list(TEMPLATES)
        current = cfg.geology.template
        chosen = st.selectbox(
            "Template", names,
            index=names.index(current) if current in names else 0,
            key="geo_template",
            help="`layer_cake` is the general one: as many units as you like, "
                 "each with its own thickness and pinchout. The rest are "
                 "ready-made structures with their own parameters.")
        st.caption(f"Model extent {extent[0]:,.0f} × {extent[1]:,.0f} × "
                   f"{extent[2]:,.0f} m on a {grid.dx:,.0f} × {grid.dy:,.0f} × "
                   f"{grid.dz:,.0f} m cell, from the Domains settings.")

        parameters = dict(cfg.geology.parameters or {})
        if chosen != current:
            parameters = {}          # a different template, not the old one's dials
        parameters.setdefault("extent", [float(v) for v in extent])

        if chosen == "layer_cake":
            new_parameters = _layer_cake_controls(parameters, extent)
        else:
            new_parameters = _template_parameters(chosen, parameters, extent)
            new_parameters["extent"] = [float(v) for v in extent]

        try:
            layers, faults = template(chosen, **new_parameters)
        except Sim3DError as exc:
            st.error(str(exc))
            return

        # The drawing needs the tops of the units *above* the one being
        # drawn, which do not depend on its own thickness - so the layers
        # built a moment ago are the right ones to convert picks against,
        # and the stack is rebuilt afterwards to show the result.
        preview: dict = {}
        if chosen == "layer_cake":
            new_parameters, preview = _horizon_editor(pipe, new_parameters, layers)
            try:
                layers, faults = template(chosen, **new_parameters)
            except Sim3DError as exc:
                st.error(str(exc))
                return

        drawing = bool(preview.get("placement"))
        figure = ui.stratigraphy_figure(
            layers, grid, axis=int(preview.get("axis", 0)),
            title="Section through the model centre", wells=pipe.wells(),
            highlight=preview.get("highlight"), picks=preview.get("picks"),
            placement=drawing)
        event = st.plotly_chart(
            figure, width="stretch", key="strat_section",
            on_select="rerun" if drawing else "ignore",
            selection_mode="points")
        if drawing and event and event.get("selection", {}).get("points"):
            point = event["selection"]["points"][-1]
            clicked = [float(point["x"]), float(point["y"])]
            picks = st.session_state.setdefault("horizon_picks", [])
            if not picks or clicked != picks[-1]:
                picks.append(clicked)
                st.rerun()
        if len(faults):
            st.caption(f"The section shows the stratigraphy before faulting. "
                       f"{len(faults)} fault(s) displace it; the offset is in "
                       f"the property volumes above, and the traces are on the "
                       f"fault map below.")
        for note in _stratigraphy_warnings(layers, grid,
                                           cfg.source.frequency):
            st.warning(note)

        apply, revert = st.columns(2)
        # "Would applying this change anything?" is a question about the
        # stratigraphy, not about the parameter dictionary. The editor fills in
        # defaults the configuration leaves out - an extent, a datum, a flat
        # structure - so comparing dictionaries reports a change on every
        # first render and the button is never honestly disabled.
        changed = _describe_layers(chosen, new_parameters) != _describe_layers(
            current, dict(cfg.geology.parameters or {}))
        if apply.button("Apply stratigraphy", type="primary",
                        disabled=not changed, key="geo_apply"):
            cfg.geology.template = chosen
            cfg.geology.parameters = new_parameters
            # The drawing is in the configuration now, so the live picks are
            # spent; leaving them would re-apply them over the next edit.
            for key in ("horizon_picks", "horizon_token"):
                st.session_state.pop(key, None)
            invalidate()
            st.rerun()
        if revert.button("Discard changes", disabled=not changed,
                         key="geo_revert"):
            st.rerun()
        if changed:
            st.caption("The section above is the edit; the volumes below are "
                       "still the applied model.")


def _layer_cake_controls(parameters: dict, extent) -> dict:
    """The units table and the one structure the whole package carries."""
    a, b, c, d = st.columns(4)
    structure = dict(parameters.get("structure") or {})
    style = str(structure.get("style", "flat"))
    style = a.selectbox("Structure", _STRUCTURES,
                        index=_STRUCTURES.index(style) if style in _STRUCTURES else 0,
                        key="cake_style")
    datum = b.number_input("Datum (m)", 0.0, 10000.0,
                           float(parameters.get("datum", 0.0)), 10.0,
                           key="cake_datum",
                           help="Depth of the top of the first unit.")
    new_structure: dict = {"style": style}
    if style == "dipping":
        new_structure["dip"] = c.number_input(
            "Dip (degrees)", 0.0, 80.0, float(structure.get("dip", 6.0)), 0.5,
            key="cake_dip")
        new_structure["azimuth"] = d.number_input(
            "Dip azimuth", 0.0, 360.0, float(structure.get("azimuth", 90.0)),
            5.0, key="cake_dip_az",
            help="Degrees clockwise from +y; 90 dips towards +x.")
    elif style in ("anticline", "syncline"):
        new_structure["amplitude"] = c.number_input(
            "Relief (m)", 0.0, 2000.0, float(structure.get("amplitude", 90.0)),
            5.0, key="cake_amp")
        new_structure["azimuth"] = d.number_input(
            "Long-axis azimuth", 0.0, 360.0,
            float(structure.get("azimuth", 0.0)), 5.0, key="cake_fold_az")

    st.caption("One row per unit, top first. Add and delete rows in the table. "
               "Blank porosity, Vsh or NTG take the facies default. "
               "**pinch out** `wedge` thins the unit along **azimuth**, from "
               "**from** to **to** as fractions of the model; `lens` tapers it "
               "away from the centre with **from** and **to** as the two radii, "
               "again as fractions.")
    rows = _units_to_rows(list(parameters.get("units") or DEFAULT_UNITS))
    edited = st.data_editor(
        rows, num_rows="dynamic", width="stretch", key="cake_units",
        column_order=_UNIT_COLUMNS,
        column_config={
            "name": st.column_config.TextColumn("name", required=True),
            "facies": st.column_config.SelectboxColumn(
                "facies", options=list(FACIES), required=True),
            "thickness": st.column_config.NumberColumn(
                "thickness (m)", min_value=0.0, max_value=5000.0, step=5.0),
            "reservoir": st.column_config.CheckboxColumn("reservoir"),
            "pinch out": st.column_config.SelectboxColumn(
                "pinch out", options=["none", "wedge", "lens"]),
            "azimuth": st.column_config.NumberColumn(
                "azimuth", min_value=0.0, max_value=360.0, step=5.0),
            "from": st.column_config.NumberColumn(
                "from", min_value=0.0, max_value=1.0, step=0.05),
            "to": st.column_config.NumberColumn(
                "to", min_value=0.0, max_value=1.0, step=0.05),
            "porosity": st.column_config.NumberColumn(
                "porosity", min_value=0.0, max_value=0.6, step=0.01),
            "Vsh": st.column_config.NumberColumn(
                "Vsh", min_value=0.0, max_value=1.0, step=0.01),
            "NTG": st.column_config.NumberColumn(
                "NTG", min_value=0.0, max_value=1.0, step=0.01),
        })
    out = dict(parameters)
    out["extent"] = [float(v) for v in extent]
    out["datum"] = float(datum)
    out["units"] = _rows_to_units(edited, extent)
    out["structure"] = new_structure
    return out


def layer_overrides() -> list[dict]:
    """The per-layer override list, straight off the configuration."""
    return config().geology.layers


def _layer_properties(geology) -> None:
    """Edit one layer's petrophysics without writing a new template.

    The template fixes a plausible earth; this is how an experiment asks a
    different question of it - a tighter seal, a cleaner reservoir - and it
    changes the property cube the flow simulation and the rock physics both
    read, so the edit reaches the sweep and the seismic.
    """
    with st.expander("Petrophysics by layer"):
        names = [layer.name for layer in geology.layers]
        chosen = st.selectbox("Layer", names, key="layer_edit")
        layer = next(l for l in geology.layers if l.name == chosen)
        facies = layer.resolved_facies()
        overrides = layer_overrides()
        current = next((e for e in overrides if e.get("name") == chosen), {})

        mid = lambda pair: 0.5 * (pair[0] + pair[1])           # noqa: E731
        st.caption(f"Facies **{layer.facies}** supplies porosity "
                   f"{mid(facies.porosity):.2f}, Vsh {mid(facies.vsh):.2f}, "
                   f"NTG {mid(facies.ntg):.2f}, "
                   f"{mid(facies.permeability):,.0f} mD. An unset field takes "
                   f"that midpoint; permeability unset is derived from "
                   f"porosity rather than held constant.")

        a, b = st.columns(2)
        new_facies = a.selectbox(
            "Facies", list(FACIES), index=list(FACIES).index(layer.facies),
            key=f"facies_{chosen}")
        reservoir = b.checkbox(
            "Is reservoir", value=bool(layer.is_reservoir
                                       if layer.is_reservoir is not None
                                       else facies.is_reservoir),
            key=f"resv_{chosen}",
            help="Whether the flow simulation solves in this unit at all.")

        c, d, e, f = st.columns(4)
        values = {
            "porosity": c.number_input(
                "Porosity", 0.0, 0.6,
                float(layer.porosity if layer.porosity is not None
                      else mid(facies.porosity)), 0.01, key=f"phi_{chosen}"),
            "vsh": d.number_input(
                "Shale volume", 0.0, 1.0,
                float(layer.vsh if layer.vsh is not None else mid(facies.vsh)),
                0.01, key=f"vsh_{chosen}"),
            "ntg": e.number_input(
                "Net-to-gross", 0.0, 1.0,
                float(layer.ntg if layer.ntg is not None else mid(facies.ntg)),
                0.01, key=f"ntg_{chosen}"),
            "permeability": f.number_input(
                "Permeability (mD)", 0.0, 20000.0,
                float(layer.permeability if layer.permeability is not None
                      else mid(facies.permeability)), 10.0, key=f"k_{chosen}"),
        }

        apply, reset = st.columns(2)
        if apply.button("Apply to layer", type="primary", key=f"apply_{chosen}"):
            entry = {"name": chosen, "facies": new_facies,
                     "is_reservoir": bool(reservoir), **values}
            overrides[:] = [o for o in overrides if o.get("name") != chosen]
            overrides.append(entry)
            invalidate()
            st.rerun()
        if reset.button("Back to the template", key=f"reset_{chosen}",
                        disabled=not current):
            overrides[:] = [o for o in overrides if o.get("name") != chosen]
            invalidate()
            st.rerun()

        if overrides:
            st.caption("Overridden: "
                       + ", ".join(sorted(o["name"] for o in overrides))
                       + ". Everything else is the template.")


def body_specs() -> list[dict]:
    """The editable geobody list, straight off the configuration."""
    return config().geology.bodies


def fault_specs() -> list[dict]:
    """The editable fault list, straight off the configuration."""
    return config().geology.faults


def _fault_trace(fault, grid) -> tuple[list[float], list[float]]:
    """The two ends of a fault's map trace, for drawing it."""
    sx, sy, _ = fault.strike_vector
    reach = fault.strike_extent
    if reach is None:
        # Unlimited along strike: long enough to leave the model either way.
        reach = float(np.hypot(grid.extent[0], grid.extent[1]))
    ox, oy, _ = fault.origin
    return ([ox - reach * sx, ox + reach * sx],
            [oy - reach * sy, oy + reach * sy])


def _add_fault_layer(figure, faults, grid, path) -> None:
    """Existing fault traces, and the one being drawn."""
    import plotly.graph_objects as go

    for fault in faults:
        xs, ys = _fault_trace(fault, grid)
        figure.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=fault.name,
            line=dict(color=theme.INK_PRIMARY, width=2.4),
            hovertemplate=f"{fault.describe()}<extra></extra>"))
    if path:
        figure.add_trace(go.Scatter(
            x=[p[0] for p in path], y=[p[1] for p in path],
            mode="lines+markers", name="new fault",
            line=dict(color=theme.SERIES[1], width=2.6, dash="dash"),
            marker=dict(size=11, color=theme.SURFACE,
                        line=dict(color=theme.SERIES[1], width=2)),
            hovertemplate="%{x:,.0f} m, %{y:,.0f} m<extra></extra>"))


def _fault_editor(pipe, geology) -> None:
    """Draw a fault trace on the map and give it a plane.

    Faults reached the model only through the `fault_compartment` template,
    which makes exactly one, at the model centre, with a fixed strike. Any
    other fault meant editing the template's source. Drawn here they are
    added to whatever the template brings rather than replacing it.

    A fault is not decoration: it displaces the stratigraphy and attenuates
    transport across its plane, so it moves the flood front, splits the
    pressure response into compartments and offsets the reflectors together.
    """
    grid = geology.grid
    specs = fault_specs()
    drawing = st.toggle("Draw a fault", key="drawing_fault",
                        help="Click two points on the map for the ends of the "
                             "fault trace.")
    path = st.session_state.setdefault("fault_path", [])

    if drawing:
        st.caption("The trace sets three things at once: the fault runs "
                   "exactly as far as the line drawn for it, its strike is "
                   "the line's bearing, and its centre is the line's midpoint "
                   "at the depth below. **The fault dips down to the right of "
                   "the direction you draw**, so drawing the line the other "
                   "way round puts the hanging wall on the other side.")
        figure = ui.map_figure(wells=pipe.wells(),
                               grid=pipe.domains.propagation,
                               pml_nodes=config().solver.pml_nodes,
                               bounds=grid.bounds)
        _add_fault_layer(figure, geology.faults, grid, path)
        _add_placement_layer(figure, grid, enabled=True)
        event = st.plotly_chart(figure, width="stretch", key="faultmap",
                                on_select="rerun", selection_mode="points")
        if event and event.get("selection", {}).get("points"):
            point = event["selection"]["points"][-1]
            clicked = [float(point["x"]), float(point["y"])]
            if not path or clicked != path[-1]:
                if len(path) >= 2:
                    path.clear()         # a third click starts a new trace
                path.append(clicked)
                st.rerun()
        st.info(f"{len(path)} of 2 points placed.", icon="🖱️")

        (_, _), (_, _), (z0, z1) = grid.bounds
        a, b, c = st.columns(3)
        name = a.text_input("Name", _unique_fault_name(), key="fault_name")
        depth = b.number_input(
            "Centre depth (m)", float(z0), float(z1),
            float(np.clip(_reservoir_top(geology), z0, z1)), 10.0,
            key="fault_depth", help="Where the trace sits; the plane passes "
                                    "through this depth at the drawn line.")
        dip = c.number_input("Dip (degrees)", 1.0, 90.0, 65.0, 1.0,
                             key="fault_dip")
        d, e, f = st.columns(3)
        throw = d.number_input(
            "Throw (m)", -500.0, 500.0, 40.0, 5.0, key="fault_throw",
            help="Positive drops the hanging wall — a normal fault. "
                 "Negative is reverse.")
        seal = e.slider(
            "Transmissibility", 0.0, 1.0, 0.0, 0.05, key="fault_trans",
            help="0 is sealing, 1 lets flow through untouched. This is what "
                 "decides whether the fault stops the flood.")
        zone = f.number_input(
            "Zone width (m)", float(grid.dx), 500.0,
            float(max(2 * grid.dx, 20.0)), 5.0, key="fault_zone",
            help="The plane is smoothed over this width. A fault sharper "
                 "than the grid gives stair steps, not resolution.")
        limited = st.checkbox("Limit how far it cuts down dip",
                              key="fault_limit_dip")
        dip_extent = None
        if limited:
            dip_extent = st.number_input(
                "Half-height down dip (m)", float(grid.dz),
                float(z1 - z0), float(0.25 * (z1 - z0)), 10.0,
                key="fault_dip_extent")

        place, clear = st.columns(2)
        if place.button("Place fault", type="primary", disabled=len(path) < 2,
                        key="fault_place"):
            if name in {spec.get("name") for spec in specs}:
                st.error(f"There is already a fault called {name!r}.")
            else:
                entry = {"name": name, "trace": [list(p) for p in path],
                         "depth": float(depth), "dip": float(dip),
                         "throw": float(throw), "zone_width": float(zone),
                         "transmissibility": float(seal)}
                if dip_extent is not None:
                    entry["dip_extent"] = float(dip_extent)
                try:
                    build_faults([entry])         # refuse it here, not later
                except Sim3DError as exc:
                    st.error(str(exc))
                else:
                    specs.append(entry)
                    st.session_state.fault_path = []
                    invalidate()
                    st.rerun()
        if clear.button("Clear trace", disabled=not path, key="fault_clear"):
            st.session_state.fault_path = []
            st.rerun()

    if specs:
        st.caption("Drawn faults — the template's own are listed above.")
        for index, spec in enumerate(list(specs)):
            row, button = st.columns([5, 1])
            row.code(build_faults([spec])[0].describe(), language="text")
            if button.button("Delete", key=f"fault_delete_{index}"):
                specs.pop(index)
                invalidate()
                st.rerun()


def _unique_fault_name() -> str:
    """The first free ``F<n>`` against the drawn faults."""
    taken = {spec.get("name") for spec in fault_specs()}
    index = 1
    while f"F{index}" in taken:
        index += 1
    return f"F{index}"


def _geobody_editor(pipe, geology) -> None:
    """Draw a channel or a lens and paint it into the property cube.

    The body edits porosity, permeability, net-to-gross and the reservoir
    mask, so it reaches the flow simulation and through it the 4D seismic.
    That is the whole point of drawing one: not to annotate the display but
    to ask what this geometry would do to the sweep.
    """
    grid = geology.grid
    specs = body_specs()
    st.subheader("Geobodies")
    st.caption("Drawn bodies are painted over the template in order, and they "
               "change the properties the flow simulation reads — so a channel "
               "drawn here moves the water, the 4D anomaly and the seismic, "
               "not just the picture.")

    drawing = st.toggle("Draw a body", key="drawing",
                        help="Then click the map to lay down its path.")
    path = st.session_state.setdefault("body_path", [])

    if drawing:
        a, b, c = st.columns([1, 1, 2])
        kind = a.radio("Kind", BODY_TYPES, horizontal=True, key="body_kind")
        facies_name = b.selectbox("Facies", list(FACIES), key="body_facies",
                                  index=list(FACIES).index("clean_sandstone"))
        need = 2 if kind == "channel" else 3
        c.info(f"Click the map to add points — a {kind} needs at least {need}. "
               f"{len(path)} so far.", icon="🖱️")

        figure = ui.map_figure(wells=pipe.wells(), grid=pipe.domains.propagation,
                               pml_nodes=config().solver.pml_nodes,
                               bounds=grid.bounds)
        _add_body_layer(figure, specs, path)
        _add_placement_layer(figure, grid, enabled=True)
        event = st.plotly_chart(figure, width="stretch", key="bodymap",
                                on_select="rerun", selection_mode="points")
        if event and event.get("selection", {}).get("points"):
            point = event["selection"]["points"][-1]
            xy = [float(point["x"]), float(point["y"])]
            if xy != (path[-1] if path else None):
                path.append(xy)
                st.rerun()

        d, e, f, g = st.columns(4)
        width = d.number_input("Width (m)", 25.0, 5000.0, 300.0, 25.0,
                               disabled=kind != "channel",
                               help="Full width of the channel, not the half.")
        (_, _), (_, _), (z0, z1) = grid.bounds
        top = e.number_input("Top (m)", float(z0), float(z1),
                             float(np.clip(_reservoir_top(geology), z0, z1)), 10.0)
        thickness = f.number_input("Thickness (m)", float(grid.dz),
                                   float(z1 - z0), 40.0, 10.0)
        name = g.text_input("Name", _unique_body_name(kind))

        # The facies fixes a default for each property; these let a body be
        # something the catalogue does not contain - a tight streak, an
        # unusually clean bar - without inventing a facies to hold it.
        rock = GeoBody(name=name, type=kind, path=[[0.0, 0.0], [1.0, 1.0]],
                       facies=facies_name).rock()
        override = st.checkbox(
            "Set properties explicitly", key="body_override",
            help=f"Otherwise {facies_name} supplies them: porosity "
                 f"{rock['porosity']:.2f}, Vsh {rock['vsh']:.2f}, NTG "
                 f"{rock['ntg']:.2f}, {rock['permeability_md']:,.0f} mD.")
        properties = {}
        if override:
            h, i, j, k = st.columns(4)
            properties = {
                "porosity": h.number_input("Porosity", 0.0, 0.6,
                                           float(rock["porosity"]), 0.01),
                "vsh": i.number_input("Shale volume", 0.0, 1.0,
                                      float(rock["vsh"]), 0.01),
                "ntg": j.number_input("Net-to-gross", 0.0, 1.0,
                                      float(rock["ntg"]), 0.01),
                "permeability_md": k.number_input(
                    "Permeability (mD)", 0.0, 20000.0,
                    float(rock["permeability_md"]), 10.0),
            }
            st.caption("Permeability drives the flow and porosity drives both "
                       "the flow and the rock physics, so these change the "
                       "sweep and the seismic together — they are not a "
                       "display setting.")

        place, clear, undo = st.columns(3)
        if place.button("Place body", type="primary",
                        disabled=len(path) < need):
            if name in {b["name"] for b in specs}:
                st.error(f"There is already a body called {name!r}.")
            else:
                specs.append({"name": name, "type": kind, "path": list(path),
                              "width": float(width), "top": float(top),
                              "thickness": float(thickness),
                              "facies": facies_name, **properties})
                st.session_state.body_path = []
                # Drop the shared cursor onto the body: it is almost always
                # thinner than the reservoir, so a slice left where it was
                # shows the template and the user concludes nothing happened.
                st.session_state.cursor_z = float(np.clip(
                    top + 0.5 * thickness, *grid.bounds[2]))
                invalidate()
                st.rerun()
        if undo.button("Undo point", disabled=not path):
            path.pop()
            st.rerun()
        if clear.button("Clear path", disabled=not path):
            st.session_state.body_path = []
            st.rerun()

    if not specs:
        st.info("No drawn bodies. The geology is the template alone.")
        return

    def row(b: dict) -> dict:
        # Show what the body will actually be made of, resolving the facies
        # defaults: a table of blanks for everything unset would hide the
        # difference between two bodies of the same facies.
        rock = GeoBody(**b).rock()
        explicit = {k for k in ("porosity", "vsh", "ntg", "permeability_md")
                    if b.get(k) is not None}
        mark = lambda k, v: f"{v}*" if k in explicit else v   # noqa: E731
        return {
            "type": b.get("type", "channel"),
            "facies": b.get("facies", "clean_sandstone"),
            "points": len(b.get("path", [])),
            "width (m)": f"{b.get('width', 0.0):,.0f}",
            "top (m)": f"{b.get('top', 0.0):,.0f}",
            "thickness (m)": f"{b.get('thickness', 0.0):,.0f}",
            "porosity": mark("porosity", f"{rock['porosity']:.2f}"),
            "perm (mD)": mark("permeability_md", f"{rock['permeability_md']:,.0f}"),
        }

    st.dataframe({b["name"]: row(b) for b in specs}, width="stretch")
    st.caption(r"\* set explicitly; everything else is the facies default.")
    remove = st.selectbox("Remove a body", [b["name"] for b in specs],
                          key="remove_body")
    if st.button("Remove"):
        specs[:] = [b for b in specs if b["name"] != remove]
        invalidate()
        st.rerun()


def _reservoir_top(geology) -> float:
    """Where the reservoir starts, as the default top for a new body."""
    z = geology.grid.axis(2)
    reservoir = geology.reservoir_mask.any(axis=(0, 1))
    return float(z[reservoir][0]) if reservoir.any() else float(z[len(z) // 2])


def _unique_body_name(kind: str) -> str:
    taken = {b["name"] for b in config().geology.bodies}
    index = 1
    while f"{kind}_{index}" in taken:
        index += 1
    return f"{kind}_{index}"


def _add_body_layer(figure, specs, path) -> None:
    """Draw the bodies that exist and the path being laid down."""
    import plotly.graph_objects as go
    for i, spec in enumerate(specs):
        pts = np.asarray(spec.get("path") or [], dtype=float)
        if pts.size == 0:
            continue
        closed = spec.get("type", "channel") == "lens"
        xs = np.append(pts[:, 0], pts[0, 0]) if closed else pts[:, 0]
        ys = np.append(pts[:, 1], pts[0, 1]) if closed else pts[:, 1]
        figure.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=spec["name"],
            line=dict(color=theme.SERIES[2 + i % 3], width=3)))
    if path:
        pts = np.asarray(path, dtype=float)
        figure.add_trace(go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode="lines+markers", name="new body",
            line=dict(color=theme.SERIES[1], width=3, dash="dot"),
            marker=dict(size=9, color=theme.SERIES[1])))


def _contact_controls(pipe, cfg) -> None:
    """Oil-water and gas-oil contacts, and the capillary transition.

    A contact is a horizontal plane cutting across the stratigraphy, so it
    is the one reflector in the model whose geometry owes nothing to the
    layering - the flat spot. Leaving them unset keeps the uniform column
    a mechanistic sweep test wants.
    """
    b = cfg.reservoir.baseline
    (_, _), (_, _), (z0, z1) = pipe.domains.geology.bounds
    with st.expander("Fluid contacts"):
        on = st.checkbox("Model fluid contacts", value=b.owc is not None,
                         key="contacts_on")
        if not on:
            if b.owc is not None or b.goc is not None:
                b.owc = b.goc = None
                invalidate()
                st.rerun()
            st.caption(f"Uniform reservoir saturation: Sw = {b.sw:g} "
                       f"everywhere, no water leg and no flat spot.")
            return

        a, c, d, e = st.columns(4)
        default = b.owc if b.owc is not None else float(0.5 * (z0 + z1))
        owc = a.number_input("OWC (m)", float(z0), float(z1), float(default), 10.0)
        gas = c.checkbox("Gas cap", value=b.goc is not None)
        goc = d.number_input("GOC (m)", float(z0), float(owc),
                             float(b.goc if b.goc is not None else max(z0, owc - 100.0)),
                             10.0, disabled=not gas)
        transition = e.number_input("Transition (m)", 0.0, 500.0,
                                    float(b.transition), 5.0,
                                    help="Height above the OWC over which Sw "
                                         "falls from 1 to Swirr. A linear "
                                         "ramp, not a J-function. 0 is sharp.")
        swirr = st.slider("Irreducible water saturation", 0.0, 0.8,
                          float(b.sw), 0.01,
                          help="With contacts this is Swirr in the "
                               "hydrocarbon column, not a uniform Sw.")
        chosen = (owc, goc if gas else None, transition, swirr)
        if chosen != (b.owc, b.goc, b.transition, b.sw):
            if gas and goc >= owc:
                st.error("The gas-oil contact must sit above the oil-water "
                         "contact — depth increases downwards.")
            else:
                b.owc, b.goc, b.transition, b.sw = chosen
                invalidate()
                st.rerun()

        state = pipe.baseline_state()
        z = pipe.domains.geology.axis(2)
        mask = state.reservoir_mask
        profile = {}
        for label, cube in (("Sw", state.sw), ("So", state.so), ("Sg", state.sg)):
            column = np.where(mask, cube, np.nan)
            with np.errstate(invalid="ignore"):
                profile[label] = np.nanmean(column, axis=(0, 1))
        st.plotly_chart(ui.series_figure(
            z, profile, xlabel="depth (m)", ylabel="saturation", height=260),
            width="stretch", key="contact_profile")
        st.caption("Reservoir-cell average against depth. The contact is flat "
                   "whatever the structure does, which is what makes a flat "
                   "spot cut across dipping reflectors.")


def page_reservoir() -> None:
    pipe = pipeline()
    cfg = config()
    st.title("Wells & reservoir state")

    _contact_controls(pipe, cfg)

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


def _facies_rockphysics_controls() -> None:
    """Give a lithology its own dry frame.

    One global frame is not a simplification but a bias, and it falls on the
    contrast the angle stacks are built on: with a single soft-sand frame
    the shale of the flat template comes out *slower* than the reservoir
    sand, so the top of the reservoir barely reflects at all.
    """
    rp = config().rock_physics
    with st.expander("Rock physics by facies"):
        present = _present_facies()
        chosen = st.selectbox("Facies", present, key="rp_facies")
        current = dict((rp.facies or {}).get(chosen, {}))
        a, b, c = st.columns(3)
        frame = a.selectbox(
            "Dry frame", list(DRY_FRAME_MODELS),
            index=list(DRY_FRAME_MODELS).index(
                current.get("dry_frame_model", rp.dry_frame_model)),
            key=f"rpf_{chosen}")
        phi_c = b.number_input(
            "Critical porosity", 0.20, 0.80,
            float(current.get("critical_porosity", rp.critical_porosity)),
            0.01, key=f"rpc_{chosen}")
        coord = c.number_input(
            "Coordination number", 4.0, 20.0,
            float(current.get("coordination", rp.coordination)), 0.5,
            key=f"rpn_{chosen}")

        apply, reset = st.columns(2)
        if apply.button("Apply to facies", type="primary", key=f"rpa_{chosen}"):
            rp.facies = dict(rp.facies or {})
            rp.facies[chosen] = {"dry_frame_model": frame,
                                 "critical_porosity": float(phi_c),
                                 "coordination": float(coord)}
            invalidate()
            st.rerun()
        if reset.button("Back to the global model", key=f"rpr_{chosen}",
                        disabled=not current):
            rp.facies = {k: v for k, v in (rp.facies or {}).items() if k != chosen}
            invalidate()
            st.rerun()

        if rp.facies:
            st.caption("Overridden: " + ", ".join(sorted(rp.facies))
                       + ". Everything else uses the global frame.")
        st.caption("Fluid properties stay global — one connected reservoir has "
                   "one fluid, so temperature, salinity, API and GOR are not "
                   "offered here and are refused if configured.")


def _present_facies() -> list[str]:
    """Facies the current model actually contains, global order preserved."""
    try:
        geology = pipeline().result.geology
    except Sim3DError:
        geology = None
    if geology is None:
        return list(FACIES)
    codes = set(np.unique(geology.facies_code).tolist())
    present = [n for n, f in FACIES.items() if f.code in codes]
    return present or list(FACIES)


def page_rockphysics() -> None:
    pipe = pipeline()
    earth = stage("Running the rock-physics chain", pipe.rockphysics)
    grid = pipe.domains.geology
    point = cursor_controls(grid)

    st.title("Rock physics")
    _facies_rockphysics_controls()
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


def _acquisition_controls(cfg, include_record: bool = True) -> None:
    """Edit the geometry the diagnostics below are about to judge.

    ``include_record`` exists because the record length belongs to the
    recording rather than to the geometry, and the migration page owns it
    there. Two widgets writing the same field on one page is a bug, not a
    convenience.

    The survey diagnostics tell you the standoff is too small or the trace
    spacing aliases the operator, and until now the page gave no way to act
    on either: everything under ``acquisition`` and ``domains`` was
    configuration-file only.  Aperture, standoff and the operator limit are
    geometry, so they recompute in milliseconds - the point of editing them
    here is to see the numbers move before spending a shot on them.

    The absorbing layer is the one constraint worth enforcing in the widget
    rather than in QC afterwards: nothing may sit inside it, and its inner
    edge moves whenever the top of the propagation domain does.
    """
    acq = cfg.acquisition
    domains = cfg.domains
    bounds = [list(b) for b in (domains.propagation_bounds or domains.geology_bounds)]
    spacing = list(domains.propagation_spacing or domains.geology_spacing)
    pml_metres = cfg.solver.pml_nodes * float(spacing[2])
    shallowest = float(bounds[2][0]) + pml_metres

    with st.expander("Acquisition geometry"):
        st.caption(
            f"The absorbing layer takes {cfg.solver.pml_nodes} nodes - "
            f"{pml_metres:,.0f} m - off every face, so nothing may sit above "
            f"z = {shallowest:,.0f} m. Raise the top of the propagation domain "
            f"to buy standoff above that.")

        a, b, c = st.columns(3)
        source_spacing = a.number_input(
            "Source spacing (m)", 10.0, 2000.0, float(acq.source_spacing), 10.0,
            key="acq_source_spacing",
            help="Along a source line. Cost is linear in the shot count: the "
                 "planner reports three propagations per shot per earth model.")
        line_spacing = b.number_input(
            "Source line spacing (m)", 10.0, 2000.0,
            float(acq.source_line_spacing), 10.0, key="acq_line_spacing",
            help="Between source lines. A separate knob - halving the source "
                 "spacing alone leaves the lines as far apart as they were.")
        receiver_spacing = c.number_input(
            "Receiver spacing (m)", 10.0, 1000.0, float(acq.receiver_spacing),
            10.0, key="acq_receiver_spacing",
            help="Nearly free: the receiver wavefield is back-propagated as "
                 "one field, so the cost does not scale with receiver count.")

        d, e, f = st.columns(3)
        domain_top = d.number_input(
            "Propagation domain top (m)", float(domains.geology_bounds[2][0]),
            float(bounds[2][1]) - 100.0, float(bounds[2][0]), 20.0,
            key="acq_domain_top",
            help="Raising this deepens the model and costs cells, and it is "
                 "the only way to put the acquisition further above the "
                 "target once the absorbing layer has taken its share.")
        new_shallowest = domain_top + pml_metres
        source_depth = e.number_input(
            "Source depth (m)", new_shallowest, float(bounds[2][1]),
            max(float(acq.source_depth), new_shallowest), 10.0,
            key="acq_source_depth")
        receiver_depth = f.number_input(
            "Receiver depth (m)", new_shallowest, float(bounds[2][1]),
            max(float(acq.receiver_depth), new_shallowest), 10.0,
            key="acq_receiver_depth")

        # The footprint: how far the survey reaches, as opposed to how
        # finely it samples. Left unset it is derived from the target plus a
        # margin, which is the right default and the wrong answer the moment
        # the target is narrowed for a 2.5D line - the inline footprint
        # shrinks with it and the survey collapses to a couple of shots.
        explicit = st.checkbox(
            "Set the survey footprint explicitly",
            value=any(e is not None for e in (acq.receiver_extent,
                                              acq.source_extent,
                                              acq.receiver_extent_y,
                                              acq.source_extent_y)),
            key="acq_explicit_extent",
            help="Unset, each footprint is the target plus "
                 f"{acq.target_margin:,.0f} m on every side. Aperture scales "
                 "with depth, not with target size: illuminating a point h "
                 "below the sources at incidence θ wants sources h·tan θ "
                 "either side of it.")
        extents = (acq.receiver_extent, acq.source_extent,
                   acq.receiver_extent_y, acq.source_extent_y)
        if explicit:
            # Seed from the footprint that is already in force - the target
            # plus a margin - rather than from the domain span. Seeding from
            # the span put 124 receivers inside the absorbing layer the
            # instant the box was ticked, which is a validation error offered
            # as a starting point.
            target = cfg.domains.target_bounds or bounds
            derived_x = (float(target[0][1]) - float(target[0][0])
                         + 2.0 * acq.target_margin)
            derived_y = (float(target[1][1]) - float(target[1][0])
                         + 2.0 * acq.target_margin)
            g, h = st.columns(2)
            receiver_extent = g.number_input(
                "Receiver footprint, inline (m)", 0.0, 100000.0,
                float(acq.receiver_extent if acq.receiver_extent is not None
                      else derived_x), 20.0, key="acq_receiver_extent")
            source_extent = h.number_input(
                "Source footprint, inline (m)", 0.0, 100000.0,
                float(acq.source_extent if acq.source_extent is not None
                      else derived_x), 20.0, key="acq_source_extent")
            i, j = st.columns(2)
            receiver_extent_y = i.number_input(
                "Receiver footprint, crossline (m)", 0.0, 100000.0,
                float(acq.receiver_extent_y if acq.receiver_extent_y is not None
                      else derived_y), 20.0, key="acq_receiver_extent_y",
                help="Narrow this and the source crossline footprint below "
                     "the source line spacing to shoot one line instead of a "
                     "carpet: the wave equation stays 3D, the cost does not.")
            source_extent_y = j.number_input(
                "Source footprint, crossline (m)", 0.0, 100000.0,
                float(acq.source_extent_y if acq.source_extent_y is not None
                      else derived_y), 1.0, key="acq_source_extent_y")
            extents = (receiver_extent, source_extent,
                       receiver_extent_y, source_extent_y)
        else:
            extents = (None, None, None, None)

        record = float(cfg.solver.record_length)
        if include_record:
            record = st.number_input(
                "Record length (s)", 0.1, 20.0, float(cfg.solver.record_length),
                0.1, key="acq_record_length",
                help="Listening time. Moving the acquisition up lengthens every "
                     "travel path, so a standoff that fixes the near-field needs "
                     "a longer record to still capture the target.")

        chosen = (source_spacing, line_spacing, receiver_spacing, domain_top,
                  source_depth, receiver_depth, record, extents)
        current = (float(acq.source_spacing), float(acq.source_line_spacing),
                   float(acq.receiver_spacing), float(bounds[2][0]),
                   float(acq.source_depth), float(acq.receiver_depth),
                   float(cfg.solver.record_length),
                   (acq.receiver_extent, acq.source_extent,
                    acq.receiver_extent_y, acq.source_extent_y))
        if chosen == current:
            return
        if min(source_depth, receiver_depth) < new_shallowest:
            st.error(
                f"With the domain top at {domain_top:,.0f} m the absorbing "
                f"layer reaches {new_shallowest:,.0f} m; sources and receivers "
                f"must sit below that.")
            return
        acq.source_spacing = source_spacing
        acq.source_line_spacing = line_spacing
        acq.receiver_spacing = receiver_spacing
        acq.source_depth = source_depth
        acq.receiver_depth = receiver_depth
        (acq.receiver_extent, acq.source_extent,
         acq.receiver_extent_y, acq.source_extent_y) = extents
        cfg.solver.record_length = record
        bounds[2][0] = domain_top
        domains.propagation_bounds = bounds
        domains.propagation_spacing = spacing
        invalidate()
        st.rerun()


def _survey_diagnostics(pipe, cfg, acquisition) -> None:
    """Aperture, standoff and sampling, against the limits that matter.

    These three numbers decide whether a migrated image is worth running:
    too narrow an aperture gives arcs instead of reflectors, too small a
    standoff lets the injection near-field sit on the target, and sampling
    coarser than the operator limit leaves every shot's isochrone in the
    image. All three were learned the expensive way on this project.
    """
    (_, _), (_, _), (z_top, _) = pipe.domains.target.bounds
    earth = pipe.result.earth
    # Aperture and standoff are geometry and are always available; only the
    # wavelength and the operator limit need a velocity, and asking for a flow
    # run before the survey can be judged has the design backwards.
    velocity = (float(np.median(earth.models["baseline"].vp))
                if earth is not None else float("nan"))
    report = sampling_report(acquisition, z_top, velocity, pipe.fmax,
                             cfg.acquisition.source_spacing,
                             cfg.acquisition.receiver_spacing)

    st.markdown("**Survey diagnostics**")
    a, b, c, d = st.columns(4)
    a.metric("Max aperture", f"{report.aperture_deg:.0f}°",
             help="Half-angle subtended at the top of the target by the "
                  "furthest source or receiver, on the diagonal. Below about "
                  "30 degrees a flat reflector images as arcs.")
    b.metric("Standoff", f"{report.standoff:,.0f} m",
             help="Acquisition to target. Below about two wavelengths the "
                  "injection near-field overlaps the target.")
    known = np.isfinite(report.wavelength)
    c.metric("…in wavelengths",
             f"{report.standoff_wavelengths:.1f} λ" if known else "—",
             help=None if known else "Needs the rock physics for a velocity.")
    d.metric("Operator limit",
             f"{report.operator_limit:,.0f} m" if known else "—",
             help="V / (4 fmax sin θ): trace spacing that does not alias the "
                  "migration operator at this aperture."
                  + ("" if known else " Needs the rock physics for a velocity."))

    st.dataframe({
        "spacing (m)": {"sources": f"{report.source_spacing:,.0f}",
                        "receivers": f"{report.receiver_spacing:,.0f}"},
        "× the operator limit": {
            "sources": f"{report.factor(report.source_spacing):.1f}×" if known else "—",
            "receivers": f"{report.factor(report.receiver_spacing):.1f}×" if known else "—"},
    }, width="stretch")

    for note in report.notes():
        st.warning(note)
    if not report.notes():
        st.success("Aperture and standoff are within their limits."
                   + (" Sampling too." if known else ""))
    if not known:
        st.info("Run the rock physics for the velocity-dependent numbers — "
                "standoff in wavelengths and the sampling limits.")
    st.caption("Measured on this project: refining the receiver spacing alone, "
               "from 3.6× to 0.9× the limit, moved the reservoir image not at "
               "all. Aperture and standoff were what mattered.")


def _show_checks(result, seen: set[str] | None = None) -> set[str]:
    """Render QC checks and return their messages.

    Deduplicating by message rather than by position: the full QC repeats the
    geometry checks, and which end of the list they land on is an ordering
    detail this page should not depend on.
    """
    seen = seen or set()
    for check in result.checks:
        if check.message in seen:
            continue
        seen.add(check.message)
        icon = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}[check.status.value]
        (st.error if check.status is Status.FAIL else
         st.warning if check.status is Status.WARNING else st.success)(
            f"{icon} {check.message}")
    return seen


def page_acquisition() -> None:
    pipe = pipeline()
    cfg = config()
    st.title("Acquisition & QC")

    _acquisition_controls(cfg)
    acquisition = stage("Building the geometry", pipe.acquisition)

    st.subheader("Aerial view")
    st.plotly_chart(ui.aerial_figure(
        acquisition=acquisition, wells=pipe.wells(), domains=pipe.domains,
        pml_nodes=cfg.solver.pml_nodes,
        show_receivers=st.checkbox("show receivers", value=True,
                                   key="aerial_receivers")), width="stretch")
    st.caption("Drawn to scale — the axes share one. Whether the spread is wide "
               "enough for the depth of the target is the question this view "
               "exists to answer, and a stretched aspect ratio hides it.")

    a, b, c, d = st.columns(4)
    a.metric("Sources", f"{acquisition.n_sources:,}")
    b.metric("Receivers", f"{acquisition.n_receivers:,}")
    c.metric("Traces", f"{acquisition.n_traces:,}")
    d.metric("Max offset", f"{acquisition.offsets().max():,.0f} m")

    _survey_diagnostics(pipe, cfg, acquisition)

    st.subheader("Common-midpoint fold")
    bin_size = st.slider("bin size (m)", 25.0, 200.0, 50.0, 25.0, key="fold_bin")
    (_, _), (_, _), (z_top, _) = pipe.domains.target.bounds
    fx, fy, fold = fold_map(acquisition, pipe.domains.propagation, z_top,
                            bin_size=float(bin_size))
    st.plotly_chart(ui.fold_figure(fx, fy, fold, wells=pipe.wells()),
                    width="stretch")
    st.caption("Straight-ray midpoint counting at the top of the target. It is a "
               "geometric proxy and says nothing about whether the wavefield "
               "actually reaches the reservoir — that is what the modelling is for.")

    st.subheader("Offset and azimuth")
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
    # Geometry first and always: it costs nothing.  The dispersion and
    # stability checks need the propagation models, which pull the flow
    # simulation and the rock physics in behind them, so they wait to be
    # asked - a survey is designed before a reservoir is simulated.
    shown = _show_checks(pipe.geometry_qc())
    if pipe.result.qc is not None or st.button("Run the full model QC"):
        result = stage("Running QC", pipe.qc)
        _show_checks(result, seen=shown)
        if result.failed:
            st.error("QC failed. sim3d will not adjust the grid, the bandwidth "
                     "or the geometry for you — change the configuration.")

        st.subheader("Computational estimate")
        try:
            st.code(pipe.plan().describe(), language="text")
            st.success("Within the configured budget.")
        except Sim3DError as exc:
            st.error(str(exc))
    else:
        st.info("The dispersion, stability and state checks need the flow "
                "simulation and the rock physics — minutes, not seconds.")


def _scenario_controls(cfg) -> None:
    """Choose which earth models get modelled, and pay for only those.

    Cost is linear in the list: four scenarios is four independent
    propagations of every shot.  The decomposition needs all four, but
    "is the image clean?" needs one, and answering it should not cost the
    other three.  Baseline is not optional - every difference is measured
    against it, and a monitor with nothing to subtract is not a 4D result.
    """
    chosen = st.multiselect(
        "Earth models to simulate", list(SCENARIO_NAMES),
        default=[n for n in SCENARIO_NAMES if n in cfg.fourd.scenarios],
        help="Each one is modelled independently, so the cost is linear in "
             "this list. baseline alone answers whether the geometry and the "
             "imaging work; pressure_only and saturation_only are what "
             "separate a fluid change from a pressure change, and the "
             "interaction term needs all four.")
    # Keep the canonical order whatever order they were clicked in: the
    # decomposition indexes SCENARIO_NAMES positionally.
    ordered = [n for n in SCENARIO_NAMES if n in chosen]
    if not ordered:
        st.error("Select at least the baseline — there is nothing to model.")
        return
    if ordered[0] != "baseline":
        st.warning("baseline is the reference every difference is measured "
                   "against; it is always modelled.")
        ordered = ["baseline"] + ordered
    if ordered != list(cfg.fourd.scenarios):
        cfg.fourd.scenarios = ordered
        invalidate()
        st.rerun()
    if len(ordered) < len(SCENARIO_NAMES):
        st.caption(f"{len(ordered)} of {len(SCENARIO_NAMES)} — "
                   f"about {len(ordered) / len(SCENARIO_NAMES):.0%} of the "
                   f"full cost. The 4D decomposition needs all four.")


def page_migration() -> None:
    """Build and validate the migration job; do not run it.

    Full-wave modelling and RTM used to happen behind two buttons on this
    page, inside the Streamlit process.  That was always the wrong place for
    them: a survey worth migrating is hours of work, a browser session is not
    a batch queue, and a page that blocks for four hours cannot show progress
    honestly or survive a reload.  What the app is good at is deciding *what*
    to run - the geometry, the sampling, the imaging choices, and the checks
    that catch a survey which will alias its operator before any of it costs
    anything.

    So this page produces a configuration file rather than an image.
    Everything the migration needs is set here, validated here, and written
    out as a complete YAML for `examples/run_migration.py` to consume on
    whatever machine has the cores.
    """
    cfg = config()
    pipe = pipeline()
    st.title("Migration setup")
    st.caption("This page configures and validates the migration. It does not "
               "run it — the run is a batch job, and the YAML below is its "
               "only input.")

    _scenario_controls(cfg)

    st.subheader("Imaging")
    _imaging_controls(cfg)

    st.subheader("Recording")
    a, b, c = st.columns(3)
    record = a.number_input(
        "Record length (s)", 0.05, 8.0, float(cfg.solver.record_length), 0.05,
        help="Long enough for the deepest reflection to arrive at the furthest "
             "offset, and no longer: every extra second is propagated for "
             "every shot.")
    pml = b.number_input(
        "Absorbing layer (nodes)", 4, 40, int(cfg.solver.pml_nodes), 1,
        help="Too few and the domain edges reflect back into the image. The "
             "layer eats this many cells from each face, and no source or "
             "receiver may sit inside it.")
    order = c.selectbox(
        "Spatial order", (2, 4, 6, 8),
        index=(2, 4, 6, 8).index(int(cfg.solver.spatial_order)),
        help="8th order needs about 5.25 cells per minimum wavelength for 1 % "
             "phase error; a lower order needs more cells, not fewer.")
    if (record != cfg.solver.record_length or int(pml) != cfg.solver.pml_nodes
            or int(order) != cfg.solver.spatial_order):
        cfg.solver.record_length = float(record)
        cfg.solver.pml_nodes = int(pml)
        cfg.solver.spatial_order = int(order)
        invalidate()

    st.subheader("Acquisition")
    st.caption("The geometry is a migration parameter, not a survey-design "
               "afterthought: the trace spacing decides whether the operator "
               "aliases, and the footprint decides whether the anomaly is "
               "illuminated at all. Both are editable here. **Acquisition & "
               "QC** has the same controls alongside the fold, offset and "
               "azimuth diagnostics.")
    _acquisition_controls(cfg, include_record=False)
    _survey_layout(cfg, pipe)
    _acquisition_summary(cfg, pipe)

    st.subheader("Validation")
    _migration_validation(cfg, pipe)

    st.subheader("The configuration")
    _config_export(cfg, name_hint="migration")


def _imaging_controls(cfg) -> None:
    """Every imaging choice the migration will make, in one place."""
    img = cfg.imaging
    c1, c2 = st.columns(2)
    condition = c1.selectbox(
        "Imaging condition", list(IMAGING_CONDITIONS),
        index=list(IMAGING_CONDITIONS).index(img.imaging_condition),
        help="source_normalized divides by the source illumination, which is "
             "what keeps a bright shallow event from setting the scale for "
             "the whole image.")
    laplacian = c2.checkbox(
        "Laplacian artefact filter", value=img.laplacian_filter,
        help="Applied as -∇², not ∇². A Laplacian turns a peak into a trough, "
             "so the unsigned operator inverts the polarity of every "
             "reflector — which it did here until it was caught.")
    taper = c1.slider(
        "Near-field taper (wavelengths)", 0.0, 2.0,
        float(img.taper_wavelengths), 0.25,
        help="Zeroes the image within this radius of any source or receiver. "
             "Those are injection points and the wavefield around one is a "
             "singularity no imaging condition removes — 57× the reservoir "
             "amplitude on the first run here.")
    start = c2.number_input(
        "Correlation start (s, 0 = from t=0)", 0.0, 2.0,
        float(img.correlation_start_time or 0.0), 0.01,
        help="Nothing reflected from the target can arrive sooner than the "
             "two-way time to it, so anything correlated before that is "
             "near-field by construction. Leave at 0 to derive it from the "
             "geometry.")

    c3, c4 = st.columns(2)
    limit_aperture = c3.checkbox(
        "Limit the aperture by offset", value=img.max_offset is not None,
        help="RTM has no per-trace aperture the way a Kirchhoff sum does, so "
             "the only place to choose which angles are migrated is the "
             "gather. The aliasing limit V/(4·f·sinθ) is set by the steepest "
             "angle actually summed, and a long line subtends a wide one at "
             "its ends whether or not those traces carry anything.")
    max_offset = img.max_offset
    offset_taper = img.offset_taper
    if limit_aperture:
        max_offset = c3.number_input(
            "Maximum offset (m)", 50.0, 20000.0,
            float(img.max_offset if img.max_offset is not None else 1000.0), 10.0,
            help="For a reflector h below the acquisition, an incidence angle "
                 "θ is an offset of 2·h·tan θ.")
        offset_taper = c4.number_input(
            "Offset taper width (m)", 0.0, 5000.0,
            float(img.offset_taper if img.offset_taper is not None
                  else 0.2 * float(max_offset)), 10.0,
            help="A hard cut is a step in the summed wavefield and images as "
                 "its own edge. 0 uses 20 % of the maximum offset.")
        offset_taper = float(offset_taper) or None
    else:
        max_offset = None

    c5, c6 = st.columns(2)
    scale = c5.slider("Migration velocity scale", 0.90, 1.10,
                      float(img.velocity_scale), 0.01,
                      help="1.0 migrates with the true model. Anything else is "
                           "a deliberate velocity-error experiment.")
    smooth = c6.slider("Migration velocity smoothing (m)", 0.0, 200.0,
                       float(img.velocity_smoothing), 10.0)
    mute = c5.checkbox(
        "Mute the direct arrival", value=img.mute_direct_arrival,
        help="A migration maps an event at time t onto |x−S| + |x−R| = v·t. A "
             "reflection has a stationary point on that surface; a direct "
             "arrival has none, so it paints the whole surface as a smile.")

    changed = (condition != img.imaging_condition
               or laplacian != img.laplacian_filter
               or taper != img.taper_wavelengths
               or start != (img.correlation_start_time or 0.0)
               or scale != img.velocity_scale
               or smooth != img.velocity_smoothing
               or mute != img.mute_direct_arrival
               or max_offset != img.max_offset
               or offset_taper != img.offset_taper)
    if changed:
        img.imaging_condition = condition
        img.laplacian_filter = bool(laplacian)
        img.taper_wavelengths = float(taper)
        img.correlation_start_time = float(start) if start > 0 else None
        img.velocity_scale = float(scale)
        img.velocity_smoothing = float(smooth)
        img.mute_direct_arrival = bool(mute)
        img.max_offset = float(max_offset) if max_offset is not None else None
        img.offset_taper = offset_taper
        invalidate()


def _survey_layout(cfg, pipe) -> None:
    """The survey drawn as it is designed, against the model it sits in.

    Every number above is a distance, and distances are read wrongly as
    numbers.  A spread that "extends 1,800 m" means nothing until it is seen
    against a 2,000 m model with an absorbing layer taking 120 m off each
    face; a 180 m separation between sources and receivers is invisible in a
    table and obvious in an elevation.  Both views are geometry only -
    milliseconds - so they redraw on every edit rather than behind a button.
    """
    try:
        acquisition = pipe.acquisition()
    except Sim3DError as exc:
        st.error(str(exc))
        return

    plan, side = st.tabs(["Plan view", "Elevation"])
    with plan:
        # A dense carpet draws tens of thousands of markers and the sources
        # disappear underneath them; the toggle is the honest way out.
        show_receivers = st.checkbox(
            "Show receivers", value=acquisition.n_receivers <= 4000,
            key="layout_show_receivers",
            help=f"{acquisition.n_receivers:,} receivers. Hiding them makes "
                 f"the shot layout visible when the carpet is dense.")
        st.plotly_chart(ui.aerial_figure(
            acquisition=acquisition, wells=pipe.wells(), domains=pipe.domains,
            pml_nodes=cfg.solver.pml_nodes, show_receivers=show_receivers),
            width="stretch")
        st.caption("Drawn to scale. The survey must reach past the imaging "
                   "target on every side — a spread that stops at its edge "
                   "lights that edge from one side only.")
    with side:
        st.plotly_chart(ui.elevation_figure(
            acquisition=acquisition, domains=pipe.domains,
            pml_nodes=cfg.solver.pml_nodes), width="stretch")
        st.caption("Depth down, same scale on both axes. What to look for: "
                   "the standoff above the target, and daylight between the "
                   "sources and the receivers — co-located, their injection "
                   "near-fields reinforce into a band brighter than the "
                   "reservoir.")


def _acquisition_summary(cfg, pipe) -> None:
    """Aperture, standoff and operator sampling, read-only.

    The same numbers the acquisition page shows, repeated here because they
    decide whether this migration is worth running at all - and because the
    one that matters most, the operator limit, was for a long time reachable
    only from a page nobody opened before a batch run.
    """
    try:
        acquisition = pipe.acquisition()
    except Sim3DError as exc:
        st.error(str(exc))
        return
    a, b, c, d = st.columns(4)
    a.metric("Sources", f"{acquisition.n_sources:,}",
             help="Cost is linear in this. The planner counts three "
                  "propagations per shot per earth model.")
    b.metric("Receivers", f"{acquisition.n_receivers:,}",
             help="Nearly free: the receiver wavefield is back-propagated as "
                  "one field however many receivers are in it.")
    c.metric("Traces", f"{acquisition.n_sources * acquisition.n_receivers:,}")
    d.metric("Shot spacing", f"{cfg.acquisition.source_spacing:,.0f} m")

    # Footprint against target, which is the coverage question: a survey that
    # stops at the target edge illuminates it from one side only.
    points = np.vstack([np.asarray(acquisition.sources, dtype=float),
                        np.asarray(acquisition.receivers, dtype=float)])
    (tx0, tx1), (ty0, ty1), _ = pipe.domains.target.bounds
    e, f, g = st.columns(3)
    e.metric("Inline footprint",
             f"{points[:, 0].min():,.0f} – {points[:, 0].max():,.0f} m",
             help=f"Target spans {tx0:,.0f} – {tx1:,.0f} m.")
    f.metric("Crossline footprint",
             f"{points[:, 1].min():,.0f} – {points[:, 1].max():,.0f} m",
             help=f"Target spans {ty0:,.0f} – {ty1:,.0f} m.")
    margin = min(tx0 - points[:, 0].min(), points[:, 0].max() - tx1)
    g.metric("Inline margin past target", f"{margin:,.0f} m",
             help="Below about 100 m the edge of the target is lit from one "
                  "side only and the image there should not be read.")

    earth = pipe.result.earth
    if earth is None:
        st.info("The operator-sampling check needs a velocity, which means the "
                "rock physics. Open **Rock Physics** once and it will appear "
                "here — the geometry above is free, that number is not.")
        return
    (_, _), (_, _), (z_top, _) = pipe.domains.target.bounds
    report = sampling_report(acquisition, z_top,
                             float(np.median(earth.models["baseline"].vp)),
                             pipe.fmax, cfg.acquisition.source_spacing,
                             cfg.acquisition.receiver_spacing)
    a, b, c = st.columns(3)
    a.metric("Aperture", f"{report.aperture_deg:.0f}°")
    b.metric("Standoff", f"{report.standoff_wavelengths:.1f} λ")
    c.metric("Operator limit", f"{report.operator_limit:,.0f} m")
    notes = report.notes()
    for note in notes:
        st.warning(note)
    if not notes:
        st.success(f"Source spacing is {report.factor(cfg.acquisition.source_spacing):.1f}× "
                   f"the operator limit and receiver spacing "
                   f"{report.factor(cfg.acquisition.receiver_spacing):.1f}× — "
                   f"within the tolerance the checks apply.")


def _migration_validation(cfg, pipe) -> None:
    """Everything that can be known before the run, and what it will cost."""
    geometry = pipe.geometry_qc()
    failed = [c for c in geometry.checks if c.status is Status.FAIL]
    warned = [c for c in geometry.checks if c.status is Status.WARNING]
    for check in failed:
        st.error(check.message)
    for check in warned:
        st.warning(check.message)
    if not failed and not warned:
        st.success("Geometry fits the propagation domain with room for the "
                   "absorbing layer.")

    st.caption("The full QC also checks dispersion and stability on every "
               "scenario, which needs the flow simulation and the rock "
               "physics behind it. Run it once the model is settled.")
    if st.button("Run full QC"):
        result = stage("Running QC", pipe.qc)
        if result is not None:
            for check in result.checks:
                if check.status is Status.FAIL:
                    st.error(check.message)
                elif check.status is Status.WARNING:
                    st.warning(check.message)
            if not result.failed:
                st.success("QC passed.")

    if st.button("Estimate the cost"):
        estimate = stage("Planning", pipe.plan)
        if estimate is not None:
            a, b, c = st.columns(3)
            a.metric("Cost class", estimate.cost_class.name)
            b.metric("Cell-steps", f"{estimate.cell_steps:.2e}")
            c.metric("Disk", f"{estimate.disk_bytes / 2 ** 30:.1f} GB")
            st.caption("Wall time depends on the machine that runs it. "
                       "`run_migration.py --check` benchmarks that machine "
                       "and reports the hours before anything is propagated.")


def _config_export(cfg, name_hint: str = "experiment") -> None:
    """Write the configuration out, complete and loadable.

    `to_dict` round-trips through `from_dict`, so what downloads here is what
    the pipeline will build - not a summary of it, and not the example file
    it started from.
    """
    import yaml

    # One slug for the file and for the output directory: a project named
    # with spaces produced a download called "my run.yaml" and a command with
    # an unquoted path in it, which is a broken command.
    slug = (cfg.project.name or name_hint).replace(" ", "_")
    filename = st.text_input("File name", f"{slug}.yaml",
                             key=f"export_name_{name_hint}")
    text = yaml.safe_dump(cfg.to_dict(), sort_keys=False, default_flow_style=False)

    a, b = st.columns([1, 3])
    a.download_button("Download YAML", text, file_name=filename,
                      mime="application/x-yaml", type="primary",
                      key=f"export_button_{name_hint}")
    b.caption(f"{len(text.splitlines()):,} lines · config hash "
              f"`{cfg.short_hash}` · loads with `ExperimentConfig.load`")

    st.markdown("**Then, on the machine with the cores:**")
    st.code(f"python examples/run_migration.py {filename} --check\n"
            f"python examples/run_migration.py {filename} "
            f"--out runs/{slug} --migrate difference",
            language="bash")
    st.caption("`--check` benchmarks that machine and reports the wall time "
               "before propagating anything. `--migrate difference` images the "
               "4D only, at half the cost of also migrating the baseline.")
    with st.expander("Under a scheduler"):
        st.code(f"sbatch examples/slurm_migration.sbatch {filename} "
                f"$PWD/runs/{slug} difference",
                language="bash")
        st.caption("One task, many cores, no GPU — the batch script explains "
                   "why in its header. It checkpoints every shot, so a job "
                   "that runs out of wall clock resumes rather than restarts.")
    with st.expander("Preview the YAML"):
        st.code(text, language="yaml")


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


def _wavelet_controls() -> None:
    """Choose the source wavelet, and show what it costs in sampling."""
    source = config().source
    with st.expander("Source wavelet"):
        kind = st.radio("Type", WAVELETS, horizontal=True,
                        index=WAVELETS.index(source.type), key="wavelet_type")
        if kind == "ricker":
            frequency = st.slider("Peak frequency (Hz)", 5.0, 80.0,
                                  float(source.frequency), 1.0)
            corners = source.corners
            if (kind, frequency) != (source.type, source.frequency):
                source.type, source.frequency = kind, frequency
                invalidate()
        else:
            current = list(source.corners or DEFAULT_ORMSBY_CORNERS)
            a, b, c, d = st.columns(4)
            corners = [
                a.number_input("f1 (Hz)", 1.0, 200.0, float(current[0]), 1.0),
                b.number_input("f2 (Hz)", 1.0, 200.0, float(current[1]), 1.0),
                c.number_input("f3 (Hz)", 1.0, 200.0, float(current[2]), 1.0),
                d.number_input("f4 (Hz)", 1.0, 200.0, float(current[3]), 1.0),
            ]
            if (kind, corners) != (source.type, source.corners):
                if corners[0] < corners[1] < corners[2] < corners[3]:
                    source.type, source.corners = kind, corners
                    invalidate()
                else:
                    st.error("Corners must increase: f1 < f2 < f3 < f4.")

        pipe = pipeline()
        dt = pipe.seismic_sample_interval()
        wavelet = pipe.wavelet(np.arange(int(0.6 / dt)) * dt)
        st.plotly_chart(ui.series_figure(
            np.arange(wavelet.size) * dt, {"wavelet": wavelet},
            xlabel="time (s)", ylabel="amplitude", height=220),
            width="stretch", key="wavelet_shape")
        st.caption(
            f"Fmax {pipe.fmax:,.1f} Hz, sample interval {1000 * dt:.0f} ms. "
            f"A Ricker is not band-limited at its peak, so its Fmax is a "
            f"spectral-fraction estimate; an Ormsby's is its top corner "
            f"exactly. Fmax sets the grid and the time sampling, so this is "
            f"not only a change to the trace.")


def _noise_controls(cfg) -> None:
    """Survey noise and how much of it repeats between surveys."""
    with st.expander("Survey noise and repeatability"):
        current = dict(cfg.noise or {})
        a, b, c = st.columns(3)
        level = a.slider("Noise (fraction of signal RMS)", 0.0, 0.5,
                         float(current.get("level", 0.0)), 0.01,
                         help="0 disables it. Without a noise term NRMS has "
                              "no floor and is not comparable to field data.")
        repeat = b.slider("Repeatability", 0.0, 1.0,
                          float(current.get("repeatability", 0.0)), 0.05,
                          help="How much of the noise is the same in both "
                               "surveys. 1 cancels completely in the "
                               "difference; 0 survives it entirely.")
        seed = c.number_input("Seed", 0, 10**6,
                              int(current.get("seed", 1)), 1)
        chosen = {"level": level, "repeatability": repeat, "seed": int(seed)}
        if chosen != current:
            cfg.noise = chosen
            invalidate()
        model = NoiseModel(**chosen)
        (st.info if model.active else st.warning)(model.describe())
        st.caption("Each survey gets sqrt(r)·shared + sqrt(1−r)·its own, so a "
                   "single survey carries the same noise whatever repeats — "
                   "only the difference changes. That makes the floor "
                   "predictable: 100·level·√(2(1−r)) percent.")


def _shift_controls(cfg) -> None:
    """Whether to estimate 4D time shifts, and over what window."""
    with st.expander("4D time shifts"):
        on = st.checkbox("Estimate time shifts and align the monitors",
                         value=bool(cfg.time_shifts))
        a, b, c = st.columns(3)
        window = a.slider("Correlation window (ms)", 20.0, 400.0,
                          1000 * float(cfg.shift_window), 10.0)
        step = b.slider("Hop (ms)", 4.0, 100.0, 1000 * float(cfg.shift_step), 2.0)
        search = c.slider("Search range (ms)", 5.0, 200.0,
                          1000 * float(cfg.max_shift), 5.0)
        chosen = (on, window / 1000.0, step / 1000.0, search / 1000.0)
        if chosen != (cfg.time_shifts, cfg.shift_window, cfg.shift_step,
                      cfg.max_shift):
            (cfg.time_shifts, cfg.shift_window, cfg.shift_step,
             cfg.max_shift) = chosen
            invalidate()
        st.caption("A softened reservoir delays everything beneath it, whether "
                   "or not that rock changed. Differencing without accounting "
                   "for it turns a shift into a derivative-shaped anomaly at "
                   "the wrong depth — on the three-layer model a 6 ms shift "
                   "alone makes 58% NRMS.")


def page_sim2seis() -> None:
    pipe = pipeline()
    st.title("Synthetic seismic volume")
    st.caption("The earth model converted to seismic, column by column, in "
               "angle stacks — the sim2seis path. Seconds, not hours, because "
               "nothing is propagated and nothing is migrated.")

    cfg = config().sim2seis
    with st.expander("Angle stacks", expanded=not pipe.result.volumes):
        rows = []
        for stack in cfg.stacks:
            lo, hi = (float(a) for a in stack["angles"])
            c1, c2 = st.columns(2)
            low = c1.slider(f"{stack['name']} — from (deg)", 0.0, 60.0, lo, 1.0,
                            key=f"a_lo_{stack['name']}")
            high = c2.slider(f"{stack['name']} — to (deg)", 0.0, 60.0, hi, 1.0,
                             key=f"a_hi_{stack['name']}")
            rows.append({"name": stack["name"], "angles": [low, high]})
        if rows != cfg.stacks:
            try:
                build_stacks(rows)
            except Sim3DError as exc:
                st.error(str(exc))
            else:
                cfg.stacks = rows
                invalidate()
        st.caption("Beyond about 45 degrees the Aki-Richards linearisation "
                   "stops being trustworthy, and the tool says so rather than "
                   "quietly extrapolating.")

    _wavelet_controls()
    _noise_controls(cfg)
    _shift_controls(cfg)

    if st.button("Build synthetic volume", type="primary"):
        progress = st.progress(0.0, text="starting — 0%")

        def report(name, done, count):
            fraction = min(done / max(count, 1), 1.0)
            progress.progress(fraction, text=f"{name} — {fraction:.0%}")
        try:
            pipe.sim2seis(progress=report)
        except Sim3DError as exc:
            st.error(str(exc))
        progress.empty()

    volumes = pipe.result.volumes
    if not volumes:
        st.info("No volume yet. It takes seconds — the rock physics above it "
                "is the part that costs time.")
        return

    base = volumes["baseline"]
    st.caption(base.label)
    for note in base.notes:
        st.warning(note)

    grid = base.grid
    point = cursor_controls(grid)
    a, b, c = st.columns(3)
    stack = a.selectbox("Angle stack", list(base.names))
    scenario = b.selectbox("Earth model", list(volumes))
    domain = c.radio("Axis", ["depth", "time"], horizontal=True, key="s2s_axis")

    st.subheader("Volume")
    if domain == "depth":
        st.plotly_chart(ui.slice_figure(
            volumes[scenario].depth_cube(stack), grid, point,
            title=f"{scenario} — {stack}", kind="diverging", unit="amplitude",
            wells=pipe.wells()), width="stretch")
    else:
        _time_slices(volumes[scenario].time_cube(stack), base, stack, scenario)

    if scenario != "baseline":
        st.subheader("Difference from baseline")
        difference = (volumes[scenario].depth_cube(stack).astype(float)
                      - base.depth_cube(stack).astype(float))
        st.plotly_chart(ui.slice_figure(
            difference, grid, point, title=f"{scenario} − baseline — {stack}",
            kind="diverging", unit="amplitude", wells=pipe.wells()),
            width="stretch")

    st.subheader("Repeatability by angle stack")
    rows = {}
    for name in list(volumes)[1:]:
        rows[name] = {
            s: round(nrms(base.time_cubes[s], volumes[name].time_cubes[s]), 3)
            for s in base.names}
    if rows:
        st.dataframe(rows, width="stretch")
        st.caption("A response that falls with angle while another rises is "
                   "the AVO discrimination between a pressure change and a "
                   "fluid one — the reason to carry more than one stack.")
    st.caption(f"{base.megabytes:,.0f} MB per earth model in memory.")

    # Sparse synthetics live here now rather than with the migration: they
    # propagate nothing either, and grouping the two propagation-free seismic
    # modes together is what makes the page a modelling page.
    st.divider()
    _sparse_section(pipe)


def _time_slices(cube, base, stack: str, scenario: str) -> None:
    """Inline and crossline sections against the two-way time axis."""
    nx, ny, _ = cube.shape
    a, b = st.columns(2)
    iy = a.slider("inline (y index)", 0, ny - 1, ny // 2, key="s2s_il")
    ix = b.slider("crossline (x index)", 0, nx - 1, nx // 2, key="s2s_xl")
    left, right = st.columns(2)
    left.plotly_chart(ui.section_figure(
        cube[:, iy, :].T, base.grid.axis("x"), base.times,
        xlabel="x (m)", ylabel="two-way time (s)",
        title=f"inline — {scenario} · {stack}"), width="stretch")
    right.plotly_chart(ui.section_figure(
        cube[ix, :, :].T, base.grid.axis("y"), base.times,
        xlabel="y (m)", ylabel="two-way time (s)",
        title=f"crossline — {scenario} · {stack}"), width="stretch")


def page_fourd() -> None:
    """4D on whatever exists: sim2seis here, migrated images from elsewhere.

    The app no longer migrates, so the migrated volumes this page used to
    read from memory now arrive as a file written by `run_migration.py` on
    whichever machine ran it. Everything that needs no propagation -
    the property-space decomposition, the sim2seis 4D - is computed here as
    it always was.
    """
    pipe = pipeline()
    st.title("4D analysis")
    _migrated_results_section(pipe)

    if not pipe.result.images:
        st.divider()
        st.info("The seismic-space decomposition below needs all four scenarios "
                "migrated, which happens outside the app. The property-space "
                "decomposition on **Rock Physics** and the sim2seis 4D on "
                "**Synthetic Volume** need neither and are available now.")
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



def _migrated_results_section(pipe) -> None:
    """Load and show an images.npz written by a migration run elsewhere.

    This is the return leg of the split: the app builds the configuration,
    a batch job somewhere else does the propagating, and the result comes
    back here as a file rather than as a four-hour page load.
    """
    st.subheader("Migrated images from a run")
    st.caption("Point this at the `images.npz` that `run_migration.py` wrote. "
               "Nothing is propagated here — the file already holds the image.")
    path_text = st.text_input("Path to images.npz", key="images_path",
                              placeholder="runs/dense/images.npz")
    if not path_text:
        return
    path = Path(path_text).expanduser()
    if not path.exists():
        st.error(f"No such file: {path}")
        return
    try:
        loaded = np.load(path)
        keys = [k for k in loaded.files if k not in ("origin", "spacing")]
        origin = tuple(float(v) for v in loaded["origin"])
        spacing = tuple(float(v) for v in loaded["spacing"])
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not read {path.name}: {exc}")
        return
    if not keys:
        st.error(f"{path.name} holds no images, only the grid.")
        return

    from sim3d.core.grid import Grid3D
    grid = Grid3D(origin, spacing, loaded[keys[0]].shape)
    st.caption(f"{path.name} · {', '.join(sorted(keys))} · grid {grid.shape} "
               f"at {spacing[0]:g} m, origin {origin}")
    which = st.radio("Image", sorted(keys), horizontal=True, key="loaded_image")
    point = cursor_controls(grid)
    st.plotly_chart(ui.slice_figure(
        loaded[which], grid, point, title=which.replace("_", " "),
        kind="diverging", unit="amplitude", wells=pipe.wells()), width="stretch")

    # NRMS needs a baseline to normalise by, and a run asked only for the
    # difference will not have one.
    base_key = "baseline" if "baseline" in keys else None
    monitors = [k for k in keys if k.startswith("monitor_")]
    if base_key and monitors:
        st.dataframe({k.replace("monitor_", ""): {
            "NRMS (%)": round(float(nrms(loaded[base_key], loaded[k])), 3)}
            for k in monitors}, width="stretch")
    elif not base_key:
        st.caption("No baseline image in this file, so NRMS is undefined — it "
                   "is normalised by the baseline. A run made with "
                   "`--migrate difference` writes the 4D alone, which is the "
                   "cheaper half and usually the point.")


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


def _new_well_spec(specs, role: str, x: float, y: float, grid) -> dict:
    """A well placed on the map, set up like the ones already there.

    Left to the automatic suggestion a new well is rated by *pattern scale*:
    the pore volume inside a drainage radius of half the distance to its
    nearest neighbour, swept over ``sweep_years``.  Drop one next to an
    existing well and that radius collapses - on the dipping wedge, placing
    a producer 200 m from P1 suggests 247 STB/day against P1's 2,250, which
    reads as a broken well rather than as the geometric consequence it is.

    So a new well copies its completion interval, control mode, target and
    limits from the first well of the same role, which is what "the same as
    the others" means. With no well of that role to copy, the fields stay
    unset and the suggestion stands - it has no pattern to be out of step
    with.
    """
    prefix = "P" if role == "producer" else "I"
    spec = {"name": unique_name(prefix, specs), "role": role, "x": x, "y": y,
            "perforation": [grid.bounds[2][0], grid.bounds[2][1]],
            "completions": [],
            "control": "liquid_rate" if role == "producer" else "water_rate",
            "target": None, "bhp_limit_psi": None,
            "start_day": 0.0, "end_day": None}
    template = next((w for w in specs if w.get("role") == role), None)
    if template is not None:
        for key in ("perforation", "completions", "control", "target",
                    "bhp_limit_psi", "start_day", "end_day"):
            if key in template:
                spec[key] = copy.deepcopy(template[key])
    return spec


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
            specs.append(_new_well_spec(specs, well_type, x, y, grid))
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
    e, f = st.columns(2)
    live = e.checkbox(
        "Solution gas", value=cfg.simulation.solution_gas,
        help="Let gas come out of solution where the pressure falls below the "
             "bubble point. The PVT comes from the rock-physics section, so the "
             "flow and the seismic describe the same oil. The liberated gas does "
             "not flow between cells, so it is trustworthy while the gas "
             "saturation stays below critical and indicative after that.")
    critical = f.number_input(
        "Critical gas saturation", 0.001, 0.30,
        float(cfg.simulation.critical_gas_saturation), 0.005, format="%.3f",
        help="Where free gas would start to move. This model holds it in place, "
             "so this is the saturation past which the answer stops being "
             "quantitative rather than a flow parameter.")
    if (duration, report, step, gravity, live, critical) != (
            cfg.simulation.duration_days, cfg.simulation.report_every_days,
            cfg.simulation.max_timestep_days, cfg.simulation.gravity,
            cfg.simulation.solution_gas, cfg.simulation.critical_gas_saturation):
        cfg.simulation.duration_days = duration
        cfg.simulation.report_every_days = report
        cfg.simulation.max_timestep_days = step
        cfg.simulation.gravity = gravity
        cfg.simulation.solution_gas = live
        cfg.simulation.critical_gas_saturation = critical
        invalidate()

    if live:
        gas = pipe.solution_gas()
        st.caption(f"Bubble point {pa_to_psi(gas.bubble_point):,.0f} psi at "
                   f"GOR {gas.initial_gor:g} m³/m³, {gas.api:g} API — gas comes "
                   f"out below that and nowhere else.")

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
    if flow.gas_saturation:
        peak = max(float(sg.max()) for sg in flow.gas_saturation)
        critical = flow.settings.solution_gas.critical_saturation
        message = (f"peak gas saturation {peak:.3f} against a critical "
                   f"{critical:g} · hydrocarbon volume closes to "
                   f"{100 * flow.volume_closure_error:.2f}% of pore volume on "
                   f"average, {100 * flow.peak_volume_closure_error:.0f}% at "
                   f"worst (almost always a well block)")
        (st.warning if peak > critical else st.info)(message)
    for note in flow.notes:
        st.caption(f"note: {note}")

    st.markdown("**Field state through time**")
    day = st.select_slider("Day", options=[float(d) for d in flow.days],
                           value=float(flow.days[-1]))
    grid = pipe.geology().grid
    point = cursor_controls(grid)
    choices = ["pressure", "water saturation", "ΔP", "ΔSw"]
    if flow.gas_saturation:
        choices += ["gas saturation", "ΔSg", "solution GOR"]
    what = st.radio("Volume", choices, horizontal=True)
    pressure, saturation = flow.at(day)
    p0, s0 = flow.at(flow.days[0])
    mask = pipe.geology().reservoir_mask
    lookup = {
        "pressure": (np.where(mask, pa_to_psi(pressure), np.nan), "sequential", "psi"),
        "water saturation": (np.where(mask, saturation, np.nan), "sequential", "fraction"),
        "ΔP": (np.where(mask, pa_to_psi(pressure - p0), np.nan), "diverging", "psi"),
        "ΔSw": (np.where(mask, saturation - s0, np.nan), "diverging", "fraction"),
    }
    if flow.gas_saturation:
        index = int(np.argmin(np.abs(np.asarray(flow.days) - day)))
        sg = flow.gas_saturation[index]
        lookup["gas saturation"] = (np.where(mask, sg, np.nan), "sequential",
                                    "fraction")
        lookup["ΔSg"] = (np.where(mask, sg - flow.gas_saturation[0], np.nan),
                         "diverging", "fraction")
        lookup["solution GOR"] = (np.where(mask, flow.solution_gor[index], np.nan),
                                  "sequential", "m³/m³")
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
    "Synthetic Volume": page_sim2seis,
    "Acquisition & QC": page_acquisition,
    "Migration Setup": page_migration,
    "4D Analysis": page_fourd,
    "Scenarios": page_scenarios,
}


def main() -> None:
    st.set_page_config(page_title="sim3d", page_icon="🌊", layout="wide")
    PAGE_FUNCTIONS[sidebar()]()


if __name__ == "__main__":
    main()
