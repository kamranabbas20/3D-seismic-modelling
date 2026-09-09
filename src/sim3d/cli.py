"""Command-line interface (spec section 118).

The scientific engine runs without any GUI.  Every command takes a
configuration file and drives :class:`sim3d.experiments.Pipeline`; no
science lives here.

    sim3d describe   model.yaml    domains, models and the content hash
    sim3d physics    model.yaml    what each simulation mode does and does not model
    sim3d build      model.yaml    geology, wells and the four reservoir states
    sim3d rockphysics model.yaml   the four earth models and the property decomposition
    sim3d qc         model.yaml    the section 127 model checks
    sim3d plan       model.yaml    the cost estimate and the budget decision
    sim3d benchmark                measure this machine's throughput
    sim3d preview    model.yaml    fast 1D convolution screening
    sim3d synthetic  model.yaml    K vertical 1D synthetic traces
    sim3d simulate   model.yaml    full-wave shot gathers, one set per earth model
    sim3d migrate    model.yaml    3D RTM of every scenario
    sim3d decompose  model.yaml    the 4D difference and the interaction term
    sim3d run        model.yaml    every stage, end to end
    sim3d gui                      launch the Streamlit research GUI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

from .core.config import ExperimentConfig
from .core.errors import Sim3DError
from .core.planning import measure_throughput
from .experiments.pipeline import Pipeline
from .fourd.decomposition import describe as describe_decomposition
from .fourd.metrics import nrms
from .fourd.scenarios import SCENARIO_NAMES
from .validation.physics import physics_table

RULE = "=" * 78


def _heading(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def _load(path: str) -> Pipeline:
    return Pipeline(ExperimentConfig.load(path))


def _progress(stage_label: str):
    def report(name, done, total):
        end = "\n" if done == total else "\r"
        print(f"  {stage_label} {name:16s} {done:4d} / {total}", end=end, flush=True)
    return report


# --- commands -------------------------------------------------------------
def cmd_describe(args) -> int:
    pipeline = _load(args.config)
    _heading("Configuration")
    print(pipeline.config.describe())
    _heading("Source bandwidth")
    print(f"  peak frequency {pipeline.config.source.frequency:g} Hz; practical Fmax "
          f"{pipeline.fmax:.1f} Hz at {pipeline.config.source.bandwidth_fraction:.0%} "
          f"of peak spectral amplitude.\n"
          f"  The grid must be sampled for Fmax, not for the peak frequency.")
    return 0


def cmd_physics(args) -> int:
    modes = ["convolution", "sparse_synthetic", "acoustic_fd", "rtm"]
    _heading("Physics of each simulation mode (spec section 83)")
    print(physics_table(modes))
    return 0


def cmd_build(args) -> int:
    pipeline = _load(args.config)
    pipeline.run(("geology", "reservoir"))
    _heading("Domains")
    print(pipeline.domains.summary())
    _heading("Geology")
    print(pipeline.result.geology.summary())
    _heading("Wells")
    print(pipeline.wells().summary())
    _heading("Reservoir states")
    print(pipeline.reservoir().baseline.summary())
    print()
    print(pipeline.scenario().describe())
    base = pipeline.reservoir().baseline
    mask = base.reservoir_mask
    for name in SCENARIO_NAMES[1:]:
        d = pipeline.reservoir()[name].difference(base)
        print(f"\n  {name}:")
        for key, scale, unit in (("dP", 1e5, "bar"), ("dSw", 1.0, ""),
                                 ("dSg", 1.0, "")):
            v = d[key][mask]
            print(f"    {key:4s} {v.min() / scale:+9.3f} to {v.max() / scale:+9.3f} {unit}")
    return 0


def cmd_rockphysics(args) -> int:
    pipeline = _load(args.config)
    pipeline.run(("geology", "reservoir", "rockphysics"))
    _heading("Rock-physics configuration")
    print(pipeline.rock_physics_config().describe())
    _heading("Baseline result")
    print(pipeline.result.earth.rock_physics["baseline"].summary())
    _heading("Four earth models")
    print(pipeline.result.earth.summary())
    _heading("Property-space decomposition (spec section 53)")
    parts = pipeline.decompose()
    mask = parts["property_mask"]
    for attribute, scale, unit in (("vp", 1.0, "m/s"), ("rho", 1.0, "kg/m3"),
                                   ("ai", 1e6, "1e6 kg/m2/s")):
        print(describe_decomposition(parts["property"][attribute], attribute.upper(),
                                     scale, unit, mask=mask))
        print()
    return 0


def cmd_qc(args) -> int:
    pipeline = _load(args.config)
    _heading("Model QC (spec section 127)")
    result = pipeline.qc()
    print(result.report())
    for note in pipeline.result.notes:
        print(f"note    - {note}")
    if result.failed:
        print("\nQC FAILED. Fix the failures above before simulating; sim3d will not "
              "adjust the grid, the bandwidth or the geometry on your behalf.")
        return 1
    print("\nQC passed.")
    return 0


def cmd_plan(args) -> int:
    pipeline = _load(args.config)
    _heading("Computational estimate (spec section 13)")
    try:
        estimate = pipeline.plan(benchmark=args.benchmark)
    except Sim3DError as exc:
        print(exc)
        return 1
    print(estimate.describe(throughput=getattr(pipeline, "_throughput", None)))
    print("\nWithin the configured budget.")
    return 0


def cmd_gui(args) -> int:
    """Launch the Streamlit front end.

    Streamlit owns the process from here; the engine is unchanged and the
    GUI is only another caller of the same Pipeline the CLI uses.
    """
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print("The GUI needs Streamlit and Plotly:\n"
              '    pip install -e ".[ui]"', file=sys.stderr)
        return 1
    app = Path(__file__).resolve().parent / "ui" / "streamlit_app.py"
    argv = ["streamlit", "run", str(app)]
    if args.port:
        argv += ["--server.port", str(args.port)]
    if args.headless:
        argv += ["--server.headless", "true"]
    sys.argv = argv
    return int(stcli.main() or 0)


def cmd_benchmark(args) -> int:
    _heading("Throughput benchmark")
    for backend in ("numpy", "numba"):
        try:
            rate = measure_throughput(backend=backend)
        except Sim3DError as exc:
            print(f"  {backend:8s} unavailable: {exc}")
            continue
        print(f"  {backend:8s} {rate / 1e6:8.2f} Mcell-steps/s")
    print("\nRuntime estimates are only quoted against a measured rate like these.")
    return 0


def cmd_preview(args) -> int:
    pipeline = _load(args.config)
    _heading("Fast convolution preview (spec section 82)")
    previews = pipeline.preview()
    for name, preview in previews.items():
        print(f"  {name:16s} {preview.describe().splitlines()[-1].strip()}")
    print(f"\n  {list(previews.values())[0].label}")
    base = previews["baseline"].depth_traces
    for name in SCENARIO_NAMES[1:]:
        if name in previews:
            d = previews[name].depth_traces - base
            print(f"  screening difference {name:16s} RMS "
                  f"{np.sqrt(np.mean(d**2)):.6f}")
    return 0


def cmd_synthetic(args) -> int:
    pipeline = _load(args.config)
    _heading("Sparse vertical synthetics")
    synthetics = pipeline.synthetic()
    base = synthetics["baseline"]
    print(f"  {base.label}")
    print(f"  {base.n_traces} traces, {base.times.size} samples at "
          f"{(base.times[1] - base.times[0]) * 1e3:.2f} ms\n")
    for location in base.locations:
        print(f"    {location.describe()}")
    print()
    for name in SCENARIO_NAMES[1:]:
        if name not in synthetics:
            continue
        d = synthetics[name].traces - base.traces
        print(f"  4D difference {name:16s} RMS {np.sqrt(np.mean(d**2)):.6e}")
    print("\n  per trace, NRMS against baseline (%)")
    header = "    " + "".join(f"{n:>16s}" for n in base.names)
    print(header)
    for name in SCENARIO_NAMES[1:]:
        if name not in synthetics:
            continue
        cells = "".join(
            f"{nrms(base.traces[i], synthetics[name].traces[i]):>16.3f}"
            for i in range(base.n_traces))
        print(f"    {name:16s}{cells}")
    return 0


def cmd_simulate(args) -> int:
    pipeline = _load(args.config)
    if pipeline.qc().failed and not args.force:
        print(pipeline.qc().report())
        print("\nQC failed; refusing to simulate. Re-run with --force only if you "
              "understand exactly which check you are overriding and why.")
        return 1
    _heading("Full-wave modelling")
    gathers = pipeline.simulate(progress=_progress("shot"))
    for name, records in gathers.items():
        peak = max(float(np.max(np.abs(r.traces))) for r in records)
        print(f"  {name:16s} {len(records)} gathers, "
              f"{records[0].traces.shape[0]} traces x {records[0].nt} samples, "
              f"peak |p| = {peak:.4g}")
    print(f"\n  elapsed {pipeline.result.timings['simulate']:.1f} s")
    return 0


def cmd_migrate(args) -> int:
    pipeline = _load(args.config)
    if pipeline.qc().failed and not args.force:
        print(pipeline.qc().report())
        print("\nQC failed; refusing to migrate.")
        return 1
    _heading("3D reverse time migration")
    images = pipeline.migrate(progress=_progress("migrating"))
    for name, image in images.items():
        print(f"  {name:16s} {image.describe().splitlines()[0]}")
    print(f"\n  elapsed {pipeline.result.timings['migrate']:.1f} s")
    return 0


def cmd_decompose(args) -> int:
    pipeline = _load(args.config)
    if args.migrate:
        if pipeline.qc().failed and not args.force:
            print(pipeline.qc().report())
            print("\nQC failed; refusing to migrate.")
            return 1
        pipeline.migrate(progress=_progress("migrating"))
    parts = pipeline.decompose()
    _heading("Property-space decomposition")
    for attribute, scale, unit in (("vp", 1.0, "m/s"), ("ai", 1e6, "1e6 kg/m2/s")):
        print(describe_decomposition(parts["property"][attribute], attribute.upper(),
                                     scale, unit, mask=parts["property_mask"]))
        print()
    if parts["seismic"]:
        _heading("Seismic-space decomposition (spec section 55)")
        print(describe_decomposition(parts["seismic"]["rtm"], "migrated amplitude"))
        print("\n  NRMS against the baseline image:")
        for name, value in parts["seismic"]["nrms"].items():
            print(f"    {name:16s} {value:7.2f} %")
    else:
        print("\nNo migrated images yet; add --migrate to run RTM first.")
    return 0


def cmd_run(args) -> int:
    pipeline = _load(args.config)
    _heading("Configuration")
    print(pipeline.config.describe())
    _heading("Model QC")
    qc = pipeline.qc()
    print(qc.report())
    if qc.failed and not args.force:
        print("\nQC failed; stopping before any simulation.")
        return 1
    _heading("Computational estimate")
    try:
        print(pipeline.plan().describe())
    except Sim3DError as exc:
        print(exc)
        return 1
    pipeline.run(("simulate", "migrate"), progress=_progress("running"))
    return cmd_decompose_from(pipeline)


def cmd_decompose_from(pipeline: Pipeline) -> int:
    parts = pipeline.decompose()
    _heading("Property-space decomposition")
    print(describe_decomposition(parts["property"]["ai"], "AI", 1e6, "1e6 kg/m2/s",
                                 mask=parts["property_mask"]))
    if parts["seismic"]:
        _heading("Seismic-space decomposition")
        print(describe_decomposition(parts["seismic"]["rtm"], "migrated amplitude"))
        print("\n  NRMS against the baseline image:")
        for name, value in parts["seismic"]["nrms"].items():
            print(f"    {name:16s} {value:7.2f} %")
    _heading("Timings")
    for stage, seconds in pipeline.result.timings.items():
        print(f"  {stage:12s} {seconds:8.1f} s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sim3d",
        description="3D Mechanistic Reservoir-to-Seismic & 4D Inversion Laboratory",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, handler, help_text, needs_config=True):
        p = sub.add_parser(name, help=help_text)
        if needs_config:
            p.add_argument("config", help="experiment YAML or JSON file")
        p.set_defaults(handler=handler)
        return p

    add("describe", cmd_describe, "show the configuration and its content hash")
    add("physics", cmd_physics, "what each simulation mode does and does not model",
        needs_config=False)
    add("build", cmd_build, "build geology, wells and the four reservoir states")
    add("rockphysics", cmd_rockphysics, "build the four earth models and decompose them")
    add("qc", cmd_qc, "run the model and geometry checks")
    add("plan", cmd_plan, "estimate cost and check it against the budget") \
        .add_argument("--benchmark", action="store_true",
                      help="measure this machine before quoting a runtime")
    add("benchmark", cmd_benchmark, "measure solver throughput", needs_config=False)
    gui = add("gui", cmd_gui, "launch the Streamlit research GUI", needs_config=False)
    gui.add_argument("--port", type=int, default=None, help="port to serve on")
    gui.add_argument("--headless", action="store_true",
                     help="do not try to open a browser")
    add("preview", cmd_preview, "fast 1D convolution screening")
    add("synthetic", cmd_synthetic, "K vertical 1D synthetic traces")
    add("simulate", cmd_simulate, "model shot gathers for every earth model") \
        .add_argument("--force", action="store_true", help="proceed despite failed QC")
    add("migrate", cmd_migrate, "run 3D RTM on every scenario") \
        .add_argument("--force", action="store_true", help="proceed despite failed QC")
    decompose = add("decompose", cmd_decompose, "4D differences and the interaction term")
    decompose.add_argument("--migrate", action="store_true",
                           help="run simulation and migration first")
    decompose.add_argument("--force", action="store_true",
                           help="proceed despite failed QC")
    add("run", cmd_run, "run every stage end to end") \
        .add_argument("--force", action="store_true", help="proceed despite failed QC")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except Sim3DError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
