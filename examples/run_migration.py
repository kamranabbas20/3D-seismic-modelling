#!/usr/bin/env python
"""Run a 4D migration end to end, checkpointed shot by shot.

``sim3d migrate`` already does this, and on a small model it is the right
tool.  This script exists for the runs that take hours: it writes every shot
to disk as soon as it exists, so an interrupted job restarts where it
stopped rather than from the beginning, and it drives the machine rather
than accepting the defaults.

    python examples/run_migration.py examples/configs/five_layer_600m_dense.yaml \
        --out runs/dense --threads 32

    python examples/run_migration.py <config> --check     # QC and cost only

On GPUs
-------
sim3d has no GPU backend.  ``compute.GPUBackend`` is a deliberate stub that
raises rather than silently falling back, and this script will refuse
``--backend gpu`` for the same reason.  The obstacle is not the two
derivative kernels - those port to CuPy in an afternoon - it is that
``AcousticSolver`` allocates its fields with ``np.zeros`` and does the CPML
memory variables, the source injection and the receiver gathers in NumPy.
A backend that only accelerated the derivatives would copy the pressure
field host-to-device and back six times per timestep: for a 1.6 M-cell grid
that is about 26 GB of PCIe traffic per shot, and it would run slower than
the CPU does now.  A real port keeps the fields on the device for the whole
propagation, which is a change to the solver rather than a new backend.

What a GPU machine does buy today is cores.  The Numba kernels parallelise
over the outer axis with ``prange``, so the wall time scales with the core
count until memory bandwidth takes over - ``--threads`` sets it, and
``--check`` reports what the machine can do before you commit hours to it.

The 4D image
------------
Two migrations are run, not three.  The migration velocity is built from the
baseline earth for *both* surveys, so the source wavefield is identical and
the imaging condition is linear in the recorded data::

    I_monitor - I_baseline = M(d_monitor - d_baseline)

Migrating the difference gather is therefore exactly the 4D image, reached
in one pass instead of by subtracting two.  It also cancels everything the
surveys share - the direct arrival, every overburden reflection, the
injection near-field - in the data domain, before any operator touches it.
It runs first because it is the pass that answers the question.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle
import sys
import time


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="experiment YAML or JSON file")
    p.add_argument("--out", default="runs/migration",
                   help="directory for checkpoints and results")
    p.add_argument("--backend", default="numba", choices=("numba", "numpy"),
                   help="compute backend; 'gpu' is not implemented (see module docstring)")
    p.add_argument("--threads", type=int, default=None,
                   help="Numba threads; default is every core the machine reports")
    p.add_argument("--check", action="store_true",
                   help="run QC and the cost estimate, then stop")
    p.add_argument("--fresh", action="store_true",
                   help="ignore any existing checkpoints and start over")
    p.add_argument("--force", action="store_true",
                   help="proceed even if QC fails")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Thread count has to be set before Numba is imported, which means before
    # sim3d is, because importing it builds the backend registry.
    if args.threads:
        os.environ["NUMBA_NUM_THREADS"] = str(args.threads)
    # The BLAS pools and the Numba pool both size themselves to the core count
    # and then fight over it. Two Numba jobs on four cores measured a 13x
    # slowdown on this project; the same happens between the pools.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    import numpy as np
    from dataclasses import replace

    from sim3d.core.config import ExperimentConfig
    from sim3d.experiments.pipeline import Pipeline
    from sim3d.fourd.metrics import nrms
    from sim3d.imaging.rtm import RTMSettings, filtered_image, migrate_shot
    from sim3d.wave.acoustic import AcousticSolver, steps_for_duration
    from sim3d.wave.sources import PointSource

    t0 = time.time()

    def log(message: str) -> None:
        print(f"[{time.time() - t0:8.1f}s] {message}", flush=True)

    def atomic(path: pathlib.Path, write) -> None:
        """Write via a temporary so a kill never leaves half a checkpoint.

        The temporary keeps the real suffix: ``np.savez`` appends ``.npz`` to
        any name that lacks it, so a ``.npz.tmp`` scratch name comes back as
        ``.npz.tmp.npz`` and the rename finds nothing to move.
        """
        tmp = path.with_name(path.name + ".tmp" + path.suffix)
        write(tmp)
        os.replace(tmp, path)

    out = pathlib.Path(args.out)
    (out / "fwd").mkdir(parents=True, exist_ok=True)
    (out / "mig").mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig.load(args.config)
    pipeline = Pipeline(config)
    # The hash covers the configuration, so a checkpoint from a different
    # geometry is ignored rather than silently mixed into this one.
    digest = config.content_hash()[:12]
    log(f"{args.config}  hash {digest}")

    try:
        import numba
        log(f"numba {numba.__version__}, {numba.get_num_threads()} threads "
            f"of {len(os.sched_getaffinity(0))} cores")
    except ImportError:
        log("numba not importable - the numpy backend will be slow")

    qc = pipeline.qc()
    for check in qc.checks:
        if check.status.value.upper() != "PASS":
            log(f"  {check}")
        elif "operator" in check.message:
            log(f"  {check}")
    if qc.failed and not args.force:
        log("QC failed; pass --force to run anyway")
        return 1

    estimate = pipeline.plan()
    acquisition = pipeline.acquisition()
    models, dt = pipeline.propagation_models()
    nt = steps_for_duration(config.solver.record_length, dt)
    scenarios = list(config.fourd.scenarios)
    n_shots = len(acquisition.sources)
    log(f"{n_shots} sources x {acquisition.n_receivers} receivers, "
        f"{nt} steps at {dt * 1e3:.3f} ms, scenarios {scenarios}")
    log(f"cost class {estimate.cost_class.name}, {estimate.cell_steps:.2e} "
        f"cell-steps, {estimate.ram_bytes / 2**30:.1f} GB RAM, "
        f"{estimate.disk_bytes / 2**30:.1f} GB disk")
    # A throughput measurement turns cell-steps into an honest wall time, and
    # on an unfamiliar machine that is the number worth having before
    # committing to the run.
    if args.check:
        from sim3d.core.planning import measure_throughput
        rate = measure_throughput(backend=args.backend, dtype=pipeline.dtype)
        hours = estimate.runtime_seconds(rate) / 3600.0
        log(f"measured {rate:.2e} cell-steps/s here -> about {hours:.1f} h "
            f"for the whole experiment on this machine")
    if args.check:
        log("--check: stopping before any propagation")
        return 0

    wavelet = pipeline.wavelet(np.arange(nt) * dt)
    settings = pipeline.solver_settings(dt)

    # ------------------------------------------------------------ forward
    gathers: dict[str, list] = {}
    for name in scenarios:
        records, solver = [], None
        for ishot, source in enumerate(acquisition.sources):
            shot = out / "fwd" / f"{digest}_{name}_{ishot:04d}.pkl"
            if shot.exists() and not args.fresh:
                with shot.open("rb") as fh:
                    records.append(pickle.load(fh))
                continue
            if solver is None:
                solver = AcousticSolver(models[name], settings,
                                        f0=config.source.frequency,
                                        backend=args.backend)
            record = solver.run(
                PointSource(pipeline.domains.propagation, source, wavelet), nt,
                receivers=acquisition.receivers)
            atomic(shot, lambda t, r=record: pickle.dump(
                r, t.open("wb"), protocol=pickle.HIGHEST_PROTOCOL))
            records.append(record)
            if (ishot + 1) % 5 == 0 or ishot + 1 == n_shots:
                log(f"  forward {name} {ishot + 1}/{n_shots}")
        gathers[name] = records
        log(f"forward {name} complete")
    pipeline.result.gathers = gathers
    pipeline._wavelet = wavelet

    # ---------------------------------------------------------- migration
    migration = pipeline.migration_model(models["baseline"])
    rtm = RTMSettings(
        imaging_condition=config.imaging.imaging_condition,
        time_decimation=config.imaging.time_decimation,
        epsilon=config.imaging.epsilon,
        laplacian_filter=config.imaging.laplacian_filter,
        taper_wavelengths=config.imaging.taper_wavelengths,
        taper_radius=config.imaging.taper_radius,
        correlation_start_time=pipeline.correlation_start_time(
            models["baseline"], acquisition),
        workdir=pathlib.Path(config.output.directory) / config.short_hash,
    )

    base_records = gathers[scenarios[0]]
    # One difference per monitor scenario, so a run configured for the
    # pressure-only and saturation-only states gets an image of each rather
    # than only of the combined one. The baseline goes last: the differences
    # are what the run is for.
    to_migrate = {}
    for name in scenarios[1:]:
        to_migrate[f"difference_{name}"] = [
            replace(b, traces=(m.traces.astype(np.float64)
                               - b.traces.astype(np.float64)).astype(b.traces.dtype))
            for b, m in zip(base_records, gathers[name])]
    to_migrate["baseline"] = base_records

    if config.imaging.mute_direct_arrival:
        to_migrate = pipeline._mute_direct(to_migrate, migration,
                                           config.imaging.mute_pad)
    if config.imaging.max_offset is not None:
        to_migrate = pipeline._mute_offsets(to_migrate, config.imaging.max_offset,
                                            config.imaging.offset_taper)
    for note in pipeline.result.notes:
        log(f"  {note}")

    grid = migration.grid
    stacks = {}
    for name, records in to_migrate.items():
        state = out / "mig" / f"{digest}_{name}.npz"
        if state.exists() and not args.fresh:
            saved = np.load(state)
            image, illum = saved["image"], saved["illum"]
            done = int(saved["done"])
            log(f"  resumed {name} at shot {done}/{n_shots}")
        else:
            image = np.zeros(grid.shape)
            illum = np.zeros(grid.shape)
            done = 0
        for ishot in range(done, n_shots):
            migrate_shot(records[ishot], migration, wavelet,
                         acquisition.sources[ishot], solver_settings=settings,
                         rtm=rtm, fmax=pipeline.fmax, f0=config.source.frequency,
                         backend=args.backend, image_out=image, illum_out=illum)
            atomic(state, lambda t: np.savez(t, image=image, illum=illum,
                                             done=ishot + 1))
            if (ishot + 1) % 5 == 0 or ishot + 1 == n_shots:
                log(f"  migrate {name} {ishot + 1}/{n_shots}")
        stacks[name] = (image, illum)
        log(f"migration {name} complete")

    # The image filters are linear, so they go after the stack rather than per
    # shot: identical result, and one pass instead of n_shots of them.
    points = np.concatenate(
        [np.asarray(acquisition.sources, float).reshape(-1, 3)]
        + [np.asarray(r.receiver_positions, float).reshape(-1, 3)
           for r in to_migrate["baseline"]])
    wavelength = float(migration.vp.min()) / config.source.frequency
    images = {}
    for name, (image, illum) in stacks.items():
        result = image
        if rtm.imaging_condition == "source_normalized":
            eps = rtm.epsilon * float(illum.max()) if illum.max() > 0 else 1.0
            result = result / (illum + eps)
        result, _ = filtered_image(result, grid, rtm, points=points,
                                   wavelength=wavelength)
        images[name] = result

    baseline = images["baseline"].astype(float)
    saved = {"baseline": baseline, "origin": np.array(grid.origin),
             "spacing": np.array(grid.spacing)}
    for name in scenarios[1:]:
        fourd = images[f"difference_{name}"].astype(float)
        saved[f"difference_{name}"] = fourd
        saved[f"monitor_{name}"] = baseline + fourd
    destination = out / "images.npz"
    np.savez_compressed(destination, **saved)
    log(f"wrote {destination}  image {baseline.shape} "
        f"spacing {grid.spacing} origin {grid.origin}")

    xs, ys, zs = grid.axis(0), grid.axis(1), grid.axis(2)
    (x0, x1), (y0, y1), (z0, z1) = pipeline.domains.target.bounds
    mask = np.zeros(baseline.shape, bool)
    mask[np.ix_((xs >= x0) & (xs <= x1), (ys >= y0) & (ys <= y1),
                (zs >= z0) & (zs <= z1))] = True
    for name in scenarios[1:]:
        monitor = baseline + images[f"difference_{name}"].astype(float)
        log(f"{name:16s} 4D NRMS {nrms(baseline, monitor):6.2f} % whole volume, "
            f"{nrms(baseline, monitor, mask=mask):6.2f} % in the target window")
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
