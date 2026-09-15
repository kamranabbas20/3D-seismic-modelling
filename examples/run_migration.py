#!/usr/bin/env python
"""Run a 4D migration end to end, checkpointed shot by shot.

``sim3d migrate`` already does this, and on a small model it is the right
tool.  This script exists for the runs that take hours: it writes every shot
to disk as soon as it exists, so an interrupted job restarts where it
stopped rather than from the beginning, and it drives the machine rather
than accepting the defaults.

    pip install -e ".[accel,viz]"        # numba for speed, matplotlib for PNGs

    python examples/run_migration.py <config> --check     # QC and cost only
    python examples/run_migration.py examples/configs/five_layer_600m_dense.yaml \
        --out runs/dense --threads 32

Run ``--check`` first on an unfamiliar machine: it benchmarks this host and
reports the wall time for the whole experiment before anything is
propagated.

Into ``--out`` it writes

    images.npz            baseline, and a difference and monitor per scenario,
                          with the grid origin and spacing
    section_<scenario>.png   baseline, monitor and 4D difference, as a depth
                          section through the middle of the target
    map_<scenario>.png    the same three in map view, sliced at the depth
                          where the 4D difference is strongest
    illumination.png      what the survey actually lit, which after an
                          aperture cut is not a detail
    fwd/, mig/            per-shot checkpoints; delete them to start over,
                          or pass --fresh

Figures need matplotlib and nothing else - no browser, no display, no X
server. ``--no-figures`` skips them.

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
                   help="Numba threads; defaults to $SLURM_CPUS_PER_TASK when a "
                        "scheduler set it, otherwise every core the machine reports")
    p.add_argument("--check", action="store_true",
                   help="run QC and the cost estimate, then stop")
    p.add_argument("--fresh", action="store_true",
                   help="ignore any existing checkpoints and start over")
    p.add_argument("--force", action="store_true",
                   help="proceed even if QC fails")
    p.add_argument("--migrate", default="both",
                   choices=("difference", "baseline", "both"),
                   help="which images to migrate. 'difference' is the 4D and is "
                        "what most runs want; 'baseline' adds the structural "
                        "image at roughly double the cost")
    p.add_argument("--no-figures", action="store_true",
                   help="write the .npz only, skipping the PNGs")
    p.add_argument("--dpi", type=int, default=140, help="figure resolution")
    return p.parse_args(argv)


def _colormaps():
    """Matplotlib versions of the palettes the GUI uses.

    Seismic amplitude is signed, so it gets a diverging ramp with a neutral
    midpoint and limits symmetric about zero - otherwise the colour of a
    sample stops meaning its polarity, which is the only thing the reader is
    looking at. Illumination is a magnitude and gets a single hue, light to
    dark. Neither is a rainbow: on a rainbow ramp the eye reads the yellow
    band as a feature of the data rather than of the colour map.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from sim3d.ui import theme

    def build(stops, name):
        return LinearSegmentedColormap.from_list(
            name, [(position, colour) for position, colour in stops])

    return build(theme.DIVERGING, "sim3d_seismic"), build(theme.SEQUENTIAL,
                                                          "sim3d_illumination")


def _section(ax, data, x, z, cmap, limit, title, horizons=(), wells=()):
    """One depth section, drawn the way a seismic section is read."""
    ax.imshow(data.T, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto",
              origin="upper", interpolation="bilinear",
              extent=[x[0], x[-1], z[-1], z[0]])
    for depth in horizons:
        ax.axhline(depth, color="#52514e", lw=0.8, ls=(0, (4, 3)), alpha=0.7)
    for position, label in wells:
        ax.axvline(position, color="#0b0b0b", lw=0.8, ls=(0, (1, 2)), alpha=0.8)
        ax.annotate(label, (position, z[0]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#52514e")
    ax.set_title(title, fontsize=10, color="#0b0b0b", pad=8)
    ax.set_xlabel("x (m)", fontsize=9)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=8)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Thread count has to be set before Numba is imported, which means before
    # sim3d is, because importing it builds the backend registry.
    threads = args.threads or os.environ.get("SLURM_CPUS_PER_TASK")
    if threads:
        os.environ["NUMBA_NUM_THREADS"] = str(int(threads))
    # Assigned, not defaulted. The parallelism here is Numba's `prange` over
    # the outer grid axis; BLAS is incidental. Both pools size themselves to
    # the core count and then fight over it - two Numba jobs on four cores
    # measured a 13x slowdown on this project, and the pools do the same to
    # each other. `setdefault` was wrong under a scheduler: Slurm exports
    # OMP_NUM_THREADS from --cpus-per-task, so the value was already set and
    # the guard did nothing on exactly the machines that needed it.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    # Some clusters mount $HOME read-only on compute nodes, where matplotlib
    # fails on its font cache rather than on anything to do with the run.
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(args.out, ".mplcache"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

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
    if args.migrate in ("difference", "both"):
        for name in scenarios[1:]:
            to_migrate[f"difference_{name}"] = [
                replace(b, traces=(m.traces.astype(np.float64)
                                   - b.traces.astype(np.float64)).astype(b.traces.dtype))
                for b, m in zip(base_records, gathers[name])]
    if args.migrate in ("baseline", "both"):
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
    # Taper geometry comes from the acquisition, not from whichever gather
    # happens to have been migrated: --migrate difference leaves no baseline
    # entry, and the taper must not change with that choice.
    points = np.concatenate(
        [np.asarray(acquisition.sources, float).reshape(-1, 3),
         np.asarray(acquisition.receivers, float).reshape(-1, 3)])
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

    saved = {"origin": np.array(grid.origin), "spacing": np.array(grid.spacing)}
    baseline = images["baseline"].astype(float) if "baseline" in images else None
    if baseline is not None:
        saved["baseline"] = baseline
    for name in scenarios[1:]:
        key = f"difference_{name}"
        if key not in images:
            continue
        fourd = images[key].astype(float)
        saved[key] = fourd
        # A monitor image only exists where the baseline was migrated too:
        # the monitor is the baseline plus the difference, and inventing one
        # from a difference alone would be a picture of nothing.
        if baseline is not None:
            saved[f"monitor_{name}"] = baseline + fourd
    destination = out / "images.npz"
    np.savez_compressed(destination, **saved)
    shape = next(v for k, v in saved.items()
                 if k not in ("origin", "spacing")).shape
    log(f"wrote {destination}  {sorted(k for k in saved if k not in ('origin', 'spacing'))} "
        f"  image {shape} spacing {grid.spacing} origin {grid.origin}")

    xs, ys, zs = grid.axis(0), grid.axis(1), grid.axis(2)
    (x0, x1), (y0, y1), (z0, z1) = pipeline.domains.target.bounds
    mask = np.zeros(grid.shape, bool)
    mask[np.ix_((xs >= x0) & (xs <= x1), (ys >= y0) & (ys <= y1),
                (zs >= z0) & (zs <= z1))] = True
    for name in scenarios[1:]:
        key = f"difference_{name}"
        if key not in images:
            continue
        fourd = images[key].astype(float)
        if baseline is not None:
            monitor = baseline + fourd
            log(f"{name:16s} 4D NRMS {nrms(baseline, monitor):6.2f} % whole volume, "
                f"{nrms(baseline, monitor, mask=mask):6.2f} % in the target window")
        else:
            # NRMS is normalised by the baseline image, so without one the
            # honest summary is the 4D energy in the target window against
            # the 4D energy outside it - a ratio of the difference to itself,
            # which needs no baseline and says the same thing about focus.
            inside = float(np.sqrt(np.mean(fourd[mask] ** 2)))
            outside = float(np.sqrt(np.mean(fourd[~mask] ** 2)))
            log(f"{name:16s} 4D RMS {inside:.4e} in the target window, "
                f"{outside:.4e} outside it, ratio {inside / outside:.1f}")

    if not args.no_figures:
        render(out, images, stacks, grid, pipeline, config, scenarios, args, log)
    log("DONE")
    return 0


def render(out, images, stacks, grid, pipeline, config, scenarios, args, log):
    """Write the PNGs: a section per scenario, a map, and the illumination."""
    import matplotlib
    matplotlib.use("Agg")                 # no display, no browser, no X server
    import matplotlib.pyplot as plt
    import numpy as np

    seismic, sequential = _colormaps()
    xs, ys, zs = grid.axis(0), grid.axis(1), grid.axis(2)
    (x0, x1), (y0, y1), (z0, z1) = pipeline.domains.target.bounds
    iy = int(np.argmin(np.abs(ys - 0.5 * (y0 + y1))))
    # Plot a little above and below the target rather than the whole domain:
    # the top of the model is acquisition depth and carries no geology.
    band = (zs >= max(zs[0], z0 - 450.0)) & (zs <= min(zs[-1], z1 + 250.0))
    z = zs[band]
    horizons = (z0, z1)
    # From the pipeline, not the config: a configuration may name its wells
    # explicitly or give a pattern for them, and only the pipeline resolves
    # both to the same thing.
    wells = [(float(w.x), float(w.y), str(w.name)) for w in pipeline.wells()]

    have_baseline = "baseline" in images
    baseline = (images["baseline"].astype(float)[:, iy, :][:, band]
                if have_baseline else None)
    limit = float(np.abs(baseline).max()) or 1.0 if have_baseline else 1.0

    for name in scenarios[1:]:
        if f"difference_{name}" not in images:
            continue
        fourd = images[f"difference_{name}"].astype(float)[:, iy, :][:, band]
        # Baseline and monitor share one scale or the eye reads the rescaling
        # as a change in the earth. The difference gets its own: it is a
        # different quantity, and on the shared scale it would be invisible.
        if have_baseline:
            panels = ((baseline, limit, "baseline"),
                      (baseline + fourd, limit, f"monitor - {name}"),
                      (fourd, float(np.abs(fourd).max()) or 1.0, "4D difference"))
        else:
            panels = ((fourd, float(np.abs(fourd).max()) or 1.0, "4D difference"),)
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.2),
                                 sharey=True, squeeze=False, facecolor="#fcfcfb")
        axes = axes[0]
        for ax, (data, scale, title) in zip(axes, panels):
            _section(ax, data, xs, z, seismic, scale, title, horizons,
                     [(x, name) for x, _, name in wells])
        axes[0].set_ylabel("depth (m)", fontsize=9)
        fig.suptitle(f"{config.project.name} - migrated section at y = "
                     f"{ys[iy]:,.0f} m", fontsize=12, color="#0b0b0b")
        fig.text(0.5, 0.015, "dashed horizons: top and base of the target window"
                 "   ·   amplitude scales are symmetric about zero, and the "
                 "baseline and monitor share theirs",
                 ha="center", fontsize=8, color="#898781")
        fig.tight_layout(rect=(0, 0.04, 1, 0.94))
        path = out / f"section_{name}.png"
        fig.savefig(path, dpi=args.dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        log(f"  wrote {path}")

        # Map view at the depth where the 4D is strongest: a section can walk
        # past an anomaly that sits off the line it was cut on.
        volume = images[f"difference_{name}"].astype(float)
        profile = np.sqrt(np.mean(volume ** 2, axis=(0, 1)))
        k = int(np.argmax(profile))
        maps = ([(images["baseline"].astype(float)[:, :, k], "baseline")]
                if have_baseline else [])
        maps.append((volume[:, :, k], "4D difference"))
        fig, axes = plt.subplots(1, len(maps), figsize=(5.5 * len(maps), 4.8),
                                 squeeze=False, facecolor="#fcfcfb")
        axes = axes[0]
        for ax, (data, title) in zip(axes, maps):
            scale = float(np.abs(data).max()) or 1.0
            ax.imshow(data.T, cmap=seismic, vmin=-scale, vmax=scale,
                      aspect="equal", origin="lower",
                      extent=[xs[0], xs[-1], ys[0], ys[-1]],
                      interpolation="bilinear")
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor="#52514e", lw=0.8,
                                       ls=(0, (4, 3))))
            for well_x, well_y, label in wells:
                ax.plot(well_x, well_y, "o", ms=5, mfc="none", mec="#0b0b0b",
                        mew=1.2)
                ax.annotate(label, (well_x, well_y), xytext=(6, 4),
                            textcoords="offset points", fontsize=8,
                            color="#0b0b0b")
            ax.set_title(title, fontsize=10, pad=8)
            ax.set_xlabel("x (m)", fontsize=9)
            for spine in ax.spines.values():
                spine.set_color("#c3c2b7")
            ax.tick_params(colors="#52514e", labelsize=8)
        axes[0].set_ylabel("y (m)", fontsize=9)
        fig.suptitle(f"{config.project.name} - depth slice at {zs[k]:,.0f} m, "
                     f"where the 4D difference peaks", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        path = out / f"map_{name}.png"
        fig.savefig(path, dpi=args.dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        log(f"  wrote {path}")

    # Illumination says which parts of the image the survey actually lit, and
    # after an aperture cut that is not a detail: an edge the operator never
    # reached images as an absence, not as a reservoir that is not there.
    illum = next(iter(stacks.values()))[1][:, iy, :][:, band]
    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor="#fcfcfb")
    image = ax.imshow(illum.T, cmap=sequential, aspect="auto", origin="upper",
                      extent=[xs[0], xs[-1], z[-1], z[0]],
                      interpolation="bilinear")
    fig.colorbar(image, ax=ax, label="source illumination")
    for depth in horizons:
        ax.axhline(depth, color="#52514e", lw=0.8, ls=(0, (4, 3)), alpha=0.7)
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("depth (m)", fontsize=9)
    ax.set_title(f"{config.project.name} - source illumination at "
                 f"y = {ys[iy]:,.0f} m", fontsize=11, pad=8)
    for spine in ax.spines.values():
        spine.set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=8)
    fig.tight_layout()
    path = out / "illumination.png"
    fig.savefig(path, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    log(f"  wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
