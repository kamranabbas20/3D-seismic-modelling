r"""3D acoustic reverse time migration (spec sections 73-79).

RTM is the platform's primary imaging method.  The workflow is not
optional and the pipeline never stops at shot gathers: a seismic *image*
is produced by propagating and migrating, never by convolving reflectivity.

Per shot
--------
1. propagate the source wavefield forward through the **migration** model
   and store it, decimated in time;
2. inject the recorded traces at the receivers, time-reversed, and
   propagate that receiver wavefield backwards;
3. correlate the two at each stored time and stack.

Imaging conditions
------------------
``crosscorrelation``
    :math:`I(\mathbf x) = \sum_t S(\mathbf x,t)\,R(\mathbf x,t)`.
``source_normalized``
    :math:`I(\mathbf x) = \dfrac{\sum_t S R}{\sum_t S^2 + \epsilon}`,
    which compensates for geometric spreading and uneven illumination and
    gives an image closer to reflectivity.

Cross-correlation RTM produces well-known low-wavenumber artefacts along
the source and receiver ray paths.  A Laplacian applied to the stacked
image suppresses them; it is on by default and is a *filter of the image*,
never a change to the physics, so it is recorded in the result.

Migration velocity
------------------
The migration model is a separate :class:`~sim3d.wave.acoustic.AcousticModel`
from the true earth model.  Migrating with smoothed, biased or deliberately
wrong velocities is a supported experiment (spec sections 76-78), not a
mistake to be guarded against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from ..core.errors import ConfigError
from ..core.grid import Grid3D
from ..wave.acoustic import AcousticModel, AcousticSolver, ShotRecord, SolverSettings
from ..wave.interp import PointSet
from ..wave.sources import MultiPointSource, PointSource, SourceTerm
from .store import WavefieldStore, safe_decimation

IMAGING_CONDITIONS = ("crosscorrelation", "source_normalized")


@dataclass
class RTMSettings:
    """Imaging controls.  Defaults are safe; the advanced knobs are exposed."""

    imaging_condition: str = "source_normalized"
    #: Time-step stride for the correlation; ``None`` uses the Nyquist-safe value.
    time_decimation: int | None = None
    #: Stabilisation as a fraction of the peak source illumination.
    epsilon: float = 1.0e-4
    #: Apply a Laplacian to the stacked image to suppress low-wavenumber artefacts.
    laplacian_filter: bool = True
    #: Snapshots above this many bytes spill to a memory-mapped file.
    max_ram_bytes: int = 2 * 2**30
    #: Directory for memory-mapped wavefields and per-shot checkpoints.
    workdir: Path | None = None

    def __post_init__(self) -> None:
        if self.imaging_condition not in IMAGING_CONDITIONS:
            raise ConfigError(
                f"unknown imaging condition {self.imaging_condition!r}; "
                f"choose from {IMAGING_CONDITIONS}"
            )
        if self.workdir is not None:
            self.workdir = Path(self.workdir)


@dataclass
class RTMResult:
    """A migrated image plus the diagnostics needed to trust it."""

    image: np.ndarray
    grid: Grid3D
    illumination: np.ndarray
    n_shots: int
    settings: RTMSettings
    dt: float
    time_decimation: int
    notes: list[str] = field(default_factory=list)

    def target_window(self, target: Grid3D) -> np.ndarray:
        """Extract the sub-cube covering ``target`` from the migrated image."""
        idx = [
            np.clip(np.round((target.axis(a) - self.grid.origin[a]) / self.grid.spacing[a]).astype(int),
                    0, self.grid.shape[a] - 1)
            for a in range(3)
        ]
        return self.image[np.ix_(*idx)]

    def describe(self) -> str:
        return (
            f"RTM image {self.image.shape} from {self.n_shots} shot(s), "
            f"condition={self.settings.imaging_condition}, "
            f"time decimation={self.time_decimation} "
            f"(dt_save = {self.dt * self.time_decimation * 1e3:.3f} ms)"
            + ("".join(f"\n  note: {n}" for n in self.notes))
        )


def laplacian(image: np.ndarray, spacing) -> np.ndarray:
    """Second-order 3D Laplacian, used as an RTM artefact filter."""
    d = np.asarray(spacing, dtype=float)
    out = np.zeros_like(image)
    for axis in range(3):
        f = np.moveaxis(image, axis, 0)
        o = np.moveaxis(out, axis, 0)
        o[1:-1] += (f[2:] - 2.0 * f[1:-1] + f[:-2]) / d[axis] ** 2
    return out


def migrate_shot(record: ShotRecord, migration_model: AcousticModel,
                 source_wavelet: np.ndarray, source_position,
                 solver_settings: SolverSettings | None = None,
                 rtm: RTMSettings | None = None,
                 fmax: float | None = None,
                 f0: float = 20.0,
                 backend=None,
                 image_out: np.ndarray | None = None,
                 illum_out: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Migrate one shot gather, returning ``(image, illumination, decimation)``.

    ``image_out`` / ``illum_out`` let a survey loop accumulate directly into
    the stacked arrays instead of allocating a cube per shot.
    """
    rtm = rtm or RTMSettings()
    solver_settings = solver_settings or SolverSettings()
    grid = migration_model.grid
    nt = record.nt

    solver = AcousticSolver(migration_model, solver_settings, f0=f0, backend=backend)
    if abs(solver.dt - record.dt) > 1e-12:
        raise ConfigError(
            f"the migration model gives dt = {solver.dt * 1e3:.5f} ms but the shot "
            f"record was written at {record.dt * 1e3:.5f} ms. Migration and "
            f"modelling must share a time step; set SolverSettings.dt explicitly."
        )

    fmax = float(fmax if fmax is not None else 2.5 * f0)
    dec_limit = safe_decimation(solver.dt, fmax)
    dec = int(rtm.time_decimation) if rtm.time_decimation is not None else dec_limit
    if dec < 1:
        raise ConfigError(f"time_decimation must be >= 1, got {dec}")
    if dec > dec_limit:
        raise ConfigError(
            f"time_decimation={dec} aliases the imaging condition: the "
            f"correlation contains frequencies up to 2*fmax = {2 * fmax:g} Hz, "
            f"so the stride must not exceed {dec_limit} "
            f"(dt_save <= {1 / (4 * fmax) * 1e3:.3f} ms). "
            f"Lower fmax or accept the finer stride - sim3d will not alias the "
            f"image on your behalf."
        )

    saved_steps = list(range(0, nt, dec))
    store = WavefieldStore(grid.shape, len(saved_steps), dtype=solver_settings.dtype,
                           max_ram_bytes=rtm.max_ram_bytes,
                           directory=None if rtm.workdir is None else str(rtm.workdir))
    slot_of_step = {s: i for i, s in enumerate(saved_steps)}

    try:
        # --- 1. source wavefield, forward in time -----------------------
        src = PointSource(grid, source_position, source_wavelet)

        def stash(istep: int, p: np.ndarray) -> None:
            slot = slot_of_step.get(istep)
            if slot is not None:
                store.save(slot, p)

        solver.run(src, nt, receivers=None, on_step=stash)
        store.flush()

        # --- 2. receiver wavefield, backward in time --------------------
        image = np.zeros(grid.shape, dtype=np.float64) if image_out is None else image_out
        illum = np.zeros(grid.shape, dtype=np.float64) if illum_out is None else illum_out
        back = MultiPointSource(grid, record.receiver_positions, record.traces[:, ::-1])

        def correlate(istep: int, p: np.ndarray) -> None:
            forward_step = nt - 1 - istep
            slot = slot_of_step.get(forward_step)
            if slot is not None:
                s = store[slot]
                # Ellipsis indexing writes through the closure variable in
                # place; a bare ``+=`` would rebind it as a local.
                image[...] += s * p
                illum[...] += s * s

        solver.run(back, nt, receivers=None, on_step=correlate)
    finally:
        store.close()

    return image, illum, dec


def migrate_survey(records: Sequence[ShotRecord], migration_model: AcousticModel,
                   source_wavelet: np.ndarray, source_positions: Sequence,
                   solver_settings: SolverSettings | None = None,
                   rtm: RTMSettings | None = None,
                   fmax: float | None = None, f0: float = 20.0,
                   backend=None,
                   progress: Callable[[int, int], None] | None = None) -> RTMResult:
    """Migrate and stack a whole survey.

    Shots are migrated one at a time and accumulated in place, so peak
    memory is set by one wavefield store rather than by the shot count.
    """
    if len(records) != len(source_positions):
        raise ConfigError(
            f"{len(records)} shot records but {len(source_positions)} source positions"
        )
    if not records:
        raise ConfigError("no shot records to migrate")

    rtm = rtm or RTMSettings()
    grid = migration_model.grid
    image = np.zeros(grid.shape, dtype=np.float64)
    illum = np.zeros(grid.shape, dtype=np.float64)
    dec = 1
    for ishot, (rec, pos) in enumerate(zip(records, source_positions)):
        _, _, dec = migrate_shot(
            rec, migration_model, source_wavelet, pos,
            solver_settings=solver_settings, rtm=rtm, fmax=fmax, f0=f0,
            backend=backend, image_out=image, illum_out=illum,
        )
        if progress is not None:
            progress(ishot + 1, len(records))

    notes: list[str] = []
    if rtm.imaging_condition == "source_normalized":
        eps = rtm.epsilon * float(illum.max()) if illum.max() > 0 else 1.0
        image = image / (illum + eps)
        notes.append(
            f"source-illumination normalised with epsilon = {rtm.epsilon:g} "
            f"of peak illumination"
        )
    if rtm.laplacian_filter:
        image = laplacian(image, grid.spacing)
        notes.append("Laplacian filter applied to suppress low-wavenumber artefacts")

    return RTMResult(
        image=image, grid=grid, illumination=illum, n_shots=len(records),
        settings=rtm, dt=float(records[0].dt), time_decimation=dec, notes=notes,
    )
