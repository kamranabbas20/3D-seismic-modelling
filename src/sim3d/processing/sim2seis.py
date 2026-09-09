"""Simulation-to-seismic: a synthetic 3D volume from the earth model.

This is the *sim2seis* path.  It takes the elastic properties the rock
physics produced for one scenario - Vp, Vs and density on the geological
grid - and turns the whole model into a synthetic seismic volume by
convolving angle-dependent reflectivity down every column.

Where it sits between the other modes
-------------------------------------
``preview``  filters the property cube at normal incidence and exists to
             screen rock-physics cases.
``sparse``   does the same for K chosen columns, for a well tie.
``sim2seis`` is this module: the full cube, in angle stacks, presented as
             the synthetic seismic volume a reservoir study actually works
             with - baseline and monitors, differenced, in time and depth.
``imaging``  is the scientific path: propagate, acquire, migrate.

What the volume contains: angle-dependent normal-and-oblique reflectivity
from the full elastic model, source bandwidth, tuning between closely
spaced interfaces, and the AVO behaviour that separates a fluid change
from a pressure change.

What it does not contain, and cannot: lateral wave propagation,
diffraction, refraction and head waves, transmission loss, geometric
spreading, multiples, illumination, and the migration operator.  Every
trace is built independently of its neighbours, so no energy ever moves
sideways.  Two consequences worth stating plainly: structure steeper than
a few degrees is mispositioned, because nothing here migrates; and the
angle stacks are *constant-angle* stacks, because a 1D column has no
offset axis to map from.

The volume is therefore a model of what the reservoir would look like in
seismic, not a model of what the survey would record.  Everything this
module returns carries :data:`SIM2SEIS_LABEL`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.errors import ConfigError
from ..core.grid import Grid3D
from .preview import convolve_reflectivity, reflectivity, time_from_depth

SIM2SEIS_LABEL = ("Synthetic 1D-Convolution Volume (sim2seis) - "
                  "Not Full 3D Wave Modelling or Migration")

#: The angle stacks a study normally starts from.
DEFAULT_STACKS = (
    {"name": "near", "angles": [0.0, 15.0]},
    {"name": "mid", "angles": [15.0, 30.0]},
    {"name": "far", "angles": [30.0, 45.0]},
)


@dataclass(frozen=True)
class AngleStack:
    """One angle range, averaged into a single stacked volume."""

    name: str
    min_angle: float
    max_angle: float

    def __post_init__(self) -> None:
        if self.min_angle < 0.0 or self.max_angle < self.min_angle:
            raise ConfigError(
                f"stack {self.name!r}: angles must satisfy "
                f"0 <= min <= max, got [{self.min_angle}, {self.max_angle}]")
        if self.max_angle >= 90.0:
            raise ConfigError(
                f"stack {self.name!r}: {self.max_angle} deg is past vertical "
                f"incidence; the linearisation is meaningless beyond about 45 "
                f"and undefined at 90")

    @property
    def centre(self) -> float:
        return 0.5 * (self.min_angle + self.max_angle)

    def sub_angles(self, count: int) -> np.ndarray:
        """The angles averaged to form the stack, in degrees."""
        if self.max_angle == self.min_angle:
            return np.array([self.min_angle])
        return np.linspace(self.min_angle, self.max_angle, max(int(count), 2))

    def describe(self) -> str:
        return f"{self.name} ({self.min_angle:g}-{self.max_angle:g} deg)"


def build_stacks(specs=None) -> tuple[AngleStack, ...]:
    """Turn configuration entries into :class:`AngleStack` objects."""
    entries = list(specs) if specs else [dict(s) for s in DEFAULT_STACKS]
    if not entries:
        raise ConfigError("at least one angle stack is required")
    out, seen = [], set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigError(
                f"stack {i} is {entry!r}; each stack is a mapping with 'name' "
                f"and 'angles': [min, max] in degrees")
        try:
            name = str(entry["name"])
            lo, hi = (float(a) for a in entry["angles"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(
                f"stack {i} is {entry!r}; each stack needs 'name' and "
                f"'angles': [min, max] in degrees ({exc})") from None
        if name in seen:
            raise ConfigError(f"duplicate stack name {name!r}")
        seen.add(name)
        out.append(AngleStack(name=name, min_angle=lo, max_angle=hi))
    return tuple(out)


# ------------------------------------------------------------ reflectivity
def aki_richards(vp, vs, rho, angle_deg: float) -> np.ndarray:
    """Angle-dependent reflectivity between vertical neighbours.

    The three-term Aki-Richards linearisation

    ``R(t) = A + B sin^2(t) + C (tan^2(t) - sin^2(t))``

    with the intercept ``A`` taken as the **exact** impedance contrast
    rather than its linearised form ``(dVp/Vp + drho/rho) / 2``.  The two
    agree to first order in the contrasts, so substituting one for the
    other costs nothing at the order the rest of the expression is valid
    at - and it buys exact agreement with the normal-incidence modes in
    :mod:`sim3d.processing.preview`, so a zero-angle sim2seis stack and a
    convolution preview of the same model are the same numbers rather than
    nearly the same numbers.

    ``vp``, ``vs`` and ``rho`` are ``(..., nz)``; the last axis is depth,
    and the result is one shorter along it, being defined on interfaces.

    Where the shear velocity vanishes on both sides of an interface - a
    fluid layer - the gradient term's ``dVs/Vs`` is undefined; it is taken
    as zero there, which is the correct limit for a medium that carries no
    shear.
    """
    vp = np.asarray(vp, dtype=float)
    vs = np.asarray(vs, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if not (vp.shape == vs.shape == rho.shape):
        raise ConfigError(
            f"Vp, Vs and density must have the same shape; got "
            f"{vp.shape}, {vs.shape} and {rho.shape}")

    theta = np.deg2rad(float(angle_deg))
    d_vp, m_vp = vp[..., 1:] - vp[..., :-1], 0.5 * (vp[..., 1:] + vp[..., :-1])
    d_vs, m_vs = vs[..., 1:] - vs[..., :-1], 0.5 * (vs[..., 1:] + vs[..., :-1])
    d_rho, m_rho = rho[..., 1:] - rho[..., :-1], 0.5 * (rho[..., 1:] + rho[..., :-1])

    intercept = reflectivity(rho * vp)          # exact, see the docstring
    if theta == 0.0:
        return intercept

    ratio = np.divide(m_vs, m_vp, out=np.zeros_like(m_vp), where=m_vp != 0.0)
    vs_term = np.divide(d_vs, m_vs, out=np.zeros_like(m_vs), where=m_vs != 0.0)
    vp_term = np.divide(d_vp, m_vp, out=np.zeros_like(m_vp), where=m_vp != 0.0)
    rho_term = np.divide(d_rho, m_rho, out=np.zeros_like(m_rho), where=m_rho != 0.0)

    gradient = 0.5 * vp_term - 2.0 * ratio**2 * (rho_term + 2.0 * vs_term)
    curvature = 0.5 * vp_term
    s2, t2 = np.sin(theta) ** 2, np.tan(theta) ** 2
    return intercept + gradient * s2 + curvature * (t2 - s2)


def stacked_reflectivity(vp, vs, rho, stack: AngleStack,
                         sub_angles: int = 5) -> np.ndarray:
    """Reflectivity averaged over one angle range.

    Averaging the coefficients before convolution rather than convolving
    each sub-angle and averaging afterwards gives the same answer for a
    linear operator and costs one convolution instead of several.
    """
    angles = stack.sub_angles(sub_angles)
    total = aki_richards(vp, vs, rho, float(angles[0]))
    for angle in angles[1:]:
        total = total + aki_richards(vp, vs, rho, float(angle))
    return total / len(angles)


# ------------------------------------------------------------------ volume
@dataclass
class Sim2SeisVolume:
    """A synthetic seismic volume, one cube per angle stack."""

    stacks: tuple[AngleStack, ...]
    time_cubes: dict[str, np.ndarray]     #: name -> ``(nx, ny, nt)``
    depth_cubes: dict[str, np.ndarray]    #: name -> ``(nx, ny, nz)``
    times: np.ndarray                     #: time axis, seconds
    twt: np.ndarray                       #: ``(nx, ny, nz)`` two-way time
    grid: Grid3D
    scenario: str = ""
    label: str = SIM2SEIS_LABEL
    notes: list[str] = field(default_factory=list)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(stack.name for stack in self.stacks)

    def time_cube(self, name: str) -> np.ndarray:
        try:
            return self.time_cubes[name]
        except KeyError:
            raise KeyError(
                f"no stack named {name!r}; have {list(self.names)}") from None

    def depth_cube(self, name: str) -> np.ndarray:
        try:
            return self.depth_cubes[name]
        except KeyError:
            raise KeyError(
                f"no stack named {name!r}; have {list(self.names)}") from None

    @property
    def megabytes(self) -> float:
        arrays = list(self.time_cubes.values()) + list(self.depth_cubes.values())
        return sum(a.nbytes for a in arrays) / 1e6

    def describe(self) -> str:
        first = next(iter(self.time_cubes.values()))
        dt = (self.times[1] - self.times[0]) * 1e3 if self.times.size > 1 else 0.0
        head = f"{self.label}"
        if self.scenario:
            head += f"\n  scenario {self.scenario}"
        return (f"{head}\n"
                f"  {len(self.stacks)} angle stacks: "
                f"{', '.join(s.describe() for s in self.stacks)}\n"
                f"  cube {first.shape}, {self.times.size} samples at "
                f"{dt:.2f} ms, {self.megabytes:,.0f} MB in memory")


def sim2seis_volume(grid: Grid3D, vp, vs, rho, wavelet, dt: float,
                    stacks=None, t_max: float | None = None,
                    sub_angles: int = 5, map_to_depth: bool = True,
                    scenario: str = "", dtype=np.float32) -> Sim2SeisVolume:
    """Convert one earth model into a synthetic seismic volume.

    Parameters
    ----------
    grid:
        The grid the properties live on.  Two-way time is integrated from
        the top of this grid, so it should span the overburden as well as
        the reservoir for the times to mean anything.
    vp, vs, rho:
        Elastic properties, each ``(nx, ny, nz)``.
    wavelet, dt:
        Source wavelet and its sample interval, seconds.
    stacks:
        :class:`AngleStack` objects, or configuration mappings for
        :func:`build_stacks`.  Defaults to near/mid/far.
    t_max:
        Length of the time axis; defaults to the deepest two-way time.
    sub_angles:
        How many angles are averaged within each stack.
    map_to_depth:
        Also resample each cube onto the depth axis, which is what lets a
        synthetic sit beside the property volume it came from.
    dtype:
        Storage type for the cubes.  Single precision by default: these
        are display and difference products, and a full-model cube in
        double costs twice the memory for digits nothing reads.
    """
    if dt <= 0:
        raise ConfigError(f"dt must be positive, got {dt}")
    vp = np.asarray(vp, dtype=float)
    if vp.shape != grid.shape:
        raise ConfigError(
            f"properties are {vp.shape} but the grid is {grid.shape}")
    angle_stacks = (tuple(stacks) if stacks and isinstance(stacks[0], AngleStack)
                    else build_stacks(stacks))

    twt = time_from_depth(vp, grid.dz)
    t_max = float(t_max if t_max is not None else twt.max())
    nt = int(np.ceil(t_max / dt)) + 1

    time_cubes: dict[str, np.ndarray] = {}
    depth_cubes: dict[str, np.ndarray] = {}
    notes = []
    for stack in angle_stacks:
        rc = stacked_reflectivity(vp, vs, rho, stack, sub_angles=sub_angles)
        traces, times, depth = convolve_reflectivity(
            twt, rc, wavelet, dt, nt, map_to_depth=map_to_depth)
        time_cubes[stack.name] = traces.astype(dtype)
        depth_cubes[stack.name] = (depth.astype(dtype) if depth is not None
                                   else np.zeros(grid.shape, dtype=dtype))
    if angle_stacks[-1].max_angle > 45.0:
        notes.append(
            f"largest stack reaches {angle_stacks[-1].max_angle:g} deg; the "
            f"Aki-Richards linearisation loses accuracy beyond about 45")
    return Sim2SeisVolume(
        stacks=angle_stacks, time_cubes=time_cubes, depth_cubes=depth_cubes,
        times=times, twt=twt, grid=grid, scenario=scenario, notes=notes)
