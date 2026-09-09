"""The rock-physics chain, with every intermediate preserved (spec section 40).

``minerals + porosity -> dry frame -> fluids -> mixing -> Gassmann -> Vp, Vs, rho``

:class:`RockPhysicsResult` keeps each stage, so a study can ask *why* a
velocity changed - a softer frame, a lighter fluid, a lower fluid modulus -
rather than only observing that it did.  That is what makes the
pressure/saturation decomposition of spec sections 48-55 interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.errors import ConfigError, ValidationError
from .dryframe import (
    DEFAULT_COORDINATION, DEFAULT_CRITICAL_POROSITY, critical_porosity_model,
    soft_sand, stiff_sand,
)
from .fluids import FluidState, brine_properties, check_validity, gas_properties, mix_fluids, oil_properties
from .gassmann import gassmann_saturated_modulus, velocities
from .minerals import mixed_mineral
from .pressure import PressureModel, effective_stress


@dataclass
class RockPhysicsConfig:
    """Which models to use, and their parameters.  Saved with every experiment."""

    mineral_model: str = "vrh"
    dry_frame_model: str = "soft_sand"
    fluid_mixing: str = "wood"
    brie_exponent: float = 3.0
    critical_porosity: float = DEFAULT_CRITICAL_POROSITY
    coordination: float = DEFAULT_COORDINATION
    pressure: PressureModel = field(default_factory=PressureModel)
    #: Fluid-property inputs, all SI except temperature (degC) and API.
    temperature: float = 70.0
    salinity: float = 35000.0
    api: float = 30.0
    gas_gravity: float = 0.65
    gor: float = 100.0
    #: Floor on effective stress, Pa.  At the free surface the confining and
    #: pore pressures are both atmospheric, so the true effective stress is
    #: zero and every grain-contact frame model degenerates there.  Cells
    #: below this floor are raised to it and *counted in the warnings*, so
    #: the shallow section is never quietly given a stiffness it has not
    #: earned.  Set to 0 to disable and let such a model raise instead.
    min_effective_pressure: float = 1.0e6

    def describe(self) -> str:
        return "\n".join([
            "Rock-physics configuration:",
            f"  mineral mixing   {self.mineral_model}",
            f"  dry frame        {self.dry_frame_model} "
            f"(phi_c = {self.critical_porosity:g}, n = {self.coordination:g})",
            f"  fluid mixing     {self.fluid_mixing}"
            + (f" (Brie e = {self.brie_exponent:g})" if self.fluid_mixing == "brie" else ""),
            f"  fluid properties Batzle-Wang; T = {self.temperature:g} degC, "
            f"salinity = {self.salinity:g} ppm, API = {self.api:g}, "
            f"gas gravity = {self.gas_gravity:g}, GOR = {self.gor:g} L/L",
            f"  {self.pressure.describe()}",
            f"  min P_eff        {self.min_effective_pressure / 1e6:g} MPa "
            f"(floor for the shallow section)",
            "  saturation:      Gassmann fluid substitution (low-frequency limit)",
        ])


@dataclass
class RockPhysicsResult:
    """Every stage of the chain, on the same grid as the inputs."""

    k_mineral: np.ndarray
    mu_mineral: np.ndarray
    rho_mineral: np.ndarray
    effective_pressure: np.ndarray
    k_dry: np.ndarray
    mu_dry: np.ndarray
    fluid: FluidState
    phases: dict[str, FluidState]
    k_sat: np.ndarray
    mu_sat: np.ndarray
    rho: np.ndarray
    vp: np.ndarray
    vs: np.ndarray
    config: RockPhysicsConfig
    warnings: list[str] = field(default_factory=list)

    @property
    def ai(self) -> np.ndarray:
        """Acoustic impedance ``rho * Vp``, kg/m^2/s."""
        return self.rho * self.vp

    @property
    def si(self) -> np.ndarray:
        """Shear impedance ``rho * Vs``, kg/m^2/s."""
        return self.rho * self.vs

    @property
    def vp_vs(self) -> np.ndarray:
        return self.vp / self.vs

    @property
    def poisson(self) -> np.ndarray:
        r = self.vp_vs**2
        return (r - 2.0) / (2.0 * (r - 1.0))

    def summary(self) -> str:
        def rng(label, arr, scale=1.0, unit=""):
            a = np.asarray(arr, dtype=float)
            return f"  {label:16s} {np.min(a) / scale:10.3f} - {np.max(a) / scale:10.3f} {unit}"
        lines = ["Rock-physics result:",
                 rng("K mineral", self.k_mineral, 1e9, "GPa"),
                 rng("K dry", self.k_dry, 1e9, "GPa"),
                 rng("mu dry", self.mu_dry, 1e9, "GPa"),
                 rng("P effective", self.effective_pressure, 1e6, "MPa"),
                 rng("K fluid", self.fluid.k, 1e9, "GPa"),
                 rng("rho fluid", self.fluid.rho, 1.0, "kg/m3"),
                 rng("K saturated", self.k_sat, 1e9, "GPa"),
                 rng("rho", self.rho, 1.0, "kg/m3"),
                 rng("Vp", self.vp, 1.0, "m/s"),
                 rng("Vs", self.vs, 1.0, "m/s"),
                 rng("AI", self.ai, 1e6, "1e6 kg/m2/s")]
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def elastic_from_state(porosity, composition: dict[str, np.ndarray],
                       saturations: dict[str, np.ndarray],
                       pore_pressure, confining_pressure,
                       config: RockPhysicsConfig | None = None) -> RockPhysicsResult:
    """Run the full chain from reservoir state to elastic properties.

    Parameters
    ----------
    porosity:
        Fraction, strictly between 0 and 1.
    composition:
        Mineral name -> volume fraction of the solid phase; must sum to 1.
    saturations:
        Phase name -> saturation; must sum to 1.  Recognised phase names are
        ``brine``, ``oil`` and ``gas``, whose properties come from
        Batzle-Wang; any other name must be supplied as a
        :class:`~sim3d.rockphysics.fluids.FluidState` in ``saturations``
        is not supported, so use one of the three.
    pore_pressure, confining_pressure:
        In Pa.  Their difference drives the frame through the configured
        pressure model.

    Returns
    -------
    RockPhysicsResult
        Carrying every intermediate, not only Vp/Vs/rho.
    """
    config = config or RockPhysicsConfig()
    phi = np.asarray(porosity, dtype=float)
    if np.any(phi <= 0) or np.any(phi >= 1):
        raise ValidationError(
            f"porosity must be strictly within (0, 1); it spans "
            f"[{np.min(phi):g}, {np.max(phi):g}]"
        )

    k_min, mu_min, rho_min = mixed_mineral(composition, config.mineral_model)
    k_min, mu_min, rho_min = (np.broadcast_to(np.asarray(v, dtype=float), phi.shape).copy()
                              for v in (k_min, mu_min, rho_min))

    p_eff = effective_stress(confining_pressure, pore_pressure, config.pressure.biot)
    p_eff = np.broadcast_to(np.asarray(p_eff, dtype=float), phi.shape).copy()
    # A *negative* effective stress is always an error: the pore pressure has
    # passed the confining stress. Exactly zero is what the free surface gives -
    # both pressures are atmospheric there - so it is handled by the shallow
    # floor below rather than treated as a failure.
    if np.any(p_eff < 0):
        raise ValidationError(
            f"effective stress is negative somewhere (minimum "
            f"{np.min(p_eff) / 1e6:.3f} MPa), so the pore pressure has exceeded the "
            f"confining stress. That is fracture territory, outside every frame "
            f"model here."
        )
    floor = float(config.min_effective_pressure)
    clamp_notes: list[str] = []
    below = p_eff < floor
    if floor > 0 and np.any(below):
        clamp_notes.append(
            f"{int(below.sum()):,} of {p_eff.size:,} cells "
            f"({100 * below.mean():.2f}%) had an effective stress below the "
            f"{floor / 1e6:g} MPa floor (minimum {np.min(p_eff) / 1e6:.3f} MPa) and "
            f"were raised to it; this is the shallow section, where confining and "
            f"pore pressure both approach atmospheric and grain-contact frame "
            f"models degenerate"
        )
        p_eff = np.maximum(p_eff, floor)

    frame_p = config.pressure.frame_pressure(p_eff)
    if config.dry_frame_model == "soft_sand":
        k_dry, mu_dry = soft_sand(k_min, mu_min, phi, frame_p,
                                  config.critical_porosity, config.coordination)
    elif config.dry_frame_model == "stiff_sand":
        k_dry, mu_dry = stiff_sand(k_min, mu_min, phi, frame_p,
                                   config.critical_porosity, config.coordination)
    elif config.dry_frame_model == "critical_porosity":
        k_dry, mu_dry = critical_porosity_model(k_min, mu_min, phi,
                                                config.critical_porosity)
    else:
        raise ConfigError(
            f"unknown dry-frame model {config.dry_frame_model!r}; choose from "
            f"soft_sand, stiff_sand, critical_porosity"
        )

    f_k, f_mu = config.pressure.multipliers(p_eff)
    k_dry = k_dry * f_k
    mu_dry = mu_dry * f_mu

    warnings = clamp_notes + check_validity(pore_pressure, config.temperature,
                                            config.salinity)
    builders = {
        "brine": lambda: brine_properties(pore_pressure, config.temperature, config.salinity),
        "oil": lambda: oil_properties(pore_pressure, config.temperature, config.api,
                                      config.gas_gravity, config.gor),
        "gas": lambda: gas_properties(pore_pressure, config.temperature, config.gas_gravity),
    }
    unknown = set(saturations) - set(builders)
    if unknown:
        raise ConfigError(
            f"unknown fluid phase(s) {sorted(unknown)}; supported: {sorted(builders)}"
        )
    phases = {name: builders[name]() for name in saturations}
    fluid = mix_fluids(phases, saturations, config.fluid_mixing, config.brie_exponent)

    k_sat = gassmann_saturated_modulus(k_dry, mu_dry, k_min, fluid.k, phi)
    mu_sat = mu_dry  # Gassmann: the pore fluid does not change the shear modulus
    rho = (1.0 - phi) * rho_min + phi * np.asarray(fluid.rho, dtype=float)
    vp, vs = velocities(k_sat, mu_sat, rho)

    return RockPhysicsResult(
        k_mineral=k_min, mu_mineral=mu_min, rho_mineral=rho_min,
        effective_pressure=p_eff, k_dry=k_dry, mu_dry=mu_dry,
        fluid=fluid, phases=phases, k_sat=k_sat, mu_sat=mu_sat, rho=rho,
        vp=vp, vs=vs, config=config, warnings=warnings,
    )
