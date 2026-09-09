"""Ready-made geological templates (spec section 148).

Each template returns a list of :class:`~sim3d.geology.builder.Layer` and a
:class:`~sim3d.geology.faults.FaultSet` for a domain of the given extent,
so a study can start from a recognisable structure and vary it, rather than
assembling horizons by hand every time.
"""

from __future__ import annotations

from typing import Callable

from ..core.errors import ConfigError
from .builder import Layer
from .faults import Fault, FaultSet
from .heterogeneity import HeterogeneitySpec
from .surfaces import Anticline, Dipping, Flat, Relief, Syncline


def _overburden(z_reservoir: float) -> list[Layer]:
    """The three units above the reservoir, common to every template."""
    return [
        Layer("upper_overburden", Flat(0.0), "shale", porosity=0.22, vsh=0.85),
        Layer("shaly_interval", Flat(0.45 * z_reservoir), "sandy_shale",
              porosity=0.18, vsh=0.55),
        Layer("seal", Flat(z_reservoir - 60.0), "shale", porosity=0.12, vsh=0.90),
    ]


def _underburden(z_base: float) -> list[Layer]:
    return [
        Layer("underburden", Flat(z_base), "sandy_shale", porosity=0.12, vsh=0.60),
        Layer("deep_high_velocity", Flat(z_base + 380.0), "carbonate",
              porosity=0.06, vsh=0.05, is_reservoir=False),
    ]


def _reservoir(top, gross: float, seed: int, n_sublayers: int = 8) -> Layer:
    return Layer(
        "reservoir", top, "clean_sandstone",
        porosity=0.26, vsh=0.10, ntg=0.85,
        porosity_heterogeneity=HeterogeneitySpec(
            std=0.030, correlation_major=650.0, correlation_minor=380.0, correlation_vertical=18.0,
            azimuth=30.0, model="exponential", seed=seed),
        vsh_heterogeneity=HeterogeneitySpec(
            std=0.05, correlation_major=650.0, correlation_minor=380.0, correlation_vertical=18.0,
            azimuth=30.0, model="exponential", seed=seed + 1),
        n_sublayers=n_sublayers, sublayer_porosity_range=0.035,
    )


def flat_reservoir(extent=(3000.0, 3000.0, 2200.0), z_reservoir=1200.0,
                   gross=150.0, seed=1):
    """Template A: horizontal layering with a tabular reservoir."""
    layers = _overburden(z_reservoir)
    layers.append(_reservoir(Flat(z_reservoir), gross, seed))
    layers += _underburden(z_reservoir + gross)
    return layers, FaultSet([])


def dipping_reservoir(extent=(3000.0, 3000.0, 2200.0), z_reservoir=1200.0,
                      gross=150.0, dip=4.0, seed=1):
    """Template B: regional dip carried by the whole stratigraphy.

    The overburden dips with the reservoir. Dipping only the reservoir top
    would drive it up through a flat seal on the shallow side, which is a
    crossing horizon, not a structure.
    """
    tilt = Relief(Dipping(0.0, dip=dip, azimuth=90.0,
                          origin=(extent[0] / 2, extent[1] / 2)))
    layers = [
        Layer("upper_overburden", Flat(0.0), "shale", porosity=0.22, vsh=0.85),
        Layer("shaly_interval", Flat(0.45 * z_reservoir) + tilt, "sandy_shale",
              porosity=0.18, vsh=0.55),
        Layer("seal", Flat(z_reservoir - 60.0) + tilt, "shale",
              porosity=0.12, vsh=0.90),
        _reservoir(Flat(z_reservoir) + tilt, gross, seed),
        Layer("underburden", Flat(z_reservoir + gross) + tilt, "sandy_shale",
              porosity=0.12, vsh=0.60),
        Layer("deep_high_velocity", Flat(z_reservoir + gross + 380.0), "carbonate",
              porosity=0.06, vsh=0.05, is_reservoir=False),
    ]
    return layers, FaultSet([])


def anticline_reservoir(extent=(3000.0, 3000.0, 2200.0), z_reservoir=1200.0,
                        gross=150.0, amplitude=90.0, seed=1):
    """Template C: a gentle four-way dip closure."""
    centre = (extent[0] / 2, extent[1] / 2)
    fold = Relief(Anticline(0.0, amplitude=amplitude, centre=centre,
                            radius=(0.32 * extent[0], 0.45 * extent[1]), azimuth=30.0))
    layers = [
        Layer("upper_overburden", Flat(0.0), "shale", porosity=0.22, vsh=0.85),
        Layer("shaly_interval", Flat(0.45 * z_reservoir) + fold, "sandy_shale",
              porosity=0.18, vsh=0.55),
        Layer("seal", Flat(z_reservoir - 60.0) + fold, "shale", porosity=0.12, vsh=0.90),
        _reservoir(Flat(z_reservoir) + fold, gross, seed),
    ]
    layers += [
        Layer("underburden", Flat(z_reservoir + gross) + fold, "sandy_shale",
              porosity=0.12, vsh=0.60),
        Layer("deep_high_velocity", Flat(z_reservoir + gross + 380.0), "carbonate",
              porosity=0.06, vsh=0.05, is_reservoir=False),
    ]
    return layers, FaultSet([])


def fault_compartment(extent=(3000.0, 3000.0, 2200.0), z_reservoir=1200.0,
                      gross=150.0, throw=40.0, transmissibility=0.0, seed=1):
    """Template D: an anticline cut by one normal fault that seals."""
    layers, _ = anticline_reservoir(extent, z_reservoir, gross, seed=seed)
    fault = Fault(
        name="F1", origin=(extent[0] / 2, extent[1] / 2, z_reservoir),
        strike=20.0, dip=68.0, throw=throw,
        strike_extent=0.7 * extent[1], dip_extent=900.0,
        zone_width=40.0, transmissibility=transmissibility,
    )
    return layers, FaultSet([fault])


def channel_reservoir(extent=(3000.0, 3000.0, 2200.0), z_reservoir=1200.0,
                      gross=90.0, seed=1):
    """Template E: a narrow, strongly anisotropic reservoir body."""
    layers = _overburden(z_reservoir)
    channel = _reservoir(Flat(z_reservoir), gross, seed, n_sublayers=3)
    channel.name = "channel"
    channel.porosity_heterogeneity = HeterogeneitySpec(
        std=0.035, correlation_major=1400.0, correlation_minor=140.0, correlation_vertical=12.0,
        azimuth=15.0, model="exponential", seed=seed)
    layers.append(channel)
    layers += _underburden(z_reservoir + gross)
    return layers, FaultSet([])


def lens_reservoir(extent=(3000.0, 3000.0, 2200.0), z_reservoir=1200.0,
                   gross=110.0, seed=1):
    """Template F: a lens that thins to nothing at its edges."""
    centre = (extent[0] / 2, extent[1] / 2)
    layers = _overburden(z_reservoir)
    layers.append(_reservoir(Flat(z_reservoir), gross, seed, n_sublayers=4))
    layers.append(Layer(
        "underburden",
        Flat(z_reservoir + gross) + Relief(
            Syncline(0.0, amplitude=gross, centre=centre,
                     radius=(0.30 * extent[0], 0.30 * extent[1]))),
        "sandy_shale", porosity=0.12, vsh=0.60))
    layers.append(Layer("deep_high_velocity", Flat(z_reservoir + gross + 380.0),
                        "carbonate", porosity=0.06, vsh=0.05, is_reservoir=False))
    return layers, FaultSet([])


def stacked_sands(extent=(3000.0, 3000.0, 2200.0), z_reservoir=1200.0,
                  gross=60.0, n_units=3, separation=90.0, seed=1):
    """Template G: several reservoir units separated by shale."""
    layers = _overburden(z_reservoir)
    z = z_reservoir
    for unit in range(n_units):
        sand = _reservoir(Flat(z), gross, seed + 10 * unit, n_sublayers=4)
        sand.name = f"reservoir_{unit + 1}"
        layers.append(sand)
        z += gross
        if unit < n_units - 1:
            layers.append(Layer(f"interbed_{unit + 1}", Flat(z), "shale",
                                porosity=0.13, vsh=0.88))
            z += separation
    layers += _underburden(z)
    return layers, FaultSet([])


#: Template name -> builder.
TEMPLATES: dict[str, Callable] = {
    "flat": flat_reservoir,
    "dipping": dipping_reservoir,
    "anticline": anticline_reservoir,
    "fault_compartment": fault_compartment,
    "channel": channel_reservoir,
    "lens": lens_reservoir,
    "stacked_sands": stacked_sands,
}


def template(name: str, **kwargs):
    """Build a named template; returns ``(layers, fault_set)``."""
    try:
        builder = TEMPLATES[name]
    except KeyError:
        raise ConfigError(
            f"unknown geological template {name!r}; choose from {sorted(TEMPLATES)}"
        ) from None
    return builder(**kwargs)
