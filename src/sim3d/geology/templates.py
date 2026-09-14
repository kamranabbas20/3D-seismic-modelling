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


def five_layer(extent=(2000.0, 2000.0, 1800.0), overburden_shale=500.0,
               upper_sand=500.0, seal=50.0, gross=30.0, base_shale=50.0,
               seed=1, heterogeneous=True):
    """Template I: a reflective overburden above a thin 4D target.

    Shale, sand, shale, sand, shale - and only the fourth unit, the thin
    one, is a reservoir.  The 500 m sand above it is thick, bright and
    completely inert: nothing is produced from it, nothing is injected into
    it, and its saturations never move.

    That inertness is the point.  ``three_layer`` puts the target under a
    featureless halfspace, so the only reflectors in the model are the
    target's own and anything a migration draws above it is artefact with
    nothing to compete against - which is what made the acquisition
    near-field and the migration smiles the dominant features of that
    image.  Here the interval between the acquisition and the target
    carries two strong interfaces that do not change between surveys, so
    the image has real structure to focus on and the 4D difference has
    somewhere to be quiet.  A monitor that reproduces those reflectors and
    differences to zero on them is the check that the modelling and the
    imaging are repeatable; a 4D anomaly on the inert sand is a bug.

    The target is deliberately thinner than the seismic can resolve - 30 m
    against a quarter-wavelength of about 50 m at these velocities - so its
    top and base interfere into one composite event and the 4D shows as an
    amplitude change rather than as two separable interfaces.  That is how
    a thin reservoir behaves in real 4D, not a limitation of the model.
    """
    z_upper = float(overburden_shale)
    z_seal = z_upper + float(upper_sand)
    z_reservoir = z_seal + float(seal)
    z_base = z_reservoir + float(gross)

    sand = _reservoir(Flat(z_reservoir), gross, seed, n_sublayers=3)
    if not heterogeneous:
        sand.porosity_heterogeneity = None
        sand.vsh_heterogeneity = None
        sand.n_sublayers = 1

    return [
        Layer("overburden_shale", Flat(0.0), "shale", porosity=0.20, vsh=0.85),
        # Marked explicitly: the facies is a clean sandstone and would be
        # treated as a reservoir by default, which would put the flow
        # simulation - and the 4D - in the wrong unit.
        Layer("upper_sand", Flat(z_upper), "clean_sandstone",
              porosity=0.24, vsh=0.12, ntg=0.90, is_reservoir=False),
        Layer("seal", Flat(z_seal), "shale", porosity=0.12, vsh=0.90),
        sand,
        Layer("base_shale", Flat(z_base), "shale", porosity=0.12, vsh=0.90),
    ], FaultSet([])


def three_layer(extent=(3000.0, 3000.0, 2200.0), z_reservoir=1200.0,
                gross=150.0, seed=1, heterogeneous=True, dip=0.0,
                azimuth=90.0):
    """Template H: shale, sand, shale, and nothing else.

    The textbook 4D case.  Every other template carries a layered
    overburden so that a migrated image has something to focus on above the
    reservoir; this one carries exactly three units, so the only reflectors
    in the model are the top and base of the sand and the 4D signal has
    nowhere to hide.  That makes it the right model for comparing what two
    seismic modes do with the *same* change, and the wrong one for judging
    how either behaves under a realistic overburden.

    ``dip`` tilts the whole package about the model centre, which is what
    turns it into a structural trap: the sand climbs updip, a contact cuts
    across it at one depth, and an injector downdip can sit in the water
    leg while a producer updip sits in oil.  The overburden dips with the
    sand - tilting only the reservoir would drive it up through a flat
    seal, which is a crossing horizon rather than a structure.

    The two shales are semi-infinite, extending to the top and base of the
    model rather than being finite beds.  A three-layer *seismic* model
    means one sand between two half-spaces: giving the shales their own
    outer boundaries would add two more reflectors and stop the response
    being the sand's alone.
    """
    tilt = Relief(Dipping(0.0, dip=dip, azimuth=azimuth,
                          origin=(extent[0] / 2, extent[1] / 2))) if dip else None
    top = Flat(z_reservoir) + tilt if tilt else Flat(z_reservoir)
    base = Flat(z_reservoir + gross) + tilt if tilt else Flat(z_reservoir + gross)
    sand = _reservoir(top, gross, seed)
    if not heterogeneous:
        sand.porosity_heterogeneity = None
        sand.vsh_heterogeneity = None
        sand.n_sublayers = 1
        sand.sublayer_porosity_range = 0.0
    return [
        Layer("overburden_shale", Flat(0.0), "shale", porosity=0.16, vsh=0.88),
        sand,
        Layer("underburden_shale", base, "shale", porosity=0.13, vsh=0.90),
    ], FaultSet([])


#: Template name -> builder.
TEMPLATES: dict[str, Callable] = {
    "flat": flat_reservoir,
    "dipping": dipping_reservoir,
    "anticline": anticline_reservoir,
    "fault_compartment": fault_compartment,
    "channel": channel_reservoir,
    "lens": lens_reservoir,
    "stacked_sands": stacked_sands,
    "three_layer": three_layer,
    "five_layer": five_layer,
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
