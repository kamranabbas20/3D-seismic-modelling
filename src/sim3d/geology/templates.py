"""Ready-made geological templates (spec section 148).

Each template returns a list of :class:`~sim3d.geology.builder.Layer` and a
:class:`~sim3d.geology.faults.FaultSet` for a domain of the given extent,
so a study can start from a recognisable structure and vary it, rather than
assembling horizons by hand every time.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..core.errors import ConfigError
from ..core.geometry import bearing_vector
from .builder import Layer, heterogeneity_specs
from .faults import Fault, FaultSet
from .heterogeneity import HeterogeneitySpec
from .facies import get_facies
from .surfaces import (Anticline, Composite, Dipping, Flat, Lens,
                       PickedThickness, Relief, SectionThickness, Surface,
                       Syncline, Wedge)


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


#: The heterogeneity a reservoir unit gets unless it asks for something else.
def _heterogeneity(seed: int, offset: int = 0):
    return HeterogeneitySpec(
        std=0.030 if offset == 0 else 0.05,
        correlation_major=650.0, correlation_minor=380.0,
        correlation_vertical=18.0, azimuth=30.0, model="exponential",
        seed=seed + offset)


def _structure_relief(structure: dict | None, extent) -> Surface | None:
    """The relief every horizon in a layer cake carries, or ``None`` for flat.

    One structure for the whole package, deliberately.  Deforming a single
    horizon and leaving its neighbours flat drives it through them, which is
    a crossing horizon rather than a structure - the same mistake
    :func:`dipping_reservoir` exists to avoid.
    """
    if not structure:
        return None
    spec = dict(structure)
    style = str(spec.pop("style", "flat")).lower()
    centre = tuple(spec.pop("centre", None) or (extent[0] / 2, extent[1] / 2))
    if style == "flat":
        return None
    if style == "dipping":
        return Relief(Dipping(0.0, dip=float(spec.pop("dip", 4.0)),
                              azimuth=float(spec.pop("azimuth", 90.0)),
                              origin=centre))
    if style in ("anticline", "syncline"):
        radius = tuple(spec.pop("radius", None)
                       or (0.32 * extent[0], 0.45 * extent[1]))
        fold = Anticline if style == "anticline" else Syncline
        return Relief(fold(0.0, amplitude=float(spec.pop("amplitude", 90.0)),
                           centre=centre, radius=radius,
                           azimuth=float(spec.pop("azimuth", 0.0))))
    raise ConfigError(
        f"unknown structural style {style!r}; choose from flat, dipping, "
        f"anticline, syncline")


def structure_summary(structure: dict | None, extent, datum: float = 0.0) -> dict:
    """What a structural style actually does to the model, in metres.

    A dip is chosen as an angle and felt as a depth range.  Nobody computes
    6 degrees over three kilometres in their head, and that number - 315 m of
    relief - is the one being chosen.  Returns ``shallowest``, ``deepest``,
    ``relief`` and a sentence saying so.
    """
    relief = _structure_relief(structure, extent)
    if relief is None:
        return {"shallowest": float(datum), "deepest": float(datum),
                "relief": 0.0,
                "text": f"Flat layering: every horizon sits at its own depth "
                        f"below {float(datum):,.0f} m."}
    # Corners and centre are enough: every style here is monotone along one
    # bearing or radially symmetric about one point.
    xs = np.array([0.0, extent[0], 0.0, extent[0], 0.5 * extent[0]])
    ys = np.array([0.0, 0.0, extent[1], extent[1], 0.5 * extent[1]])
    values = np.asarray(relief.depth(xs, ys), dtype=float) + float(datum)
    shallowest, deepest = float(values.min()), float(values.max())
    style = str((structure or {}).get("style", "flat")).lower()
    span = deepest - shallowest
    if style == "dipping":
        dip = float((structure or {}).get("dip", 0.0))
        azimuth = float((structure or {}).get("azimuth", 90.0))
        reach = abs(extent[0] * np.sin(np.radians(azimuth))) + \
            abs(extent[1] * np.cos(np.radians(azimuth)))
        text = (f"{dip:g}° towards {azimuth:g}° = **{span:,.0f} m** of relief "
                f"across {reach:,.0f} m of model. The first horizon runs "
                f"{shallowest:,.0f} m at the shallow edge to {deepest:,.0f} m "
                f"at the deep one.")
    else:
        text = (f"{span:,.0f} m of relief: the first horizon runs "
                f"{shallowest:,.0f} m at the crest to {deepest:,.0f} m off it.")
    return {"shallowest": shallowest, "deepest": deepest, "relief": span,
            "text": text}


def _pinch_thickness(unit: dict, extent) -> Surface:
    """One unit's thickness, constant or tapering to zero.

    A pinchout is expressed as a thickness that reaches zero, never as a
    horizon that crosses another: the base is the top plus this, so where
    the thickness is zero the two touch and the unit is simply absent.
    """
    # A drawn profile says everything about the thickness, so it wins over
    # both the constant and the parametric pinchout rather than combining
    # with either - two sources for one number is how they disagree.
    profile = unit.get("thickness_profile")
    if profile:
        name = unit.get("name", "?")
        axis = int(profile.get("axis", 0))
        sections = profile.get("sections")
        if sections:
            # Several sections, interpolated between: an edit made on one
            # stays put and only its neighbourhood moves.
            everything = [float(t) for section in sections
                          for _, t in (section.get("points") or [])]
            if not everything:
                raise ConfigError(
                    f"unit {name!r} has sections but no knee points on any of "
                    f"them; draw at least one or remove the profile")
            if max(everything) <= 0.0:
                raise ConfigError(
                    f"unit {name!r} was drawn with no thickness on any section; "
                    f"a unit that is absent everywhere should be removed rather "
                    f"than drawn flat against its own top")
            return SectionThickness(
                sections=[{"at": float(section["at"]),
                           "points": [(float(a), float(b))
                                      for a, b in section["points"]]}
                          for section in sections], axis=axis)

        points = profile.get("points") or []
        if len(points) < 1:
            raise ConfigError(
                f"unit {name!r} has a drawn thickness with no "
                f"points; draw at least one or remove the profile")
        if max(float(t) for _, t in points) <= 0.0:
            raise ConfigError(
                f"unit {name!r} was drawn with no thickness "
                f"anywhere; a unit that is absent everywhere should be removed "
                f"rather than drawn flat against its own top")
        return PickedThickness(points=[(float(a), float(b)) for a, b in points],
                               axis=axis)

    thickness = float(unit.get("thickness", 100.0))
    if thickness <= 0:
        raise ConfigError(
            f"unit {unit.get('name', '?')!r} needs a positive thickness, got "
            f"{thickness}; a unit that is absent everywhere should be removed "
            f"rather than given zero thickness")
    pinch = unit.get("pinch_out")
    if not pinch:
        return Flat(thickness)

    spec = dict(pinch)
    shape = str(spec.pop("shape", "wedge")).lower()
    if shape == "wedge":
        azimuth = float(spec.pop("azimuth", 90.0))
        # `start` and `end` are fractions of how far the model reaches along
        # that bearing, so they mean the same thing whichever way it points.
        ux, uy = bearing_vector(azimuth)
        corners = [cx * ux + cy * uy
                   for cx in (0.0, extent[0]) for cy in (0.0, extent[1])]
        lo, span = min(corners), max(corners) - min(corners)
        start = float(spec.pop("start", 0.3))
        end = float(spec.pop("end", 0.8))
        if not 0.0 <= start < end <= 1.0:
            raise ConfigError(
                f"unit {unit.get('name', '?')!r}: a pinchout must run from "
                f"`start` to a larger `end`, both within 0 to 1 as fractions "
                f"of the model, got start {start} and end {end}")
        return Wedge(thickness, azimuth=azimuth, start=lo + start * span,
                     end=lo + end * span, origin=(0.0, 0.0))
    if shape == "lens":
        centre = tuple(spec.pop("centre", None)
                       or (extent[0] / 2, extent[1] / 2))
        radius = tuple(spec.pop("radius", None)
                       or (0.30 * extent[0], 0.30 * extent[1]))
        return Lens(thickness, centre=centre, radius=radius,
                    azimuth=float(spec.pop("azimuth", 0.0)),
                    taper=float(spec.pop("taper", 1.0)))
    raise ConfigError(
        f"unknown pinchout shape {shape!r}; choose from wedge, lens")


#: What a layer cake looks like when nothing is asked for.
DEFAULT_UNITS: list[dict] = [
    {"name": "overburden", "facies": "shale", "thickness": 900.0},
    {"name": "seal", "facies": "shale", "thickness": 60.0, "porosity": 0.12},
    {"name": "reservoir", "facies": "clean_sandstone", "thickness": 120.0,
     "is_reservoir": True},
    {"name": "underburden", "facies": "sandy_shale", "thickness": 400.0},
]


def layer_cake(extent=(3000.0, 3000.0, 2200.0), datum=0.0, units=None,
               structure=None, seed=1):
    """Template J: as many units as you like, stacked by thickness.

    The general case the other templates are special cases of.  Each entry in
    ``units`` is a dictionary with a ``name``, a ``facies`` and a
    ``thickness``, plus anything a
    :class:`~sim3d.geology.builder.Layer` accepts - ``porosity``, ``vsh``,
    ``ntg``, ``permeability``, ``is_reservoir``, ``n_sublayers`` - an
    optional ``pinch_out``, and an optional ``heterogeneity`` giving the
    correlated field its own controls instead of the default a reservoir
    unit would otherwise get.

    Horizons are built by *accumulating thickness* rather than by naming
    depths:

        top[0] = datum + relief,    top[k+1] = top[k] + thickness[k](x, y)

    That is what makes pinchouts work.  A thickness may vary across the map
    and reach zero, but it is never negative, so the horizons below a
    pinching unit rise to meet it and no horizon can ever cross another -
    which is the condition :func:`~sim3d.geology.builder.build_geology`
    enforces and which naming depths per horizon makes very easy to break.

    ``structure`` deforms the whole package together - ``{"style":
    "dipping", "dip": 8}``, ``{"style": "anticline", "amplitude": 90}`` - and
    ``pinch_out`` thins one unit out, either directionally
    (``{"shape": "wedge", "azimuth": 90, "start": 0.3, "end": 0.8}``, as
    fractions of the model) or radially
    (``{"shape": "lens", "radius": [900, 500]}``).

    ``thickness_profile`` is the drawn alternative and replaces ``thickness``
    and ``pinch_out`` for that unit rather than combining with them.  One
    section, ``{"axis": 0, "points": [[500, 90], [2000, 0]]}``, interpolates
    between knee points along that axis and holds constant across the other.
    Several, ``{"axis": 0, "sections": [{"at": 500, "points": [...]},
    {"at": 2500, "points": [...]}]}``, interpolate between the sections as
    well, which is how a unit is shaped in both map directions.

    The last unit has no base: like every other template here, it extends to
    the bottom of the model.
    """
    units = [dict(u) for u in (units if units is not None else DEFAULT_UNITS)]
    if not units:
        raise ConfigError("a layer cake needs at least one unit")
    names = [u.get("name") or f"unit_{i + 1}" for i, u in enumerate(units)]
    if len(set(names)) != len(names):
        duplicated = sorted({n for n in names if names.count(n) > 1})
        raise ConfigError(
            f"layer names must be unique; repeated: {', '.join(duplicated)}")

    relief = _structure_relief(structure, extent)
    parts: list[Surface] = [Flat(float(datum))]
    if relief is not None:
        parts.append(relief)

    layers: list[Layer] = []
    for index, (unit, name) in enumerate(zip(units, names)):
        top = Composite(list(parts))
        facies_name = unit.get("facies", "shale")
        explicit = unit.get("is_reservoir")
        is_reservoir = (get_facies(facies_name).is_reservoir if explicit is None
                        else bool(explicit))
        # A reservoir is heterogeneous unless told otherwise; an overburden
        # unit is not, because heterogeneity there costs time and changes
        # nothing anyone is asking about.
        heterogeneous = bool(unit.get("heterogeneous", is_reservoir))
        sublayers = int(unit.get("n_sublayers", 4 if heterogeneous else 1))
        layer = Layer(
            name, top, facies_name,
            porosity=unit.get("porosity"), vsh=unit.get("vsh"),
            ntg=unit.get("ntg"), permeability=unit.get("permeability"),
            n_sublayers=max(sublayers, 1),
            sublayer_porosity_range=float(
                unit.get("sublayer_porosity_range",
                         0.035 if heterogeneous else 0.0)),
            is_reservoir=explicit if explicit is None else bool(explicit),
        )
        spec = unit.get("heterogeneity")
        if spec is not None:
            # Spelled out on the unit, so it wins over the default the
            # reservoir flag would otherwise pick - including `false`, which
            # is how a reservoir layer is made uniform.
            layer.porosity_heterogeneity, layer.vsh_heterogeneity = \
                heterogeneity_specs(spec)
        elif heterogeneous:
            layer.porosity_heterogeneity = _heterogeneity(seed + 10 * index, 0)
            layer.vsh_heterogeneity = _heterogeneity(seed + 10 * index, 1)
        layers.append(layer)
        parts.append(_pinch_thickness(unit, extent))

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
    "three_layer": three_layer,
    "five_layer": five_layer,
    "layer_cake": layer_cake,
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
