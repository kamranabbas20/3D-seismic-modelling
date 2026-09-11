"""Assemble a 3D geological model from horizons, faults and heterogeneity.

Spec sections 17-25.  The builder produces the *static* model: facies,
porosity, shale volume, net-to-gross and the mineral composition of every
cell.  It knows nothing about fluids, pressure or elastic properties -
those come later in the chain, which is what lets the same geology be
re-used across every 4D scenario without rebuilding it.

Construction order
------------------
1. Restore each grid node through the fault set, giving the depth it would
   have had before faulting.
2. Look up which layer that restored depth falls in, from the horizon
   surfaces.
3. Assign each layer's facies and property statistics, adding a correlated
   random field where the layer specifies one.

Doing the fault restoration first is what keeps every horizon offset
consistently across a fault, instead of displacing them one at a time and
hoping they stay parallel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.errors import ConfigError, ValidationError
from ..core.grid import Grid3D
from .facies import FACIES, Facies, get_facies
from .faults import FaultSet
from .heterogeneity import HeterogeneitySpec, random_field
from .surfaces import Surface


@dataclass
class Layer:
    """One stratigraphic unit, defined by the surface that forms its top."""

    name: str
    top: Surface
    facies: str
    #: Overrides of the facies defaults; ``None`` uses the facies midpoint.
    porosity: float | None = None
    vsh: float | None = None
    ntg: float | None = None
    #: Constant permeability in mD; ``None`` derives it from porosity.
    permeability: float | None = None
    #: Optional correlated heterogeneity added to porosity and shale volume.
    porosity_heterogeneity: HeterogeneitySpec | None = None
    vsh_heterogeneity: HeterogeneitySpec | None = None
    #: Split the layer into this many sub-layers with alternating properties.
    n_sublayers: int = 1
    #: Peak-to-peak porosity variation between sub-layers.
    sublayer_porosity_range: float = 0.0
    is_reservoir: bool | None = None

    def resolved_facies(self, catalogue: dict[str, Facies] | None = None) -> Facies:
        return get_facies(self.facies, catalogue)


@dataclass
class GeologyModel:
    """A built static geological model on a grid."""

    grid: Grid3D
    layer_index: np.ndarray
    facies_code: np.ndarray
    porosity: np.ndarray
    vsh: np.ndarray
    ntg: np.ndarray
    #: Absolute permeability in m^2 (display it in mD).
    permeability: np.ndarray
    reservoir_mask: np.ndarray
    layers: list[Layer]
    faults: FaultSet
    horizons: dict[str, np.ndarray]
    catalogue: dict[str, Facies]

    @property
    def permeability_md(self) -> np.ndarray:
        """Permeability in millidarcy, for display and reporting."""
        from ..core.units import MILLIDARCY
        return self.permeability / MILLIDARCY

    @property
    def composition(self) -> dict[str, np.ndarray]:
        """Mineral volume fractions of the solid phase, cell by cell.

        Built from the facies composition, then adjusted so that the clay
        fraction honours the cell's shale volume.  Non-clay minerals keep
        their relative proportions.
        """
        clay = np.clip(self.vsh, 0.0, 1.0)
        others: dict[str, np.ndarray] = {}
        for name, facies in self.catalogue.items():
            mask = self.facies_code == facies.code
            if not np.any(mask):
                continue
            non_clay = {m: f for m, f in facies.composition.items() if m != "clay"}
            total = sum(non_clay.values())
            for mineral, fraction in non_clay.items():
                share = fraction / total if total > 0 else 0.0
                others.setdefault(mineral, np.zeros(self.grid.shape))[mask] = share
        out = {"clay": clay}
        for mineral, share in others.items():
            out[mineral] = share * (1.0 - clay)
        return out

    def summary(self) -> str:
        lines = [f"Geological model on {self.grid.describe()}",
                 f"  {len(self.layers)} layers, {len(self.faults)} fault(s)"]
        for i, layer in enumerate(self.layers):
            mask = self.layer_index == i
            if not np.any(mask):
                lines.append(f"    {i}: {layer.name:22s} ({layer.facies}) - absent")
                continue
            lines.append(
                f"    {i}: {layer.name:22s} ({layer.facies:16s}) "
                f"{100 * mask.mean():5.1f}% of cells, "
                f"phi {self.porosity[mask].min():.3f}-{self.porosity[mask].max():.3f}, "
                f"Vsh {self.vsh[mask].min():.3f}-{self.vsh[mask].max():.3f}, "
                f"k {self.permeability_md[mask].min():.3g}-"
                f"{self.permeability_md[mask].max():.3g} mD"
            )
        lines.append(f"  reservoir cells: {100 * self.reservoir_mask.mean():.1f}%")
        if len(self.faults):
            lines.append("  " + self.faults.describe().replace("\n", "\n  "))
        return "\n".join(lines)


def build_geology(grid: Grid3D, layers: list[Layer], faults: FaultSet | None = None,
                  catalogue: dict[str, Facies] | None = None) -> GeologyModel:
    """Build a static geological model.

    ``layers`` must be ordered from shallowest to deepest by their tops; the
    first layer's top is taken as the top of the model and everything above
    it is assigned to that layer.
    """
    if not layers:
        raise ConfigError("a geological model needs at least one layer")
    catalogue = catalogue or FACIES
    faults = faults or FaultSet([])

    x, y, z = np.meshgrid(grid.axis(0), grid.axis(1), grid.axis(2), indexing="ij")
    z_restored = faults.restore(x, y, z)

    horizons: dict[str, np.ndarray] = {}
    tops = []
    for layer in layers:
        surface = layer.top.on_grid(grid)  # (nx, ny)
        horizons[layer.name] = surface
        tops.append(surface[:, :, None])
    for i in range(1, len(tops)):
        if np.any(tops[i] < tops[i - 1] - 1e-9):
            raise ValidationError(
                f"layer {layers[i].name!r} has a top above layer "
                f"{layers[i - 1].name!r}; horizons must be ordered downwards "
                f"(a truncation or pinchout needs an explicit unconformity, "
                f"not a crossing horizon)"
            )

    layer_index = np.zeros(grid.shape, dtype=np.int16)
    for i, top in enumerate(tops):
        layer_index = np.where(z_restored >= top, i, layer_index)

    facies_code = np.zeros(grid.shape, dtype=np.int16)
    porosity = np.zeros(grid.shape)
    vsh = np.zeros(grid.shape)
    ntg = np.zeros(grid.shape)
    permeability = np.zeros(grid.shape)
    reservoir = np.zeros(grid.shape, dtype=bool)

    for i, layer in enumerate(layers):
        facies = layer.resolved_facies(catalogue)
        mask = layer_index == i
        if not np.any(mask):
            continue
        facies_code[mask] = facies.code

        phi = _layer_property(grid, layer, "porosity", facies, layer.porosity_heterogeneity)
        if layer.n_sublayers > 1 and layer.sublayer_porosity_range > 0:
            phi = phi + _sublayer_modulation(grid, tops[i], tops[i + 1] if i + 1 < len(tops) else None,
                                             z_restored, layer)
        # Clip to the facies range widened by half its own width: heterogeneity
        # may take a cell outside the nominal range, but not to an unphysical value.
        lo, hi = facies.porosity
        margin = 0.5 * (hi - lo)
        porosity[mask] = np.clip(phi, max(0.005, lo - margin), min(0.50, hi + margin))[mask]

        shale = _layer_property(grid, layer, "vsh", facies, layer.vsh_heterogeneity)
        vsh[mask] = np.clip(shale, 0.0, 1.0)[mask]

        ntg[mask] = layer.ntg if layer.ntg is not None else facies.midpoint("ntg")
        permeability[mask] = _permeability(porosity, facies, layer)[mask]
        is_res = facies.is_reservoir if layer.is_reservoir is None else layer.is_reservoir
        reservoir[mask] = is_res

    return GeologyModel(
        grid=grid, layer_index=layer_index, facies_code=facies_code,
        porosity=porosity, vsh=vsh, ntg=ntg, permeability=permeability,
        reservoir_mask=reservoir, layers=list(layers), faults=faults,
        horizons=horizons, catalogue=catalogue,
    )


def _permeability(porosity: np.ndarray, facies: Facies, layer: Layer) -> np.ndarray:
    r"""Permeability from porosity, log-linear within the facies' own range.

    .. math:: \log_{10} k = \log_{10}k_{lo}
              + rac{\phi - \phi_{lo}}{\phi_{hi} - \phi_{lo}}
                ig(\log_{10}k_{hi} - \log_{10}k_{lo}ig)

    Permeability spans orders of magnitude for a few porosity units, so the
    interpolation is in log space; doing it linearly would put most of a
    reservoir at implausibly high permeability.

    This is a placeholder transform, not a measurement: it is monotonic,
    honours the facies table's endpoints, and is stated here rather than
    buried, so a study whose conclusions depend on the porosity-permeability
    relation can replace it with a calibrated one.  An explicit
    ``layer.permeability`` overrides it entirely.
    """
    from ..core.units import MILLIDARCY

    if layer.permeability is not None:
        return np.full(porosity.shape, float(layer.permeability) * MILLIDARCY)
    phi_lo, phi_hi = facies.porosity
    k_lo, k_hi = facies.permeability
    span = max(phi_hi - phi_lo, 1e-9)
    fraction = np.clip((porosity - phi_lo) / span, 0.0, 1.0)
    log_k = np.log10(k_lo) + fraction * (np.log10(k_hi) - np.log10(k_lo))
    return 10.0**log_k * MILLIDARCY


def _layer_property(grid, layer: Layer, attribute: str, facies: Facies,
                    spec: HeterogeneitySpec | None) -> np.ndarray:
    override = getattr(layer, attribute)
    base = override if override is not None else facies.midpoint(attribute)
    if spec is None:
        return np.full(grid.shape, float(base))
    return random_field(grid, HeterogeneitySpec(
        mean=float(base), std=spec.std,
        correlation_major=spec.correlation_major,
        correlation_minor=spec.correlation_minor,
        correlation_vertical=spec.correlation_vertical,
        azimuth=spec.azimuth, model=spec.model, seed=spec.seed,
    ))


def _sublayer_modulation(grid, top, base, z_restored, layer: Layer) -> np.ndarray:
    """Cyclic porosity variation within a layer, for tuning and interference studies."""
    if base is None:
        thickness = np.full_like(top, grid.extent[2])
    else:
        thickness = np.maximum(base - top, grid.dz)
    fraction = np.clip((z_restored - top) / thickness, 0.0, 1.0)
    return 0.5 * layer.sublayer_porosity_range * np.cos(
        2.0 * np.pi * layer.n_sublayers * fraction
    )
