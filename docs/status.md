# Implementation status against the specification

Section numbers refer to the project specification. Three states:
**done** (implemented and tested), **partial** (usable, with a stated gap),
**deferred** (interface designed, physics not implemented).

## Core architecture — Phase 0

| § | Requirement | State | Notes |
|---|---|---|---|
| 6 | Geology / propagation / target domains kept distinct | done | `DomainSet` enforces containment |
| 8 | No silent degradation | done | `InfeasibleExperiment` carries the alternatives |
| 9 | Multi-resolution workflow | partial | Preview and full-wave exist; the four named quality levels are configuration presets, not a first-class type |
| 10 | Grid definition | done | `Grid3D`, SI internally |
| 11 | Frequency-dependent sampling | done | Solved from the scheme's dispersion relation, not assumed |
| 12 | CFL stability | done | Derived limit, reported, never auto-corrected |
| 13 | Computational estimate | done | Runtime only against a measured benchmark |
| 14 | CPU-first | done | NumPy + Numba |
| 15 | Backend abstraction | done | `ComputeBackend`; GPU raises rather than falling back |
| 16 | Future GPU | deferred | Interface fixed |
| 109 | Dependency-aware recomputation | partial | Stage-level caching in `Pipeline`; not a general graph |
| 110 | Caching by configuration hash | partial | Hash implemented and excludes display settings; no on-disk cache yet |
| 111 | Checkpointing | partial | Per-shot wavefield store; no restart-after-failure |
| 114 | xarray data model | deferred | Volumes are NumPy with explicit grids; xarray not yet the container |
| 115 | SI units | done | Explicit `Quantity` conversion at the boundaries |
| 116 | Provenance | partial | `ReservoirState.provenance` and pipeline notes; no full lineage graph |
| 117 | YAML/JSON configuration | done | Unknown keys are errors |
| 118 | CLI | done | Twelve commands, no GUI dependency |
| 119 | GUI as a frontend only | done | Streamlit calls `Pipeline`; a test asserts the UI imports no solver or rock-physics model |
| 121 | Main GUI pages | partial | Seven pages covering the fourteen the spec lists; no dedicated Experiments page |
| 122 | Visualisation libraries | partial | Plotly sections and maps; no PyVista/VTK 3D rendering |
| 123 | Interactive slicing with a linked cursor | done | One cursor in metres shared across grids and pages |
| 124 | 3D reservoir display with opacity | deferred | Needs the volume renderer |
| 108 | Interactive editing without automatic reruns | done | Sliders update state and rock physics; modelling and RTM stay on buttons |

## Geology and wells — Phases 1–2

| § | Requirement | State | Notes |
|---|---|---|---|
| 17–18 | Volumetric model, layering, folds | done | Composable surfaces |
| 19 | Normal/reverse/sealing faults | done | Planar and kinematic; listric, networks and damage zones deferred |
| 20–21 | Reservoir geometries from horizons | partial | Tabular, wedge, lens, channel and stacked via templates; no interactive editing |
| 22–23 | Stratigraphy and sub-layering | done | Cyclic sub-layer modulation for tuning studies |
| 24 | Facies | done | Eight defaults, extensible |
| 25 | Heterogeneity | done | Gaussian, exponential, spherical; anisotropic; seeded |
| 26–29 | Wells, spacing, patterns | done | Vertical only; close spacing warns, never forbids |
| 30–38 | Mechanistic reservoir state | done | Independent halos and fronts, fault compartmentalisation, pseudo-time |
| 39 | State variables and closure | done | `Sw + So + Sg = 1` enforced |
| 148 | Templates A–G | done | Seven templates |

## Rock physics — Phase 3

| § | Requirement | State | Notes |
|---|---|---|---|
| 40 | Rock physics as a core engine | done | Every intermediate preserved |
| 41 | Mineral mixing | done | Voigt, Reuss, VRH, Hashin-Shtrikman |
| 42 | Dry-rock models | partial | Hertz-Mindlin, soft sand, stiff sand, critical porosity; contact-cement deferred |
| 43 | Fluid properties | done | Batzle-Wang, validated against measured water and seawater |
| 44 | Fluid mixing | done | Wood and Brie |
| 45 | Gassmann | done | Forward and inverse, with validity guards |
| 46 | Pressure substitution | done | Four approaches, including a table that refuses to extrapolate |
| 47 | Combined pressure and fluid effects | done | Never assumed additive |
| 58 | Sensitivities | deferred | Computable by finite difference through the chain; no dedicated API |

## Wave, imaging and 4D — Phases 4–8

| § | Requirement | State | Notes |
|---|---|---|---|
| 60–62 | 3D acoustic solver | done | Validated against the analytic Green's function |
| 63 | Absorbing boundaries | done | Unsplit CPML |
| 64 | Source model | partial | Ricker and Ormsby; Klauder and imported signatures deferred |
| 65 | Wavefield snapshots | done | Selected steps only |
| 66–68 | OBN acquisition and display | partial | Geometry, fold, offset and azimuth; no 3D viewer |
| 69 | Other geometries | deferred | Named and refused, not silently substituted |
| 70–71 | Shot records | partial | Gathers produced and inspectable; no wiggle/variable-density viewer |
| 72 | Illumination | partial | Geometric fold only; wavefield illumination deferred |
| 73–79 | 3D RTM | done | Validated by diffractor collapse and reflector depth |
| 76–78 | Migration velocity experiments | done | Scaling and smoothing, recorded as deliberate |
| 80 | Imaging-condition extensions | deferred | Cross-correlation and source-normalised only |
| 81 | Kirchhoff | deferred | Named and refused |
| 82–83 | Preview mode and physics transparency | done | Labelled everywhere |
| — | Sparse-synthetic mode (K vertical traces) | done | `imaging.method: sparse_synthetic`; layouts wells/grid/points; pinned equal to the full cube at the same column |
| 84 | Light processing | done | Band-pass, mutes, display-only gain |
| 85 | Baseline/monitor repeatability | done | One pinned `dt`, one migration operator |
| 86 | Non-repeatability | deferred | Noise and geometry perturbation not yet modelled |
| 87–92 | 4D outputs, metrics, well-centric analysis | partial | All metrics implemented; no synchronised viewer |
| 48–57 | Pressure/saturation decomposition | done | Four independent earth models, isolation asserted |

## Validation — sections 127–134

| § | Requirement | State | Notes |
|---|---|---|---|
| 127 | Automatic model QC | done | `sim3d qc`, with FAIL blocking simulation |
| 128 | Rock-physics validation | done | Guards on moduli, fractions and the Gassmann denominator |
| 129 | Numerical benchmarks | done | Direct wave, two-layer reflection, diffractor, RTM diffractor collapse, flat-reflector depth |
| 130 | Rock-physics benchmarks | done | Voigt, Reuss, VRH, Gassmann round-trip, fluid density, pressure response |
| 131 | 4D null test | done | Identical baseline and monitor, run independently through the full chain |
| 132–133 | Pressure-only and saturation-only isolation | done | Asserted at construction, not only in a test |
| 134 | Combined interaction test | done | Interaction grows with perturbation size |

## Additional requirements (second specification)

| § | Requirement | State | Notes |
|---|---|---|---|
| 1 | Interactive well placement by mouse | done | Add Well, pick type, click the map. Select, move, rename, retype, delete and edit completions and controls. Producers green, injectors blue, everywhere |
| 2 | Completions by geological unit | done | Layer intersection honours dip, fold and fault throw; manual sub-intervals validated against their unit |
| 3 | Structural geology editing | partial | Engine supports all the listed structures and they are visible in 3D; parameters are edited in the configuration, not by dragging surfaces |
| 4 | Fault definition | partial | Planar faults with strike/dip/throw/extent/transmissibility, feeding both the geometry and the flow model, drawn in 3D; no fault editor UI |
| 5 | Interactive 3D model with visibility toggles | done | Rotate, pan, zoom, reset, perspective/orthographic; per-layer, per-well and per-component visibility; layer transparency; decimated property volumes |
| 6 | Well representation in 3D | done | True vertical trajectories with open intervals drawn thick in the well's own colour, and a filled wellhead circle |
| 7 | User-defined simulation duration and timestep | done | Days internally, adaptive timestep under a saturation-change limit |
| 8 | Pressure in psi everywhere | done | Single display unit, no toggle |
| 9 | Sensible production and injection rates | done | Suggestion is the smaller of deliverability and pattern scale, with the reasoning recorded; sanity checks warn and explain |
| 10 | Rate and well-control modes | done | Liquid/oil/water rate and BHP, with limit switching |
| 11 | Dynamic well behaviour | done | Rates, BHP, water cut and cumulative volumes as time series |
| 12 | Injection/production schedule | partial | Per-well start and end days, editable in the GUI; no mid-run rate or completion changes |
| 13 | Saveable scenario setup | done | Definition, flow results and seismic results in three separate files, plus display state; a configuration edit never discards stored results |
| 14 | Save/load/duplicate/delete | done | Plus rename. Duplicate copies the definition only by default |
| 15 | Scenario comparison | partial | Two setups compared section by section, with the stages that differ; no side-by-side result panels yet |
| 16 | Recommended workflow | done | All twelve steps are reachable from the GUI |
| 17 | Dependency-aware recalculation | done | Explicit graph in `core.graph`, tested against the requirement's own examples |
| 18 | Core UX principle | partial | Depths, thicknesses, pore volume and rates are derived automatically and overridable; geology is still built from templates rather than drawn |

## Not started

Sections 93–103 (detectability and experiment engine), 105–108
(demonstration outputs and interactive editing), 120–126 (visualisation and
I/O), 135–147 (assistant, elastic, anisotropy, attenuation, Born, LSRTM,
FWI, simulator import), and the Streamlit front end (§121).

## Known physical limitations

- **No free surface or water layer.** Every face is absorbing, so there are
  no ghosts and no surface multiples. This is the most significant gap for
  a marine OBN study and is stated at every run by `sim3d physics`.
- **Acoustic only.** No S waves, no mode conversion, no AVO.
- **Lossless.** No intrinsic attenuation.
- **Isotropic.** No VTI or TTI.
- **Kinematic faults.** Displacement and transmissibility, not stress.
- **Batzle-Wang extrapolation** below 5 MPa, reported per run.
