# Governing equations, assumptions and validation

Every scientific module states its equation, its assumptions, its units,
its numerical constraints and its validation test. This file collects them.

## Wave propagation

The solver integrates the first-order velocity–pressure acoustic system,
SI throughout:

```
∂v/∂t = −(1/ρ) ∇p
∂p/∂t = −κ ∇·v + s          κ = ρ Vp²
```

**Assumptions.** Acoustic (no shear stress, hence no S waves and no mode
conversion), lossless (Q = ∞), isotropic, spatially variable density, and
every outer face absorbing — so there is no free surface, no water layer,
no ghost and no surface multiple.

**Discretisation.** Staggered grid; the pressure sits on integer nodes and
each velocity component on the half node along its own axis. Space is 2N-th
order,

```
(∂ₓf)_{i+½} = (1/Δx) Σ_{k=1..N} c_k ( f_{i+k} − f_{i−k+1} )
```

and time is 2nd-order leapfrog. Buoyancy 1/ρ is averaged onto the staggered
locations — equivalently the harmonic mean of ρ — which is the choice that
keeps the scheme's effective impedance right across a sharp interface.

**Stability.** From `|c·Δt·k̃| ≤ 2` with `max|k̃ᵢ| = (2/Δxᵢ)Σ|c_k|`:

```
Δt ≤ 1 / ( Σ|c_k| · Vmax · sqrt(Σ 1/Δxᵢ²) )
```

For 8th order on an isotropic grid this is `Δt ≤ 0.449 Δx / Vmax`.

**Dispersion.** The fully discrete relation is
`sin(ωΔt/2) = ½ c Δt |k̃|`. The points-per-wavelength requirement is solved
from it by bisection rather than assumed, so it depends on the operator
*and* the Courant number.

| order | cells/λ at 1% error, at the CFL limit | spatial error alone |
|---|---|---|
| 2 | 10.99 | 12.81 |
| 4 | 5.56 | 5.08 |
| 6 | 5.42 | — |
| 8 | 5.25 | 3.32 |
| 10 | 5.13 | — |

The practical consequence: at the stability limit the time-stepping error
dominates and raising the spatial order past 4th buys almost nothing unless
`Δt` also drops.

**Boundaries.** Unsplit CPML (Roden & Gedney 2000; Komatitsch & Martin
2007), with `d(u) = d₀uᵐ`, `α(u) = α_max(1−u)`, `κ(u) = 1+(κ_max−1)uᵐ`, and
memory variables stored only inside the layers.

**Source.** Injected into the pressure equation, never used as a
convolution kernel:
`p += Δt · κ · s(t) · w / (ΔxΔyΔz)` with `w` the trilinear weights.

### Validation

| test | result |
|---|---|
| Whole-space Green's function `p = ρ ṡ(t−r/V)/(4πr)` | amplitude to 5%, arrival to < 1 sample |
| Geometric spreading | 1/r to 5% |
| Two-layer reflection traveltime | within one cell of two-way time |
| Normal-incidence coefficient `(Z₂−Z₁)/(Z₂+Z₁)` | to 15%, both polarities |
| CPML residual after the wavefront leaves | < 0.5% of peak |
| Energy decay after the source is silent | monotonic, < 1e-4 of peak |
| FD operator convergence | 2nd, 4th, 6th order confirmed |
| Scatter/gather adjointness | exact |

## Imaging

Per shot: propagate the source wavefield forward through the **migration**
model and store it with a Nyquist-safe time stride; inject the recorded
traces time-reversed at the receivers; correlate.

```
I(x) = Σ_t S(x,t) R(x,t)                          cross-correlation
I(x) = Σ_t S R / ( Σ_t S² + ε )                   source-normalised
```

The correlation product contains frequencies up to `2·fmax`, so the stored
stride must satisfy `Δt_save ≤ 1/(4·fmax)`. A coarser stride is refused,
not aliased.

**Validation.** A point diffractor images within 0.12 wavelengths of its
true position, with 52% of the image energy inside a sphere of 0.3
wavelengths that occupies 1.4% of the analysed volume. A flat reflector
images at the correct depth.

## Rock physics

```
minerals + porosity → dry frame → fluids → mixing → Gassmann → Vp, Vs, ρ
```

**Mineral mixing.** Voigt and Reuss are rigorous bounds; VRH is their mean
and carries no bound status (a test records that it can sit outside the
Hashin–Shtrikman bounds at 90% clay). Hashin–Shtrikman uses Berryman's
n-phase form.

**Dry frame.** Hertz–Mindlin at critical porosity,

```
K_HM = [ n²(1−φ_c)²μ² P / (18π²(1−ν)²) ]^(1/3)
```

with the `P^(1/3)` stiffening that is the physical origin of the pressure
half of a 4D signal. Soft sand and stiff sand (Dvorkin & Nur 1996)
interpolate from that point to the mineral point along the modified HS
lower and upper bounds respectively; the stiff-sand model is markedly less
pressure-sensitive, so the choice between them sets the predicted pressure
response. Nur's critical-porosity model has no pressure dependence at all
and serves as a control.

**Fluids.** Batzle & Wang (1992), evaluated in the authors' units (MPa,
°C, g/cm³) so the printed coefficients stay checkable against the paper,
then converted back to SI. Validity is 5–100 MPa and 10–350 °C; outside
that the result is an extrapolation and says so.

**Mixing.** Wood/Reuss (uniform, the standard choice — a few percent of gas
collapses the fluid modulus) or Brie (patchy). Density always mixes
linearly.

**Gassmann.**

```
K_sat = K_dry + (1 − K_dry/K_min)² / ( φ/K_fl + (1−φ)/K_min − K_dry/K_min² )
μ_sat = μ_dry
```

Low-frequency limit: the pore fluid is assumed to equilibrate over a
seismic period. That the fluid does not change μ is what makes the
pressure/saturation decomposition tractable — saturation acts only on K,
while effective stress acts on both.

**Pressure.** `P_eff = P_conf − α·P_pore`. At the free surface both
pressures are atmospheric, so `P_eff` is genuinely zero and every
grain-contact model degenerates; an explicit floor is applied there and the
number of affected cells is reported.

### Validation

| test | result |
|---|---|
| Pure water, 20 °C, 0.1 MPa | 1482.4 m/s, 997 kg/m³ (measured 1482, 998) |
| Seawater, 20 °C, 0.1 MPa | 1520.3 m/s (measured 1521) |
| Hertz–Mindlin pressure exponent | P^(1/3) to nine digits |
| Gassmann forward ∘ inverse | identity to 1e-9 |
| Mixing bound ordering | Reuss ≤ HS⁻ ≤ HS⁺ ≤ Voigt |

## 4D decomposition

Four independent earth models, each run through the complete chain:

```
ΔX_pressure    = X_pressure   − X_baseline
ΔX_saturation  = X_saturation − X_baseline
ΔX_combined    = X_combined   − X_baseline
ΔX_interaction = ΔX_combined − (ΔX_pressure + ΔX_saturation)
```

In property space the interaction measures the nonlinearity of the rock
physics alone — Gassmann's response to a fluid change depends on the frame,
and the frame depends on stress. In seismic space it measures that plus
everything the wave equation and the imaging chain add: finite-frequency
interference, tuning, illumination and the migration operator. Comparing
the two says whether an observed nonlinearity is rock physics or wave
physics.
