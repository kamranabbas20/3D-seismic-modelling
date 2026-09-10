# Shale–sand–shale: the two seismic modes on one 4D change

A flat three-layer model — shale, a 150 m sand at 1200 m, shale — with one
injector and one producer 800 m apart. Nothing else is in the model, so the
only reflectors are the sand's top and base and a 4D anomaly has nowhere to
hide. Both seismic modes run on the same geology, the same rock physics and
the same monitor day, so every difference between the two pictures is the
acquisition and the imaging operator.

Config: `examples/configs/three_layer_4d.yaml`.

## Choosing the monitor interval

Not picked — derived, from two criteria measured on this model.

**Resolvability binds.** The first Fresnel zone at 12 Hz and 1200 m is 350 m.
Below that radius seismic sees a blur, not an anomaly with mappable shape.

**Amplitude never binds.** The 4D NRMS is already 11.1 % at one year and grows
about two points a year, roughly linearly. Against the 5–10 % NRMS
repeatability floor of a good OBN survey — which is what this config acquires
— the change is detectable almost from the start.

| day | years | flood radius | ÷ Fresnel | swept | NRMS (near) |
|---|---|---|---|---|---|
| 365 | 1.0 | 285 m | 0.81 | 3.5 % | 11.07 % |
| 548 | 1.5 | 351 m | 1.00 | 5.1 % | 12.60 % |
| **730** | **2.0** | **382 m** | **1.09** | 6.8 % | **13.59 %** |
| 1095 | 3.0 | 476 m | 1.36 | 10.0 % | 15.69 % |
| 1460 | 4.0 | 576 m | 1.65 | 13.0 % | 17.29 % |

Day 548 is the true minimum, but it meets the Fresnel criterion by 0.3 % —
met by luck rather than margin. **Day 730** clears it at 1.09× with NRMS at
13.6 %, and is the conventional field repeat interval.

There is an upper bound too: the flood reaches the model's far corner, 1720 m
from the injector, at around day 2650. Later monitors are contaminated by the
no-flow boundaries, so waiting for years three or four buys two or three NRMS
points and spends that margin.

## What the two modes cost

| | wall clock |
|---|---|
| flow, states, rock physics (shared) | 390 s |
| sim2seis, 2 models × 3 angle stacks | **5 s** |
| full-wave, 2 models × 9 shots | 135 s |
| RTM, 2 models | 291 s |

## What they produce

**NRMS depends entirely on where you measure it.**

| | whole volume | reservoir window |
|---|---|---|
| sim2seis (near) | 13.6 % | 14.7 % |
| RTM | **1.6 %** | **26.2 %** |

The whole-volume RTM figure is the trap. It is not that the migration sees
less: its denominator is inflated by the source-side artefact at ~950 m, which
is 20× the reservoir amplitude and *identical* in both surveys, so it cancels
in the numerator while dominating the divisor. Measured where the change
actually is, the RTM 4D difference is nearly twice sim2seis's — because it
carries the imaging operator's own variance on top of the reservoir signal.

**Where each mode puts the anomaly**, against the simulated flood:

| | centroid | x half-width |
|---|---|---|
| flood (ΔSw, the truth) | (630, 1000, 1262) m | 425–825 m |
| sim2seis | (679, 1009, 1298) m | 450–825 m |
| RTM | (746, 1005, 1254) m | 746–858 m |

sim2seis reproduces the flood's shape and lateral extent almost exactly, which
it must: it is the rock physics filtered by a wavelet, with no lateral operator
to move anything. Its 36 m vertical offset is wavelet asymmetry across a
150 m sand.

The RTM gets the depth right to 8 m but places the anomaly's strongest part
116 m from the flood centroid and concentrated into a compact spot, with weaker
rings spreading well beyond the true flood. At 9 shots and this aperture that
is fold and illumination, not a coding error — and it is exactly the kind of
mispositioning a convolutional synthetic cannot warn you about, because it has
no operator to mis-position anything.

## The RTM image: what was wrong with it

The first version of this comparison produced a migrated image that was
unreadable, and the diagnosis took four steps, two of them in code and two
in the survey geometry.

| defect | evidence | fix |
|---|---|---|
| polarity inverted | correlation −0.707 against a band-limited reference | apply −∇², not ∇²: a Laplacian turns a peak into a trough |
| injection near-field 57× the reservoir | acquisition band max 11.4 against 0.2 | taper the image one wavelength around every source and receiver |
| correlating before the target can reflect | artefact ratio 2.9 → 1.5 | skip the imaging condition before the earliest possible target arrival, 152 ms here |
| sources only 1.6 wavelengths above the target | near-field overlapped the reservoir top and no filter separated them | standoff 260 m → 660 m |
| that move cut the aperture to 21 degrees | image became migration arcs rather than reflectors | spread 500 m → 1000 m, restoring 37 degrees |

Whole-volume 4D NRMS across those steps: 1.6, 5.6, 14.1, 28.1, 16.9 %.
The measurement to trust is the reservoir window, which went 26.2 → 39.5 %.

Two things worth recording because they cost time. The obvious suspect —
the direct arrival — was not the cause: muting it moved the acquisition
band's share of image energy from 95.9 % to 93.0 % while discarding 67 % of
the data. And moving the acquisition up was only half a fix, because
standoff and aperture trade against each other for a fixed spread; the run
in between was worse than the one before it.

What is still limited: correlation with the independent synthetic is +0.512
inside the illuminated footprint, and the acquisition band still carries
3.7× the reservoir RMS. That is 16-shot fold with a hard aperture edge, and
the reflectors stop where the spread stops illuminating. Neither
contaminates the reservoir window; neither is production quality.

## Reading this honestly

sim2seis answers *what change did the rock physics put into the earth model*.
The acquisition-driven path answers *what would a survey of that earth model
show*. On this model those two answers differ by 116 m of lateral position and
a factor of nearly two in measured 4D amplitude. Neither is wrong; they are
answers to different questions, and the gap between them is the value of
running the expensive one.

Caveats that apply to the RTM leg here: 9 shots is very low fold, there is no
direct-arrival or aperture mute, and the source-side artefact is untreated.
Those are known gaps, not properties of migration.
