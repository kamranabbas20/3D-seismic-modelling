# A 40-degree dipping trap: downdip injection, updip production

A three-layer shale–sand–shale system dipping at 40 degrees, with the sand
crossing 2,000 m at the model centre, an oil-water contact cutting it, water
injected downdip into the water leg and oil produced updip, over three years.

## The model

| | |
|---|---|
| dip | 40 degrees, towards +x |
| sand | 20 m gross, crossing 2,000 m at x = 500 m |
| relief | 839 m over 1,000 m of dip length: sand runs 1,580 m updip to 2,420 m downdip |
| shales | semi-infinite above and below, so the only reflectors are the sand's own |
| OWC | 2,050 m, which cuts the sand at x = 560 m |
| transition | 15 m capillary ramp, Swirr 0.18 |
| P1 | updip at x = 200 m, sand at 1,748–1,768 m, oil leg |
| I1 | downdip at x = 800 m, sand at 2,252–2,272 m, water leg |
| duration | 1,095 days, monitor at the end |

## Three things the dip decided, before any physics

**The cell size is set by the dip, not by the seismic.** A bed of thickness
`g` dipping at `theta` drops `dx tan(theta)` per lateral cell, so adjacent
columns of it overlap by only `g - dx tan(theta)`. At 40 degrees with
`dx = 20` m and a 20 m bed that is 3 m, and on a 5 m vertical grid a third
of the column pairs shared no cell at all. The reservoir was a staircase of
disconnected blocks: both wells railed against their BHP limits inside a
year, produced and injected essentially nothing — and the material balance
closed to 4e-08, because nothing was conserved incorrectly. The fluid simply
had nowhere to go. Every symptom looked like a badly chosen rate.

`dx = 10` m and `dz = 4` m leave about three cells of overlap and every
adjacent column pair connected. `check_layer_connectivity` now measures this
and fails on it.

**A 20 m sand at 2,000 m is below tuning.** Vp in the sand is 2,597 m/s, so
at 30 Hz the wavelength is 87 m and the bed is a quarter of it. The top and
base do not resolve as two events; they interfere into one composite
wavelet. That is the physics of the model as specified, not a limitation of
the modelling.

**40 degrees is steeper than a 1D convolution can place.** sim2seis builds
every trace independently, so no energy moves sideways and a dipping
reflector is mispositioned by construction. The 4D *amplitudes* below are
still meaningful — they compare the same column against itself before and
after — but the structural position of the events is not, and only a
migration would fix that.

## What the rock physics does

One global dry frame is not a simplification here, it is a bias. With a
single soft-sand frame the shale comes out slower than the reservoir sand
and the top of the sand barely reflects at all. A stiff frame for the shale
restores it:

| | Vp |
|---|---|
| shale (stiff frame) | 3,102 m/s |
| sand | 2,597 m/s |

Soft sand under hard shale, so the top of the sand is a negative reflection
coefficient — the ordinary class of response for this section.

## The flow result

| | |
|---|---|
| P1, day 1095 | 2,250 STB/day oil, **no water**, BHP 1,968 psi |
| I1, day 1095 | 2,200 STB/day water, BHP 3,259 psi |
| material balance | 1.7e-12 over 288 timesteps |
| swept | 1,851 of 14,645 sand cells, max dSw 0.546 |

The injector sits in water, so every swept cell is the contact itself
moving. Averaging Sw over the sand in each column puts the oil-water
boundary at **x = 540 m at the start and x = 430 m after three years**: the
flood has pushed the contact 110 m updip, and the producer 230 m further
updip has not yet seen it.

## The 4D result

NRMS against the baseline, per scenario and angle stack:

| scenario | near | mid | far |
|---|---|---|---|
| pressure only | 8.72 | 8.25 | 7.97 |
| saturation only | 22.53 | 24.38 | 27.57 |
| combined | 23.15 | 25.02 | 28.38 |

Noise floor 4.6 % (6 % of signal RMS at 70 % repeatability), so every number
above is signal.

An earlier geometry, differing only in how widely and finely y was sampled,
gave 12.78 / 24.90 / 27.18 for the three near-stack figures. Nothing in the
model varies along y — the dip is along x — so the per-column physics is
identical and only the cube-wide average moved, by two to four NRMS points.
Worth stating plainly: a whole-volume NRMS is a property of the volume you
chose as much as of the change you modelled, and it is not comparable
between two runs on different grids.

**The AVO separates the two causes, and in opposite directions.** The
saturation response *grows* with angle, 24.90 to 30.52; the pressure
response *falls*, 12.78 to 10.58. That is the discriminator the angle stacks
exist for: a far-stack difference that brightens is fluid, one that dims is
pressure.

**The time shifts are negligible and that is the useful finding.** The true
shift peaks at 1.28 ms and the windowed estimate recovers it to 0.61 ms RMS.
Aligning the monitor before differencing moves the combined near-stack NRMS
from 23.15 % to 19.94 % — a real but secondary correction. A 20 m reservoir
changes too little of the travel path to shift the section below it, so this
4D is an amplitude signal almost entirely. On a thick or compacting
reservoir the same code gives the opposite answer: 6 ms of shift alone
manufactures 58 % NRMS.

## Figures

![the model in the app](figures/wedge-02_geology.png)

*The dipping package in section and map view. The sand runs 1,580 m updip
to 2,420 m downdip; the wells sit either side of the contact.*

![the section through time](figures/wedge-20_section_timelapse.png)

*The synthetic near stack along the P1 -> I1 line at each survey date.*

![the 4D difference](figures/wedge-21_section_difference.png)

*The same sections differenced against day 0. The anomaly tracks the sand
and its updip edge advances with the flood.*

Also in `figures/`: `wedge-01_project.png`, `wedge-03_model3d.png` and
`wedge-04_wells.png`, showing the configuration, the 3D view and the
completions resolved against the dipping sand.

## The flood through time, as a section between the wells

The wells share y = 700 m, so the P1 -> I1 line is a constant-y inline and
the section is a straight slice rather than an interpolated traverse. Four
surveys - day 0, 365, 730 and 1095 - on one shared time axis set by the
slowest of them, because each survey has its own velocity and so its own
deepest two-way time, and letting each end at its own would put the
monitors on axes the baseline cannot be subtracted from.

| day | traces with a 4D response above 10 % of peak | flow contact |
|---|---|---|
| 365 | 50 | x = 500 m |
| 730 | 56 | x = 460 m |
| 1095 | 62 | x = 430 m |

**The peak saturates and the extent grows.** The largest 4D amplitude is
28.4 %, 28.1 % and 28.2 % of the baseline peak at the three dates -
essentially unchanged. That is not a flat response: wherever the front has
passed, the oil-to-water substitution is *complete*, so the amplitude
change there is maxed out from the first year. What advances is the edge of
the anomaly, and it advances in step with the contact the flow simulation
puts there - two calculations that share no code agreeing on the same
front.

Reading the anomaly's updip edge directly is harder than it looks: a
threshold on trace amplitude picks up a small edge-of-model artefact at
x = 0 before it reaches the front, which is why the table counts affected
traces rather than quoting an edge position.

## The migration

It ran: 9 shots per survey, two surveys, on the isotropic 10 m grid at a
16 Hz Ricker.

| | |
|---|---|
| forward | 1,272 s |
| imaging | 2,835 s |
| total | ~68 min on an idle 4-core machine |
| memory | 3.56 GB, flat across all 18 shots |
| RTM 4D NRMS | 8.96 % |

![the migrated 4D difference](figures/wedge-32_rtm_4d.png)

![the migrated section](figures/wedge-33_rtm_section.png)

**The dipping reflector is in the right place.** That is the whole reason to
pay for a migration here: the 1D convolution mispositions a 40-degree dip by
construction, and the migrated inline puts the 4D anomaly along the sand
where the flow simulation says the flood is, terminating updip near
x = 500 m rather than smeared across the section.

**The boundaries are not the problem.** Measured rather than assumed: the
same homogeneous model in a 600 m box against an 1,800 m reference box whose
faces are too far away for anything to return inside the record, so any
difference after the small box's first face echo *is* the boundary's
leakage.

| PML | thickness | max leak | RMS |
|---|---|---|---|
| 10 nodes (shipped) | 100 m, 0.79 wavelengths | 0.028 % | 0.009 % |
| 20 nodes | 200 m, 1.58 wavelengths | 0.004 % | 0.001 % |
| 30 nodes | 300 m, 2.37 wavelengths | 0.001 % | 0.000 % |

All three are negligible, and the shipped layer is already inside 0.03 % of
a domain with no boundary at all - despite being thinner than the usual
one-to-two-wavelength guidance. Doubling it would double the domain to
remove 0.024 % of an artefact.

**The artefacts are large and worth naming.** Sources sit at 600 m and
receivers at 620 m, which is 100 m below the top of the propagation domain
and 20 m inside the absorbing layer's inner edge, so the top of every image
carries strong near-field ringing. That is the *source*, not the boundary: a
source is a singularity and no imaging condition removes it. This project
already learned that once and added `taper_wavelengths` and
`correlation_start_time` to `RTMSettings` for it; this configuration sets
neither, which is the cheapest available improvement to the image and costs
no propagation, because the taper applies to the image rather than to the
wavefield. The right-hand edge carries the
same from the domain boundary. With 9 shots the illumination of a
40-degree dip is sparse, and the QC said so before the run: the survey
clears the target by 90 m on its narrowest edge, under the 100 m the check
wants.

**On the 8.96 %.** It is not comparable with the 23.15 % the sim2seis near
stack reports, because the two are measured over different volumes: the
migrated image spans the whole propagation domain, most of which is barren
overburden with no 4D change at all, and a whole-volume NRMS is diluted by
exactly that. This project has the lesson already - *NRMS depends entirely
on where you measure it* - and the honest comparison is in a reservoir
window, which is work still to do.

## What was not run

The migrated image. The acquisition is defined and passes QC — 30 sources,
361 receivers, 10,830 traces, clearing the target by 110 m on its narrowest
edge — but a full-wave run at this bandwidth is the expensive path, and the
dip is exactly what makes it so: imaging a 40-degree reflector without
operator aliasing wants trace spacing under about 18 m at 55 Hz. The
sim2seis numbers above are amplitude-correct per column and structurally
wrong; a migration would fix the second and change the first.
