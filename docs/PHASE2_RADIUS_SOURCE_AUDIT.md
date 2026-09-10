# Radius estimator and Kimimaro investigation — 2026-09-10

## What exists and what was reproduced

Original files under `data` were read without
modification. Byte-preserved script/statistics snapshots are in
`reference/radius-estimator-20260910`; installed Kimimaro 4.1.2 `trace.py` and
`intake.py` snapshots are in `reference/kimimaro-4.1.2`. SHA256 provenance is saved in
`outputs/radius-estimator-audit/source-provenance.json`. No dependency changes.

The workflow is:

1. `estimate_radii.py`: physical GPU EDT, 512 linear erosion thresholds,
   26-connected-component counts K(r), surviving volume V(r), smoothed peaks/valleys,
   and a three-Gaussian fit to repeated radii weighted by shifted log1p(K).
2. `multi_class_dusting.py`: retain components accounting for 99% of each larger
   class's volume, reassign smaller remnants to the next lower class. Binary
   foreground is preserved; these are caliber reassignments, not voxel deletion.
3. `shell_recovery.py`: promote foreground voxels near each larger-class core using
   that class's fitted mean as a Euclidean expansion radius, thick to thin.

Saved stats and outputs are for a `(212,950,700)` ZYX chunk with spacing
`(1280,256,256)` nm. `chunk2_dist_seg.h5` contains 36 positive labels; the estimator
uses **all nonzero values**, despite the unused `mask_value=1` setting. This is
consistent with the saved experiment and must not be mechanically changed to label 1
when reproducing it. The small macaque input remains explicitly label 1 only.

The larger mouse input is `(624,3570,5144)` uint8 (11.46 billion voxels); only its
metadata was read. Parent crop coordinates for chunk 2 are not recorded in the HDF5
attributes or the supplied scripts. Do not infer full parent lineage from filenames.
The macaque file exists at the previously recorded path without literal backslashes
before underscores and is the same phase-1 test volume.

The two chunk inputs, raw GMM classes, dust-cleaned classes and recovered classes
have exactly the same foreground: **3,585,679 voxels**. The chunk inputs' nonzero
values differ and should not be interpreted as interchangeable caliber labels.

One CPU EDT of the chunk, with the saved physical spacing, reproduces maximum EDT
**5976.380371 nm** and the entire saved raw GMM class volume with **zero voxel
mismatches** when using the saved thresholds. This establishes the thresholded mask
result, not that the original 512-pass curve or GMM fitting was rerun. K(r)/V(r) arrays
were not saved in the statistics NPZ; plots alone cannot reproduce the fit.

## Corrected intersections and recovered classes

The weighted-Gaussian intersection helper has the documented sign error. Solving
equality of the two log densities between their means gives:

| Quantity | Saved value (nm) | Corrected value (nm) |
|---|---:|---:|
| Class 1/2 boundary | 1648.705 | 1929.638 |
| Class 2/3 boundary | 3709.704 | 3860.153 |

These corrections use the saved means/sigmas/weights; no refitting occurred.
They change **73,015 raw class voxels**, approximately 2.04% of foreground. Existing
volumes were not overwritten. An independent unequal-weight/equal-variance case
gives the analytic crossing 2.34657359; the original helper gives 1.65342641.
The audit uses a bracketed log-density root and refuses an absent between-means
crossing instead of silently manufacturing a midpoint.

| Class | Raw GMM voxels | After class dusting | After shell recovery |
|---|---:|---:|---:|
| 1 | 3,249,302 | 3,252,316 | 2,563,306 |
| 2 | 305,643 | 302,930 | 609,744 |
| 3 | 30,734 | 30,433 | 412,629 |

Dusting demotes 301 voxels from class 3 to 2 and 3,014 from 2 to 1. Recovery promotes
856,024 voxels in total. It produces a spatial caliber hypothesis, which could inform
a representative-radius field `R_hat(x)=mean_radius[class(x)]`; it is not a measured
local radius at every voxel. Its Euclidean distance operation is not constrained to
travel inside a vessel component. It can transfer classes across background gaps to
nearby foreground without changing binary connectivity. That possibility follows
from the code; it was not measured as an actual error in these saved volumes.

## Controlled check of the radius interpretation

`scripts/probe_caliber_initialization.py` reproduces the original erosion/GMM recipe
on small, flat-ended digital cylinders with known nominal radii, using the installed
CPU EDT and CC implementations. Two probes, each 48×72×72 voxels, take about seconds;
no skeletonization is needed.

| Geometry | Nominal radii (nm) | Forced three-GMM means (nm) |
|---|---|---|
| One round tube | 1536 | 481, 913, 1337 |
| Three disjoint round tubes | 768, 1536, 2304 | 589, 1260, 1960 |

Neither curve has an interior peak under the original peak-finding settings.
For an ideal tube, K(r) stays constant until its core disappears, rather than
providing a vote concentrated at the tube radius. For disjoint tubes it resembles a
survival count. For a branching network it also incorporates splitting events.
Consequently, the GMM means are **features of the weighted erosion curve**, not
automatically independent vessel-population radii. This does not erase the observed
utility of the real-data method; it means the measurement-to-radius interpretation
requires calibration. Automatically selecting GMM class count alone does not solve
this semantic issue.

The current cross-section estimator gives scale 1.110 on the single round tube and
1.186 on three round tubes, compared with 3.714 on macaque. This basic check works
reasonably on ideal round geometry; it does not explain the real-data high scale.
Orientation, elongated profiles, junctions, tapering and anisotropic sampling still
need controlled tests. No synthetic result here validates the production estimator.

## Kimimaro execution and field implications

The installed `intake.py:skeletonize_subset` crops each connected component's DBF,
then calls `trace.trace` with the same parameter dictionary. `trace.py` computes a
component-specific maximum, root-distance field DAF, and

`P(x) = A * (1 - D(x)/M)^e + DAF(x)/max(DAF)`, with `M = max(D)^1.01`.

`compute_paths` selects remaining targets using DAF and traces toward existing
zero-cost paths when `fix_branching=True`. Scale/const affect invalidation of target
candidates; they do not directly strengthen the centering penalty. Thus excess
tracing and centerline placement need separate measurements.

An installed-extension probe rejects an array-valued invalidation scale with
`TypeError: only length-1 arrays can be converted to Python scalars`. A spatial
exclusion field is therefore not a drop-in scale argument. The public wrapper has
no supplied PDRF-field argument either. A field-based experiment needs an explicit
adapter with component slicing and tests; changing the global installed module is
not appropriate. The Python PDRF computation itself is available for controlled
field experiments without first splitting the input into separate caliber masks.

The installed wheel does not contain the Cython/C++ source. Attempts to fetch the
previously recorded pinned commit failed (web cache miss; shell network restricted).
This turn reverified the installed Python path and extension behavior, not the
underlying C++ implementation. Earlier notes remain the source for the detailed
connected invalidation mechanism. This limitation does not block the present probes.

### Why exponent alone is insufficient

Using the saved representative radii 1076/2819/4734 nm, saved global maximum 5976 nm,
`A=100000`, and an illustrative off-center value `D_o=0.8 D_c`, the actual installed
float32 penalty computation gives:

| Exponent | Thin contrast | Thick contrast | Thick absolute penalty difference |
|---|---:|---:|---:|
| 4 | 1.168 | 5.487 | 2521.25 |
| 16 | 1.860 | 906.21 | 0.0903 |
| 32 | 3.460 | 821,218 | 0.0000000817 |

The normalized DAF term ranges from 0 to 1. Huge ratios can therefore accompany
negligible absolute centering differences. This is a parameter-response probe using
assumed transverse displacement, not measured centering error in the mouse volume;
component-specific maxima may differ from the global value used here.

A better mathematical initializer should jointly assess exponent and amplitude.
Given measured center/off-center samples, let `b_c=1-D_c/M`, `b_o=1-D_o/M`.
For each candidate exponent, inspect both `b_o^e/b_c^e` and
`A*(b_o^e-b_c^e)`. A desired minimum absolute contrast delta would imply
`A >= delta/(b_o^e-b_c^e)` at each reliable sample. Delta, displacement tolerance,
and acceptable numerical dynamic range still need calibration against paths;
this pointwise inequality alone is not a path-optimality guarantee.

## Earlier iterative proposal recovered

`reference/caliber_skeletonization.md`, section 4,
contains an explicit earlier proposal: a grid over scale and exponent, a few dozen
runs, scored by component mismatch, coverage and parsimony. The project notes had
retained the general calibration strategy but omitted this concrete proposal.

That advisory document also overstates several conclusions: connected-component
agreement cannot detect a missed side arm; coverage/parsimony cannot establish
anatomical correctness; scale=1 does not guarantee exact vessel consumption; fitted
GMM means are not proven vessel-population estimates. Its spacing comparison mixes
the two confirmed datasets. Its hole-filling advice does not match the clean-input
contract. Do not adopt its numeric candidate or grid objective unchanged.

## Resulting development direction

1. Keep the erosion/caliber workflow as an experimental feature extractor, fix its
   intersection/plotting/selection contracts, and save full curves and provenance.
   Supplement it with sampled physical ridge/core radii; compare those measurements
   across erosion-derived bands rather than assuming each Gaussian mean is truth.
   Component-disappearance and persistence information is a candidate improvement,
   but fragmentation must be distinguished from disappearance before calling it a
   radius distribution.
2. Sample reliable transverse profiles across caliber bands, including thin vessels,
   then derive **joint centering exponent/amplitude** candidates and exclusion
   constraints. Check transverse coverage together with nearby branch preservation;
   radii alone cannot specify the length of anatomy safe to invalidate.
3. Test the formulas on controlled tapering/elliptical/anisotropic tubes and
   mixed-caliber junctions. Use known centerlines and outgoing-arm recovery to
   calibrate constants and reject failure modes before full macaque comparisons.
4. Run a small, purposeful macaque sequence with centering and exclusion changes
   separated. Use visual inspection and missing-arm/extra-trace measurements; do
   not optimize branch count alone. Authorizing reruns does not require arbitrary
   additional runs before there is an informed candidate.
5. First seek a scalar rule compatible with the packaged pipeline. If controlled
   mixed-caliber tests demonstrate a conflict that scalar settings cannot resolve,
   test a caliber-conditioned PDRF adapter on the intact mask. Hard class cuts and
   automatic joins are not prerequisites. Freeze the successful rule before mouse
   transfer evaluation.

This investigation did not change production parameters or any original data. It
produced an auditable reproduction, corrected split calculations and controlled
counterexamples that directly constrain the next estimator implementation.

## Outputs

- `scripts/audit_radius_estimator.py`: reproducible saved-chunk and PDRF audit.
- `scripts/probe_caliber_initialization.py`: reproducible known-radius probes.
- `outputs/radius-estimator-audit/audit.json`: hashes, counts, transition matrices,
  corrected splits and numeric PDRF survey (16.66 s measured audit runtime).
- `outputs/radius-estimator-audit/synthetic-caliber-probe.json`: full synthetic curves
  and fit/proposal outputs.

Original full mouse input and the phase-1 frozen artifacts were not recomputed.
