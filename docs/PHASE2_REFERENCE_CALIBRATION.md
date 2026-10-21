# First reference-level estimator result — 2026-09-10

**Superseded interpretation:** the user clarified that this reference-derived coefficient does
not meet the binary-volume-only requirement. This document preserves the experiment; it must
not be described as achieving the automation goal. Current work is in PHASE2_BINARY_ONLY.md.

The user set a practical first target: automatically obtain a skeleton roughly as
useful as the accepted hardcoded phase-1 reference. The reference is a calibration
target, with known imperfections; exact biological ground truth is not required for
this development milestone.

**A fresh macaque skeletonization using the first reference-calibrated exclusion
estimate reproduced the frozen baseline graph byte for byte.** This is a successful
training-dataset reproduction, not independent validation or a general solution to
TEASAR parameter estimation.

## Measured rule

During one-time calibration, for every selected foreground voxel x, find the nearest
accepted reference vertex v(x). Let d(x) be their physical Euclidean separation and
r(v) the saved reference radius. Hold the reference scale s=1.5 fixed and calculate

`c_reference = max_x max(0, d(x) - s*r(v(x)))`.

This is a sufficient additive margin for Euclidean coverage by the nearest-reference
balls. It is not the globally smallest union-of-balls margin, not a simulation of
Kimimaro's connected invalidation, and not evidence of anatomical completeness.
The fresh skeletonization tests whether it works as a parameter proposal.

Measure physical EDT maximum R from the selected input and save the dimensionless
calibration coefficient `alpha = c_reference / R`. Subsequent estimation uses only
the new input mask and saved model:

`scale = 1.5`

`const_nm = alpha * EDT_max_nm + 0.5 * norm(voxel_size_nm)`

The half-voxel-diagonal term is an explicit resolution-margin heuristic. On macaque:

| Measurement | Value |
|---|---:|
| Selected foreground | 4,066,930 voxels, label 1 only |
| Input EDT maximum | 4413.680664 nm |
| Largest distance to reference | 15298.008236 nm |
| Required nearest-reference additive margin | 10936.887059 nm |
| Learned alpha | 2.477951598993 |
| Resolution margin | 241.594702 nm |
| Estimated const | **11178.481761 nm** |
| Previous hardcoded const | 30000 nm |

The estimate does not copy the old 30000 nm constant. It learns a dimensionless
relationship from the reference and remeasures EDT at application time. However,
only one training dataset supports that relationship. Its predictive value on new
geometry remains unknown; extreme EDT/coverage values may be sensitive to local
geometry. Scale, pdrf_scale=100000, and pdrf_exponent=4 remain inherited settings.

The experiment also retains legacy soma and max_paths settings to isolate exclusion.
That includes soma detection/acceptance 750/3500 and max_paths=300. This is compatibility
with the accepted reference, not endorsement as clean-vessel defaults. The model
records these fixed settings and rejects application under a different context.

## Fresh-run comparison

| Metric | Accepted baseline | Previous cross-section heuristic | Reference-calibrated estimate |
|---|---:|---:|---:|
| Vertices | 4520 | 5225 | **4520** |
| Edges | 4517 | 5222 | **4517** |
| Components | 3 | 3 | **3** |
| Endpoints | 11 | 90 | **11** |
| Junctions | 5 | 81 | **5** |
| Branches | 13 | 168 | **13** |
| Length (um) | 1609.732 | 1909.099 | **1609.732** |
| Branches shorter than 5 um | 0 | 129 | **0** |

All baseline vertices and edges are retained exactly, with no extra vertices or edges.
Mapped radii match exactly. The candidate graph SHA256 equals the frozen reference:
`f5926f4739941d2df8cbb6e5dce0228b630c22710fdbd1859de77df8ec207e1e`.
The frozen baseline skeleton, graph, branch table and metrics hashes were verified.
The candidate ran with `skeleton.source=null`; it did not load the reference skeleton.

One candidate skeletonization took **37.14 s** (stage timer; outer progress displayed
37.15 s), graph extraction about 0.14 s. The packaged calibration command took
11.28 s; reference-free estimation took **8.15 s**. Conservative memory preflight was
used; peak RSS was not measured. No dense segmentation or extra full-volume candidate
was needed after exact reproduction was established.

## Packaged use

The workspace environment provides two explicit operations:

```powershell
# One-time development calibration using an accepted run.
python -m VesselEMPy calibrate --config config.yaml --reference runs/macaque --calibration calibrations/macaque-reference-v1.json

# Application: reads the input segmentation and saved model, not the reference graph.
python -m VesselEMPy estimate --config config.yaml --calibration calibrations/macaque-reference-v1.json
```

Estimation writes `runs/macaque/estimation-reference/measurements.json` and
`candidate.yaml`, selecting a separate `runs/macaque-reference-calibrated-v1` output.
It does not skeletonize. Resource settings are inherited, not silently increased.
The real run used a 16 GiB process budget and the existing 2 GiB reserve. Its exact
estimation input config is `outputs/baseline-calibration/estimation-input.yaml`.
Using that config in the estimate command generates the exact tested configuration.
The main config's 10 GiB budget is enough for estimation but not the conservative
full macaque skeletonization preflight.

The executed scientific stage command was:

```powershell
python -m VesselEMPy graph --config outputs/baseline-calibration/candidate.yaml
```

The packaged estimator subsequently generated an identical resolved configuration;
this was checked without running skeletonization again. Provenance links are in
`runs/macaque-reference-calibrated-v1/estimator-link.json`.

`estimate` without `--calibration` still invokes the earlier experimental cross-section
heuristic. This new model is explicit opt-in; it is not an unvalidated global default.
The reference-free application needs the saved model, including its inherited
parameter settings. Training metadata may name reference files, but application
does not open them.

## Verification and remaining work

23 tests pass, including physical-length scaling of the exclusion formula, explicit
label selection, calibration/application round trip, application after the reference
graph is deleted, and context rejection before EDT allocation. Two existing
Neuroglancer deprecation warnings remain. `scripts/verify_reference_candidate.py`
reproduces the saved-graph comparison and checks the frozen reference hashes.

The immediate reference-level macaque target is met. Broader estimator work remains:

- Evaluate the frozen coefficient on independent geometry, with a useful reference
  or controlled known answer. Same-dataset exact reproduction does not establish transfer.
- Determine whether a maximum-EDT scaling is sufficient or needs robust caliber and
  geometry features. Do not retune on each target and label that transfer validation.
- Estimate centering/scale only when evidence requires it; the current accepted
  settings already meet the first target and were intentionally held fixed here.
- Keep adaptive refinement bounded and driven by identifiable failures. A second
  candidate on this dataset was unnecessary after exact baseline reproduction.

All original data and accepted outputs are preserved. No pruning, joining, publication,
dependency changes, or automatic class-based partitioning were introduced.
