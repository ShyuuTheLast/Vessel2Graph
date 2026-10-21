# Binary-volume-only TEASAR estimator — 2026-09-10

## Larger closing trial: 1024 nm

User visual acceptance after branch overlays: the current skeleton looks perfect from
every observable side. Old branch 8 occupied a thick junction region; the new result
correctly repartitions that region among all three adjoining branches, extending the
other two neighbors. Exact concatenation of old 8 and 13 is therefore not the desired
evaluation criterion. User confirms the exterior-connected gap/sharp surface indentation
has disappeared in the closed binary mask. This is direct local visual evidence supporting
the preprocessing experiment, not independent validation across datasets. Preserve the
accepted run and use it only for evaluation, never to derive estimator coefficients.
The closing radius remains user-selected; automatic preparation-scale selection is unsolved.

User requested a 3–5 voxel radius: chose 4 XY voxels = 1024 nm physical radius,
3.2 Z voxels. The original label-1 mask is closed and optionally enclosed-filled;
the resulting saved mask is used for BOTH parameter estimation and skeleton tracing.
Command: `estimate --config configs/macaque-closing-1024nm.yaml --fill-holes --closing-radius-nm 1024`.
Closing adds 147,128 voxels (3.618%), removes none; subsequent filling adds zero.
One component remains. 381 profiles derive scale 2.7668182781209683, const
9649.594701928663 nm, exponent 8, amplitude 16384, in 95.257 seconds.
Candidate config: runs/macaque-closing-1024nm/estimation-volume-v2/candidate.yaml.
Monitored cropped tracing completes in 23.205 seconds, sampled peak RSS 10.322 GiB.
Result: 11 branches, 7 endpoints, 5 junctions, one component; no branches under 10 um
and no terminal endpoint within 15 um of the user-identified old spur junction.
New branch 11 follows old branch 13 closely (98.23% of old vertices within 1 um);
the path near old branch 8 differs substantially. This supports removal of the local
terminal spur, not confirmed anatomical accuracy or exact old 8/13 reunion.
Baseline proximity within 1 um is 80.69%; all 4551 vertices lie inside prepared mask.
Root viewer config selects the trial; older masks and graphs are preserved.

## Scope correction

The goal is to estimate useful TEASAR parameters from a binary vessel mask and its
physical spacing. A pre-existing accepted skeleton is **evaluation only**. The
earlier reference-calibrated coefficient does not meet that goal and is not used
by this method. No reference skeleton, reference measurements, or calibration file
is read by `volume_estimation.py`.

The user clarified that macaque has holes and authorized explicit preparation to
address them. This supersedes the earlier blanket exclusion of input correction
for this investigation. Original input files remain unchanged. Label 1 is selected
before any preparation; the unrelated label 2 is excluded.

## Hole-filling investigation

The original radius script contains optional binary closing (dilation followed by
erosion), but its saved `closing_radius_vox` is zero, so it was disabled. Its shell
recovery script expands caliber labels inside the vessel mask; that is not binary
hole filling. Installed Kimimaro also calls `fill_voids.fill` when its soma-detection
threshold is exceeded, even before soma acceptance is decided.

Explicit enclosed-hole filling on macaque adds **39,112 voxels** (0.962% of original
foreground), changing the selected count from 4,066,930 to 4,106,042. Both masks have
one 26-connected component. Maximum physical EDT rises from 4413.681 to 4890.873 nm.
No dilation or closing is applied in this experiment. Exterior-connected cavities
remain open; the implementation rejects preparation that merges input components.

The prepared mask is saved separately and is the exact input to both parameter
measurement and tracing. Soma detection and acceptance are disabled explicitly,
so no hidden preparation or soma-specific root/path trimming is mixed into tracing.
`max_paths=None` and `dust_threshold=0` prevent arbitrary path/component removal.
These are vessel-processing policies, not estimated biological parameters.

## Actual parameter derivation

One physical EDT is computed. The prior erosion-count/GMM workflow is retained as
a 16-threshold diagnostic with full curve and corrected Gaussian intersections.
The fitted Gaussian means are not treated as independent vessel-population radii.
Direct transverse measurements drive the parameters instead.

1. Sample 1024 interior seeds, ascend EDT, and use the existing ridge/Hessian filter.
2. Estimate a local axis from foreground-coordinate PCA within three local EDT radii.
   Reject regions lacking a dominant axis (largest/second covariance eigenvalue <1.5).
3. Trace 32 rays in the transverse plane at half-minimum-voxel steps. Reject unresolved
   wall exits and crop-boundary exits. Record central EDT r and largest transverse
   exit distance R. Measure median EDT o at half-r displacement across ray directions.
   Reject profiles for which o is not below r.
4. Derive exclusion from the accepted profiles:

   `scale = max(1, Q90(R/r))`

   `const = Q90(R) + 0.5 * voxel_diagonal`

   The additive transverse-radius term is an explicit tip-clearance heuristic. It
   prevents the exclusion radius from collapsing with DBF at boundary targets.
   Its multiplier is one, chosen before the candidate run, not fitted to a reference.
5. Split sampled central radii into three equal-count strata. These balance the
   centering calculation; they are not asserted anatomical classes. For each
   supported exponent e in 1,2,4,8,16,32,64, use the actual penalty bases
   `b_c=1-r/M`, `b_o=1-o/M`, with `M=maxEDT^1.01`:

   - Require the lower-decile ratio `(b_o/b_c)^e` in every populated stratum to be ≥4.
   - Require a lower-decile absolute difference `A*(b_o^e-b_c^e)` of ≥32 in each
     stratum, relative to Kimimaro's normalized DAF range of 0–1.
   - Round required A up to a power of two, reject values above 2^24, and choose
     the smallest tested exponent satisfying both constraints.

These numerical tolerances are explicit design assumptions, not constants learned
from the accepted skeleton and not path-optimality guarantees. A failed profile or
centering check raises an error rather than silently using the old parameters.

### Macaque measurements and proposal

303 profiles passed the checks. Q90(R) is 8896 nm; the resolution margin is 241.595 nm.
Sampled-radius stratum boundaries are 1999.424, 3334.759, 3878.209 and 4890.873 nm.
Exponent 4 fails the fourfold ratio requirement in the lower-radius stratum (2.800).
Exponent 8 passes with stratum lower-decile ratios 7.838, 66.182 and 644.051.

| Parameter | Derived value |
|---|---:|
| scale | 2.95932409848 |
| const | 9137.59470193 nm |
| pdrf_exponent | 8 |
| pdrf_scale | 16384 |

No previous values of these four parameters are used. A blind replay deliberately
set all incoming values to 999999, pointed `skeleton.source` at a nonexistent file,
and supplied no calibration. It reproduced the same proposal, centering survey and
prepared-mask SHA256 exactly. The end-to-end unit test also forbids skeleton-array
loading. Thus this is distinct from the prior reference-calibrated experiment.

## Fresh run and controlled ablation

The full proposal was frozen before evaluating against the baseline. One fresh
skeletonization ran, then one controlled ablation removed only the Q90(R) additive
tip margin (leaving const=241.595 nm). No coefficient was fitted from the reference.

| Metric | Accepted reference | Binary-only proposal | Without tip margin |
|---|---:|---:|---:|
| Vertices | 4520 | 4587 | 6866 |
| Edges | 4517 | 4586 | 6865 |
| Components | 3 | 1 | 1 |
| Endpoints | 11 | 8 | 353 |
| Junctions | 5 | 6 | 337 |
| Branches | 13 | 13 | 689 |
| Branches below 5 um | 0 | 0 | 626 |
| Total length (um) | 1609.732 | 1632.153 | 2610.382 |
| Skeletonization time (s) | — | 26.09 | 169.59 |

The first candidate is retained for inspection. Removing the tip margin creates
many extra traces with little improvement in reference proximity. This supports
keeping a nonzero, radius-derived tip-clearance floor on this dataset. It does not
establish its globally optimal value or validity on other data.

For the retained candidate, **95.40%** of baseline vertices are within 1 um of a
candidate vertex; the median distance is zero and p95 is 0.859 um. All 4587 candidate
vertices lie within the prepared mask. The longest edge is 483.189 nm, one physical
voxel diagonal: its connected result was produced by local tracing, not post-hoc
joining. The prepared binary input has one component; matching that count alone
does not prove correct branching or complete arm recovery.

### Remaining discrepancy

User interpretation update: the user identifies the shortest baseline branch as an
artifact. The saved table confirms this is branch 12 (11.712 um), itself an isolated
31-vertex component. Its absence should therefore be regarded as an improvement
under that identification, not a failed recovery target. Do not tune the estimator
to restore it.

The other isolated baseline piece is branch 13 (72.640 um, 168 vertices). It is not
the second-shortest branch by the saved physical-length table: that is branch 3
(28.813 um), which belongs to the main baseline component. Candidate geometry follows
branch 13 (p95 nearest-vertex offset 1.523 um; distal endpoint identical) and connects
it into the main skeleton. It has not removed that vessel. The user's reference to
another unrelated component has not yet been mapped unambiguously to branch 13 or
an input object. Do not equate reconnection with correction of an unrelated-object
artifact. Input label 2 remains excluded in both baseline and candidate.

The following geometric discrepancy remains useful as a description, but its earlier
interpretation as possibly missing desired anatomy is superseded for branch 12:

Reference branch 12 is an 11.712 um short path whose maximum distance from the
candidate is 9.820 um. Its discrepancy is **unchanged in the no-tip-margin ablation**,
so the added tip margin alone does not explain its omission. Changed root/soma
handling, centering, and exclusion scale remain possible contributors. The overlay
shows this reference spur within a broad vessel-region projection; projection alone
cannot establish whether it is a true arm or unnecessary interior tracing. Branch 1
also has local differences up to 4.322 um. Do not claim all reference branches were
recovered or exact baseline accuracy was demonstrated.

`outputs/binary-estimation-v2/comparison.png` shows three full-skeleton projections
and three close-ups of branch 12 over the prepared mask. Orange is the reference;
blue is the candidate. It was visually inspected for rendering and the discrepancy.
The baseline remains a practical reference, not ground truth.

## Packaged usage

```powershell
# Activate the existing workspace environment first.
. .\scripts\activate.ps1

# Binary mask + spacing; no skeleton or calibration input.
python -m VesselEMPy estimate --config configs/macaque-binary.yaml --fill-holes

# Run the emitted candidate, in its own output directory.
python -m VesselEMPy graph --config runs/macaque-binary/estimation-volume-v2/candidate.yaml
```

`estimate` now defaults to the volume method. `--fill-holes` is explicit; without it,
the input mask is preserved. `--method cross-section` retains the earlier heuristic
for reproduction. The old `--calibration` route remains explicit legacy experimental
functionality and cannot be combined with volume-method or filling flags.

The actual evaluated candidate is in `runs/macaque-volume-estimated-v2`, generated
from `runs/macaque/estimation-volume-v2/candidate.yaml`. To inspect that existing run:

```powershell
python -m VesselEMPy view --config runs/macaque/estimation-volume-v2/candidate.yaml
```

The new clean config writes future experiments under `runs/macaque-binary` so it does
not overwrite the evaluated run. Estimation does not launch skeletonization or dense
assignment. Dense assignment using the candidate config would target the prepared
mask; mapping back to the original mask has not been implemented in this experiment.

## Verification, provenance and limits

- 27 tests pass, including enclosed versus exterior-connected cavities, original-mask
  immutability, parameter constraints, failure cases, and a no-reference end-to-end run.
- The accepted baseline skeleton/graph/branches/metrics hashes were checked unchanged.
- Final-code blind replay: 35.36 s; initial estimation: 35.12 s. Peak RSS was not
  measured. Conservative memory checks apply; the local skeleton budget was 16 GiB
  with 2 GiB reserve. No environment/dependency changes occurred.
- `runs/macaque/estimation-volume-v2/measurements.json` records initial measurements;
  `runs/macaque-volume-replay/estimation-volume-v2/measurements.json` records final-code
  replay with component guards. The final implementation and the initial proposal
  produce identical mask bytes and parameter values.
- `outputs/binary-estimation-v2/blind-replay-verification.json` links both reports.
- `scripts/evaluate_volume_candidate.py` reproduces evaluation without modifying or
  tuning the estimator. Both candidate directories contain `baseline-evaluation.json`.

Sample selection still favors larger volumes and rejects very thin/ambiguous profiles.
The accepted minimum sampled radius is about 2 um; this is not proof that thinner
vessels are absent. Global-maximum centering needs additional multi-component tests.
The tip floor, axis filter, quantiles and penalty tolerances need validation on more
geometry. Hole filling plus changed soma policy plus centering changed together in
the first proposal, so their individual contributions have not been isolated.

This is a promising binary-only candidate with a useful ablation and removal of the
user-identified shortest-branch artifact, not a validated general automatic estimator.
Next identify the second reported artifact precisely and evaluate the fixed rules on
independent geometry before tuning further. Do not restore branch 12 to improve a
reference-distance score.
# Explicit closing trial — 2026-09-10

User reports a false 5.118 um branch 12 in the binary-only candidate, creating a junction
that subdivides branches 8 and 13, and suspects an exterior-connected segmentation gap.
This is distinct from branch 12 in the phase-1 reference. Enclosed filling cannot repair
such gaps, so an authorized opt-in physical-ball closing precedes filling and measurement.
CLI: `estimate --config configs/macaque-closing-320nm.yaml --fill-holes --closing-radius-nm 320`.
At spacing (320,256,256), the footprint is seven voxels (center plus six axis neighbors).
Padding before both morphological operations preserves original foreground at crop edges.
Original selected mask stays untouched. Closing adds 90,275 voxels; filling afterward adds
3,755. Components remain one. 322 profiles yield scale 2.7660304, const 9380.794702 nm,
PDRF exponent 8 and amplitude 16384 in 42.335 s. Radius 320 nm is an explicit development
choice, not an automatically inferred quantity; the four tracing parameters remain binary-only.
Candidate: runs/macaque-closing-320nm/estimation-volume-v2/candidate.yaml. Initial tracing
attempts failed memory checks. Opt-in empty-background cropping with one zero halo and
restored original coordinates allowed normal preflight and a monitored run to complete:
25.146 s tracing, sampled peak RSS 10.320 GiB, minimum free 2.760 GiB. Synthetic tests
preserve exact Kimimaro geometry, including vessels touching original volume faces.
Result has 13 branches, 8 endpoints, 6 junctions, one component. The old junction moves
9.836 um but a 5.076 um spur remains and old branches 8/13 do not map to one continuous
branch. Closing is not yet an improvement. Baseline proximity within 1 um drops to 86.88%
(not an accuracy score). Full audit files are in runs/macaque-closing-320nm-volume-estimated-v2.
