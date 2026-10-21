# Phase 2: measurement-derived TEASAR parameters

Handoff checkpoint: local development paused after visual acceptance of the 1024 nm
closing candidate. See PHASE2_BINARY_ONLY.md and CLUSTER_HANDOFF.md for the current
implemented method and limitations. Earlier proposal sections below are historical;
mixed-caliber investigation is reserved for the cluster, not this workstation.

Status: active experimental development, authorized 2026-09-10 after phase-1 acceptance. The user prefers
a mathematical parameter estimate using inexpensive measurements, aiming for one production
skeletonization. Do not substitute a costly whole-volume hyperparameter search as the default.
The main deliverable is automated TEASAR parameter estimation in the now-packaged pipeline;
saved-graph diagnostics are evaluation tools for that deliverable, not a separate pruning stage.

Latest required target: estimate useful parameters from binary input and spacing alone; accepted
skeletons are evaluation references only. The user rejected the reference-calibrated approach as
not satisfying this requirement. It is preserved in PHASE2_REFERENCE_CALIBRATION.md as historical
work, not the accepted estimator strategy.

Current implementation: PHASE2_BINARY_ONLY.md and volume_estimation.py. Explicit optional hole
filling, physical EDT/erosion features and transverse profiles derive all four scale/const/PDRF
parameters. A fresh macaque candidate has 13 branches, no branches below 5 um, and 95.40% of
reference vertices within 1 um. Independent-input replay confirms no prior skeleton/parameter
dependence. An ablation demonstrates excess tracing without the radius-derived tip floor.
Some local reference discrepancies remain; accuracy and transfer are not validated. User's latest
hole-filling authorization supersedes older clean-input restrictions for this preparation experiment.

Latest investigation: see PHASE2_RADIUS_SOURCE_AUDIT.md for exact saved-chunk reproduction,
corrected class boundaries, controlled-radius counterexamples, installed PDRF response tests,
and a more concrete development direction. The first heuristic does not yet implement that work.

## Development iteration and current authorization — 2026-09-10

User explicitly permits further skeletonization runs to figure out good automatic parameter
estimation. The earlier no-rerun instruction concerned the completed saved-graph analysis.
The one-production-pass aim does not prohibit multiple development/calibration experiments.

The original notes below document mathematical initialization and synthetic/macaque calibration,
but do not specify an adaptive parameter-update algorithm, an objective, or a stopping rule.
The current implementation only proposes scale/const; it does not implement iterative refinement
or the proposed centering calculation. A usable automated estimator remains the main deliverable.
Subsequent source investigation recovered an explicit grid proposal in the advisory
Downloads/caliber_skeletonization.md (scale × exponent, topology/coverage/parsimony scoring).
It was not fully transcribed here and is not implemented. Its objective and claimed mathematical
guarantees need corrections described in PHASE2_RADIUS_SOURCE_AUDIT.md.

Proposed development outline (newly made explicit here, not a previously validated method):

1. Verify measurement-derived initialization on controlled geometries: cross-sectional extent,
   inscribed radius, and center/off-center EDT profiles, covering resolution and caliber changes.
2. Calibrate a small number of formula constants with targeted skeletonization experiments.
   Separate centering changes from exclusion changes to identify which mechanism improves or
   worsens missing arms, duplicate/offshoot tracing, and centerline position.
3. Test a small, explicitly motivated set of full macaque candidates with resource preflights,
   preserved artifacts and fixed evaluation sites. Use the frozen baseline as a reference,
   not a branch-count target or ground truth. User visual assessment can supply local evidence.
4. If adaptive refinement is useful, define the measurable failure signal, parameter-update
   direction and bounded trial budget before implementing it. Stop when evidence supports
   acceptance, improvement stalls, or the budget is reached; report unresolved tradeoffs.
   These criteria still need operational thresholds and validation.
5. Freeze the measurement-to-parameter rule and assess mouse transfer. Keep development cost
   separate from routine estimation cost; avoid making full-volume search the production default.

## Existing radius method

The local prototype computes physical EDT, sweeps 512 linear erosion thresholds with 26-connected
component labeling, and records K(r) and surviving volume V(r). It smooths log1p(K), detects
peaks/valleys, and fits a three-component Gaussian mixture to repeated linear-radius samples
weighted by shifted/scaled log1p(K). The user considers this a useful caliber estimator under
the clean-input contract. Keep it as the initial method and test its predictive usefulness.

Nuances: K(r) measures fragmentation/disappearance, not one independent vote per vessel; GMM
weights are not demonstrated vessel-population fractions. A log transform of K is not a model
of log(radius). Raw EDT classes describe shells/cores until region recovery is applied.

Saved mouse chunk-2 summaries: means 1076.445, 2818.572, 4734.375 nm; original split thresholds
1648.705 and 3709.704 nm; maximum EDT 5976.380 nm. These do not describe macaque. Thresholds need
recalculation because the Gaussian-intersection helper has a sign error in its log term.

Confirmed isolated check: means 0,4; sigmas 1,1; weights .8,.2. Helper gives 1.6534; correct crossing
is 2.3466. The plot also independently peak-normalizes Gaussian PDFs, so is not a valid fit overlay.
Saved mouse Gaussian statistics now give corrected boundaries 1929.638/3860.153 nm. Reproducing
the saved raw classes with one CPU EDT and old thresholds gives zero mismatches; applying the
corrected thresholds changes 73,015 voxels. No original output was overwritten. Synthetic probes
show fitted K(r) Gaussian means cannot automatically be interpreted as physical vessel radii:
a single 1536 nm tube produces three fitted means 481/913/1337 nm. Supplement and calibrate
these features with direct core/profile measurements before using them to set a penalty field.

## Source-grounded mechanisms (Kimimaro 4.1.2)

1. Exclusion limit at traced points is `scale * DBF + const`, in the units of anisotropy.
   The executed Cython/C++ path is `roll_invalidation_ball_inside_component`: foreground-connected
   expansion bounded by straight-line physical Euclidean distance from a source point. It is not
   unrestricted cube clearing, nor accumulated geodesic distance. Earlier cube descriptions were wrong.
2. Centering term is `A * (1 - D / M)**e`, with `M = dbf_max**1.01`, float32 arithmetic, plus
   normalized distance-from-root (DAF). A is pdrf_scale; e is pdrf_exponent. The component maximum
   affects thin-vessel guidance. Keep physical units fixed (nm): the 1.01 exponent means changing
   numerical units can alter this normalization slightly beyond simple geometric rescaling.
3. Soma detection can trigger hole filling before acceptance. Disable both soma detection and
   acceptance for the clean-vessel baseline; do not enable repair unintentionally.
4. `max_paths=None` avoids an arbitrary cap. `dust_threshold` is in voxels and should not silently
   discard valid components. Public defaults differ from low-level function signature defaults;
   specify the chosen settings explicitly.
5. No native GPU backend is exposed. CPU multiprocessing is mainly across connected components,
   so one connected vascular network is not automatically distributed across cores during tracing.
6. Public skeletonize does not accept an externally prepared EDT. Reuse across calls would require
   a deliberate adapter/upstream change, not an assumed optimization.

Source locations: `kimimaro/trace.py`, `kimimaro/intake.py`,
`ext/skeletontricks/skeletontricks.pyx`, `ext/skeletontricks/dijkstra_invalidation.hpp`.

## Proposed measurements and calculations — NOT validated

Reuse EDT and caliber estimates. Compute per-component extrema by reductions. Add a bounded sample
of candidate interior locations across calibers, with local profiles and approximate cross-sections.
Avoid a full preliminary skeleton or full-volume multiscale Hessian analysis merely to choose params.

### Centering

At reliable tubular samples, measure center EDT Dc and EDT Do at a specified transverse displacement.
For desired contrast q in the penalty term alone:

`e >= log(q) / log((1 - Do/M) / (1 - Dc/M))`

Use valid ordered values `0 <= Do < Dc < M`; handle flat/quantized profiles and unreliable junctions.
Aggregate across calibers and round toward supported fast power-of-two exponents where sensible.
Displacement tolerance and q need one-time calibration. This ratio omits the DAF contribution and
does not guarantee path accuracy. Large e may diminish absolute penalties; do not derive A
independently as though contrast were the entire objective. Initially hold A fixed.

### Exclusion

EDT gives nearest-wall distance, which can underrepresent full transverse extent in elongated
cross-sections. For sampled approximate local axes, estimate inscribed radius ri, required transverse
extent Ri and uncertainty epsilon_i. Fit a conservative rule:

`scale * ri + const approximately Ri + epsilon_i`

This is a hypothesis for deriving settings from shape and resolution. Local axis estimation and
cross-section sampling must be benchmarked and validated. Exclude or separately treat junctions;
their large extent must not inflate all exclusion settings. `scale=1` is not guaranteed to exactly
consume one vessel; const, geometry, path location and voxel discretization still matter.

### Erosion and cost

Persistent eroded cores can supply stable sampling locations and caliber evidence. Tracking them
has a cost and is optional. The original 512 full-volume CC passes may be expensive. Validate a
coarse sweep with adaptive refinement against the dense macaque sweep before adopting it.

## Validation strategy

Develop a small number of explicit formulas on macaque, using synthetic tubes/junctions to verify
mechanisms. Calibrate fixed constants, freeze them, and test mouse transfer. Assess centering,
missing branches, duplicate traces and graph connectivity; component counts alone cannot detect a
missing arm of a connected Y. Reconstruction coverage and stability are also insufficient alone.
Cycles and intended branch semantics need explicit assessment.

Separate hard class masks alter EDT and introduce artificial endpoints. If a global rule fails,
consider overlapping class passes with explicit reconciliation, only after characterizing failure.
Cropping also changes dbf_max, so crop tuning does not automatically transfer to the full volume.

No numeric macaque starting parameters have been validated. No evidence yet proves the formula
approach succeeds or fails. Avoid claims of mathematical optimality or automatic accuracy guarantees.

## First implemented experiment — 2026-09-10

`VesselEMPy estimate --config config.yaml` computes one physical EDT, a 16-threshold linear
erosion/26-connected-component curve, and samples 1,024 random interior seeds (EDT > twice
minimum spacing). Up to 32 local EDT ascent steps move seeds toward ridges. Duplicates are
removed. A finite-difference Hessian rejects samples without two negative transverse eigenvalues
and a relatively flat longitudinal eigenvalue. Sixteen rays in the estimated transverse plane
measure first foreground exits up to 4 local radii; unresolved exits/boundary exits are rejected.

The exclusion proposal is the 90th percentile of maximum transverse ray extent / local EDT,
bounded below by 1, with half the voxel diagonal as const. These choices are heuristics, not
calibrated guarantees. This version does NOT fit radius classes, tune centering, change soma/path
caps, or implement the old forced three-Gaussian fit. The erosion curve is diagnostic only.
Sampling is volume-weighted, excludes the thinnest resolved voxels, and rejects broad profiles;
Hessian estimates are sensitive to grid/ridge geometry. These are material limitations to assess.

Macaque measurements: maximum EDT 4413.681 nm; one coarse smoothed erosion peak near 1765.47 nm;
227 accepted cross-sections; scale 3.713996, const 241.594702 nm. The high scale relative to the
4r ray search limit is a reason to inspect sampling bias/axis quality before accepting this rule.
Measurement took 21.49 s; sampled peak process RSS 4,152,823,808 bytes (~3.87 GiB).
Raw samples and full curve: runs/macaque/estimation/. The candidate leaves baseline science
settings fixed except scale/const and selects fresh skeleton generation in a separate run.
Its RAM budget was set to 16 GiB solely for the full-volume preflight. Two attempts stopped before
Kimimaro because available memory was insufficient. With 17.14 GiB available, one candidate
skeletonization completed in 74.02 s. It produced 5,225 vertices, 5,222 edges, 168 branches,
90 endpoints and 81 junctions, versus baseline 4,520 vertices, 13 branches, 11 endpoints and
5 junctions. Both have three components; frozen gap inspection sites remain disconnected.
All baseline vertex positions occur in the candidate. Length increased from 1.610 to 1.909 mm.
The extra tracing could include spurious branches; these metrics do not validate the formula.
Inspect the candidate before further experiments. No candidate dense segmentation was generated.
Comparison saved at runs/macaque-estimated-v1/baseline-comparison.json.

User visualization and subsequent saved-graph diagnostics refine the failure description:
all baseline edges (not only vertices) are retained. Added edges form 12 single-attachment
offshoot trees. 129/168 candidate branches are shorter than 5 um and entirely added, with median
nearest short-branch center separation 0.656 um. This supports clustered excess offshoots rather
than widespread replacement of the backbone. However, only two attachment sites are within
30 um geodesically of baseline endpoints. "Terminal offshoot" does not imply "near an original
endpoint." Use short-branch concentration and backbone-relative offshoot topology as diagnostics;
neither alone distinguishes artifacts from real fine vessels or justifies automatic pruning.
Script: scripts/analyze_terminal_branches.py; results: terminal-branch-analysis.json and
branch-diagnostics.csv in runs/macaque-estimated-v1. No further full-volume computation needed.

## Original branch subdivision — 2026-09-10

Completed exact saved-graph lineage in docs/PHASE2_BRANCH_SUBDIVISION.md. Twelve new interior
junctions split eight original branches into 25 descendants. The other 143 candidate branches
are entirely added; 64 of 76 new junctions are off the original backbone. All original endpoints
remain degree 1. This confirms that edge preservation coexists with branch-ID subdivision.
Branch 1 has terminal-facing splits at 72.21% and 90.88%; however only one of eight new junctions
on original terminal branches is in its terminal-most 20%. No general concentration near original
endpoints is demonstrated. Distinguish original terminals from tips in newly added offshoot trees.

Across saved accepted sample radii, the candidate exclusion distances are 4.906–16.634 um versus
baseline 31.884–36.621 um: the constant decrease dominates the scale increase. This is consistent
with extra tracing, not proof of artifact identity or the correct parameter choice. Assess the
axis/extent measurements and residual-path suppression assumptions before deriving another rule.
No new skeletonization, pruning, parameter change, or acceptance of the first estimate occurred.
