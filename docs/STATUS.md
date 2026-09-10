# Status — 2026-09-09

## Phase 2 begun — 2026-09-10

Publication authorized by owner: preserve the previous public version with the tag
legacy-before-v0.2.0 and update public main directly. Prepared using a clean clone of
public commit fbaa80e1437aaeb6b5e30d99a5ef13f6aad5c9b9, with sanitized current files
and no imported workstation history. MIGRATION.md documents replacement of src/main.py
and requirements.txt by the installed package/configuration workflow. Legacy images stay.

Showcase restoration: user wants the original project's philosophy and operation figures
preserved. README now leads with graph-based vascular analysis, three segmentation modes,
and experimental binary-only TEASAR estimation with a workflow diagram; operational
instructions remain in README and docs/QUICKSTART.md. Four original PNGs downloaded
unchanged from public commit fbaa80e1437aaeb6b5e30d99a5ef13f6aad5c9b9 and inspected.
images/README.md records provenance/hashes. Original runtime claims and automatic joining
are not presented as current behavior; original images are labeled as showcase examples.
Recommendation: retain existing public Vessel2Graph repo, preserve/tag legacy history,
apply sanitized snapshot on a branch/PR without importing local private Git history.
No publishing, remote setup, tag or branch action performed.

Public-path cleanup: machine-specific absolute paths removed from publication configs,
notes, historical metadata and helper scripts. Portable data examples use ../data/;
project artifacts use relative runs/outputs paths. Historical source-audit and baseline
recording scripts now require explicit --source / --input arguments. Absolute working
configs remain in ignored configs/*.local.yaml; default.local.yaml keeps the active viewer.
Pre-cleanup source backup: outputs/pre-publication-path-cleanup.zip (ignored).
The previous Git commit still contains old paths: use the sanitized source snapshot
for a fresh public repository, or explicitly review/sanitize history before publishing it.
Do not push existing history merely because current files pass the path scan.

GitHub preparation checkpoint: user requests wrapping up local work; mixed-caliber
experiments are reserved for Misha. Version 0.2.0 packages the current experimental
binary-only estimator. README and CLUSTER_HANDOFF.md describe setup, accepted result,
full-volume memory limits, and pending Linux/mixed-caliber validation. Direct estimator
dependencies edt/cc3d/fill-voids are now explicit; no environment dependencies upgraded.
Root config selects ignored configs/default.local.yaml, preserving the current branch
comparison locally. Fresh users copy configs/example.yaml to that path and edit it.
PHASE2_ACCEPTED_MANIFEST.json freezes hashes of accepted artifacts (data stays outside Git).
31 tests pass, pip check clean, environment synthetic tube 34 vertices/33 edges passes.
Source and wheel prepared under ignored dist/. No remote configured, no commit/push or
publication performed. Historical research notes/configs retain workstation paths;
portable usage starts from configs/example.yaml. Windows locks remain historical.

VISUAL ACCEPTANCE — user inspected the old/new branch overlays and the closed mask and
reports the 1024 nm closing result is a perfect skeleton from every observable side.
Accepted run: runs/macaque-closing-1024nm-volume-estimated-v2; its prepared mask and
measurement-derived parameters are under runs/macaque-closing-1024nm/estimation-volume-v2.
Preserve this run as the visually accepted phase-2 development reference; do not overwrite.
User explains old branch 8 occupied a thick junction region. The new skeleton correctly
repartitions that region among three adjoining branches and extends the other two
topological neighbors. Therefore absence of an exact old 8+13 merged path is NOT a
failure: deleting 12 and merging 8/13 was an incomplete expectation. Earlier geometric
comparison cautions remain historical, superseded locally by this visual assessment.
User also confirms the visible gap is gone in the closed mask: it was a narrow, sharp
local surface indentation/open cleft, rather than an enclosed cavity. Closing smooths
that defect. Their local-convexity description is observational, not a universal vessel
constraint or an implemented convexification rule. Do not enforce convexity at real
branch junctions or infer global biological accuracy from this one inspected volume.
All four TEASAR settings were derived from binary measurements. The 1024 nm preparation
radius was explicitly selected during development, not automatically estimated yet.

Viewer correction: user wants original branch colors and independently selectable branch
IDs, not solid recoloring. Root comparison config now has only OLD - by branch (hole
filled), IDs 1–13, and NEW - by branch (4 voxel closing), IDs 1–11, plus hidden current
input mask. visualization.skeleton_overlays supports mode: branch, loading each run's
branches.json and preserving the existing branch-color function. The old run is the
hole-filled-only candidate where user identified 8/12/13, not the later 320 nm closing.
Viewer layer state and all per-branch geometries verified; no science rerun.

User visually reports the 1024 nm closing skeleton looks perfect and requested overlays
to inspect changes. Root config now selects configs/macaque-skeleton-comparison.yaml:
old hole-filled candidate orange + current 1024 nm cyan visible, intermediate 320 nm
purple hidden. Current branch-colored skeleton and prepared mask are available hidden.
Viewer supports explicit visualization.skeleton_overlays (name/path/color/visible),
using saved physical coordinates with no retracing. Source geometry checked:
old 4587/4586 vertices/edges, current 4551/4550, intermediate 4583/4582.

Latest closing trial: user requested radius 3–5 voxels; used 1024 nm physical radius
(4 XY voxels, 3.2 Z voxels), footprint 207 voxels in 7x9x9 bounding box. Raw label 1
was closed, then enclosed-filled: +147,128 voxels (3.618%), zero removed, zero further
fill additions, one component preserved. THE CLOSED MASK IS BOTH THE ESTIMATION AND
TRACING INPUT, as user emphasized. Original input and all previous trials preserved.
configs/macaque-closing-1024nm.yaml produces 381 profiles: scale 2.7668182781209683,
const 9649.594701928663 nm, exponent 8, amplitude 16384. Estimation 95.257 s;
tracing 23.205 s, sampled peak RSS 10.322 GiB, minimum available 7.180 GiB.
New run runs/macaque-closing-1024nm-volume-estimated-v2: 4551 vertices, 4550 edges,
11 branches, 7 endpoints, 5 junctions, one component, 1611.473 um total length.
No branches below 10 um, no endpoint within 15 um of the user-identified old junction.
Old branch 13 maps closely to new branch 11 (98.23% within 1 um, p95 832 nm), but
old branch 8 has shifted substantially (only 26.67% within 1 um of new branch 11).
Do not claim exact reunion of old 8/13 or globally improved accuracy. Baseline proximity
within 1 um is 80.69%; baseline is imperfect and this is not an accuracy percentage.
All vertices inside prepared mask, maximum edge 483.189 nm. Frozen baseline hashes
verified by evaluation script. Root config now selects this 1024 nm trial for review.
No estimator formula or dependencies changed; closing radius remains user-directed.

Closing experiment (2026-09-10): user identifies branches 8 and 12 of the binary-only
candidate as artifacts; branch 12 is a 5.118 um spur whose junction divides 8 and 13.
These IDs belong to runs/macaque-volume-estimated-v2, not the frozen baseline.
User authorized small binary closing because defects connect to exterior background.
Added explicit estimate --closing-radius-nm, physical ball, zero padding before closing
and cropping afterward, with foreground preservation and component-count rejection.
320 nm trial from original label 1: closing adds 90,275 voxels, filling adds 3,755;
total +94,030 (2.312%); original/prepared components both 1. 322 profiles derive
scale=2.7660304136746587, const=9380.794701928668 nm, exponent=8, amplitude=16384.
Estimation took 42.335 s. Input config configs/macaque-closing-320nm.yaml;
candidate runs/macaque-closing-320nm/estimation-volume-v2/candidate.yaml.
Tracing completed after adding opt-in skeleton.crop_empty_border (zero halo retained,
physical coordinates restored). Candidate enables it; original inputs remain full-sized.
Empty-background crop reduces (450,1271,995) to (447,1219,746), offset ZYX (2,22,90).
Uncropped live trial stopped at 11.228 GiB RSS as free memory crossed the 2 GiB reserve;
saved at outputs/closing-320nm-uncropped-memory-monitor.json. Cropped trial passes normal
preflight, finishes tracing in 25.146 s, sampled peak RSS 10.320 GiB, minimum free 2.760 GiB.
scripts/run_monitored_skeleton.py now retains normal preflight and adds live limits.
New skeleton: runs/macaque-closing-320nm-volume-estimated-v2, 4583 vertices/4582 edges,
one component, 13 branches, 8 endpoints, 6 junctions. All vertices inside prepared mask;
all edges local. Nearest junction moved 9.836 um from the user-identified old junction;
a 5.076 um spur remains and old 8/13 are not reunited as one branch. This is not a
confirmed improvement. Baseline vertex proximity within 1 um falls from 95.40% to 86.88%;
proximity is not accuracy. Root viewer config now selects closing trial for user review;
previous candidate and frozen baseline preserved. See closing-evaluation.json and
baseline-evaluation.json in the new run. Pipeline code change invalidates old cache
fingerprints if computation is requested; do not recompute frozen baseline artifacts.
31 tests pass, including identical cropped/uncropped synthetic Kimimaro vertices/edges/radii
with and without boundary-touching vessels, closing exterior channels, crop-edge preservation, and rejection
of inter-component merging. Component counts cannot rule out new connections within one
component; local visual evaluation is still required. Closing remains opt-in, not an
automatically estimated preprocessing setting. scripts/evaluate_closing_candidate.py
compares the user-identified old branches geometrically after tracing becomes possible.

Previous viewer selection (superseded by the closing trial above): root config.yaml pointed
to runs/macaque/estimation-volume-v2/candidate.yaml, whose output directory is
runs/macaque-volume-estimated-v2. `vessel2graph view --config config.yaml` displays its saved
branch skeleton; input_mask is the explicitly hole-filled mask and initially hidden. Original
phase-1 config remains configs/macaque.yaml. No scientific stage rerun for this switch.

Latest user interpretation: the shortest baseline branch was an artifact. Saved graph confirms
branch 12 is shortest (11.712 um) and an isolated 31-vertex component. Its absence in the binary-only
candidate is therefore an improvement under the user's identification, not a missing-arm target;
do not optimize to restore it. Another isolated baseline piece is branch 13 (72.640 um, 168 vertices),
which the candidate follows and connects into the main skeleton rather than removing. It is not
the second-shortest branch by length (branch 3, 28.813 um, in main component). The user's second
reported unrelated-component artifact is not yet unambiguously identified. Label 2 is excluded
in both runs. See PHASE2_BINARY_ONLY.md for corrected interpretation; no parameters changed.

LATEST USER CORRECTION: useful parameters must be derived from binary input + spacing only.
An accepted skeleton is evaluation only. The previous reference-calibrated experiment does
NOT satisfy that goal; prior claims of meeting the user's target through it were overstated.
User also reports holes and explicitly permits investigating/using filling/binary extension.

Current binary-only result: docs/PHASE2_BINARY_ONLY.md. Implemented volume_estimation.py and
made it the default estimate method; --fill-holes explicitly prepares a separate mask. Macaque
enclosed filling adds 39,112 voxels, preserving one input CC; EDT max increases to 4890.873 nm.
303 physical profiles derive scale=2.959324, const=9137.594702 nm, pdrf_exponent=8, pdrf_scale=16384.
No reference graph, learned coefficient, or prior values of these four parameters are used.
Soma handling disabled, max_paths=None, dust=0. Full final-code blind replay with a nonexistent
skeleton and all four incoming parameters set to 999999 reproduces the identical proposal,
centering survey and prepared-mask hash. Evidence: outputs/binary-estimation-v2/blind-replay-verification.json.

Fresh binary-only candidate (runs/macaque-volume-estimated-v2): 4587 vertices, 4586 edges,
13 branches, 8 endpoints, 6 junctions, 1 component, 1632.153 um length; zero branches below 5 um.
95.40% of baseline vertices within 1 um; all new vertices inside prepared mask; all edges local
voxel steps. Skeleton 26.09 s; initial estimate 35.12 s, final-code replay 35.36 s. Peak RSS not
measured; 16 GiB budget/2 GiB reserve used. This is promising, not independent accuracy validation.

One ablation removes only the transverse-radius tip floor: 689 branches, 626 below 5 um,
169.59 s skeletonization. It is not preferred. Short reference branch 12 has the same 9.820 um
max discrepancy in both candidates, so the added floor alone does not explain it. Do not claim
all reference arms recovered. Overlay: outputs/binary-estimation-v2/comparison.png. Both runs
contain baseline-evaluation.json; scripts/evaluate_volume_candidate.py reproduces evaluation.
27 tests pass. New clean input config: configs/macaque-binary.yaml. Existing candidate viewer:
python -m vessel2graph view --config runs/macaque/estimation-volume-v2/candidate.yaml.
Original inputs/baseline preserved; no joining, pruning, publishing or dependency changes.
Next: inspect local discrepancy in 3D and evaluate the fixed rule on independent geometry.

Historical result (not the current goal): first reference-calibrated exclusion estimate reproduces the accepted macaque
graph BYTE FOR BYTE in a fresh skeletonization (source=null). Same 4520 vertices, 4517 edges,
13 branches, 11 endpoints, 5 junctions, 3 components and radii; zero added short branches.
This was same-dataset reference calibration, not successful binary-only estimation or transfer
validation. Details: docs/PHASE2_REFERENCE_CALIBRATION.md. Candidate directory:
runs/macaque-reference-calibrated-v1; comparison and estimator-link JSON record verification.

Packaged calibrate command learns alpha=2.477951598993 from maximum foreground distance minus
1.5*nearest-reference radius, divided by input EDT maximum. estimate --calibration applies
const=alpha*EDT_max + half voxel diagonal using only the input segmentation and model. On macaque,
const=11178.481761 nm versus hardcoded 30000; scale=1.5 and all other TEASAR settings held fixed.
Model: calibrations/macaque-reference-v1.json, with fixed context guards and training provenance.
No independent centering/scale estimator implemented in this step; legacy soma/path-cap settings
remain explicit. Commands without --calibration retain the earlier experimental heuristic.

Fresh skeleton stage 37.14 s, graph ~0.14 s. Packaged calibration 11.28 s, reference-free
estimation 8.15 s. Generated config exactly matches tested config. Conservative preflight used,
no peak RSS measured; actual resource config uses 16 GiB with 2 GiB reserve. 23 tests pass,
including application without an accessible reference graph and physical exclusion scaling.
Frozen baseline skeleton/graph/branches/metrics hashes verified. No dense run, pruning, joining,
second candidate, publication or dependency change. Next assess a frozen model on independent
geometry; do not confuse this exact training reproduction with a general estimator guarantee.

Latest source investigation: docs/PHASE2_RADIUS_SOURCE_AUDIT.md records the original radius,
class-dusting and shell-recovery workflow, installed Kimimaro Python execution, and controlled
probes. Original scripts/statistics snapshotted with hashes under reference/; originals unchanged.
One CPU EDT of saved mouse chunk (212,950,700), spacing (1280,256,256), exactly reproduces all
raw GMM class voxels using saved thresholds. Corrected Gaussian boundaries are 1929.638 and
3860.153 nm (73,015 raw class assignments change); no saved volume overwritten. Chunk input and
all three class stages share 3,585,679 foreground voxels. Parent crop coordinates remain unknown.
The large mouse input was inspected by metadata only. The original script deliberately executes
vol>0 despite unused mask_value=1; preserve all nonzero chunk labels for reproduction, while
macaque remains explicit label 1.

Known-radius probes expose an estimator assumption: forcing three GMMs onto a single 1536 nm
radius tube's erosion curve gives means 481/913/1337 nm, not three vessel populations. Keep this
workflow as experimental caliber evidence, supplement with ridge/core radius measurements, and
calibrate the interpretation. Actual float32 PDRF probes confirm exponent alone can improve ratios
while collapsing absolute thick-vessel penalties; next initialization should assess exponent and
amplitude jointly. Installed invalidation rejects array-valued scale. A spatial PDRF/exclusion
adapter needs explicit integration, not hard class splitting or global monkeypatching.

Recovered earlier iterative proposal from Downloads/caliber_skeletonization.md: scale/exponent
grid scored by topology, coverage and parsimony. This was missing from project-note detail;
source audit records it and its limitations. No new skeleton run was needed for these checks;
reruns remain authorized for informed estimator candidates. Scripts audit_radius_estimator.py
and probe_caliber_initialization.py completed; JSON/provenance in outputs/radius-estimator-audit.

Latest user direction: further skeletonization runs are authorized for developing a good automated
TEASAR estimator. The no-rerun restriction was specific to the completed graph diagnostic.
Mathematical initialization plus purposeful iterative calibration is in scope; approximately one
production skeletonization remains an efficiency aim, not a one-experiment development limit.
Existing notes contain centering/exclusion formulas and calibration strategy, but no implemented
adaptive parameter-update/stopping loop. A newly documented proposed iteration outline is in
PHASE2_PARAMETER_ESTIMATION.md. Prioritize estimator development over further descriptive graph
analysis unless a diagnostic directly informs a parameter decision.

Latest completed diagnostic: exact saved-graph branch lineage confirms 12 new interior junctions
split 8/13 original branches into 25 backbone descendants; 143 candidate branches are entirely
added, with no mixed branches. Of 76 new junctions, 64 are off the backbone. All 11 original
endpoints remain degree 1. Branch 1 splits at 72.21% and 90.88% toward its original endpoint,
leaving a 7.403 um terminal piece. Only 1/8 splitting junctions on original terminal branches
lies within the terminal-most 20% of its branch length; no general terminal concentration is
demonstrated. Backbone descendant minimum/median length is 7.403/39.287 um. See
docs/PHASE2_BRANCH_SUBDIVISION.md for all 13 lineages, exposure-adjusted counts, and limitations.
Script: scripts/analyze_branch_subdivision.py; JSON and three CSVs saved in the candidate run.
Frozen baseline graph/branch hashes verified; no skeletonization, dense computation, pruning,
or parameter change. 20 tests pass (two existing Neuroglancer warnings).

User reaffirmed that phase 2's main purpose is automated TEASAR parameter estimation integrated
into the packaged pipeline. This diagnostic informs that estimator, not a new cleanup stage.
Saved sample radii imply baseline exclusion 31.884–36.621 um versus candidate 4.906–16.634 um,
despite the larger scale. Next inspect sampling/axis reliability and residual-path suppression
with inexpensive controlled geometry before proposing another full-volume candidate. Current
rule remains experimental, not accepted as an automatic default. Phase 1 is closed; phase 2
remains unvalidated and uncommitted.

User inspected the candidate: majority of skeleton looks fine, with many tiny branches clustered
around terminal regions. Saved-skeleton analysis (scripts/analyze_terminal_branches.py) confirms
all 4,517 baseline edges are preserved exactly; 705 additional edges form 12 connected offshoot
trees, each attached at one baseline vertex. Added length is 299.37 um. Candidate median branch
length is 1.376 um versus baseline 82.620 um. 129/168 branches are below 5 um, all entirely added;
their median nearest other short-branch center distance is 0.656 um (90th percentile 1.335 um).
101 branches are below 2 um. Length thresholds are descriptive, not validated pruning rules.
Do not conflate terminal offshoot trees with proximity to original baseline endpoints: only 2/12
attachment sites are within 30 um along the baseline of an original endpoint; median distance
is 127.96 um. Original backbone preservation supports a localized offshoot failure interpretation,
not globally disrupted tracing, but baseline itself is not ground truth. No pruning or new
skeletonization performed. JSON and branch CSV are in runs/macaque-estimated-v1.

User authorized parameter generation after accepting checkpoint 9dddef5. Added experimental
estimate command, measurement report/sample artifact, candidate YAML and three estimation tests.
17 tests pass. No full-volume parameter sweep or new dependencies. Macaque measurement completed
in 21.49 s, ~3.87 GiB sampled peak RSS; proposal scale=3.713996, const=241.594702 nm, 227 sampled
cross-sections. Centering/soma/path caps held fixed. See PHASE2_PARAMETER_ESTIMATION.md for
assumptions and sampling limitations. Candidate config: runs/macaque/estimation/candidate.yaml;
candidate output: runs/macaque-estimated-v1. Phase-1 artifacts were not overwritten.
After two preflight refusals, the user freed memory to 17.14 GiB available. The single candidate
skeletonization completed in 74.02 s, followed by graph extraction in 0.16 s. Candidate: 5,225
vertices, 5,222 edges, 3 components, 90 endpoints, 81 junctions, 168 branches, 1,909,099.33 nm length.
Baseline: 4,520 vertices, 13 branches, 3 components, 1,609,731.78 nm length. All baseline vertex
positions occur in the candidate; the three frozen component-pair inspection sites remain
disconnected. Extra tracing is not proof of recovered anatomy and may be spurious branching.
scripts/compare_candidate.py wrote runs/macaque-estimated-v1/baseline-comparison.json.
No dense candidate segmentation was generated. Inspect the saved candidate in Neuroglancer;
do not accept this heuristic or launch additional full-volume candidates without a specific rationale.

## Phase-1 checkpoint — 2026-09-10

User accepted phase 1, including visualization, and authorized a local Git checkpoint only.
Final checks: 14 pytest tests pass; pip check passes. Neuroglancer emits two upstream tostring
deprecation warnings in the test suite, without failures. Scientific fingerprints now exclude
viewer/CLI/config source; effective scientific values remain fingerprinted. Old stage markers
are stale under the new format; accepted outputs are preserved and were not recomputed/restamped.
docs/PHASE1_BASELINE.json records artifact/input hashes, five junction sites, the smallest-median
branch midpoint and candidate inter-component endpoint pairs. These sites are inspection aids,
not confirmed ground truth. docs/PHASE2_EVALUATION.md defines the first comparison procedure.
Generated outputs, runs, reference copies and environments are excluded from Git. This checkpoint
does not back up data volumes. No GitHub push or phase-2 computation is authorized by this step.

## Update — 2026-09-10: radius zoom artifact

Radius image averaging mixed positive radii with zero background at coarse scales, causing
false small-radius borders. Disabled radius LocalVolume downsampling (None, maximum factor 1)
for quantitative inspection. Image remains scalar float32; saved data are unchanged. Full-resolution
viewing costs more chunk traffic at wide zoom. Radius 3D surface mesh is still not implemented;
the existing image layer displays slices, including planes in the 3D panel.

## Update — 2026-09-10: viewer shutdown

Viewer CLI now supervises an isolated child process. Parent polls with short interruptible
waits; Ctrl+C terminates only the read-only viewer worker, with a bounded kill fallback.
This avoids indefinite Event.wait on Windows and waiting on native/executor mesh work at exit.
Thirteen tests pass, including interrupt handling with a real long-lived worker process.
Existing running viewers must be restarted to use the new supervisor; scientific files unchanged.

## Update — 2026-09-10: dense radius visualization

Generated runs/macaque-radius/segmentation.h5 (float32 nm), using the existing legacy skeleton
and fixed parameters, with no new Kimimaro pass. Repro config: configs/macaque-radius.yaml.
Dense stage 42.46 s; report 16.81 s. Added radius_volume ImageLayer with continuous shader,
transparent zero background and adjustable maxRadius. Main macaque config includes it initially
hidden and preserves branch/caliber outputs. Twelve tests pass. Browser verified radius-colored
slice rendering. This is a scalar image view, not a per-radius segmentation mesh.

## Current outcome

Miniconda and the isolated `vessel2graph` environment are installed and verified natively on Windows.
No global PATH changes or standalone Python updates were made. Phase 1 package is installed
editable and has completed a full macaque run; phase 2 is now experimental (see latest entry above).

Latest implementation: src/vessel2graph, pyproject.toml, config.yaml and configs/macaque.yaml.
CLI: vessel2graph run --config config.yaml. Documentation: docs/PIPELINE.md.
Eight pytest tests pass. Saved skeleton reused unchanged (4,520 vertices / 4,517 edges),
3 components preserved, 13 branches. runs/macaque/segmentation.h5 contains uint32 branch IDs
for all 4,066,930 selected voxels, with zero nonzero voxels outside label 1.
Dense stage completed in 41.57 s; read-back report in 16.82 s. No peak RSS was measured.
Isolated loops are reported separately and excluded from ordinary branches per user direction.
Baseline limitations and differences from legacy are documented; biological accuracy and
Misha execution remain unvalidated. No automated TEASAR parameter tuning has been added.

Latest user-directed update: junctions have no branch-ID seed (branch=0); caliber seed labels
are stored separately as segmentation_label. Multiclass branch-end correction is enabled by
graph.correct_branch_ends, with snapshot-based decisions and per-node audit. It does not change
branch paths/statistics or original caliber labels. Whole-branch overrides remain omitted.
Macaque graph reports 48 corrected end nodes and 5 separately labeled junctions.
Both updated full runs completed: runs/macaque/segmentation.h5 (branch) and
runs/macaque-caliber/segmentation.h5 (caliber). Each assigns all 4,066,930 selected voxels
and has zero nonzero output outside label 1. Dense stages took ~42 s each, reports ~17 s.
The caliber run's resolved-config.yaml can be passed directly to the CLI to reproduce it.
See docs/PIPELINE.md for exact semantics.

Neuroglancer viewer is implemented: `vessel2graph view --config config.yaml`. Optional dependency
neuroglancer==2.41.2 is installed locally, core pins preserved; pip check passes. Configuration
selects layers, initial visibility and explicit branch/caliber HDF5 paths. Read-only HDF5 sources,
selected-label mask, deterministic skeleton/volume colors, physical ZYX coordinates, and diagnostic
annotations are implemented. No scientific stages run from view. Ten tests pass, including viewer
state/selection and incompatible-volume rejection. Browser verification on macaque confirmed
rendered slices/3D content, branch and caliber skeleton overlays, and corrected-node markers.
Instructions and mesh memory limitations: docs/VISUALIZATION.md. Statistical plots remain deferred.

Viewer mesh fix: h5py Dataset and SelectedMask lacked transpose(), required by Neuroglancer's
3D mesh generator. VolumeArray now serves slices lazily and materializes an array in slabs only
when meshing, with a memory preflight. Added an actual mesh-generation regression test; 11 tests
pass. Full macaque mask, branch and caliber mesh generation and all four skeleton encoders were
verified. Browser verification separately showed the caliber surface without skeleton overlays
and the radius skeleton alone. Earlier browser smoke checks had not established mesh success.
Restart an existing viewer process to load this fix; rerunning segmentation is unnecessary.

Further user-reported viewer failure: PolyMeshT complex-vertex warnings followed by process exit.
Exact native termination cause is unconfirmed; do not label it OOM or input corruption.
Viewer now disables native mesh simplification (max_quadrics_error=-1), serializes native mesh
requests globally, and resets failed generator initialization state. Input opacity raised from
0.25 to 0.5. Faulthandler enabled for future native-crash diagnostics. Eleven tests pass, including
concurrent mesh requests and slice access after meshing. Full macaque stress test requested all
16 objects with four callers (1 mask, 13 branches, 2 calibers): all returned mesh bytes, no
PolyMesh messages or exit. Browser confirmed input mask in all three slice panels and 3D.
Compared Downloads/neuroglancer_script.py: it loads a single NumPy array eagerly, transposes
ZYX to XYZ, uses [1,1,1] nm rather than actual spacing, and does not preselect all mesh objects.
The package's multi-source/preselected workload is therefore not an equivalent speed benchmark.

## Installed environment

- Manager: `.tools/miniconda3`, conda 26.7.1.
- Environment: `.tools/miniconda3/envs/vessel2graph`, Python 3.11.16 from conda-forge.
- Installer: Miniconda3-py311_26.7.1-1-Windows-x86_64.exe, official repo.anaconda.com download.
- Verified SHA256: `9022c76c3907b19131107c76c2beaae04e737d1c4c260e50b9398b655bfd28c5`.
- User explicitly authorized Miniconda after declining the earlier micromamba proposal.
- Core pins in requirements-core.txt. Exact installed snapshot in locks/requirements-win-py311.txt;
  base Conda environment reconstruction in locks/conda-win-64.txt. Windows-specific locks.
- Activate with `. .\scripts\activate.ps1`; no permanent shell hook was installed.

## Verification

- Conda and Python version commands pass; activation selects the intended environment.
- All core library imports pass; `python -m pip check` reports no broken requirements.
- Tiny synthetic tube: Kimimaro 4.1.2 produces 34 vertices / 33 edges.
- Initial test with automatically resolved cloud-volume 12.14.4 failed in skeleton merging through
  osteoid (uint32/int64 casting). Pinning upstream cloud-volume 10.4.0 resolved it. Preserve that pin.
- `outputs/environment-check.json` records the runtime, package versions and HDF5 metadata.
- `outputs/macaque-input-summary.json` records bounded-memory input label counts.
- Real-data legacy baseline now run; see the section below. No accuracy benchmark or phase-2
  computation has run.

## Macaque input

`data\macaque_mpi_176-240nm_bv_gt_cc.h5`

- HDF5 dataset `main`: shape (450,1271,995) ZYX, uint8, gzip.
- Uncompressed: 569,090,250 bytes (~543 MiB), despite a ~1.6 MB compressed file.
- Label counts: 0=564,860,982; 1=4,066,930; 2=162,338.
- Confirmed spacing (320,256,256) nm ZYX.
- User confirmed label 1 is correct. Label 2 is a mistakenly included portion of an unrelated
  object; exclude it. The pipeline input contract applies after explicit label selection.

## Device / other installations

Ryzen 7 7800X3D, 8 cores / 16 threads, ~31.1 GiB usable RAM, RTX 5070 Ti with 16,303 MiB VRAM,
NVIDIA driver 591.86. Available disk space was sufficient at inventory. WSL absent and not needed for this setup.
Standalone Python 3.10.6, 3.11.9, 3.14.3 exist under AppData/Local/Python and were left unchanged.
The old the legacy Windows py launcher launcher does not detect them; absence from PATH was not absence from disk.

## Original setup checklist (superseded by implementation above)

1. Bring in the reviewed Vessel2Graph source with upstream provenance; current root Git repo is empty
   of commits/remotes. Do not accidentally overwrite upstream or publish work.
2. Implement phase 1 per PROJECT_CONTEXT.md: config, headless CLI, IO, skeleton artifacts,
   graph/branch fixes, explicit assignment baseline, provenance and meaningful tests.
3. Estimate memory before full macaque skeletonization. Test slab/ROI only when needed and record
   changed context. A large fraction of cost follows array size, not compressed size or foreground alone.
4. Run a reproducible macaque baseline selecting label 1, recording configuration and results.
5. Misha integration later: Linux environment lock, resource configuration, no cluster changes yet.

## Legacy real-data baseline (latest)

User requested testing the existing code before packaging. Local source snapshot is at
reference/Vessel2Graph/src, copied unchanged from the previously reviewed upstream commit.
scripts/run_legacy_baseline.py runs an AST-adapted main for local paths, no GUI, cluster count 2,
new skeleton generation and dense segmentation. Source functions/TEASAR settings are unchanged.
Additional baseline imports are in requirements-baseline.txt (matplotlib and headless OpenCV);
the exact pip lock has been refreshed and pip check passes.

Full macaque label-1 skeletonization completed in 31.92 s: 4,520 vertices / 4,517 edges. Raw graph
had three components; original joining creates one. Branch extraction reports 11 branches.
Skeleton NPZ, skeleton HDF5, branches PKL and stats NPZ were saved and read back successfully.
38 skeleton points lie outside the selected original mask; cause is not yet established.

Dense segmentation stopped in the memory guard at ~13.84 GiB process RSS, after 51.47 seconds
total. No dense instance volume exists. Do not report the entire pipeline as completed, or claim
an OOM occurred; the conservative guard proactively stopped it. Saved artifacts are intact.
Detailed report and preview: outputs/macaque-legacy-baseline/README.md and skeleton-preview.png.
Next real-data work should reuse the saved skeleton, not repeat skeletonization unnecessarily.
