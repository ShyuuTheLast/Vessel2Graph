# Project context and decisions

## Purpose and scope

Convert a clean binary vessel volume into a skeleton, graph, branch instances and useful
metrics. Sparse graph analysis drives dense voxel labels, enabling topology-based selection
and measurements beyond geometric regions of interest.

The input segmentation is assumed clean. Repairing input artifacts is outside scope. Corrections
to skeleton/graph implementation errors are within scope. Input connected components may be real
and must not be automatically connected merely because they are spatially close.

2026-09-10 user clarification supersedes the blanket repair exclusion for the current experiment:
macaque has holes; explicit filling/binary-extension investigation is authorized. Current code
uses enclosed-hole filling only, preserves originals, records voxel/component changes, and rejects
component merging. Do not silently enable broad closing or automatic vessel joining.

User subsequently authorized an explicit small closing trial for exterior-connected gaps.
The estimator now accepts --closing-radius-nm (default zero), a physical ball with padded
boundaries. The 320 nm macaque preparation is saved separately; all four parameters are
recomputed from its binary mask. Equal component counts do not prove unchanged topology
within a component. The trial completed but did not cleanly remove the spur. Opt-in
skeleton.crop_empty_border reduces actual working memory by excluding empty outer background;
it preserves one zero halo and restores original physical coordinates. See STATUS for evaluation.

The user maintains magneton and wants its configuration/stage architecture as a reference;
its neuron/organelle research methods are unrelated. Major computation is on Yale Misha.

## Confirmed datasets

Coordinates and spacing below are ZYX; physical units are nanometers.

| Dataset | Input file | Spacing |
|---|---|---|
| Macaque, initial small test | `data\macaque_mpi_176-240nm_bv_gt_cc.h5` | `(320, 256, 256)` |
| Mouse | `data\mouse_microns-phase2_256-1280nm_bv_gt_crop_v4.h5` | `(1280, 256, 256)` |

User explicitly confirmed these spacings. Filename tokens are not authoritative. The older
Vessel2Graph `(320,256,256)` setting belongs to macaque; local `(1280,256,256)` scripts belong
to the mouse work. These are not conflicting settings for one volume.

User also confirmed that **label 1 is the correct macaque input**. Label 2 is a mistakenly
included portion of an unrelated object and should be excluded. This illustrates the project's
motivation: geometric proximity should not cause topologically unrelated vessel voxels to be
selected together. It does not authorize a new general input-artifact repair stage.

`chunk2_dist_seg.h5` is the active input in the local radius script. Do not assume its precise
derivation from the larger mouse file without checking lineage. Existing code uses HDF5 key
`main`; dataset keys/shapes still require inspection. Neuroglancer export converts ZYX to XYZ.
2026-09-10 audit verified chunk shape (212,950,700), 36 positive input labels, and identical
foreground across chunk inputs and saved caliber outputs. The original radius estimator uses
all nonzero chunk labels. One CPU EDT exactly reproduces the saved raw GMM classes using their
recorded thresholds. Full mouse shape is (624,3570,5144); parent crop coordinates remain unknown.
See PHASE2_RADIUS_SOURCE_AUDIT.md; macaque selection remains label 1 only.

## Reviewed sources

- Vessel2Graph: https://github.com/ShyuuTheLast/Vessel2Graph
  Reviewed commit `fbaa80e1437aaeb6b5e30d99a5ef13f6aad5c9b9`.
- Magneton: https://github.com/kuan-lab/magneton
  Reviewed commit `4793158fc2cd933ff0c918d05f3f9d95271f25e5`.
- Kimimaro pinned by Vessel2Graph: 4.1.2, commit `3a289bc7116d192a8b6116dc36b1486b50f135a7`.
- Local prototypes: Desktop/bloodvessel `estimate_radii.py`, `multi_class_dusting.py`,
  `shell_recovery.py`, `to_precomputed.py`, `make_mesh.py`.
- Claude reference: `reference\caliber_skeletonization.md`.
  It is advisory, not binding; several scientific and implementation claims need correction.

## Phase 1 — authorized

Package existing functionality into an installable `vessel2graph` module. YAML configurations,
headless CLI, explicit stages, reusable functions, saved intermediate skeleton/graph/results,
resolved configuration and dependency provenance. Support fixed parameters and existing skeletons.
Make optional viewing libraries lazy imports; do not require Mayavi/Napari for computation.

Initial milestones: load/validate → skeletonize/load skeleton → graph/branches → dense assignment
→ metrics/export. Get one reproducible macaque baseline. Keep local execution first, with code
portable to Linux and configuration separating scientific settings from HPC resource settings.
Plan resume validation against inputs/settings, not just existence of a `.done` marker.

Known source issues to fix with focused tests:

- Hardcoded paths, parameter `eval`, interactive cluster-count prompt; Boolean names shadow viewers.
- Unbounded nearest-endpoint component joining and stale graph degree bookkeeping after edits.
- Branch traversal misses unbranched components, pure cycles, and directly adjacent junctions;
  stored paths omit junction endpoints, impacting length. Empty graph/path cases need handling.
- `label_branch_points` fallback assigns a radius where a class label was intended.
- `segment_volume` uses ellipsoid-surface seeds plus chamfer propagation. It is not geodesic,
  physically sampled propagation. It inherits input dtype for output labels and mutates the input.
  Preserve the method as an explicitly named baseline before claiming improvements.

Distinguish branch IDs from caliber classes and graph-hop distance from physical distance.
The exact major-vessel-through-junction instance convention remains undecided. Preserve explicit
topological branch decomposition as a documented baseline, not a claim of settled biology.

## Phase 2 — experimental development authorized 2026-09-10

Use inexpensive volume measurements to calculate good starting TEASAR parameters. The user seeks
results comparable to manual tuning, with little human time and no many-pass full-volume search.
Develop/calibrate on macaque, freeze rules, then evaluate transfer to mouse. See the separate note.
The user explicitly rejected using an accepted skeleton to derive estimator coefficients. Only
binary input and physical spacing may supply inference measurements; accepted skeletons remain
evaluation references. Current implementation and evidence: docs/PHASE2_BINARY_ONLY.md.
