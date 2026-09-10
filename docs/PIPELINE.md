# Configuration-driven pipeline

Current entry point: copy configs/example.yaml to ignored configs/default.local.yaml;
root config.yaml selects that local override. Historical macaque paths below document
phase 1. For binary-only estimation, preparation, generated candidate configs and
cluster limitations, follow README.md and docs/CLUSTER_HANDOFF.md. Optional
skeleton.crop_empty_border removes only empty outer background before tracing and
restores physical coordinates afterward. No accepted baseline should be recomputed.

## Stages and files

1. `skeleton`: load selected HDF5 labels and run Kimimaro, or import a saved skeleton.
   Writes `skeleton.npz`: vertices (physical ZYX nm), edges (zero-based vertex indices),
   radius (nm), object_id, schema_version. New archives do not require pickle.
2. `graph`: writes `graph.npz`, `branches.json`, and `metrics.json`.
3. `segment`: writes `segmentation.h5` (dataset main) and `segmentation.json`.
4. `report`: reads the dense result back in slabs, checks selection coverage, and writes report.json.

Each stage records completion, elapsed time, fingerprint, and artifact SHA256 hashes in
`<stage>.stage.json`. Input bytes, settings, upstream outputs, Python/platform, dependency
versions and scientific module source influence reuse. Changing only segmentation settings reuses
skeleton and graph. Changes to pipeline.py, io.py, graph.py or segment.py conservatively invalidate
all stages. Viewer/CLI edits do not; effective scientific configuration values are fingerprinted
separately, so configuration defaults that change those values still invalidate affected stages.
The new fingerprint format makes pre-checkpoint stage markers stale once; it does not rewrite
their provenance or require regenerating accepted baseline outputs for viewing. Resource budgets
are execution controls rather than scientific cache keys. Hashing a large input still requires
one sequential read on each invocation. Failed stages rerun; previous output files may remain
but are not treated as successful. Do not run concurrent writers into one output directory.

`resolved-config.yaml` records effective settings; `provenance.json` records the runtime.
These describe the latest invocation; stage markers describe successful stage execution.

## Configuration

See `configs/macaque.yaml` for the tested local case. `config.yaml` contains a single
`main: configs/macaque.yaml` pointer. Direct dataset YAML is also accepted. Unknown keys fail.
No current-working-directory dependency exists for paths declared inside the dataset config.

- `input.path`, `dataset`, `labels`, `voxel_size_nm`: HDF5 input, label selection and mandatory
  ZYX spacing in nanometers. Macaque label 1, spacing [320,256,256]; mouse [1280,256,256].
- `output.directory`: run directory. Keep separate directories for experimental variants.
- `skeleton.source`: null to generate, or path to an existing skeleton.
- `skeleton.format`: `arrays` for new portable NPZ; `legacy_npz` explicitly enables pickle
  loading of the old `skeletons` dictionary. Use legacy mode only for trusted files.
  Imported skeletons must already use physical ZYX nm and correspond to the configured input;
  old archives cannot verify spacing or input lineage themselves.
- `skeleton.teasar_params`: fixed Kimimaro parameters. When source is set these are recorded
  but not executed. The supplied macaque values describe the legacy run. For new datasets,
  choose values explicitly; defaults are not an automatically calibrated vessel model.
- `skeleton.parallel`: positive process count; `dust_threshold` in voxels; `fix_branching`
  and `fix_borders` Boolean. Soma thresholds default to infinity for new configurations.
- `graph.radius_clusters`: requested KMeans classes over branch median radius. Effective
  count is capped by distinct medians; classes are ordered thin (1) to thick (K).
- `graph.correct_branch_ends`: true by default. Corrects only segmentation caliber labels
  near junctions. Set false for comparison; junction caliber assignment remains active.
- `segment.attribute`: branch, label (caliber class), radius (nm), or dist_from_largest.
- `segment.method`: surface_chessboard (baseline), or surface_euclidean (physical distance).
- `segment.surface_samples`: angular samples per axis, default 100, matching legacy sampling.
- `segment.query_batch_size`, `read_slices`, `compression`: bounded query/HDF5 controls.
- `resources.max_memory_gib`, `reserve_memory_gib`: advisory process checks and available
  memory reserve. These are not OS-enforced limits. Skeletonization has a conservative
  24 bytes/voxel preflight estimate; actual Kimimaro peak depends on geometry and parallelism.
  A generation run on full macaque may require raising the default 10 GiB budget after
  checking available memory. The tested config imports a skeleton and avoids this allocation.

To generate a fresh skeleton, copy the macaque config, set `skeleton.source: null`, select
a new output directory, and review parameters/resource budget. Do not overwrite the baseline.
On Misha, create a Linux environment from requirements-core.txt, install this package, and
substitute cluster paths/resources. Windows lockfiles are not Linux environment specifications;
cluster execution has not yet been tested.

## Branch semantics and changes from legacy

Branches follow degree-2 chains between junctions or endpoints, including the endpoints in
length calculations. A standalone endpoint-to-endpoint component is retained. Attached cycles
can return to their starting junction. Isolated closed loops and isolated vertices remain in
the skeleton, are reported separately, and do not enter ordinary branch clustering. If present,
dense assignment pauses with an explanatory error rather than borrowing unrelated branch IDs.

Disconnected components are never automatically joined. Every ordinary edge is traversed once,
including edges between adjacent junctions. Junctions have branch=0 (absent ownership), and are
skipped entirely when constructing branch-ID seeds, including collisions. Branch JSON retains
the shared endpoint in each incident path for lengths/connectivity. Thus per-path radius/length
statistics still use the stored paths; they are not recalculated from projected voxel ownership.
This is a topological decomposition, not a claim that an entire biological vessel through a
junction is a single instance. Automatic component joining, whole-branch caliber overrides
and component cleanup are not enabled.

`graph.npz` preserves original node `label` and adds `segmentation_label`. Original junction
label is 0 (no unique branch class). Its segmentation label comes from the incident edge whose
immediate neighbor has the largest individual radius, using that edge's original branch class;
ties choose the smaller branch ID. This also handles adjacent junctions without cascading labels.
The old fallback bug that assigned a physical radius as a label is fixed.

For each junction, smaller-class incident paths are traversed outward and their initial nodes
inside the junction's physical radius receive the highest incident class for caliber projection.
Traversal stops at the first outside node or another junction. Every decision uses the original
branch classes, not already corrected labels. Overlapping corrections choose the highest proposed
class. All classes participate, not just the global largest. Nodes far from the ends are never
promoted merely because both ends qualify. A short branch can still lie entirely inside the
local correction regions; that is the geometric rule, not the omitted whole-branch override.
`metrics.json` records each changed node, original/segmentation classes and responsible junctions.
The `label` segmentation mode uses `segmentation_label`; branch IDs, path measurements, raw radii
and graph-hop calculations are unchanged by this correction. Radius mode continues to use raw radii.

`dist_from_largest` is one-based graph hops from any largest-caliber branch: 1 at the source,
0 for unreachable components. It is not a physical distance. Branch/class volumes use uint32
instead of inheriting uint8 input; radius output uses float32.

Dense assignment samples the same ellipsoid surfaces as legacy, retaining the last node at
seed collisions. A sparse KD-tree then queries only selected voxels. Chessboard distance is
in voxel coordinates and can cross background; this remains a geometric assignment baseline,
not geodesic propagation. Equal-distance tie selection can differ from SciPy CDT, so outputs
are not promised bitwise identical. The Euclidean option uses physical spacing and changes
the distance model. Neither option constitutes validated topology-aware dense recovery.

## Validation on 2026-09-09

Eight tests passed: branch traversal including adjacent junctions and isolated/attached cycles;
small-volume Kimimaro generation; stage reuse/invalidation and corrupt-artifact recovery;
input label exclusion; strict config validation; sparse chessboard distances against SciPy CDT,
including seed collisions with IDs above 255; multiclass non-cascading end corrections, unchanged
branch statistics/IDs, and omission of junction branch seeds even at collisions.

Full macaque import: 4,520 vertices, 4,517 edges; graph: 3 components, 13 branches, 2 caliber
classes. Dense assignment: 2,488,547 surface seeds; 4,066,930 selected voxels assigned;
zero nonzero output outside label 1. Dense stage took 41.57 s, read-back report 16.82 s.
This verifies execution and structural checks, not biological instance accuracy. The legacy
run's 11-branch merged graph used different traversal/joining and is not an equality target.
