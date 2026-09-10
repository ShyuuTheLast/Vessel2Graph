# Project handoff instructions

Local phase-2 development is paused after user visual acceptance. Prepare source for
GitHub; do not publish without a requested destination. Large mixed-caliber work belongs
on Yale Misha. Read docs/CLUSTER_HANDOFF.md before resuming. Root config now selects
ignored configs/default.local.yaml; portable users copy configs/example.yaml there.
Preserve the accepted phase-2 artifacts listed in docs/PHASE2_ACCEPTED_MANIFEST.json.

Read `docs/STATUS.md`, `docs/PROJECT_CONTEXT.md`, and `docs/PHASE2_PARAMETER_ESTIMATION.md`
before making scientific or environment changes. Update STATUS after substantive work.

- The user authorized infrastructure setup and phase 1 (configuration-driven packaging).
  On 2026-09-10 the user authorized beginning phase 2 parameter generation. Keep proposals
  explicitly experimental and evaluate against the frozen phase-1 baseline.
- Latest user clarification: estimate useful TEASAR settings from a binary volume alone.
  Accepted skeletons are evaluation references only; do not derive parameters or coefficients
  from them. The reference-calibrated experiment does not satisfy this requirement.
- User reports holes in macaque and authorizes investigating/using hole filling or binary
  extension for estimation/tracing. Keep preparation explicit, preserve original masks, record
  voxel/component changes, and do not silently join disconnected vessels.
- Confirmed spacing in ZYX nanometers: macaque `(320, 256, 256)`; mouse `(1280, 256, 256)`.
  Do not infer spacing from filenames or flag the different datasets as inconsistent.
- Production computation is on Yale Misha; this Windows workstation is for development/small tests.
- An earlier micromamba download was declined. The user subsequently authorized Miniconda
  installation and project setup explicitly. Follow current STATUS for installation progress.
- Keep environments and tools local to this workspace. Do not alter global PATH, shell profiles,
  system Python, or install WSL without a concrete reason and appropriate authorization.
- Use local reference snapshots for inspection. Preserve the upstream source separately from
  refactoring and record upstream commit/version. Do not publish changes unless requested.
- Avoid automatic joining of disconnected vessel components, topology-changing cleanup, or
  expensive whole-volume sweeps as unannounced defaults.
- Phase 2 aims at measurement-derived parameters and roughly one production skeletonization;
  repeated full-volume tuning is unacceptable as the default. Hypotheses are not validated facts.
- User subsequently authorized further skeletonization runs for phase-2 estimator development
  (2026-09-10). The earlier no-rerun instruction applied to the saved-graph diagnostic only.
  Use purposeful iterative experiments to develop good automated TEASAR estimates; distinguish
  development/calibration runs from the desired low-cost production estimation workflow.
- Do not spawn subagents unless explicitly requested or required by applicable instructions.
- Current binary-only implementation: src/vessel2graph/volume_estimation.py, documented in
  docs/PHASE2_BINARY_ONLY.md. It derives all four tracing parameters from physical profiles,
  with explicit optional enclosed-hole filling. Reference-calibrated v1 is historical and
  does not satisfy the current goal. Preserve all experiments and accepted baseline artifacts.

Miniconda is installed under `.tools/miniconda3`; the isolated environment is
`.tools/miniconda3/envs/vessel2graph` (Python 3.11.16). Activate by dot-sourcing
`scripts/activate.ps1`. Core imports and a tiny Kimimaro test pass. User prefers Miniconda.
Keep cloud-volume==10.4.0 with Kimimaro 4.1.2: the newer automatically resolved version failed
the synthetic test. Consult locks and STATUS before changing dependencies.
The macaque input contains labels 1 and 2; user confirmed selection is label 1. Label 2 is an
unrelated object mistakenly included in the file. Exclude it and keep selection explicit.
Verified standalone interpreters exist at `<local-python-installations>\`
in `pythoncore-3.10-64`, `pythoncore-3.11-64`, and `pythoncore-3.14-64` (each has python.exe).
Versions are 3.10.6, 3.11.9, 3.14.3. The older system py launcher misses these installations;
absence from PATH/py discovery must not be reported as absence from the device.
A Codex-bundled Python also exists but must not become an implicit project dependency.
