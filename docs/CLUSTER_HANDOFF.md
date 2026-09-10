# Phase-2 cluster handoff

Local development stops at the visually accepted macaque skeleton. Further mixed-caliber
evaluation is reserved for Yale Misha; no larger volume should run on this workstation.

## Accepted experiment

Original macaque label 1, spacing ZYX (320,256,256) nm. Explicit 1024 nm ball closing
(4 XY voxels / 3.2 Z voxels) and enclosed filling, then binary-only parameter estimation.
Closing adds 147,128 voxels; no subsequent filling needed; one component remains.
Derived scale 2.7668182781209683, const 9649.594701928663 nm, exponent 8, amplitude 16384.
Result: 4551 vertices, 4550 edges, 11 branches, 7 endpoints, 5 junctions, one component.
User confirms correct junction repartitioning and disappearance of the open cleft.
Local tracing: 23.205 s; sampled peak RSS 10.322 GiB. This is one development volume.

Preserved run: `runs/macaque-closing-1024nm-volume-estimated-v2`.
Prepared mask/measurements: `runs/macaque-closing-1024nm/estimation-volume-v2`.
`docs/PHASE2_ACCEPTED_MANIFEST.json` records hashes for transfer verification.
These artifacts and input data are excluded from Git; transfer separately if needed.

## Linux setup

Use the cluster-supported Conda/Python installation:

```sh
conda env create -f environment.yml
conda activate vessel2graph
python -m pip install -e ".[test]" -c requirements-core.txt
python -m pip check
python scripts/check_environment.py
python -m pytest -q
python -m pip freeze > requirements-linux-observed.txt
```

Linux execution is pending verification. Windows lockfiles are historical, not Linux
specifications. Current estimation is CPU-only. Site modules, scheduler partition/account,
and allocation sizes must be selected on Misha; no cluster job has been submitted.

Copy `configs/example.yaml` to `configs/default.local.yaml`, edit input paths, labels,
spacing, memory and a unique output directory. Mouse spacing is (1280,256,256) nm.
On an allocated compute node, start with a representative crop:

```sh
python -m vessel2graph estimate --config configs/default.local.yaml
python -m vessel2graph graph --config runs/dataset/estimation-volume-v2/candidate.yaml
```

The candidate path assumes the example output directory; the measurement report records
the actual path. Add `--fill-holes --closing-radius-nm VALUE` only when preparation is
explicitly chosen. The candidate uses the prepared mask for BOTH estimation and tracing.
Request one CPU initially (`skeleton.parallel: 1`). Budget maximum process memory plus
reserve/overhead in the scheduler allocation. Do not infer large-job memory from macaque.

## Limits and next investigation

- EDT, preparation, erosion analysis and tracing allocate full-volume arrays. The
  16-level erosion diagnostic makes multiple passes. Large closing footprints are costly.
  This is not an out-of-core or distributed estimator.
- Empty-border cropping reduces Kimimaro allocations, retains a background halo and
  restores coordinates. It does not remove interior empty space.
- Memory preflight is an estimate. The serial development monitoring script samples RSS
  and system free memory; allocation bursts can occur between samples. Use scheduler limits.
- Current sampling favors large vessels. Three sampled-radius strata balance centering
  but cannot recover thin vessels absent from the sample. Evaluate caliber coverage first.
- One global additive exclusion constant may suppress thin branches. Check per-caliber
  requirements before developing local/adaptive exclusion or reusing the costly radius estimator.
- Closing radius remains user-selected. Preserved component counts do not prove preserved
  topology within a component; strict convexity is inappropriate at real branch junctions.
- Accepted graphs are evaluation references only, never inference inputs or coefficient
  fitting targets. Reference-calibrated v1 remains historical.

Freeze this implementation for the first cluster comparison. Record input lineage,
spacing, preparation, parameter report, environment, resource use and visual findings.
Use new output directories. Dense segmentation and mixed-caliber generalization are
not validated for the accepted phase-2 skeleton.
