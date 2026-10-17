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

Use Misha's Miniconda module, with the environment and caches inside the workspace.
From the repository root, create and install once:

```sh
module load miniconda
CONDA_PKGS_DIRS="$PWD/.tools/conda-pkgs" XDG_CACHE_HOME="$PWD/.tools/cache" \
  conda env create --prefix "$PWD/.tools/conda-envs/vessel2graph" -f environment.yml --yes
conda activate "$PWD/.tools/conda-envs/vessel2graph"
PIP_CACHE_DIR="$PWD/.tools/pip-cache" python -m pip install -e ".[test]" -c requirements-core.txt
python -m pip check
python scripts/check_environment.py
python -m pytest -q
```

For subsequent Bash sessions, `source scripts/activate.sh` loads the module and
activates this environment. Equivalently, run `module load miniconda` followed by
`conda activate "$PWD/.tools/conda-envs/vessel2graph"` from the repository root.
The optional Neuroglancer viewer is installed separately with `".[view]"` using
the same constraints; its tests are skipped in a core-only environment.

Misha core environment verified on 2026-10-09: Python 3.11.17, pip check clean,
synthetic tube 34 vertices / 33 edges, 28 tests passed and 3 optional viewer tests
skipped. Observed dependencies are recorded in `locks/requirements-linux-py311.txt`
(project installed separately with `-e .`) and `locks/conda-linux-64.txt`.
Windows lockfiles remain historical, not Linux specifications. Current estimation
is CPU-only. Real-data Linux execution remains pending; select scheduler resources
for the dataset before submitting a production job. The initial development allocation
has 64 GiB; the inspected full mouse exceeds that budget even after empty-border cropping.

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

## Large-memory Slurm jobs

Tracing one connected vessel is a single in-memory Kimimaro call, so each stage is one
large-memory job, not an array. Any compute command accepts `--slurm`, which writes
`<output.directory>/slurm/<command>.sbatch` and prints the `sbatch` line, or `--submit`,
which also submits it:

```sh
python -m vessel2graph estimate --config configs/default.local.yaml --slurm
python -m vessel2graph graph --config runs/dataset/estimation-volume-v2/candidate.yaml --slurm
```

The `hpc:` config section sets partition (default `bigmem`; Misha has two 2 TB nodes),
CPUs, time and optional account. The request is `resources.max_memory_gib +
reserve_memory_gib` unless `hpc.mem_gib` is set, so raise `max_memory_gib` to what the
stage needs. Keep `hpc.time` quoted. The job runs under `/usr/bin/time -v`, so the log
ends with the measured peak RSS and wall time. Inside a job, the memory preflight now
compares against the Slurm allocation instead of host-wide free memory.

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
