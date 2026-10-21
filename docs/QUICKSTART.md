# Installation and operation

Configuration-driven HDF5 vessel skeletonization, branch graphs, measurements and optional
dense branch/caliber labels, with saved stages, content-checked resume and Neuroglancer.

Version 0.2 adds **experimental TEASAR estimation from binary input and physical spacing**,
explicit physical-radius closing, and branch-wise skeleton comparisons. The macaque result
was visually accepted by its owner. Linux core imports and synthetic tests pass on Misha;
mixed-caliber generalization and full-volume cluster execution remain to be evaluated.

## Install

The package and command are `VesselEMPy` (case-sensitive on Linux).
Existing Vessel2Graph installations should follow [migration](MIGRATION.md).

Use an isolated Python 3.11 environment:

```sh
python -m pip install -e ".[test]" -c requirements-core.txt
python -m pip check
python scripts/check_environment.py
python -m pytest -q
```

Install `".[view]"` with the same constraints for Neuroglancer. Kimimaro 4.1.2 and
cloud-volume 10.4.0 are pinned for compatibility. `environment.yml` creates a minimal
Conda environment; Windows lockfiles are historical, not Linux specifications.

## Estimate and trace

Copy `configs/example.yaml` to `configs/default.local.yaml`, then edit the input path,
selected labels, **ZYX spacing in nanometers**, output directory and memory budget.
The root `config.yaml` selects that ignored local config. Paths resolve relative to
its YAML file. Historical macaque configs depend on the original workstation data.

```sh
python -m VesselEMPy estimate --config configs/default.local.yaml
python -m VesselEMPy graph --config runs/dataset/estimation-volume-v2/candidate.yaml
```

The generated candidate uses the prepared mask and newly estimated parameters. The path
above assumes the example output directory; the measurement report records the actual path.
Preparation is opt-in. The accepted macaque experiment used:

```sh
python -m VesselEMPy estimate --config configs/default.local.yaml --fill-holes --closing-radius-nm 1024
```

**1024 nm is a development choice, not a general default.** The closed/filled mask is
used for BOTH estimation and tracing. Original input stays intact. No accepted skeleton
is used to derive parameters. Closing-radius selection is not automated yet.

`graph` runs through skeletonization and branch extraction. `segment` adds dense assignment;
`run` executes through the report. Dense assignment requires its own evaluation and memory
budget. Use one writer and a unique output directory per experiment. `--force` recomputes
prerequisites; do not use it against preserved baselines.

## View

```sh
python -m VesselEMPy view --config runs/dataset/estimation-volume-v2/candidate.yaml
```

Open the printed URL; keep the process running. Multiple saved graphs can be displayed
with their original selectable branch IDs. See [viewer settings](VISUALIZATION.md).

## Handoff

- [Cluster handoff](CLUSTER_HANDOFF.md): installation, limits and next evaluation.
- [Binary-only estimator](PHASE2_BINARY_ONLY.md): formulas and accepted experiment.
- [Pipeline](PIPELINE.md): stages, caching and configuration.
- [Status](STATUS.md): research history and local artifacts.
- [Upstream provenance](PROJECT_CONTEXT.md): reviewed repositories and commits.

The estimator and tracing allocate full-volume arrays: **this is not an out-of-core
pipeline**. Large mixed-caliber work belongs on the cluster. Data, generated runs,
environments and local overrides are excluded from Git. Historical reference-calibrated
tools remain for reproducibility and do not satisfy the binary-only estimation goal.

