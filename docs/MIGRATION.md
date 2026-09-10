# Migrating from the original scripts

The original implementation is preserved by the `legacy-before-v0.2.0` Git tag.
Version 0.2 replaces the editable `src/main.py` workflow with the installed
`vessel2graph` package and configuration-driven commands.

1. Create an isolated Python 3.11 environment and install with
   `python -m pip install -e ".[test]" -c requirements-core.txt`.
2. Copy `configs/example.yaml` to `configs/default.local.yaml`. Set input HDF5 path,
   selected labels, actual ZYX spacing in nanometers, and a new output directory.
3. For binary-only initialization, run `python -m vessel2graph estimate --config config.yaml`.
   Explicit preparation uses `--fill-holes --closing-radius-nm VALUE` when appropriate.
4. Run `graph`, `segment`, or `run` with the generated candidate config. See
   [quick start](QUICKSTART.md) for example paths and commands.

Fixed settings remain available through `skeleton.teasar_params`. For existing
skeletons, use `skeleton.source`; trusted original pickle-based NPZ archives require
explicit `skeleton.format: legacy_npz`. New artifacts use portable arrays.

Disconnected vessel components are no longer joined automatically. Anisotropic spacing
is supported explicitly. Dense outputs and stage records have new filenames; old run
directories should be preserved rather than reused. View comparisons can retain the
original branch IDs of each saved graph, but IDs are not stable across separate runs.

The original operation figures are retained. Old runtime claims are not new-version
benchmarks. The binary-only estimator is experimental, with one visually accepted
macaque result; mixed-caliber and cluster validation remain pending.
