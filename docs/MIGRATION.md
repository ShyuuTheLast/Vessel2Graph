# Migrating to VesselEMPy

## Version 0.3 naming update

The repository, Python distribution, import package and command are now named
`VesselEMPy`. Use this exact capitalization for imports and commands on Linux.
The repository is https://github.com/ShyuuTheLast/VesselEMPy. Configurations,
saved graphs, branch records and volume formats are unchanged by this rename.
Version 0.3 also adds a Windows fallback for automatic worker selection. That
source change makes older stage markers stale under the existing code-hash check;
preserve completed run directories and use saved artifacts or new output paths
when running the updated stages.

From an existing checkout on Misha, finish any running jobs before updating the
editable installation. Commit or otherwise preserve local source edits first.
Then update the remote and fast-forward the checkout:

```sh
git remote set-url origin https://github.com/ShyuuTheLast/VesselEMPy.git
git pull --ff-only
source scripts/activate.sh
python -m pip uninstall --yes vessel2graph
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python -m VesselEMPy --help
VesselEMPy --help
```

This reinstall reuses the dependencies in the verified environment. For a fresh
installation, use the complete setup in [CLUSTER_HANDOFF.md](CLUSTER_HANDOFF.md)
or [LOCAL_SETUP.md](LOCAL_SETUP.md). PowerShell users activate with
`. .\scripts\activate.ps1` and then use the same pip and help commands.

| Previous usage | Current usage |
| --- | --- |
| `import vessel2graph` | `import VesselEMPy` |
| `from vessel2graph.graph import build_graph` | `from VesselEMPy.graph import build_graph` |
| `python -m vessel2graph ...` | `python -m VesselEMPy ...` |
| `vessel2graph ...` | `VesselEMPy ...` |

Update custom analysis scripts and regenerate **unsubmitted** Slurm scripts with
the current `--slurm` command. New jobs use `-m VesselEMPy` and the job-name prefix
`VesselEMPy-`. Previously generated job files are saved artifacts and are not
rewritten automatically. The retired import and command are not compatibility
aliases; removing the old distribution avoids accidentally running an old copy.

New environments use the directory name `VesselEMPy`. Both activation helpers
also recognize an existing `vessel2graph` environment in its original location.
Keep that environment and the checkout directory in place: renaming Conda
directories or moving an editable checkout can break embedded paths. A directory
name does not determine the installed package or command name. Historical run
paths, lockfiles and accepted-artifact manifests keep their original names.

## Migrating from the original scripts

The original implementation is preserved by the `legacy-before-v0.2.0` Git tag.
Version 0.2 replaced the editable `src/main.py` workflow with an installed package
and configuration-driven commands, now named `VesselEMPy` in version 0.3.

1. Create an isolated Python 3.11 environment and install with
   `python -m pip install -e ".[test]" -c requirements-core.txt`.
2. Copy `configs/example.yaml` to `configs/default.local.yaml`. Set input HDF5 path,
   selected labels, actual ZYX spacing in nanometers, and a new output directory.
3. For binary-only initialization, run `python -m VesselEMPy estimate --config config.yaml`.
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
