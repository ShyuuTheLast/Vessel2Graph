# Public source preparation

Current name: **VesselEMPy**, repository https://github.com/ShyuuTheLast/VesselEMPy.
Version 0.3 adopts this name for the Python package and command as well. Earlier
archive names and preparation records below refer to the historical releases.
See [migration](MIGRATION.md) for existing local and cluster installations.

Owner-approved publication plan: preserve public commit
fbaa80e1437aaeb6b5e30d99a5ef13f6aad5c9b9 as `legacy-before-v0.2.0`, then update `main`
directly with the sanitized package. This supersedes the branch/PR recommendation below.
The update is based on public history only; local private commits are not imported.
See MIGRATION.md for the retired script entry points and replacement commands.

The current source snapshot uses relative project/data paths. Workstation configs are
preserved as ignored `configs/*.local.yaml`; the root switchboard selects
`configs/default.local.yaml`. A new user copies `configs/example.yaml` there and edits it.
Data examples assume a user-provided `data/` directory; no dataset is bundled.

Historical helpers accept explicit inputs:

```sh
python scripts/audit_radius_estimator.py --source /path/to/radius-experiment
python scripts/record_phase1_baseline.py --input /path/to/original-volume.h5
```

The second command records the phase-1 baseline and should not be used to overwrite
the frozen record unintentionally. Historical metadata path labels were sanitized;
the hashes of scientific input and accepted output files are unchanged.

**Existing local Git history contains pre-cleanup machine paths.** The sanitized
`dist/vessel2graph-0.2.0-source.zip` contains current source without `.git`, environments,
local overrides or research data. Use that snapshot for a fresh public repository,
or separately review and sanitize history before publishing existing commits.
No history rewrite or upload has been performed.

## Recommended transition for the existing public project

Keep https://github.com/ShyuuTheLast/VesselEMPy as the canonical repository. Preserve
its public history and original figures. Before the overhaul, tag the old public commit
as a legacy checkpoint, then create a branch from that public history and copy in the
sanitized current files. Review the replacement through a pull request and publish a
versioned release after validation. Old script entry points should be retired explicitly
in that PR, with migration instructions pointing to the packaged CLI.

Do not merge this workstation's independent, path-containing Git history into the
public repository. A clean snapshot can become a new commit on the existing public
history; a new public repository is not required for privacy. No tags, branches, PRs,
remote changes or publishing operations have been performed during this preparation.

Public upstream repository URLs remain as attribution. Relative project directories
such as `runs/` and `src/` are intentionally documented. These are useful conventions,
not private filesystem locations. Historical experiment notes remain labeled as such.

Before a toolbox-paper release, resolve licensing/upstream attribution, add citation
metadata and redistributable examples, validate Linux installation, and archive the
evaluation datasets/results separately under appropriate distribution terms. Hash
manifests verify those files but do not back them up.
