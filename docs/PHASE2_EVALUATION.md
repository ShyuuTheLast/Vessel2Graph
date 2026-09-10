# Phase-2 starting point

Phase 1 was accepted by the user on 2026-09-10. This checkpoint freezes the implementation,
settings and identity of existing outputs, not a claim of biological segmentation accuracy.
PHASE1_BASELINE.json records SHA256 hashes of the input and all three run artifact sets.
Large outputs remain local and ignored by Git; this commit is not a data backup.

## Comparison sites

Use the physical/voxel ZYX coordinates in PHASE1_BASELINE.json to revisit:

- Nearest endpoint pairs between each of the three skeleton components. The user identified
  fragmentation as a known problem with this parameter setting. These automatically selected
  pairs are candidate sites, not confirmed missing connections and not instructions to join.
- All five junctions: assess centering, preserved outgoing arms, duplicate traces, and whether
  changes to the skeleton improve or worsen the local caliber-projection result.
- The midpoint of the branch with smallest median radius: an initial thin-branch inspection
  site, not a representative statistical sample of all thin vessels.

Use physical coordinates to compare runs; node/branch IDs will change. Keep automatic component
joining disabled. A reduced component count alone is not success: false connections also reduce it.

## First experiment

1. Establish cheap radius/caliber measurements on the selected macaque label-1 mask, using the
   confirmed 320/256/256 nm spacing. Record measurement runtime and peak memory separately.
2. Derive one explicit candidate fixed TEASAR parameter set, with its rationale and uncertainties.
   Do not silently change input selection, spacing, soma handling or path caps as part of tuning.
   Any deliberate change to those controls must be recorded as part of the experiment.
3. Run one full candidate skeletonization in a new output directory. Preserve the baseline.
4. Compare gaps, centering, missing/duplicate branches and junction structure, including the sites
   above. Record skeletonization time/memory separately from measurements and dense projection.
5. Record an accept/reject/uncertain judgment with observations, not a synthetic optimality score.
   Coverage and stability are supporting diagnostics, not sufficient evidence of correct topology.

No numeric quality threshold or ground-truth skeleton has been established. Select calibration
constants on macaque before assessing transfer to mouse. Broad repeated full-volume searches
are not the intended approach. See PHASE2_PARAMETER_ESTIMATION.md for the research hypotheses.
