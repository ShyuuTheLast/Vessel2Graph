# Saved-graph branch subdivision — 2026-09-10

Phase 2's main deliverable is automated TEASAR parameter estimation integrated with
the packaged pipeline. This comparison diagnoses the first experimental estimate;
it does not introduce pruning or accept the estimate as validated.

## Result

New junctions do end original branch IDs early while their geometry continues under
other IDs. Eight of 13 baseline branches split: 12 new interior junctions produce
25 candidate backbone branches. All original edges and all 11 original degree-1
endpoints remain unchanged. None of the candidate branches mixes baseline and added
edges. The candidate's 168 branches consist of **25 backbone descendants + 143 entirely
added branches**. Thus the increase of 155 is **12 subdivisions + 143 added branches**.
Of 76 new junctions, 12 lie on the backbone and 64 lie off it.

The 25 backbone descendants have minimum/median/maximum lengths of
7.403/39.287/322.040 um. Combined with the earlier short-branch analysis, this means
the 129 branches below 5 um are added tracing, not tiny fragments of original paths.
The two processes coexist and should be assessed separately.

## Original branch lineage

J–E paths are oriented from their original junction toward their original endpoint;
100% is that endpoint. J–J and E–E paths retain saved orientation and have no preferred
terminal direction. E–E terminal-neighborhood statistics use the nearer endpoint.
Candidate IDs below are ordered along each oriented original path, not matched by ID.

| Original ID | Type | Length (um) | New junction positions (%) | Candidate IDs in path order | Terminal piece (um) |
|---|---|---:|---|---|---:|
| 1 | J–E | 81.164 | 72.21, 90.88 | 16 → 2 → 1 | 7.403 |
| 2 | J–E | 60.187 | none | 31 | 60.187 |
| 3 | J–J | 28.813 | 61.63 | 49 → 88 | — |
| 4 | J–J | 132.559 | 26.98 | 91 → 93 | — |
| 5 | J–E | 107.482 | 50.85 | 92 → 95 | 52.827 |
| 6 | J–E | 91.865 | none | 97 | 91.865 |
| 7 | J–J | 43.153 | none | 116 | — |
| 8 | J–E | 214.950 | 5.66 | 117 → 133 | 202.789 |
| 9 | J–E | 360.545 | 6.35, 16.82, 43.85 | 99 → 94 → 37 → 38 | 202.449 |
| 10 | J–J | 82.620 | 38.39, 52.45 | 120 → 129 → 121 | — |
| 11 | J–E | 322.040 | none | 122 | 322.040 |
| 12 | E–E | 11.712 | none | 161 | — |
| 13 | E–E | 72.640 | 45.66 | 162 → 164 | — |

## Are splits concentrated toward original terminals?

Branch 1 is a clear local example: its final split leaves a 7.403 um terminal
segment, only 9.12% of the original length. Branch 9 has the most splits (three),
but all occur in its first 44%; its last 202.449 um remain one branch.

Across seven J–E branches, the numbers of new junctions in consecutive 20%-length
bins from junction to endpoint are **3, 0, 2, 1, 1**. Each bin represents 247.647 um
of aggregate path length, giving densities **1.211, 0, 0.808, 0.404, 0.404 per
100 um**. Four J–J branches contain four splits over 287.145 um (1.393/100 um);
J–E branches contain seven over 1,238.234 um (0.565/100 um). Two E–E branches contain
one over 84.353 um (1.185/100 um).

Combining J–E and E–E branches, only **1 of 8** new interior junctions falls within
20% of its own original branch length of an endpoint. Those neighborhoods contain
21.28% of the total terminal-branch length (the E–E paths have two ends).
Inside/outside densities are 0.355/0.672 per 100 um. Within 10 um of original endpoints
there is one split; within 30 um there are two. These are direct along-path measurements,
not Euclidean proximity or distance through another original branch.

The saved graphs support localized terminal-facing subdivision, but do not demonstrate
a general concentration there. Counts are small and clustered; these are descriptive
comparisons, not independent samples or a statistical significance claim. A skeleton
endpoint is not necessarily an anatomical vessel termination. The user's visually
identified terminal regions may also include the tips of newly added offshoot trees;
those are distinct from original baseline endpoints.

## Implication for automated parameter estimation

The first rule has not demonstrated acceptable control of extra offshoots. Do not
promote it as the automatic default or compensate with an unvalidated pruning rule.
Retain separate diagnostics for original-path subdivision, added-tree branching,
centering, and missed anatomy. Preserving the reference backbone is useful but
insufficient evidence of parameter quality; the baseline is not ground truth.

The saved configs change exclusion from `1.5 * DBF + 30000 nm` to
`3.713996 * DBF + 241.594702 nm`. At the saved accepted cross-section radii
(1.256–4.414 um), the respective limits are **31.884–36.621 um** and
**4.906–16.634 um**. At the median sampled radius they are 34.268 and 10.810 um.
Thus a larger numeric scale accompanies substantially smaller exclusion distances
in the measured range. This is consistent with more residual tracing, but does not
prove those traces are false or establish the right exclusion margin.

Next estimator work should assess local-axis/transverse-extent reliability and the
relation between transverse coverage and residual path suppression, using saved
samples and inexpensive controlled geometry before selecting another candidate.
It should derive and calibrate an explicit parameter rule, not optimize branch counts
to reproduce the baseline. No new numeric setting is validated by this analysis.
Actual saved soma and path-cap settings remain the legacy values; the first experiment
held them fixed, despite the separate proposal to disable soma handling/use no cap.
Changing those would be a distinct experiment.

## Reproduction and checks

Run `python scripts/analyze_branch_subdivision.py` with the workspace environment.
It reads only `graph.npz` and `branches.json` from the saved baseline/candidate runs.
Unique physical-coordinate mapping and exact baseline edge preservation are required;
branch tables must cover all graph edges. It validates the baseline inputs against
the frozen checkpoint SHA256 values and records/checks source hashes around analysis.
No dense volume, EDT, skeletonization, or graph modification occurs.

Outputs in `runs/macaque-estimated-v1/`:

- `branch-subdivision-analysis.json`: lineage, coordinates, degrees, arclength bins,
  exposure-adjusted terminal neighborhoods and source hashes.
- `subdivision-branches.csv`, `subdivision-junctions.csv`, `subdivision-segments.csv`:
  per-branch lineage, inspection sites, and descendant lengths.

Three synthetic tests cover terminal orientation, splitting versus off-backbone
junctions, rejection of missing reference edges, and overlapping endpoint neighborhoods.
Full suite: 20 passed; two existing Neuroglancer deprecation warnings.
