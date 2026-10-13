# Claim–Support Matrix

## Current claims

### Completed real-data study: 13 October 2026

| Claim ID | Finding | Strength | Evidence ID | Run/ref/artifact | Boundary | Status |
|---|---|---|---|---|---|---|
| TODO | Typed decisions remain in the declared closed-set output space | strong | E1 | src/S04_Tests/test_jev.py; RUN-001; main Sections 3.1 and 4.3 | Construction also holds for ordinary classification; not truth, calibration or standalone novelty | supported |
| TODO | Unit calibration implements the declared unit-uniform objective | strong | E2 | Main Section 3.2; src/S01_Package/jev_phm.py; paper/assets/tables/jev_unit_roles.csv | Elementary identity; deployment prevalence unknown; equal counts give identical objectives | supported |
| C3 | No established equal-bearing calibration increment in the four E1 comparisons | moderate | E3 | RUN-003; RUN-004; paper/assets/tables/jev_study_summary.json | Point differences within 0.00018 NLL and all conditional intervals include zero; five nearly equal-sized calibration bearings; no equivalence claim | supported |
| C1 | No primary Qwen probability advantage over the same numerical state | moderate | E4 | RUN-005; paper/assets/tables/jev_metrics.csv; main Section 6 | Numerical-factorized NLL 1.018337594 is below every Qwen arm; linear compact-state controls are not a mature PHM ceiling; broader superiority needs tuned application baselines | supported |
| TODO | Conditional judgments compose into a normalized leaf distribution | strong | E5 | src/S04_Tests/test_jev.py; RUN-001; RUN-005; main Section 4.3 | Established mathematics and verified construction, not probability-quality evidence or new physical-concept supervision | supported |
| C1 | Stable question-conditioned increment is not established | moderate | E6 | RUN-005; paper/assets/tables/jev_study_summary.json; paper/assets/tables/jev_unit_nll.csv | Primary delta -0.001102028, conditional interval [-0.008577209, 0.005163370]; factor-minus-shared +0.007865619; rotations three gains/three losses for each Qwen control; no equivalence or independent-rotation claim | supported |
| C2 | Probability/decision gap and unsupported useful primary Qwen coverage | moderate | E7 | RUN-005; paper/assets/tables/jev_metrics.csv; main Sections 3.3 and 6 | Primary Qwen coverage 0/cost 0.25; rotation 5 factor cost 0.280825 at 5.1395% coverage; lower ECE coexists with worse AURC; no maintenance benefit | supported |
| C3 | Amplitude restoration is a separate pooled-calibrated observation effect | moderate | E8 | RUN-003; RUN-004; paper/assets/tables/jev_study_summary.json; main Section 6 | Pooled E1 NLL decreases 0.073421307 numerical and 0.036992668 legacy Qwen at identical windows/roles; one primary split, not decomposition evidence | supported |

## Execution boundary

### Required versus conditional

E0, both E1 protocols and all seven E2 protocols completed under [jev_goal.md](jev_goal.md). Saved-logit analysis checks report agreement before generating figures. Exact cohort, roles, frozen protocol, compact metrics and per-bearing NLL are versioned in paper/assets/tables; detailed arrays and attempts remain in the existing local run directory.

E2 is implemented in the sole existing model module and study entrypoint. The empirical contribution is the completed controlled attribution and decision-boundary finding, rather than a new probability factorization or a promised diagnostic gain. E1 is not decomposition evidence. All primary intervals condition on the fitted heads and split; secondary intervals are descriptive and unadjusted for multiplicity. Refitted rotations are not six independent replications.

State enrichment, physical-symptom supervision, Score/RUL, coarse-label fallback, unseen classes, extra datasets, and generative speed comparisons remain conditional. A gain over all classical or industrial methods cannot be asserted from the current linear-only control. Existing analytical identities and synthetic tests do not count as real-data effect evidence.
