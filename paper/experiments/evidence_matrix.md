# Claim–Support Matrix

## Current claims

### Scientific refinement: 14 October 2026

The sole manuscript now presents the general-method formulation before the completed pilot. Existing run IDs, numerical results and E0/E1/E2 completion are unchanged. C1–C3 are pilot findings; C4–C5 are new-method hypotheses, not validated contributions.

| Claim ID | Finding | Strength | Evidence ID | Run/ref/artifact | Boundary | Status |
|---|---|---|---|---|---|---|
| TODO | Typed pilot decisions remain in the declared closed-set output space | strong | E1 | src/S04_Tests/test_jev.py; RUN-001; main Appendix C.2 | Also true of ordinary classification; not truth or standalone novelty | supported |
| TODO | Equal-bearing calibration implements its declared objective | strong | E2 | Main Appendix C.4; existing code and bearing-role table | Elementary identity; deployment prevalence unknown | supported |
| C3 | No established equal-bearing calibration increment in four pilot E1 contrasts | moderate | E3 | RUN-003; RUN-004; paper/assets/tables/jev_study_summary.json | Point differences within 0.00018 NLL and intervals include zero; almost equal counts; no equivalence | supported |
| C1 | No primary pilot Qwen probability advantage over the same numerical state | moderate | E4 | RUN-005; paper/assets/tables/jev_metrics.csv; main Section 6 | Numerical-factorized NLL 1.018337594; not a full-signal PHM ceiling or a general-method test | supported |
| TODO | Pilot conditional judgments form a normalized leaf distribution | strong | E5 | Existing test_jev.py; RUN-001; RUN-005; Appendix C.2 | Established algebra, not additional physical-concept supervision | supported |
| C1 | Stable pilot question-conditioning increment is not established | moderate | E6 | RUN-005; paper/assets/tables/jev_study_summary.json; paper/assets/tables/jev_unit_nll.csv | Delta -0.001102028; interval [-0.008577209, 0.005163370]; factor-minus-shared +0.007865619; three favorable and three unfavorable rotations per Qwen contrast | supported |
| C2 | Lower pilot ECE does not imply better AURC or useful primary coverage | moderate | E7 | RUN-005; paper/assets/tables/jev_metrics.csv; Sections 3.3 and 6 | Primary Qwen coverage 0 and cost 0.25; rotation 5 cost 0.280825 at 5.1395% coverage; no maintenance gain | supported |
| C3 | Amplitude preservation is a separate pilot observation effect | moderate | E8 | RUN-003; RUN-004; paper/assets/tables/jev_study_summary.json | Pooled NLL changes -0.073421307 numerical and -0.036992668 legacy Qwen; same windows/roles; not decomposition evidence | supported |
| TODO | A verified deterministic coarsening has observed-event likelihood; an unresolved partition cannot identify all fine probabilities | analytical | Source/derivation | Main Sections 3.2 and 4.2–4.3; Cour 2011; Fotakis 2021 | Existing coarse-label foundations; no new general theorem, cross-domain guarantee, or inheritance of their algorithmic guarantees | supported |
| C4 | Compatible-label supervision improves excluded-dataset NLL or classification error | hypothesis | Pending | Main Sections 4.2 and 7.2; existing Goal to be extended only after source semantics and runtime are verified | Compare exact-only, justified common-coarse, and ordinary semantic controls; mixed/dominant faults are not ambiguous pure faults | planned |
| C5 | Direct readout improves error–latency tradeoffs and transfers across backbone implementations | hypothesis | Pending | Main Sections 4.4 and 7.3; no matched latency run exists | Zero response tokens and O(number of candidates × embedding width) scoring do not imply end-to-end superiority; no all-defer success criterion | planned |

## Execution boundary

### Completed and pending are separate

All existing E0, E1 and E2 evaluations remain completed under [jev_goal.md](jev_goal.md). The original run ledger and compact tables remain the result sources. Existing intervals are conditional on fits/splits and rotations are overlapping, not independent confirmations. The original data and runtime were not changed during this manuscript revision.

The new semantic method is specified, not implemented or validated. Before extending the existing Goal, verify native annotation meaning, observation compatibility, physical units and target exclusion. Do not reuse E0/E1/E2 completion as evidence for C4 or C5. No new execution command is supplied as if tested.

The only new validation needs are those supporting C4–C5: a defensible source/target vocabulary and matched learning/readout comparisons. Further datasets, fine-tuning, additional losses or mechanisms are not automatically mandatory. If selected annotations are all exact equivalents, remove the granularity-specific empirical claim rather than fabricate ambiguous labels. Controlled artificial coarsening remains a sensitivity experiment, not native annotation evidence.
