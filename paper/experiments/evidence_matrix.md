# Claim–Support Matrix

Current decisions and the single next action are maintained only in [paper/STATUS.md](../STATUS.md). This file is the existing claim/artifact index, not a parallel execution plan.

## Current claims

### Scientific refinement: 14 October 2026

The sole manuscript presents the general-method formulation before the completed pilot. Existing run IDs, numerical results and E0/E1/E2 completion are unchanged. C1–C3 are pilot findings; C4–C5 are new-method hypotheses, not validated contributions.

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
| C4 | The proposed pure-site canonicalization cannot cover all native PU samples; adding one undifferentiated both-rings state still cannot deterministically produce dominant-location annotations | analytical and native-metadata | Native compatibility | [Complete criterion and proof](../method/native_annotation_compatibility.md); [native analysis output](../assets/tables/jev_native_annotation_compatibility.json); analyze_jev_annotation_compatibility.py | KB23/KB24/KB27: 3 bearings, 240 records; refutes these candidate mappings, not the existing dominant-label pilot or all possible refined ontologies; CWRU local mapping UNKNOWN | supported_scoped_counterexample |
| C4 | Compatible-label supervision improves excluded-dataset NLL or classification error | hypothesis | Pending | Main Sections 4.2 and 7.2; [STATUS E](../STATUS.md#e-唯一下一步) | Native deterministic-map condition must first be satisfied; compare exact-only, justified common-coarse, and ordinary semantic controls; no new training result | planned |
| C5 | Direct readout improves error–latency tradeoffs and transfers across backbone implementations | hypothesis | Pending | Main Sections 4.4 and 7.3; no matched latency run exists | Zero response tokens and O(number of candidates × embedding width) scoring do not imply end-to-end superiority; no all-defer success criterion | planned |

## Execution boundary

### Completed and pending are separate

All existing E0, E1 and E2 evaluations remain completed under [jev_goal.md](jev_goal.md). The original run ledger and compact tables remain the result sources. Existing intervals are conditional on fits/splits and rotations are overlapping, not independent confirmations. The original data and runtime were not changed during this compatibility analysis.

The new semantic method remains unimplemented and unvalidated in the inspected revision. The native metadata result constrains its observation-to-label likelihood; it is not a diagnostic score, signal experiment, or proof of C4 improvement. The model-specific 87-test and saved-logit checks are historical and were not repeated here. No pilot run was relabeled or excluded.

The original main Sections 3.2/4.2 retain their deterministic-partition assumptions. The [new method supplement](../method/native_annotation_compatibility.md) supplies the source-specific derivation, complete proof, observed counterexample and permissible interpretations. No new main-text performance claim, table of classifier results, figure, or PDF was generated.

If selected annotations are all exact equivalents, the granularity-specific empirical claim does not apply. Controlled artificial coarsening is a sensitivity experiment, not evidence of a naturally coarsened source. Current pending actions and budgets are exclusively in STATUS Section E.
