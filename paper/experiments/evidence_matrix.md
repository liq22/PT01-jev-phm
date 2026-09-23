# Claim–Support Matrix

## Current claims

### Status after source absorption: 23 September 2026

| Claim | Method location | Existing support | Missing evidence / minimum action | Boundary / competing explanation | Status |
|---|---|---|---|---|---|
| Typed decisions stay within the declared closed-set output space | Main Sections 3.1 and 4.3; existing typed_decisions | Interface construction and prior-PR numerical tests | E0 real PHM/Qwen export and prediction smoke | Also true of ordinary classification; not truth, calibration, or standalone novelty | Interface-level support only |
| Unit calibration matches a declared unit-uniform objective | Main Section 3.2 | Elementary objective identity; prior equal-count/replication tests | E0 verification of units and deployment population | Pooled weighting may be appropriate; equal counts give identical objectives | Analytical identity supported |
| Unit weighting improves probability quality beyond pooled scaling | Main Section 4.3 | No empirical support | Existing E1 matched-logit contrast | No objective mismatch, limited calibration units, or sampling variation can eliminate benefit | Planned secondary hypothesis |
| Qwen adds value beyond the same numerical state | Main Section 4.2 | No empirical support | Existing E1; broader utility claims also need a competitive nonlinear numerical baseline | A tuned linear classifier is not a supervised ceiling; representation may omit key evidence | Planned, unverified |
| Two scoped judgments compose into a normalized leaf distribution | Main Section 4.4 | Probability factorization, with explicit hypothetical conditioning | E2 implementation and targeted numerical/interface tests | Normalization is elementary and does not establish prediction quality; plain hierarchy has the same property | Design only; not implemented |
| Question-conditioned decomposition improves held-out decisions | Main Sections 4.4 and 5.2 | Community non-PHM studies motivate a test, not support this claim | E2 flat/factorized numerical and Qwen comparisons plus same-question-count direct control | Ordinary hierarchy, extra computation, prompt definitions or supervision may explain any apparent gain | Design-only pre-result hypothesis |
| Calibrated probabilities support useful deferral | Main Sections 3.3 and 5.3 | Model-relative cost rule only | E1 and, if implemented, E2 fixed-cost and matched-coverage evaluation | Rejection is not unknown-fault recognition, a safety guarantee or demonstrated maintenance savings | Planned, unverified |

## Execution boundary

### Required versus conditional

E0 and E1 remain the existing executable-path handoff in [jev_goal.md](jev_goal.md), subject to their unresolved real-data and model prerequisites. Their configuration and code are not changed by this revision. No new PHM or Qwen result is added.

E2 is required only to support the newly designed decomposition claim, and is **not executable by the present CLI**. Its design and implementation prerequisites are appended to the same Goal rather than creating a second run system. Do not use E1 results to claim E2 was tested.

State enrichment, physical-symptom supervision, Score/RUL, coarse-label fallback, unseen classes, extra datasets, and generative speed comparisons remain conditional. A gain over all classical or industrial methods cannot be asserted from the current linear-only control. Existing analytical identities and synthetic tests do not count as real-data effect evidence.
