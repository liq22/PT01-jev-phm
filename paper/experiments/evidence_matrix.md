# Claim–Support Matrix

## Current claims

### Evidence status at initialization

| Claim | Method location | Existing support | Missing evidence / minimum action | Boundary / competing explanation | Status |
|---|---|---|---|---|---|
| Typed decisions stay within the declared closed-set output space | Main §3.1, §4.4; `typed_decisions` | Numerical/interface unit tests | Actual PHM/Qwen export and prediction smoke | Structural property, also true of ordinary classification; not diagnostic correctness or standalone novelty | supported at interface level |
| Unit calibration estimates the declared unit-uniform objective | Main §3.2, §4.3 | Exact covariance identity; equal-count and duplication tests | Audit whether units/counts match the intended deployment population | Pooled calibration is appropriate for a window-weighted target; equal counts give the same method | analytical identity supported |
| Unit calibration improves held-out probability quality beyond pooled scaling | Main §4.3, §5 | None; pre-result hypothesis | Required E1 in `jev_goal.md`: identical logits, distinct calibration weighting, paired test-unit NLL | Sample variance or no objective mismatch may eliminate the benefit | planned, unverified |
| Qwen representation contributes beyond the same numeric state | Main §4.2, §5 | None; pre-result hypothesis | Required E1: numeric/Qwen heads, equal search opportunities and identical downstream calibration | Numeric features may be sufficient; six features may discard key signal information | planned, unverified |
| Better calibrated probabilities yield useful selective decisions | Main §3.3, §4.4 | Only the model-relative minimum-cost rule | Required E1: held-out coverage/error and fixed scenario cost | Cost depends on probability quality and declared costs; no real maintenance savings established | planned, unverified |

## Experiment priority

### Required versus optional

E0 (required execution prerequisite): verified PHMFactory data/labels/units/configuration and one real Qwen embedding batch. E1 (required empirical evidence): one locked two-by-three comparison. No result exists yet.

Optional until motivated by a core gap: Qwen fine-tuning, generative output comparisons, additional datasets, RUL, open-set recognition, physical/temporal constraints, and motivation/overview figures. Do not add these to the current run queue. Ordinary accuracy must be identical across the three calibration variants sharing a head; any discrepancy invalidates the run.
