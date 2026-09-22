# Closest Sources and Novelty Boundary

## Load-bearing sources

### Verified scope, 2026-09-22

| Source | Material inspected | Relevant finding and present distinction |
|---|---|---|
| TypeSafe, Introducing System One Models & Jev, 2026-09-15 | Official indexed announcement; direct page retrieval failed in this environment | State -> typed probabilistic decision motivates the interface. Vendor RLCD, speed, calibration and hallucination claims are not reproduced or adopted as PHM findings. |
| Qwen3 official model card; Qwen3 Technical Report, arXiv:2505.09388 | Model card, supported Transformers version, checkpoint name, non-thinking template | Use `Qwen/Qwen3-0.6B`. Actual weights and a resolved immutable revision still require local access. The card is not PHM validation. |
| Guo et al., ICML 2017, PMLR 70:1321–1330 | Publisher metadata and abstract | Temperature scaling is prior art. The relevant control uses the same head with ordinary pooled scaling. No claim to invent calibration. |
| Geifman & El-Yaniv, arXiv:1705.08500 | Original abstract and metadata | Selective classification is prior art. We do not reuse or assert its formal risk-control guarantee. |
| Chen et al., FaultGPT, arXiv:2502.15481v1 | Original HTML, method and experimental protocol | Industrial diagnostic question answering through vision-language learning. Different evidence representation and output task; reported scores are not same-protocol baselines. |
| Forest and Fink, CAT, Sensors 2024, DOI:10.3390/s24237539 | Original publisher-indexed methods and experiment text; direct page rate-limited | PHM calibration and covariate-shift weighting already exist. CAT calibrates an adaptive teacher; our test fixes the supervised logits and specifies independent-unit rather than domain-discriminator weights. |
| Yoon et al., CLSGen, arXiv:2604.11801v1 | Original HTML Sections 3–5 | Shared LM with discriminative and generative heads already exists. Our frozen categorical head is not architectural novelty; unit-population calibration is the adaptation to test. |

## Current novelty assessment

### What the first experiment can establish

Jev-inspired output typing, a softmax head, temperature scaling, and deferral are established ingredients. The candidate contribution is a measured PHM-specific benefit from matching calibration to verified independent units, together with an honest test of whether a small language backbone adds value over numeric features. The covariance identity in the manuscript is elementary analysis, not a new general calibration theorem. Equal-size units collapse the proposed weighting to ordinary scaling.

The reviewed sources do not establish this method's efficacy. They also do not justify a first-ever claim for group-weighted calibration. CAT rules out a generic claim to introduce calibration or importance weighting into PHM. Any later first-ever clustered-calibration claim would require additional, focused evidence; the present draft makes no such claim. These sources support the comparison design, not an exhaustive absence claim. The current target venue remains unselected rather than silently assigned or downgraded.
