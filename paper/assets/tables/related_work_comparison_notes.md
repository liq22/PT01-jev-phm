# Related-work comparison: definitions and source rationale

Date: 2026-10-17. Manuscript baseline: `dev@3e7541d08f9c4923a68a449ff3737564b7da9687`, following merge of simulation PR #14. This note is the provenance of one table, not a second manuscript or experimental plan. The English table and scientific interpretation are in `paper/draft/main.md`, Section 2.5; the typesetting source is `related_work_comparison.tex` beside this note.

## 1. What the six dimensions mean

| Dimension | Operational definition | What is not inferred |
|---|---|---|
| Supervision | Labels or predictions available to fitting/adaptation, and their granularity | More varied labels do not imply a stronger model; pretraining that is not disclosed remains unknown |
| Semantic candidates | Candidate descriptions enter the prediction rule as text/embeddings, rather than being only class-index output rows | Exact-label transfer is not unseen-fault discovery; using a language model does not automatically provide a description-conditioned head |
| Retention mechanism | An explicit operation limiting change in an earlier prediction or in within-set relative probabilities | EMA smoothing, augmentations, and calibration are not automatically the same mechanism; soft constraints and first-order projections are not exact global invariance |
| Shift evaluation | The actual population/task transition evaluated, together with target-data access | Target-labelled or unlabelled adaptation and sequential task learning are not target-free deployment; simulation is not real equipment validation |
| Probability assessment | Reported held-out proper probability scores/calibration or, partially, a direct study of probability dynamics | Cross-entropy used only in training and raw accuracy/F1 are insufficient; normalized outputs are not a calibration guarantee |
| Inference measurement | Numerical latency/throughput measured with stated execution conditions | Training time, asymptotic complexity, fewer tokens, and a qualitative efficiency statement are not measured inference gains |

Symbols: ✓ addressed in the stated scope; ○ partial, adjacent or design-only coverage, specified in the cell; ✗ absent in the fully specified evaluated architecture; — not established in the inspected source. The last does not mean impossible or unsupported by every version. No total score, rank or count of checkmarks is meaningful. All rows represent concrete methods or deployed decision systems; the two row groups distinguish diagnostic/direct-prediction methods from retention/loss methods rather than mixing in surveys.

## 2. Cell-level rationale and original sources

### GLiClass — Stepanov et al., arXiv:2508.07662v1 (2025)

Source: https://arxiv.org/pdf/2508.07662v1. Read model variants in Section 2.1 (PDF pp.2–5), training in Sections 2.2–2.3, and experiments in Section 2.4 (pp.7–10, Tables 2–6). The joint encoder uses text and candidate labels; a bi-encoder is separately specified. Thus semantic candidates are ✓, but bi-encoder caching must not inherit the joint encoder's results. Single-/multi-label training is not coarse-label likelihood. Its heterogeneous text-task evaluation is adjacent to, not a test of, equipment-domain exclusion (○). A KL reference appears in optional policy training, with the described default disabling it; this does not establish the fine-conditional retention intervention studied here (—). F1 is the main output-quality metric; out-of-sample probability calibration is not established (—). A6000 batch-one throughput, input-length and candidate-count sweeps are measured (✓). These text-task throughput values are not Jev-PHM vibration measurements.

### UniFault — Eldele et al., arXiv:2504.01373v2 (2025 edition read)

Source: https://arxiv.org/pdf/2504.01373v2. Read Sections 3.4–3.5 (PDF pp.6–7) and Sections 4–5 (pp.8–13, including Table 3 and Table 4). Self-supervised pretraining is followed by supervised target adaptation with a task classifier; that evaluated classifier is not a semantic candidate matcher (✗). Target-label adaptation is explicit (○ shift evaluation), while prediction-retention controls and held-out probability assessment are not established (—). Table 4 reports training cost, not deployment latency (—). The pretraining-exclusion wording differs: p.8 refers to excluding fine-tuning samples, whereas p.9 states exclusion of all samples from three datasets. The matrix neither certifies whole-dataset exclusion nor asserts contamination. The source's target-adaptation results are not zero-target-data results regardless of that ambiguity. An author-reported later TII edition is not treated as a verified publisher text; the cited and inspected edition remains v2.

### YOTOnet — Wang et al., arXiv:2605.04528v1 (2026)

Source: https://arxiv.org/pdf/2605.04528v1. Read Sections 2 and 3.1–3.4 (PDF pp.3–13), Tables 1/3/4 and the separate adaptation results in Tables 5–7 (pp.15–16). The evaluated model predicts common exact inner/outer fault labels through a fixed head, without textual candidates (✗). Entire datasets are held out in the zero-target-data track (✓); optional target-label adaptation is separate. Two-type diagnosis conditional on fault is not health detection or mixed-fault localization. Predictive-retention constraints, held-out calibration/proper-score evaluation, and numerical inference timing are not established in these reported comparisons (—). The row credits its real diagnostic transfer scope; it is not made weaker because it studies a different valid task. No data from this work are downloaded or evaluated here.

### Jev–HVAC — Jung, arXiv:2610.09937v1 (2026)

Source: https://arxiv.org/pdf/2610.09937v1. Read Sections 3.3–3.8 (PDF pp.16–20), calibration/results including Table 5 and Figure 3 (pp.24–26), and Section 4.7 (p.31). This is an inference study; proprietary pretraining supervision is undisclosed, not known absent. Textual answer alternatives are used (✓). Empirical season/control-system/building changes are assessed (✓), with source training, prompt-development and calibration access stated separately. Brier/ECE and reliability analysis are reported (✓). **Runtime is also measured (✓):** Jev mean 0.19 s and p95 0.26 s per request; local Qwen timing uses different execution conditions and a per-window workload. These are not matched-hardware causal estimates of a particular readout's savings. This corrects the earlier blueprint's incomplete emphasis on next-token scoring while omitting Section 4.7. The source does not establish a fixed within-fault retention training mechanism (—). The paper remains cited as v1, whose submission statement is not acceptance.

### CAT — Forest and Fink, Sensors 24(23):7539 (2024)

Source: https://doi.org/10.3390/s24237539; original publisher HTML Sections 3.1–3.5 and Sections 4–5. PDF retrieval was unavailable; no PDF page number is invented. Labelled source and unlabelled target observations are used for teacher/student domain adaptation. The fixed class head is not description-conditioned (✗). Target-data adaptation is reported (○), not target-excluded inference. Temperature/other calibration choices and target ECE are evaluated (✓). An EMA teacher is not an evaluated guarantee or matched control of a fixed fine-conditional predictor (—). The inspected results do not establish numerical deployment timing (—). Its existing PHM calibration contribution is preserved; calibration is not claimed as new in Ours.

### Libra loss — Zombori et al., JMLR 25(412):1–56 (2024)

Source: https://jmlr.org/papers/volume25/23-0868/23-0868.pdf. Read Section 4.3 (Definitions 5–7 and Theorem 8) and Sections 6.3/6.5 (including Table 6). Supervision is a set containing one true label. Indexed softmax architectures do not themselves encode semantic descriptions (✗). Allowed-label probability-ratio preservation is an architecture-dependent property and is not a universal old-predictor retention guarantee (○). Probability trajectories are explicitly investigated, but that is not held-out probability calibration (○). Dataset benchmarking does not establish excluded-equipment transfer or numerical inference savings (—/—). The journal year follows the publisher record, despite differing repository deposit dates. The released weighted loss, count factor and detached probability weight remain necessary controls for the simulation.

### Learning without Forgetting — Li and Hoiem, IEEE TPAMI 40(12):2935–2947 (2018)

Publisher metadata: https://doi.org/10.1109/TPAMI.2017.2773081. Extended methods inspected in https://arxiv.org/pdf/1606.09282v3, Section 3 (PDF pp.4–5, Eqs.2–4), with the sequential visual-task evaluations. The preprint is not asserted byte-identical to the publisher PDF. New-task exact labels and old-model soft predictions on new inputs train task-specific heads, not textual candidates (✗). Distillation explicitly penalizes old-output change (✓), but is soft and sampled-input, not an exact conditional-invariance guarantee. Sequential adaptation is adjacent (○), not an unobserved diagnostic target. Distillation's training cross-entropy does not alone establish held-out probability quality; numeric deployment timing is not established in the inspected results (—/—). A conditional KL penalty in Jev-PHM therefore cannot claim to introduce predictive preservation by distillation.

### OGD — Farajtabar et al., AISTATS / PMLR 108:3762–3773 (2020)

Source: https://proceedings.mlr.press/v108/farajtabar20a/farajtabar20a.pdf. Read Section 3, Algorithm 1 and Eqs.7–9 (PDF pp.3–4), and Section 4. The evaluated task outputs are index-based rather than text-conditioned (✗). Projection on stored previous-output gradients is an explicit retention mechanism (✓); linearization, finite stored gradients and their possible staleness limit the guarantee. Exact sequential-task labels differ from the present coarse-event labels. Sequential image evaluations are adjacent (○), not real unseen-equipment diagnosis. Accuracy/forgetting evaluation and projection complexity do not establish held-out calibration or numerical deployment timing (—/—). The present fixed-semantic-head factorization result does not claim to reproduce OGD or refute it.

## 3. What Ours actually contains

The canonical merge assessed here is PR #14: `outputs/run-sim-002`, 46 features, 16-kHz sampling, four windows per virtual unit, cohorts 101/202/303, soft conditional KL and hard conditional-freeze controls. The full experiment is documented in `paper/experiments/jev_simulation_poc.md`; compact per-condition values are in `paper/assets/tables/jev_simulation_poc.csv`. This is not the earlier 16-feature `sim-20261017-001` conversation artifact. Its scores, units and selected interventions are not pooled with that artifact or with the old PU experiment.

- Supervision is exactly/coarsely labelled **simulation**, not documented industrial label coarsening.
- Semantic candidates are specified in manuscript Section 4 but not trained in the evaluated simulation (○).
- Soft and hard retention controls were executed, but source selection chose lambda=0 and additional step=0 for the soft model (○). A reduction in drift in an unselected trial is not a selected-model benefit.
- Excluded simulated operating ranges and declared stresses were evaluated; real cross-dataset transfer of this intervention was not (○).
- NLL/Brier and the NLL decomposition were computed (✓). This is coverage of probability evaluation, not a claim of universally good probabilities.
- No matched inference experiment or large-model run was executed (—). The old PU fixed-statistic pilot does not fill these gaps.

The merged report's primary NLL is 0.0021396993 with 99.9074% accuracy for four identical selected predictors; the hard control is 0.0013205430. Identical predictions yield a zero paired difference; they do not demonstrate equivalence of distinct active algorithms. Combined simulated mismatch reduces the common linear accuracy to 40.3704%. All figures and claims must retain those scope restrictions.

## 4. Gap and contribution alignment

The gap is not a missing combination of a small backbone, typed output, distillation and coarse labels. All have concrete precedents. The scoped question is whether constraining fine conditional changes provides a diagnostic-risk increment when an additional source exposes only a verified coarsening, with representation, information and selection controlled. Current support consists of an annotation-applicability condition, fixed-head preservation/expressivity boundaries, and an executed study that failed to select a nonzero soft treatment. These are qualification and negative-result findings, not a new generally superior optimization principle.

The two retained evidence-backed contributions are: (1) explicit applicability/expressivity limits that prevent invalid diagnostic label mappings or unwarranted projected-head claims; and (2) a source-selected simulation comparison distinguishing trial-path regularization from the predictor actually chosen, while retaining failure under declared stresses. Neither establishes full semantic-backbone efficacy, real target-free transfer, deployment advantage, or publication sufficiency. The sole next scientific action remains the saved-source information diagnostic, not a new training suite or restoration of CWRU.

## 5. Verification boundary

The matrix is based on original-method text and the reported evaluation sections above, not abstracts alone. The relevant UniFault experimental page, OGD algorithm and LwF method pages were inspected as rendered PDFs; other table cells use original parsed text or publisher HTML. Runtime claims from Jev–HVAC are read from Section 4.7 text, not an invented reconstruction of a figure. No experiment was rerun for this table, and no score from the eight references is treated as a same-protocol Jev-PHM baseline. The LaTeX table was compiled and its rendering inspected separately from the unavailable full-manuscript build.
