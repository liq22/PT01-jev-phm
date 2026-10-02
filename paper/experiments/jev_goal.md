# Goal: First Real Jev-PHM Comparison

## Scientific purpose and existing evidence

### Required gap

Test whether independent-unit calibration improves held-out probability quality beyond ordinary pooled temperature scaling, and whether frozen Qwen adds value over the same numeric state. Without this comparison, neither empirical contribution can be claimed. Existing evidence consists only of source inspection, the objective identity, and numerical/interface tests; no real PHM/Qwen result exists. This Goal is a bounded first experiment, not a submission-ready evidence claim. Do not write Results until the actual artifacts are available.

The first run is closed-set, single-channel diagnosis with six deterministic signal features, a frozen Qwen3-0.6B representation, and an L2 logistic head. No Jev API, proprietary weights, RLCD training, agent orchestration, or fallback model is required.

## Inputs and protocol

### E0: Required local inputs, currently unresolved

Work from the project checkout, on a work branch based on `dev`. Install `src/S02_Configs/jev_requirements.txt` in an isolated environment compatible with the existing PHMFactory environment. Record the actual environment with `python -m pip freeze` in the run directory.

Use the existing local PHMFactory checkout, not a copied loader. The inspected source is `PHMbench/PHM-Vibench` at `cdc0669167ca25ac848e34c880875ca39c6aed89`. Its verified entry is `src.data_factory.build_data(args_data, args_task)`; it constructs `train_dataset`, `val_dataset`, `test_dataset`, whose items preserve `file_id`. The narrow export adapter imports this upstream entry. It does not modify it.

Before execution, the local Agent must locate actual datasets/metadata and a working resolved PHMFactory configuration. Export its complete resolved `data` and `task` objects as JSON; do not hand-invent missing data/task options. Confirm the returned item keys and pass their actual signal and scalar-label names. The first probe rejects multi-channel tensors rather than flattening them silently. Dataset names, paths, sampling rate, label ontology, normalization, window length, stride, and actual raw-data availability are **not yet verified here**.

Create `units.csv` with `file_id,unit_id,role`. Each selected source file occurs once. Roles are `train`, `tune`, `cal`, `test`; upstream train maps only to train, upstream validation only to tune/cal, upstream test only to test. Assign whole verified independent units, never windows. Files from the same bearing/acquisition that are not independent must share a unit. The table must cover exactly the upstream selected files and every role must cover the declared classes. Insufficient units/class coverage is a blocker, not permission to use random window splitting.

Create `labels.json` as the actual raw-label-to-name mapping. Dictionary order fixes class indices; all labels must have distinct names. Confirm label meanings from metadata/source documentation. No file identifier or label enters the textual state.

Resolve `Qwen/Qwen3-0.6B` to its actual 40-character Hugging Face commit. The embedding command requires that revision. We verified the official checkpoint name and API support, but did not download or execute the model here. GPU access is unverified; CPU is accepted for a very small smoke, while the full local run should use the available GPU. No fixed hardware claim is made.

### E1: Comparisons and fixed conditions

Use the complete selected first dataset/protocol, not a favorable subset chosen after prediction. Freeze `units.csv`, labels, resolved factory config, model revision, and `src/S02_Configs/jev_probe.json` before inspecting test predictions. Class counts, unit counts, windows per unit, and operating-condition assignments must be recorded first. When calibration counts are equal, declare beforehand that the unit/pooled contrast is an expected null; do not fabricate unequal counts to obtain a benefit.

Compare numeric and Qwen representations, each with raw, pooled-scaled, and unit-scaled probabilities. Both heads use training-only standardization and the same declared C grid. Choose C using tuning-unit NLL. Fit temperatures only on calibration units. Use no test labels in preprocessing, tuning, calibration, or threshold choice. The shared scenario cost is 0/1 error with deferral cost 0.25; it is not a real economic estimate.

Primary endpoint: paired unit-minus-pooled test NLL, unit-weighted. Report Brier, ten-bin ECE, accuracy, macro-F1, coverage, selective error, and scenario cost. Also compare Qwen with the equivalently calibrated numeric control using paired unit-level analysis from saved logits. Do not claim Qwen benefit from unrelated model budgets or unmatched paper scores.

Statistical unit: the verified independent asset/acquisition, not a window or a random seed. One frozen-encoder extraction and one deterministic head search per representation are sufficient for this feasibility run; no mandatory three-seed rule. Use the predeclared 2,000 paired unit-bootstrap draws, seed 20260922, for conditional NLL uncertainty. These intervals do not include training/split variability or validate uncertain independence. With too few units, report the limitation and avoid generalization claims.

## Execution

### Verified project commands and prerequisites

The following CLI parsers and numerical path were tested during the initialization PR, not rerun during the subsequent documentation revision. The real PHMFactory import/export and Qwen forward pass remain **source-checked but not end-to-end executed** in the evidence available here. The environment variables below must point to real inputs resolved in E0; they are not suggested data paths.

```bash
# From the PT01-jev-phm checkout:
python -m unittest discover -s src/S04_Tests -p test_jev.py -v
python src/S01_Package/jev_phm.py --help

# Set PHMFACTORY_ROOT, FACTORY_JSON, UNITS_CSV, LABELS_JSON,
# SIGNAL_KEY, LABEL_KEY, QWEN_REVISION and RUN_ROOT to verified local values.
# RUN_ROOT must be a NEW directory; preserve any previous failed run.
set -euo pipefail
: "${PHMFACTORY_ROOT:?}" "${FACTORY_JSON:?}" "${UNITS_CSV:?}" "${LABELS_JSON:?}"
: "${SIGNAL_KEY:?}" "${LABEL_KEY:?}" "${QWEN_REVISION:?}" "${RUN_ROOT:?}"
mkdir "$RUN_ROOT"
python -m pip freeze > "$RUN_ROOT/environment.txt"
PYTHONPATH="$PHMFACTORY_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  python src/S01_Package/jev_phm.py export \
  --factory-root "$PHMFACTORY_ROOT" --factory-json "$FACTORY_JSON" \
  --units-csv "$UNITS_CSV" --labels-json "$LABELS_JSON" \
  --signal-key "$SIGNAL_KEY" --label-key "$LABEL_KEY" \
  --out "$RUN_ROOT/data.npz" 2>&1 | tee "$RUN_ROOT/export.log"

python src/S01_Package/jev_phm.py embed \
  --data "$RUN_ROOT/data.npz" --revision "$QWEN_REVISION" \
  --device cuda --out "$RUN_ROOT/embeddings.npz" \
  2>&1 | tee "$RUN_ROOT/embed.log"

python src/S01_Package/jev_phm.py run \
  --data "$RUN_ROOT/data.npz" --embeddings "$RUN_ROOT/embeddings.npz" \
  --config src/S02_Configs/jev_probe.json --out "$RUN_ROOT/evaluation" \
  2>&1 | tee "$RUN_ROOT/evaluation.log"
```

Run each stage only after the preceding stage succeeds. For the first real model smoke, invoke `embed` on a small E0 copy that preserves all roles/classes, retain the smoke separately, and do not mix its outputs with the full run. Do not change the frozen main protocol based on smoke prediction quality. Dataset/feature or model-interface failures return to the relevant implementation step; a valid negative result is not an infrastructure failure.

### Artifacts and acceptance

Retain the resolved inputs, environment and command/code versions, data features with upstream provenance, revision-pinned embeddings, `evaluation/predictions.npz`, `metrics.json`, `config.json`, `provenance.json`, six typed decision JSONL files, and logs including failures. The prediction archive includes class logits on all roles, class order, group/role/window identifiers, scaler statistics, and linear-head coefficients. This is sufficient for post-hoc metric checks without another model call. Keep large artifacts in the established local/external storage and commit only compact summaries plus accessible references; do not publish raw data or local paths without checking their sensitivity and dataset terms.

Validate that all role units are disjoint, classes are correctly mapped, records align between features/embeddings/logits, weights sum to one, probabilities are normalized, test labels were not used for selection, and raw/pooled/unit accuracy and macro-F1 match within each head. Recompute core metrics from saved logits. Temperature boundary solutions, unavailable selective risk at zero coverage, small independent-unit counts, and convergence failures must be reported. The adapter deliberately errors on unknown labels, missing units, wrong import/version, and conflicting roles.

The equal-count and complete-unit-duplication tests establish numerical behavior only. They do not count as real PHM evidence. Do not manufacture result tables or motivation curves. Any result figures must read the saved artifacts only.

## Failure, interpretation, and synchronization

### Predeclared outcome meanings

A reproducible decrease in held-out unit-weighted NLL versus pooled scaling supports the population-specific calibration hypothesis, not universal calibration. Equivalent results leave pooled scaling sufficient in that setting. A Qwen model that is uncompetitive with the calibrated numeric control does not support language-model necessity. Either valid outcome completes the experiment; do not rerun until positive. If the state representation is inadequate, record the result and propose one separate, justified representation revision rather than silently replacing features.

Write results back to `paper/experiments/evidence_matrix.md` and the existing research state in `paper/paper.yaml`. Only after actual analysis should the manuscript acquire results and supported empirical contributions. Full publication claims still require adequate data scope and a focused closest-neighbor check of weighted/clustered calibration; no first-ever claim is authorized by this initialization.

Record failures and protocol deviations without deleting runs. Continue independent required work when one dependency blocks; stop when nothing else can run. Use a work branch and PR into `dev`, no force-push and no `master` changes. Report the actual merged revision or why the PR remains open. Do not redesign the paper, add optional experiments, or overwrite failed outputs during execution.

## Source-informed extension design

### E2: Implementation prerequisite, not a command supported by the current CLI

This section is a design handoff prompted by the 23 September source review, not permission to substitute a new protocol into E0/E1. Inspect existing local outputs first; reuse valid E1 artifacts rather than rerunning the same snapshot. Do not claim E1 tests question decomposition. The current `jev_phm.py` has no question-conditioned or factorized-head experiment command, and none is invented here.

Required scientific gap: whether splitting a PHM decision into scoped judgments helps beyond ordinary hierarchy or extra computation. It is required to support that new claim, not to complete the earlier calibration-only experiment.

Before implementation, verify a taxonomy containing healthy plus at least two mutually exclusive fault classes. A healthy/inner-race/outer-race/rolling-element example is illustrative, not an assertion that such a dataset is loaded. A binary taxonomy gives a degenerate conditional-identity problem; do not manufacture categories to avoid this null case. No fault-presence mapping may be inferred merely from file names.

Reuse the original six-feature state first. Supply all label definitions and the same healthy/fault grouping to each language arm. Numerical arms receive the same measured evidence and hierarchy; no equality of pretrained semantic knowledge is claimed. Never place a sample's ground truth, file ID or source dataset identity in the model state.

Freeze the following question templates before reading test predictions, substituting only the verified common taxonomy:

| Judgment | Fixed meaning |
|---|---|
| Presence | Using only the observed state and supplied category definitions, is a declared fault present rather than the healthy condition? |
| Conditional identity | Assuming this observation belongs to one of the supplied fault categories, which fault category best fits? This is a conditional judgment, not evidence that a fault is present. |
| Flat diagnosis A | Using only the observed state, select the best-fitting category from the complete healthy-plus-fault taxonomy and supplied definitions. |
| Flat diagnosis B | Which category in the complete supplied taxonomy is most consistent with this observed state? Use the same healthy/fault grouping and definitions. |

The presence and identity queries are independent model evaluations over the same state and do not consume each other's answers. Both run on every test state. Target presence is derived from the training label; the identity head is trained only on faulty training observations. These are diagnosis-derived targets, not observed physical symptoms.

### E2: Minimum comparisons, fitting and estimands

| Arm | Purpose |
|---|---|
| Numerical flat | Existing measured-state control; reuse compatible artifacts |
| Numerical factorized | Same numerical evidence, presence and conditional identity heads, and the same probability composition |
| Qwen flat | Frozen Qwen conditioned on flat question A and the common definitions |
| Qwen factorized | Frozen Qwen conditioned on presence/identity questions; proposed composition |
| Qwen two-question direct | Flat A/B heads; fixed 50/50 average of their class probabilities before final calibration |

Select from the existing C grid using workflow-level tuning-unit NLL. Use one shared C for the two factorized heads, avoiding a larger Cartesian-product search. The two direct heads likewise share one C. Fit every standardizer on the corresponding training observations only. Record head parameter counts and actual token/forward counts; the two-question control matches question evaluations, not necessarily exact time or FLOPs.

First assemble the complete leaf distribution. Use ordinary pooled temperature scaling as the predeclared primary calibration regime for every E2 arm, with the same single scalar T, bounds and calibration units. Keep raw distributions for the without-calibration comparison. Unit-weighted scaling is a secondary analysis only if retaining the separate population-weighting claim; do not choose a calibration regime by its test result. Do not give only the proposed method node-specific temperatures. Recompute any exposed calibrated parent or conditional probability from the final leaf distribution, and retain the uncalibrated factors separately.

Primary endpoint: paired independent-unit-weighted NLL difference between Qwen factorized and Qwen flat. Supporting contrasts are numerical factorized minus numerical flat, and Qwen factorized minus the two-question direct control. Report the same Brier, class metrics, fixed-cost decisions and coverage/error measures as E1. Reuse the declared 2,000 unit-bootstrap draws and seed for conditional paired intervals; report unit counts and do not describe nonsignificance as equivalence. One deterministic frozen-feature fit per candidate is sufficient for this pilot. If future backbone training is introduced, declare its repetition plan before outcomes instead of borrowing this assumption.

The linear arms establish a bounded comparison, not broad superiority. Before claiming practical superiority over mature PHM classifiers, identify and tune a nonlinear numerical/signal baseline using the actual PHMFactory-supported implementation and the same evidence/split. Its command and configuration must be source-checked in the implementation stage; they are unresolved here, not guessed.

### E2: Artifacts, tests and interpretation

Extend the existing implementation and output contract rather than creating a second runner. Retain observed-state fields, query text and candidate definitions, raw presence/identity scores, composed and calibrated leaf probabilities, source units/roles, fitted parameters and selection metrics. Ground-truth test labels are used only for scoring. Save all attempts and failures. A numeric field should not be called an evidence probability without a defined event and target.

Before full execution, test normalization and parent/leaf coherence, healthy-example exclusion from conditional-identity training, both-query evaluation regardless of predicted or true class, absence of sample-label leakage, reuse of the same observed state, and preservation of the declared calibration split. Validate actual CLI commands only after those capabilities exist. Structural tests are not efficacy results.

Interpret a numerical hierarchy gain as evidence for hierarchy unless a language-specific increment is separately supported. If the two-question direct control removes the advantage, do not credit decomposition instead of extra computation. If all numerical methods are poor, investigate the state representation using development data before making language-model claims. Compare matched coverage and fixed scenario costs, not simply lower error obtained by rejecting more cases.

Changing evidence fields or selecting prompts after examining E1 test errors makes the resulting analysis exploratory on those same units. Independent confirmation needs new untouched units; do not silently reuse inspected tests as confirmation. Update the existing evidence matrix and state once valid results exist, not a second claim table.

### Claim-triggered work, not an automatic expansion queue

State enrichment is warranted only when missing evidence threatens the core diagnosis question. Examples are measured sampling rate, preprocessing semantics, spectrum/envelope descriptors, and operating conditions actually available through PHMFactory. RPM and bearing geometry are prerequisites for defect-frequency interpretation, not values for the Agent to invent. Apply the same enrichment to direct, factorized and numerical controls; separate its effect from question design.

No symptom annotation is assumed. Code-derived threshold labels test recovery of a known rule and require that rule as a baseline; they are not independent physical truth. Score-based severity needs genuine ordered targets; RUL needs a suitable degradation task. A low Noul value is not low severity, and low confidence alone does not establish an unseen fault.

An efficiency claim requires end-to-end measurements on matched hardware, precision, batch and output semantics, including preprocessing, repeated question work, calibration, rejection and any expert escalation. No escalation is implemented in the current leaf-or-defer prototype. Do not import Jev's advertised speedups, price advantages or shared-prefix behavior. A generation comparison is optional unless the manuscript actually claims an advantage over generation; when required, include an efficient constrained-output baseline as well as ordinary generation.
