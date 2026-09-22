# Typed, Unit-Calibrated Small Language Models for Selective Fault Diagnosis

## Abstract

Industrial fault diagnosis requires a decision that downstream software can interpret, together with uncertainty appropriate to the population being monitored. We investigate a Jev-inspired decision interface using a frozen Qwen3-0.6B backbone, a supervised categorical head, and an explicit deferral policy. The first implementation operates on a compact, deterministic state extracted from PHMFactory vibration windows rather than generating diagnostic prose. Its PHM-specific adaptation is to fit calibration against an explicitly declared independent-unit population instead of implicitly weighting recordings by their number of windows. A controlled comparison holds the trained logits fixed while changing only the calibration weighting. Numeric-feature classifiers receive the same tuning and calibration opportunities. We derive the relationship between window-weighted and unit-weighted calibration objectives and specify a grouped evaluation protocol for discrimination, probability quality, and selective decision cost. The empirical effectiveness of this method remains to be established; no performance improvement or deployment guarantee is asserted here.

## 1. Introduction

### 1.1 From diagnostic descriptions to executable decisions

Language-model approaches have broadened the form of industrial diagnosis. FaultGPT, for example, uses vision-language modeling and instruction supervision to produce diagnostic answers from time-frequency representations [@chen2025faultgpt]. Such a report and an executable fault decision are different products. A monitoring application may require a class from a known ontology, probabilities for every class, and an explicit decision to defer. Fluent output does not by itself establish the quality of those probabilities.

TypeSafe introduced Jev on 15 September 2026 as a System One model: state enters and typed probabilistic decisions leave, rather than unrestricted text [@typesafe2026jev]. This interface motivates the present study, but does not supply a PHM method or evidence of diagnostic reliability. We do not reproduce Jev's architecture, weights, parallel sampler, or Reinforcement Learning for Calibrated Decisions. Our open implementation uses Qwen3-0.6B as a frozen representation model [@qwen2025], with conventional supervised estimation and explicitly documented calibration.

### 1.2 The PHM-specific question

A vibration recording can produce many correlated windows. The number of retained windows need not represent how frequently its source asset will matter in deployment. When the intended population is a uniformly sampled independent unit followed by a window from that unit, pooled window calibration estimates a different objective whenever window counts and losses covary. This is an objective mismatch, not evidence that either weighting is universally correct. If deployment genuinely samples windows in proportion to recording length, the pooled objective may be preferable.

We therefore ask: under a stated independent-unit population, does a typed small-language-model decision layer provide useful diagnostic probabilities, and does unit-weighted calibration improve their held-out quality beyond ordinary temperature scaling? A second comparison asks whether the language backbone contributes anything beyond the same numeric state given to a well-tuned classifier. Both questions can be answered negatively. Making a categorical output well formed is not, by itself, the contribution.

### 1.3 Contributions and scope

The proposed contribution is a task-specific, testable integration of native categorical decisions with calibration matched to the PHM evaluation unit. The method separates a fixed diagnostic distribution from a transparent deferral rule, enabling matched comparisons without changing the backbone, input evidence, or head between calibration variants. We also provide an exact objective identity and its equal-window-count null case, which constrain what an observed improvement could mean.

The intended empirical contribution is a characterization of whether this adaptation offers a reproducible increment over pooled calibration and competitive numeric baselines. That contribution is pending experiments, not an established result. The first study concerns single-channel, closed-set fault classification. It does not establish remaining useful life prediction, unseen-fault detection, mechanistic explanations, autonomous maintenance safety, or cross-machine generalization without corresponding independent test units.

## 2. Related Work

### 2.1 Language models for industrial diagnosis

FaultGPT combines time-frequency image representations, a multi-scale cross-modal decoder, prompt learning, and instruction-based diagnostic question answering [@chen2025faultgpt]. Its output task and multimodal training differ from the present fixed-ontology decision problem. Its reported results are not interchangeable with the proposed protocol: input representations, supervision, model size, and evaluation objectives differ. We use it to establish the adjacent PHM literature, not as an unmatched numerical baseline. The present six-feature probe is deliberately narrower and may discard important fault-frequency information.

### 2.2 Probabilistic language-model heads and typed outputs

Jev motivates a decision-only interface, but its public launch description is not an independently validated PHM evaluation [@typesafe2026jev]. Discriminative heads on language models are also established. CLSGen jointly optimizes classification and explanation generation using a shared language backbone and separate heads [@yoon2026clsgen]. Our study instead freezes the backbone and omits generation. Consequently, attaching a head or returning a probability vector cannot be claimed as a new architecture. The distinction to test is the PHM sampling-unit adaptation and its measured utility, not a change of terminology from classifier to decision model.

### 2.3 Calibration and selective prediction

Calibration already appears in PHM. The Calibrated Adaptive Teacher (CAT) couples calibration with teacher-student domain adaptation for fault diagnosis, comparing temperature scaling and covariate-shift importance weighting among its calibration choices [@forest2024cat]. Our initial setting is different: a fixed supervised head, no target-domain adaptation, and weights defined by the declared independent-unit population rather than a domain discriminator. Thus neither calibration in PHM nor weighted calibration in general is a new contribution here.

Temperature scaling is an established post-hoc calibration method [@guo2017]. Selective classification explicitly trades coverage for predictive risk [@geifman2017]. Both are foundations here, not claimed inventions. Our closest executable control is the identical Qwen head with ordinary pooled temperature scaling and the identical deferral cost. A calibrated numeric classifier is necessary to test whether the language representation justifies its additional computation. A comparison only against uncalibrated outputs would not isolate the proposed increment.

## 3. Problem Formulation and Necessary Foundations

### 3.1 Data, independent units, and the output contract

Let independent unit g provide n_g vibration windows x_gi with labels y_gi in a fixed K-class ontology. Independence is a dataset property to verify, not something created by naming a file. Several files from the same bearing or acquisition must share one unit identifier when they are not independently sampled. Training, hyperparameter tuning, calibration, and test sets contain disjoint units. Labels and unit identifiers are never part of the model's textual input.

The diagnostic output is a categorical probability vector. The operational output is either a declared fault label or deferral:

$$
p(x)\in\Delta^{K-1},\qquad a(x)\in\{1,\ldots,K,\bot\}.
$$

A valid vector and an allowed action establish structural validity only. They do not establish that the label is correct, that the probabilities are calibrated, or that deferral detects unknown faults.

### 3.2 Calibration population and a useful identity

For a fixed classifier and temperature T, let L_g(T) denote the mean negative log-likelihood over calibration windows of unit g. The two empirical objectives are

$$
L_g(T)=\frac{1}{n_g}\sum_i-\log p_T(y_{gi}\mid x_{gi}),\qquad
R_{\mathrm{unit}}(T)=\frac{1}{G}\sum_gL_g(T),\qquad
R_{\mathrm{window}}(T)=\frac{\sum_g n_g L_g(T)}{\sum_g n_g}.
$$

Writing the covariance with divisor G and the mean count as bar n gives

$$
R_{\mathrm{window}}(T)-R_{\mathrm{unit}}(T)
=\frac{\operatorname{Cov}_g(n_g,L_g(T))}{\bar n}.
$$

This follows by expanding Cov_g(n_g,L_g)=G^{-1} sum_g n_g L_g - bar n G^{-1} sum_g L_g and dividing by bar n. In particular, equal counts make the objectives identical for every T. Duplicating every window of one unit changes its weight in the pooled objective but not in the unit objective. Neither identity proves that unit calibration improves held-out performance. Fitting calibration on held-out units also does not establish calibration under an arbitrary distribution shift.

### 3.3 Decision cost and limits

For a stated cost matrix C, the model-relative minimum-risk action is

$$
a^*(x)=\arg\min_{a\in\{1,\ldots,K,\bot\}}\sum_{k=1}^K C_{ak}p_k(x).
$$

The initial implementation uses a declared normalized scenario: correct classification costs zero, an incorrect classification costs one, and deferral costs c=0.25. It accepts the maximum-probability label only when 1-max_k p_k<c, deferring on a tie. These are experimental scenario costs, not measured maintenance economics. The action minimizes estimated cost under p, not necessarily the true deployment cost.

## 4. Method

### 4.1 Deterministic PHM state

PHMFactory owns reading, transforms, windowing, and its train/validation/test assignment. A narrow adapter consumes its per-window dictionaries and retained file_id, without copying a raw-data loader. The upstream validation units are assigned once to disjoint tuning and calibration roles. A user-verified table maps source files to independent units and these four roles; a conflicting assignment fails rather than falling back to random window splitting.

The first probe summarizes each univariate window by RMS, standard deviation, mean absolute value, peak-to-peak amplitude, crest factor, and standardized fourth moment. For a constant zero window the last two quantities are defined as zero. Feature names and eight-significant-digit values are serialized in a fixed order. Sampling and normalization choices remain part of the upstream protocol. This state is not claimed to preserve all diagnostic information or to encode a physical explanation.

### 4.2 Frozen representation and native categorical head

A frozen Qwen3-0.6B processes the state once using a fixed non-thinking chat template; no autoregressive response is generated. The last non-padding hidden state forms h(x). A regularized logistic head produces

$$
z(x)=Wh(x)+b,\qquad p_T(k\mid x)=\frac{\exp(z_k(x)/T)}{\sum_j\exp(z_j(x)/T)}.
$$

Feature standardization is fitted on training data only. The head is trained on training windows and its regularization is chosen on tuning units. The numeric-feature control uses the original six-dimensional state with the same preprocessing and regularization search. This is a frozen-encoder probe, not zero-shot diagnosis or Qwen fine-tuning. Its success is not assumed.

### 4.3 Unit-calibrated probability estimation

After freezing the selected head, pooled and unit temperatures minimize their respective calibration objectives. Both search T in [0.05,20] and explicitly retain T=1 when the fitted candidate does not improve its calibration objective. The unit estimator assigns window i of unit g weight 1/(G n_g). A boundary solution must be reported rather than described as a calibration guarantee.

The raw, pooled, and unit variants share exactly the same logits. Positive scalar temperature preserves class argmax, so they cannot differ in ordinary closed-set accuracy or macro-F1. They can differ in NLL, Brier score, confidence-based coverage, and the induced deferral cost. Accuracy improvements attributed to this calibration contrast would indicate an implementation or reporting error.

### 4.4 Inference and executable outputs

Inference follows the chain: PHMFactory window -> deterministic state -> frozen Qwen representation -> supervised categorical logits -> fixed temperature -> categorical probabilities -> cost-based diagnosis or deferral. Every decision stores its source-window identifier, allowed label or null, action kind, and complete categorical probabilities. The identifier supports evaluation but never enters the model input. The numeric control uses the same downstream decision code.

## 5. Experimental Protocol

### 5.1 Minimal comparisons and evidence still required

The required first comparison is a two-by-three design: numeric versus Qwen representation, crossed with raw, pooled-calibrated, and unit-calibrated probabilities. Each representation receives the same regularization candidates; calibration variants reuse its selected head. Core comparisons concern unit versus pooled calibration and each Qwen calibration variant versus its identically calibrated numeric control. No published score from a different protocol is presented as a reproduced result.

Before running, the local data audit must establish available assets, class labels, independent-unit identities, operating conditions, and disjoint role assignments. No named dataset, file path, class ontology, or PHM result is assumed to be available. The first experiment is an in-scope feasibility study, not evidence of general performance across PHM datasets.

### 5.2 Metrics and statistical units

The primary calibration endpoint is independent-unit-weighted test NLL. Brier score, ten-bin ECE, accuracy, macro-F1, coverage, selective error, and the normalized scenario cost provide complementary descriptions. ECE is a descriptive binned statistic, not a guarantee. All methods use identical test-unit weights. Raw predictions are retained for additional paired comparisons without rerunning a model.

The implementation reports paired unit-bootstrap uncertainty for the unit-minus-pooled NLL contrast, conditional on the fitted models and the selected split. It does not count windows as independent observations or equate bootstrap draws with training seeds. The frozen representation and deterministic convex head need no automatic multi-seed quota. Broad generalization or training-variability claims require additional evidence chosen before inspecting confirmatory outcomes.

### 5.3 Interpretation and boundaries

Equal calibration counts should recover the pooled method; this is a null check, not a favorable result to manufacture. A stable reduction in held-out unit-weighted NLL would support the declared population-specific calibration claim. No improvement leaves the matched pooled baseline sufficient for that setting. Failure of Qwen to compete with the numeric baseline prevents a claim that the language representation is necessary. Poor six-feature discrimination would motivate a separate representation-stage revision, not result-driven changes within the locked experiment.

No Results section is included because real PHMFactory/Qwen predictions have not yet been produced. RUL, open-set recognition, temporal or physical constraints, extra datasets, generative explanations, and elaborate figures remain outside this first experiment unless a core evidence gap makes one necessary.

## References
