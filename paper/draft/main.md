# Typed Probabilistic Decisions for Selective Fault Diagnosis

## Abstract

A diagnostic interface must distinguish measured evidence, a prediction about equipment condition, and the decision to act or defer. Inspired by Jev's state-to-typed-decision interface, this study investigates these distinctions using PHMFactory data and Qwen3-0.6B. The implemented starting point is a frozen language representation of a deterministic vibration state, a supervised categorical head, and post-hoc calibration. We additionally specify a question-conditioned extension that separates fault presence from conditional fault identity and composes their outputs into one coherent distribution. Its supervision is derived from a verified fault taxonomy rather than invented symptom annotations. Matched numerical, direct-language, and factorized controls are designed to distinguish the effects of representation, decomposition, calibration, and additional computation. The calibration population is specified at the independent-unit level to avoid implicitly treating window counts as deployment prevalence. The study currently provides a prototype and a falsifiable extension design; their empirical value requires real-data evaluation.

## 1. Introduction

### 1.1 Evidence, predictions, and actions

Language models have broadened the form of industrial diagnosis. FaultGPT, for example, combines time-frequency representations and vision-language modeling to answer diagnostic questions [@chen2025faultgpt]. A generated explanation and an executable diagnostic decision are nevertheless different outputs. Software may need an allowed fault category, a complete probability vector, and a separate deferral decision. None of these requirements is equivalent to fluent text.

TypeSafe introduced Jev on 15 September 2026 as a System One model with state inputs and typed probabilistic decisions rather than unrestricted text [@typesafe2026jev]. Its official guidance assigns semantic judgments to the model and calculations, composition, and execution to code [@typesafe_skill2026]. This motivates our interface, not a claim to reproduce Jev's proprietary architecture, parallel sampler, or Reinforcement Learning for Calibrated Decisions (RLCD). Qwen3-0.6B supplies the open backbone in this study [@qwen2025].

### 1.2 The PHM-specific question

The useful transfer is not simply to attach a softmax head or rename classification as decision making. PHM must establish which signal evidence reaches the model, what each probability describes, and whether the resulting decisions outperform competitive controls on independent equipment observations. A compact state can omit diagnostically important information; a well-formed answer can be wrong; a calibrated confidence score need not remain calibrated after distribution shift or workflow composition.

We ask whether separating a bounded diagnosis into explicit subquestions and code-owned probability composition improves held-out probability quality or selective decision utility at comparable evidence, supervision, and computation. Ordinary hierarchical classification and a direct Qwen classifier are necessary controls. A second, narrower question concerns calibration population: when the intended population gives equal weight to independent units, does unit-weighted calibration improve on pooled window calibration? The second question is an evaluation and estimation issue, not the defining mechanism of Jev.

### 1.3 Intended contributions and scope

The candidate contribution is a PHM decision design that connects verified signal states, explicitly scoped diagnostic subquestions, a coherent fault distribution, and a separately specified deferral policy. Its minimal extension uses fault-presence and conditional fault-identity judgments, with a direct predictor and an identically factorized numerical predictor as controls. The intended empirical contribution is to identify whether this design offers useful gains beyond those controls, and when it fails.

The current implementation establishes a reproducible starting point, not the effectiveness of the extension. Type safety, classification heads, hierarchical factorization, temperature scaling, and rejection are existing foundations. A publishable contribution requires a meaningful, adequately supported PHM increment. The initial scope is single-channel, supervised closed-set diagnosis; it does not establish RUL prediction, previously unseen fault recognition, causal explanation, or autonomous maintenance safety.

## 2. Related Work

### 2.1 Language models for industrial diagnosis

FaultGPT uses a multi-scale cross-modal decoder, prompt learning, and diagnostic instruction supervision [@chen2025faultgpt]. Its multimodal question-answering task differs from our fixed-ontology probability estimation. Published scores are not interchangeable with results under our input representation, supervision, and split. CLSGen uses a shared language backbone with classification and explanation-generation heads [@yoon2026clsgen]. Consequently, a discriminative language-model head is not an architectural novelty here.

### 2.2 Jev and open decision-model approximations

Jev's public materials distinguish typed questions from unrestricted generation, and encourage independent judgments followed by explicit code composition [@typesafe2026jev; @typesafe_skill2026]. NanoJev already uses Qwen3-0.6B with question/candidate-conditioned decision heads and direct probability outputs. Its documented maze example separates local Boolean judgments from a code planner [@nanojev2026]. These are closer architectural and workflow references than generic chat systems.

We do not claim the first small-language-model decision head, dynamic candidate interface, or atomic-judgment workflow. NanoJev's gameplay results also do not establish performance on vibration signals. Our first extension retains a fixed, supervised PHM taxonomy rather than claiming general-purpose inference over arbitrary unseen candidates. Batching separate question sequences is not the same as implementing a shared-prefix parallel sampler.

### 2.3 Calibration, hierarchy, and selective prediction

Temperature scaling and selective classification are established foundations [@guo2017; @geifman2017]. CAT applies calibration, including temperature and importance-weighting variants, to teacher-student domain adaptation for fault diagnosis [@forest2024cat]. Our initial probe instead fixes the supervised logits and compares calibration populations without target-domain adaptation.

Hierarchy does not automatically improve a classifier. Valmadre reports settings in which a flat softmax predictor outperforms top-down alternatives [@valmadre2022]. Goren et al. calibrate leaf probabilities, sum descendant probabilities, and use hierarchical selection rules to trade specificity against correctness [@goren2024]. Our proposed factorization changes the prediction parameterization while retaining the same leaf-or-defer output task. It does not invent hierarchical rejection or inherit a distribution-free risk guarantee.

Concept bottleneck models predict annotated concepts before labels [@koh2020]. In contrast, fault presence and fault identity in our minimal extension are derived from the same diagnostic label. They are not independently annotated physical symptoms. A later model of impact periodicity or defect-frequency evidence would require appropriate signal evidence and separate target validation before making a concept-grounding claim.

### 2.4 Lessons from community evaluations

Community Jev studies motivate controls, not PHM performance claims. The phishing benchmark reports that decomposed signals followed by supervised regression outperform a direct verdict in its setting; its subsequent controls include regex features, a held-out split, and the same questions supplied to an LLM [@phishing2026]. This contrast changes decomposition and downstream supervision, so it does not isolate a universal property of Jev.

The spam study documents improvements after restoring omitted evidence, but also acknowledges exploratory specification changes, public-corpus exposure uncertainty, and task-specific distribution-shift limitations [@spam2026]. These studies motivate equal-evidence comparisons, classical baselines, and separation of prompt development from confirmation. We do not transfer their accuracy, latency, or robustness claims to PHM.

## 3. Problem Formulation and Necessary Foundations

### 3.1 Observed state and output semantics

Let independent unit g supply vibration windows x_gi with labels y_gi in a verified K-class ontology. Files from the same bearing or dependent acquisition share a unit identifier. Training, tuning, calibration, and test partitions contain disjoint units. A file name alone does not establish independence.

The state s(x) contains only observations available at inference. Global task definitions may name all allowed categories; the individual sample's label, file identifier, dataset identity, and split assignment are not model inputs. The output is

$$
p(x)\in\Delta^{K-1},\qquad a(x)\in\{1,\ldots,K,\bot\}.
$$

Choice represents mutually exclusive alternatives. Noul represents the probability that a stated event is true; 0.5 is not medium severity and there is no separate Noul confidence field in the referenced interface. Score represents an expectation over described ordered levels, not an automatically validated health index or RUL estimate. Choice/Score confidence in the official guidance concerns distribution concentration, not permission to execute an action [@typesafe_skill2026]. Our own heads expose categorical probabilities; their maximum is not presented as Jev's proprietary confidence computation.

### 3.2 Calibration population

For fixed logits and temperature T, define the mean NLL of calibration unit g as L_g(T). Then

$$
L_g(T)=\frac{1}{n_g}\sum_i-\log p_T(y_{gi}\mid x_{gi}),\quad
R_{\mathrm{unit}}(T)=\frac{1}{G}\sum_g L_g(T),\quad
R_{\mathrm{window}}(T)=\frac{\sum_g n_gL_g(T)}{\sum_g n_g}.
$$

With covariance defined using divisor G,

$$
R_{\mathrm{window}}(T)-R_{\mathrm{unit}}(T)
=\frac{\operatorname{Cov}_g(n_g,L_g(T))}{\bar n}.
$$

Expanding the covariance proves the identity. Equal window counts make the two objectives identical. Uniform replication of all windows within one unit leaves its unit-weighted objective unchanged. These are elementary properties, not a new calibration theorem or a guarantee of held-out improvement. The correct weighting depends on the stated deployment population; pooled weighting can be appropriate.

### 3.3 Probability estimation and action cost

Log loss and Brier score are proper scoring rules [@gneiting2007]. Optimizing such an objective is not equivalent to reproducing RLCD and does not guarantee finite-sample or shifted-domain calibration. Reporting both probability quality and downstream decision behavior is necessary.

For declared cost matrix C, the model-relative decision is

$$
a^*(x)=\arg\min_{a\in\{1,\ldots,K,\bot\}}\sum_{k=1}^K C_{ak}p_k(x).
$$

The initial scenario assigns zero cost to a correct diagnosis, one to an error, and c=0.25 to deferral. It accepts the maximum-probability label only when 1-max_k p_k<c, deferring on a tie. These are normalized experimental costs, not measured maintenance economics. Deferral is a policy outcome, not an additional diagnosed fault.

## 4. Method

### 4.1 Implemented signal-state interface

PHMFactory owns reading, preprocessing, windowing, and train/validation/test assignment. The adapter consumes its window dictionaries and retained file IDs without copying a raw-data loader. Whole validation units are assigned to disjoint tuning and calibration roles.

The implemented state contains RMS, standard deviation, mean absolute value, peak-to-peak amplitude, crest factor, and standardized fourth moment. A zero RMS or standard deviation gives the corresponding undefined ratio a declared zero convention. Values use a fixed order and eight significant digits. This compact state is a feasibility representation, not a sufficient statistic for fault diagnosis.

A richer state, when justified, should preserve measured context such as sampling rate, channel, window duration, normalization, and available operating conditions. Defect-frequency claims additionally require verified speed and geometry. Missing fields remain explicitly unavailable, rather than being assigned fabricated values. Code computes known signal quantities; a language model is not used to recalculate RMS or validate a simple numeric threshold. Enrichment is a separate protocol change and must be made available to every comparison arm.

### 4.2 Implemented direct predictor

Frozen Qwen3-0.6B processes the state using a fixed non-thinking chat template, without generating answer tokens. The final non-padding hidden state h(x) feeds a regularized logistic head:

$$
z(x)=Wh(x)+b,\qquad p_T(k\mid x)=\frac{e^{z_k(x)/T}}{\sum_j e^{z_j(x)/T}}.
$$

Standardization uses training data only. Regularization is selected on tuning-unit NLL. A numerical classifier receives the same six measured values and the same regularization search. This is a supervised frozen-encoder probe, not zero-shot diagnosis, Qwen fine-tuning, or a generic question-conditioned Jev replacement.

### 4.3 Implemented calibration and deferral

The selected head is held fixed while raw, pooled-calibrated, and unit-calibrated variants are evaluated. Both calibration variants search T in [0.05,20], retain T=1 when the candidate does not improve the relevant calibration objective, and report boundary solutions. A positive scalar T preserves class argmax; ordinary accuracy and Macro-F1 must therefore agree across variants sharing the same head.

Inference stores the complete class distribution and a diagnosis-or-defer decision. The source-window identifier is retained only for evaluation. Typed validity, diagnostic correctness, probability calibration, and operational utility are assessed separately. Changing a decision cost can reuse saved probabilities, provided the state and question meanings have not changed.

### 4.4 Designed extension: two scoped diagnostic judgments

The next design applies only when the verified taxonomy contains a healthy class H and at least two mutually exclusive fault classes F. Fault presence and conditional fault identity are separate questions over the same observed state:

$$
h_F=F_{\theta_0}(s,q_F),\qquad h_L=F_{\theta_0}(s,q_L),\qquad
r(s)=\sigma(w_F^\top h_F+b_F),\qquad
u(s)=\operatorname{softmax}(W_Lh_L+b_L).
$$

The language backbone remains frozen. The first question asks whether a declared fault is present rather than the healthy condition. The second explicitly asks which fault fits, conditional on membership in F. It does not see the first answer. Both are evaluated for every test state; ground-truth test labels never select a branch. Independent question evaluation does not imply statistical independence of outputs.

Code constructs one coherent leaf distribution:

$$
p(H\mid s)=1-r(s),\qquad
p(k\mid s)=r(s)u_k(s),\quad k\in F.
$$

Its components sum to one because u is normalized. This is conditional factorization, not multiplication of unrelated marginal probabilities. The factorization may be wrong about the equipment even when internally coherent.

Training labels supply b=1[y\ne H] and the fault identity for faulty training examples. No new physical-symptom labels are implied. The unregularized per-example objective is

$$
-\log p(y\mid s)=
-\mathbf{1}_{y=H}\log(1-r)
-\mathbf{1}_{y\ne H}\bigl(\log r+\log u_y\bigr).
$$

It is ordinary leaf NLL expressed through conditional factors. The scientific intervention is the question-conditioned parameterization, not a new scoring rule. Conditional identity is trained only on faulty training examples, not on fabricated identities for healthy samples.

For the primary comparison, a single temperature is fitted to the assembled leaf distribution, just as for the flat control:

$$
\widetilde p_T(k\mid s)=
\frac{p(k\mid s)^{1/T}}{\sum_jp(j\mid s)^{1/T}}.
$$

If calibrated fault-presence or conditional-identity values are exposed, they are recomputed from this final distribution; raw factors are retained separately. Separately calibrated nodes are not presumed to produce a calibrated workflow. This extension is specified here but is not implemented by the current direct-predictor code.

## 5. Experimental Protocol

### 5.1 Existing feasibility experiment

The existing first experiment is retained: numerical versus frozen-Qwen representation, crossed with raw, pooled, and unit calibration. Its purpose is to verify real-data feasibility and quantify the narrower calibration and representation contrasts. It does not test question decomposition.

Actual data, labels, operating conditions, and independent units must be verified before prediction. The first experiment does not establish general performance across PHM datasets. A tuned linear baseline is not automatically a competitive ceiling for signal classification; a broader superiority claim requires a verified, tuned nonlinear PHM baseline under the same evidence and partition.

### 5.2 Required contrast for the designed extension

A decomposition claim requires a direct-versus-factorized contrast with a fixed state, taxonomy, label budget, and evaluation protocol. Numerical-flat and numerical-factorized controls determine whether hierarchy alone explains the difference. Qwen-flat and Qwen-factorized controls distinguish the language implementation. A two-question direct Qwen ensemble controls for the extra question evaluations; it uses fixed equal averaging, not a test-selected fusion weight.

The same regularization candidates are evaluated at the workflow level, with one shared regularization value for the two factorized heads. All arms receive the same opportunity for final leaf calibration. The flat language prompts contain the same category definitions and healthy/fault grouping. All methods use the same observations; differences in pretrained knowledge and head capacity are reported rather than described as perfectly identical models.

The extension experiment is conditional on its implementation and interface tests. The initial experiment must not be relabeled as evidence for it. Hypotheses and prompts motivated by earlier observed test errors are exploratory unless evaluated on new, untouched units.

### 5.3 Metrics, uncertainty, and costs

The primary probability endpoint is independent-unit-weighted test NLL. Brier score, ten-bin ECE, accuracy, Macro-F1, coverage, selective error, and normalized decision cost supply complementary descriptions. Comparing error at different coverages alone is misleading; compare fixed-cost utility and matched-coverage behavior. Risk-coverage analysis uses saved predictions, not new inference. ECE is descriptive, not a guarantee.

Paired unit-bootstrap intervals are conditional on the fitted models and chosen split. Windows are not independent replications, and bootstrap draws are not training seeds. Deterministic frozen-feature head fitting does not require a blanket multi-seed quota. Too few independent units limit the strength of a conclusion.

Latency claims require the same hardware, precision, batch size, warmup, and output contract, with preprocessing, encoding, decision, and any escalation costs separately visible. Two batched questions are not automatically faster than one, and no proprietary Jev speedup is inherited. A generative comparison is necessary only for a specific generation-versus-decision claim; it must also include an efficient constrained alternative rather than only verbose reasoning.

### 5.4 Interpretation and limits

A gain shared by numerical and Qwen factorization supports an ordinary hierarchy explanation, not a language-specific mechanism. A gain matched by the two-question direct ensemble can reflect extra computation. A gain that appears only after state enrichment cannot be attributed to decomposition without a same-state comparison. Equal calibration counts should recover the pooled objective. None of these outcomes warrants deleting a valid negative result.

Score-based severity, RUL, physical-symptom questions, adaptive candidate sets, coarse-label fallback, and automatic expert routing remain outside the initial experiment. Each needs appropriate targets or a changed task definition. A confidence gate does not replace engineering interlocks, and an explanation generated after a decision is not evidence that the decision was grounded. No real-data results are reported at this stage.

## References
