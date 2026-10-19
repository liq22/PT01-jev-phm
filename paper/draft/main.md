# Jev-PHM: Non-Generative Fault Diagnosis Across Datasets with Heterogeneous Annotations

## Abstract

Vibration-based diagnosis across datasets faces two distinct limitations: the meanings of fault annotations may differ, and the available measurements may not distinguish the requested states. We formulate Jev-PHM as a backbone-independent, non-generative interface that scores declared fault descriptions using a shared signal representation. Direct semantic classification and event likelihood for verified coarse annotations are established foundations, not algorithms introduced here. The empirical questions are whether correct semantic correspondence improves excluded-dataset negative log-likelihood (NLL) beyond a matched indexed readout and strong signal baselines, and whether direct scoring offers a useful error–latency trade-off relative to exact label-token scoring. A deterministic annotation-map condition excludes invalid mappings of mixed Paderborn damage to pure single-ring faults. A completed 32-bearing, six-statistic pilot finds no stable additional benefit from question factorization over direct prediction: NLL difference −0.00110, conditional 95% interval [−0.00858, 0.00516], with six refitted rotations changing direction and all primary calibrated Qwen arms deferring every observation. In a separate 46-feature simulation, source-only selection chooses zero conditional-preservation weight and no additional update. A further ground-truth-controlled vibration experiment, performed with numerical models rather than Jev-PHM, exposes the importance and limits of observable fault information: under matched shaft-speed and resonance shifts, an established order-normalized linear reference attains NLL 0.000190 versus 0.682596 for a fixed-frequency reference; deliberately identical waveform observations from different fault states yield an irreducible balanced binary conditional NLL of at least log 2 without distinguishing metadata. These results specify validity conditions and competitive controls, not semantic-model superiority. Real source-to-target diagnosis and matched inference latency remain unverified.

## 1. Introduction

### 1.1 Diagnosis across measurements and label definitions

Within prognostics and health management (PHM), a fault classifier deployed on another machine or dataset encounters more than a new distribution of vibration amplitudes. Sampling, sensor configuration, operating conditions, and the definition of the target fault can all change. A label denoting predominantly inner-ring damage need not mean an isolated inner-ring defect, and a coarse bearing-fault label does not identify a particular damaged component. Physical-bearing separation is also necessary when recordings from one bearing recur across operating conditions [@lessmeier2016; @vieira2026]. A shared model must preserve these distinctions rather than treat dataset-specific class numbers as interchangeable targets.

Existing fault-diagnosis models address important parts of this problem. Heterogeneous Signal Embedding (HSE) addresses heterogeneous signal inputs [@hse2025]. UniFault learns representations from multiple bearing datasets and evaluates downstream adaptation, whereas YOTOnet evaluates transfer to excluded datasets under a unified fault vocabulary [@unifault2025; @yoto2026]. These settings distinguish representation reuse, target-data adaptation, and target-free transfer. None is synonymous with applying the same software separately to several datasets.

Language-based diagnosis introduces a second design choice. FaultGPT produces diagnostic answers from time-frequency representations, while HCAA uses a language-model arbitrator after a preliminary probabilistic diagnosis [@chen2025faultgpt; @wu2025trustworthy]. A fixed-category diagnosis can instead return class probabilities directly. This avoids response-token decoding and answer parsing by construction, but does not imply lower end-to-end latency than every numerical model, higher diagnostic accuracy, or calibrated probabilities.

### 1.2 Scientific questions and motivation

**The primary scientific question is whether correct fault-description correspondence provides a measurable diagnostic increment on an entirely excluded dataset, relative to an indexed output head on the same backbone and carefully tuned signal-only predictors, under identical source supervision and observable measurements.** This comparison must distinguish a linguistic inductive bias from additional parameters, source information and favorable model selection. A fixed incorrect assignment of the same candidate vectors provides a necessary semantic control; a simple high-quality signal model is the practical competitor. A shared model instance, rather than independently fitted dataset-specific heads, must be evaluated on each admissible source and target.

This question has two preconditions. First, each diagnostic label must describe the same physical event when its probabilities are compared. Verified deterministic coarsenings may support event likelihood, but a union of pure single-ring faults cannot replace a recorded mixed/dominant-fault category without a valid map. Second, the measurement must actually resolve the requested distinction. A descriptor or larger pretrained backbone cannot recover two physical labels from identical waveform observations with identical available metadata. These are validity and identifiability questions; they do not themselves establish improved generalization. The currently approved first real-data comparison uses equivalent exact fault classes and therefore cannot confirm a native coarse-label learning contribution.

The secondary question is computational: **at fixed signal information, diagnostic granularity, and acceptable error or coverage, how does direct candidate scoring compare with exact single-token label scoring and shortest valid answer generation as token budget and candidate count vary?** One-token classification can itself use a single forward computation, so decoding avoidance alone is not evidence of a speedup. The proposed signal adapter and candidates are not defined by a particular pretrained model. Backbone independence is an interface contract; portability and cost must be established separately for each tested instance.

Jev motivates typed probabilistic judgments separated from response generation [@typesafe2026jev; @typesafe_skill2026]. GLiClass, CLIP and NanoJev establish semantic matching and direct decision heads [@gliclass2025; @radford2021; @nanojev2026]. Accordingly, the research value cannot follow from component assembly or a new name. It must emerge from valid external comparisons and a demonstrated information or cost difference that remains after these controls.

### 1.3 Contributions and current support

First, we specify a shared non-generative fault-candidate decision interface and its annotation-compatibility condition. The Paderborn mixed-damage counterexample establishes a concrete limit of the proposed event likelihood. Candidate scoring and coarse-label likelihood are established techniques; their combination is a method hypothesis whose diagnostic increment remains to be tested, not a proven new learning mechanism.

Second, a completed six-statistic Paderborn pilot distinguishes question conditioning from ordinary hierarchy and two direct questions. The predeclared test interval includes zero, fitted rotations disagree in sign, and the calibrated language-model arms produce no accepted primary diagnoses. A distinct 46-feature simulation compares source-selected conditional preservation with ordinary event likelihood, weighted Libra and freezing; its selected soft regularizer is inactive. These negative and inconclusive findings prevent treating interface design as a performance result.

Third, a separately executed, ground-truth-controlled vibration study establishes necessary measurement controls for the planned method comparison. Matched perturbations show that an ordinary order-normalized predictor can already be highly effective in the reduced simulator, while an observational collision makes fault location unidentifiable from waveform alone. These are characterization and limitation results for the stated simulated observations, not accuracy, semantic-interpretation or speed evidence for Jev-PHM. Real excluded-dataset prediction, backbone portability and inference latency remain open. The scope is closed-set fault diagnosis, not prognostics, unseen-fault detection or validated maintenance decisions.

## 2. Related Work

### 2.1 Transferable fault representations and diagnostic interfaces

HSE addresses heterogeneous signal inputs [@hse2025]. UniFault v2 combines multi-dataset pretraining with target-task adaptation [@unifault2025]. Its experiment passages differ in wording: one describes exclusion of fine-tuning samples, whereas another states exclusion of all samples from the three adapted datasets. We do not resolve that provenance ambiguity by assuming either contamination or verified whole-dataset exclusion. Target-label adaptation is explicit and remains different from target-free deployment. YOTOnet instead holds out datasets, but studies an inner-ring/outer-ring binary fault task and reports optional target-labeled adaptation separately [@yoto2026]. Discriminating two known fault types does not evaluate healthy-versus-faulty detection or mixed-damage recognition. These are distinct estimands, not interchangeable levels of a single benchmark. Our pending comparison must therefore match both target-data access and the physical meaning of the requested classes.

FaultGPT uses vision-language instruction supervision, and HCAA supplies a preliminary diagnosis, features, and diagnostic images to an arbitrator [@chen2025faultgpt; @wu2025trustworthy]. Their outputs and supervision differ from shared non-generative candidate prediction; their published scores are not same-protocol baselines. CLSGen jointly trains classification and explanation-generation heads, whereas the completed pilot uses a frozen encoder [@yoon2026clsgen]. A discriminative language-model head is established prior art.

### 2.2 Non-generative semantic prediction

CLIP supplies the observation–text compatibility principle [@radford2021]. GLiClass uses joint text/label encoding as its main design and separately specifies a bi-encoder with shared scoring [@gliclass2025]. Candidate precomputation follows the latter design; it cannot inherit the label interactions or measured results of the joint encoder automatically. Our dot-product readout belongs to this established family. Using the same architecture, supervision and optimization would reproduce an ordinary semantic baseline rather than introduce a different algorithm. The proposed comparison changes which verified annotation information is retained, while keeping the readout fixed.

NanoJev supplies direct probability heads [@nanojev2026]. Jung studies Jev on heating, ventilation, and air-conditioning (HVAC) faults, comparing independent decisions with language models instructed to output one word; label probabilities are recovered from returned next-token alternatives [@jung2026]. Thus short-answer controls already exist in Jev diagnosis. Our evaluation requires exact scores for every candidate token rather than a truncated top-token list, and separates source-only calibration from target calibration. The HVAC results do not supply vibration-transfer or matched-latency evidence. These precedents rule out a broad first-Jev-diagnosis claim.

### 2.3 Partial labels, coarsened labels, and incompatible faults

Partial-label learning supplies a set containing one correct class and studies when that ambiguity can be resolved [@cour2011]. Coarse-label learning instead formalizes observations induced by partitions of a finer label space and requires sufficient information preservation for recovery [@fotakis2021]. We use the latter interpretation only for a verified deterministic annotation partition. We do not reproduce Cour et al.'s convex surrogate or inherit Fotakis et al.'s statistical-query guarantee for a neural empirical optimizer under domain shift.

A mixed-damage observation is not uncertainty about which single pure defect occurred. Likewise, a dominant-damage annotation need not be a coarsening of pure single-fault labels. These distinctions decide whether source targets can share a probability space at all. The proposed contribution, if validated, would be the effect of preserving genuine PHM annotation semantics during transfer, not the invention of a set-likelihood expression. Fault-presence and fault-identity targets derived from a single diagnosis are also not independently measured physical concepts of the kind supervised in concept bottleneck models [@koh2020].

Predictive-retention mechanisms are also established. Libra loss studies allowed-label probability ratios under explicit architectural assumptions [@zombori2024]. Learning without Forgetting (LwF) distils an earlier predictor on new inputs, while Orthogonal Gradient Descent (OGD) projects updates using stored output gradients [@li2018lwf; @farajtabar2020]. Their soft or first-order preservation mechanisms cannot be equated with exact global conditional freezing, but they rule out claiming that distillation or projection itself is new. Our simulation tests whether such control survives fair source-side selection; it does not transfer their guarantees to a new semantic backbone.

### 2.4 Calibration, selective prediction, and independent evaluation

Temperature scaling, conditional classifiers, and rejection are established methods [@guo2017; @valmadre2022; @chow1970; @geifman2017]. Hierarchical selective classification can vary prediction specificity, but all comparisons here must use the same requested label granularity [@goren2024]. A coarse answer cannot be credited as a correct fine diagnosis only for the proposed method. CAT combines calibration with teacher–student domain adaptation in fault diagnosis, rather than the pilot's fixed-classifier post-hoc calibration [@forest2024cat].

Probability losses, binned calibration error, and confidence ordering evaluate different properties [@gneiting2007; @cattelan2024]. Their joint reporting is especially important when low scenario cost can be obtained by always deferring. Physical-unit grouping follows the Paderborn dataset's bearing identities and the concerns established in bearing-level evaluation research [@lessmeier2016; @vieira2026]. More overlapping windows do not create more independent equipment, and target adaptation must not be mislabeled as generalization.

### 2.5 Comparison of supervision, retention, and validation scope

Table 1 compares eight concrete methods at their stated level of analysis. Direct-classification and diagnostic-transfer methods establish the input/output and evaluation precedents; loss and continual-learning methods establish the relevant retention mechanisms. The table is not a performance ranking across incompatible tasks. Source-specific definitions and original-method locations are provided in the [comparison notes](../assets/tables/related_work_comparison_notes.md).

**Table 1. Supervision, prediction mechanisms, and validation scope of related methods.** Symbols indicate whether a dimension is addressed, restricted or not established; they do not indicate that a method is superior.

| Work / version | Supervision | Semantic candidates | Retention mechanism | Shift evaluation | Probability assessment | Inference measurement |
|---|---|---|---|---|---|---|
| GLiClass (2025, v1) [@gliclass2025] | Exact single-/multi-label | ✓ Label descriptions | — | ○ Text-task transfer | — | ✓ GPU throughput |
| UniFault (2025, v2) [@unifault2025] | Self-supervised; target fine-tuning | ✗ Indexed target head | — | ○ Target-label adaptation | — | — |
| YOTOnet (2026, v1) [@yoto2026] | Common exact fault labels | ✗ Indexed fault head | — | ✓ Excluded datasets; two faults | — | — |
| Jev–HVAC (2026, v1) [@jung2026] | Inference study; pretraining undisclosed | ✓ Textual alternatives | — | ✓ Seasonal/system/building | ✓ Brier, ECE | ✓ Hosted/local timing |
| CAT (2024) [@forest2024cat] | Labelled source; unlabelled target | ✗ Fixed class head | — | ○ Target-data adaptation | ✓ Calibration/ECE | — |
| Libra loss (2024) [@zombori2024] | Sets containing one true label | ✗ Indexed softmax | ○ Allowed-label ratio property | — | ○ Probability dynamics | — |
| LwF (2018) [@li2018lwf] | Exact new-task labels; old predictions | ✗ Task-specific heads | ✓ Soft output distillation | ○ Sequential image tasks | — | — |
| OGD (2020) [@farajtabar2020] | Exact sequential-task labels | ✗ Task-specific outputs | ✓ First-order output projection | ○ Sequential image tasks | — | — |
| **Ours (evaluated scope)** | **Exact/coarsened simulated labels** | **○ Design only** | **○ Soft/hard tested; selected soft off** | **○ Simulated target exclusion** | **✓ NLL, Brier** | **—** |

✓ denotes addressed coverage in the stated scope; ○ denotes restricted, adjacent or design-only coverage; ✗ denotes absence from the specified evaluated architecture; — denotes not established in the inspected source. Task qualifiers are essential: sequential learning and target adaptation are not target-free diagnosis, and synthetic coarsening is not industrial annotation provenance. CAT's EMA teacher is not a demonstrated fixed fine-conditional retention control. The LaTeX source separates method families and the final row with horizontal rules.

Jev–HVAC reports inference measurements, including hosted-request latency and local-model timing [@jung2026, Section 4.7]. They must not be marked unreported, although the different execution conditions do not isolate a matched-hardware readout effect. The Ours row concerns the simulation of Section 7.4; the earlier PU study evaluates a different fixed-statistic predictor. Neither experiment implements the full semantic learner of Section 4.

**Specific gap.** Direct prediction from semantic candidates, diagnostic transfer, probability calibration and predictive retention each have established methods. GLiClass and Jev–HVAC provide direct-decision and timing precedents; UniFault, YOTOnet and CAT study distinct target-access regimes. Libra controls allowed-label probability dynamics, while LwF and OGD limit changes in previous predictions through distillation or projection. These results do not directly establish whether preserving a within-fault conditional distribution improves excluded-domain diagnostic risk when a new source supplies only a verified coarse event. That effect must be separated from added source information, ordinary event likelihood, model-selection behavior and a simpler conditional-freezing head. Our current contribution is a scoped qualification of this question: an explicit annotation condition and controlled tests that distinguish a mathematically defined or trial-active mechanism from one actually selected for prediction. The present simulation selects no nonzero soft-preservation treatment, so it cannot support a claim of semantic-transfer or runtime superiority.

## 3. Problem Formulation and Notation

### 3.1 Observations, source annotations, and target exclusion

An observation is a vibration window $x\in\mathbb R^{n\times C_x}$ and its available measurement metadata $m$. The positive integers $n$ and $C_x$ denote time samples and measured channels. Dataset $d$ supplies an annotation $y_d$ whose meaning is established from its documentation, not from its numerical code. A canonical finite set $\mathcal Y=\{1,\ldots,K\}$ contains mutually exclusive diagnostic states, and $\mathcal C\subseteq\mathcal Y$ is the declared candidate set for a particular classification task. Candidate descriptions are $t_k$, with $k\in\mathcal C$. Aliases for one event are not additional classes.

The initial closed-set assumption is that the true state belongs to $\mathcal C$. Removing a state from the denominator is justified only by the declared task population, not by absence of its annotation in one dataset. Missing labels are not known negative classes. A target dataset is excluded from task-specific training, preprocessing fits, prompt refinement from its samples, checkpoint selection, calibration, and threshold selection. Public class definitions may specify the task; target observations may not tune it. The same exclusion must be checked for domain-specific pretraining. Disjoint training and test windows, or excluding only fine-tuning samples, does not exclude the target collection. Unknown general-backbone pretraining exposure is disclosed rather than treated as verified absence.

A *backbone-independent method* has a common mathematical interface across compatible backbone families. It does not imply transferable weights between architectures. *Cross-dataset generalization* means evaluation on a dataset excluded as above. *Adaptation* means use of target data for updates or selection and is reported separately. *Non-generative readout* means no autoregressive response-token decoding at diagnosis time; a causal-attention backbone may still be used.

| Term or symbol | Definition and scope | Distinction |
|---|---|---|
| $x,m$ | Measured signal and available metadata | Not a predicted equipment condition |
| $s(x)\in\mathbb R^6$ | Six temporal statistics used only in the pilot | Not a sufficient signal representation |
| $\mathcal C,t_k$ | Closed-set candidates and their descriptions | Not dataset-specific class IDs |
| $g_d,S_d(y_d)$ | Deterministic annotation map and its preimage event | A coarse event is a set of exclusive diagnostic states, not a claim that only one physical lesion exists |
| $h,e_k\in\mathbb R^{d_h}$ | Signal and candidate representations | $d_h$ is embedding width, not fault probability |
| $p_\Theta,\widetilde p_T$ | Model probabilities before/after post-hoc temperature scaling | Maximum probability is a score, not a safety guarantee |
| $\tau,T$ | Training score scale and post-hoc calibration temperature | Neither is inference latency |
| $v_k,\ell_{v_k}$ | Distinct single-token label code and its exact next-token logit, used only in the readout control | A code is not a fault description; top-token truncation is not exact scoring |
| $\Lambda,c_{\mathrm{def}}$ | Action-cost matrix and deferral cost | Separate from regularization parameter $C_{\mathrm{reg}}$ |
| Acceptance coverage | Weighted fraction of accepted diagnoses | Not a prediction-set coverage guarantee |

### 3.2 What annotation compatibility means

For the likelihood case, dataset $d$ has a known deterministic map $g_d:\mathcal C\to\mathcal L_d$ from diagnostic states to its annotation vocabulary. Define

$$
S_d(y_d)=\{k\in\mathcal C:g_d(k)=y_d\}.
$$

These sets form a disjoint partition of the relevant candidate population. An exact annotation is a singleton. A coarse annotation denotes the union of its compatible states. The source map is used to interpret supervision, not as a dataset-ID input to the predictor. If the annotator chooses among overlapping descriptions using an unmodeled selection mechanism, or a set only weakly contains the true label, this deterministic likelihood interpretation does not hold. Such data require a stated alternative weak-supervision model and are excluded from the initial likelihood experiment rather than silently repaired.

A diagnostic state may include co-occurring lesions while remaining exclusive of other *complete states*. The partition assumption concerns those states, not mutual exclusivity of individual damaged components. To check whether the state definition is sufficient, let $\Omega$ be admissible physical states, $\pi:\Omega\to\mathcal C$ their canonicalization, and $a_d:\Omega\to\mathcal L_d$ the deterministic source annotation rule. Then

$$
\exists g_d:\pi(\Omega)\to\mathcal L_d,\quad a_d=g_d\circ\pi
\quad\Longleftrightarrow\quad
\pi(\omega_1)=\pi(\omega_2)\Rightarrow a_d(\omega_1)=a_d(\omega_2).
$$

**Proof.** Necessity follows by applying $g_d$ to equal canonical states. For sufficiency, define $g_d(k)=a_d(\omega)$ for any preimage with $\pi(\omega)=k$. Constancy on each preimage makes this definition independent of the choice, and hence $a_d=g_d\circ\pi$. States outside $\pi(\Omega)$ are not identified by the argument. This elementary factorization condition establishes applicability, not a statistical generalization bound.

The distinction is concrete in Paderborn: KB23 and KB24 have both-ring damage with inner-ring dominance, whereas KB27 has both-ring damage with outer-ring dominance [@lessmeier2016]. The retained cohort contains 240 recordings from these three bearings. A vocabulary limited to health and pure single-ring damage does not cover them. Adding one undifferentiated both-ring state still merges different annotations, violating the displayed condition. Putting that state into both the inner-dominant and outer-dominant compatibility sets gives annotation probabilities summing to $1+p_{\mathrm{both}}$, where $p_{\mathrm{both}}$ is its predicted probability in the four-state vocabulary. Those overlapping events are not the assumed partition. A dominance-refined state space or an explicitly dominant-location task requires separate semantic justification. No pilot record is removed or relabeled. The source-specific verification is retained in the [annotation analysis](../method/native_annotation_compatibility.md).

Canonicalization does not assert invariant conditional signal–fault relationships. Informative annotations and transferable measurement relationships remain separate requirements; valid source maps alone do not identify target posteriors.

### 3.3 Risk, acceptance, and evaluation units

Negative log-likelihood (NLL), measured with natural logarithms in nats, is the main probability endpoint. For $B$ test bearings with $n_b$ exported windows each,

$$
R_{\mathrm{unit}}(p)=\frac1B\sum_{b=1}^{B}\frac1{n_b}\sum_{i=1}^{n_b}-\log p(y_{bi}\mid x_{bi}).
$$

Here $i$ flattens recordings and windows within a bearing. This is an equal-bearing mean of window losses, not a claim that windows are independent. A fine-label test requires independently available fine ground truth; evaluation against a coarse label can assess only its corresponding event probability. Risk–coverage curves, binned ECE, and realized action cost are separate endpoints, not synonyms for general reliability.

For a fixed action-cost matrix $\Lambda$, the action is

$$
a^*(x)=\arg\min_{a\in\mathcal C\cup\{\bot\}}\sum_{k\in\mathcal C}\Lambda_{ak}\,\widetilde p_T(k\mid x).
$$

The pilot uses correct-diagnosis cost zero, error cost one, and $c_{\mathrm{def}}=0.25$ for deferral. It accepts only when $1-\max_k\widetilde p_T(k\mid x)<c_{\mathrm{def}}$, and defers on a tie. These are normalized experimental costs, not maintenance economics. Deferral is an action, not an unknown-fault class. Accepted-sample error is undefined when acceptance coverage is zero; rejection does not erase the underlying classifier's ordinary prediction error.

## 4. Proposed Semantic Decision Method

### 4.1 Bounded signal encoding and shared candidate scoring

This section specifies the new method; the completed experiments in Sections 5–6 evaluate the earlier compact-state pilot, not this implementation. Let $E_\phi$ map $(x,m)$ to $L\le M$ signal tokens, already projected to the selected backbone's input width. Let $f_\theta$ denote that backbone, $P$ its stated pooling/projection operation, and $G_\psi$ the candidate-description encoder. The symbol $\Theta$ denotes parameters actually fitted in a given instance. In the first implemented instance, the pretrained backbone $f_\theta$ and description encoder $G_\psi$ are frozen; candidate vectors $e_k$ are computed from source-declared descriptions and cached. The signal adapter $E_\phi$ is trainable, while the matched indexed-head baseline also trains its own output head. This is not equivalent to end-to-end adaptation of $(\theta,\psi)$. Then

$$
Z=E_\phi(x,m),\qquad h=P(f_\theta(Z))\in\mathbb R^{d_h},\qquad e_k=G_\psi(t_k)\in\mathbb R^{d_h},
$$

$$
p_\Theta(k\mid x,m,\mathcal C)=
\frac{\exp(h^\top e_k/\tau)}{\sum_{j\in\mathcal C}\exp(h^\top e_j/\tau)},\qquad \tau>0.
$$

The initial readout is a dot product, not an additional mixture, retrieval system, or generated explanation. One signal encoding supports all candidates; no dataset-specific classifier rows are fitted. Candidate embeddings can be precomputed for a fixed model and candidate definitions. Reordering distinct candidates reorders the probability vector, whereas duplicating an alias would incorrectly change its normalization and is disallowed. Changing the candidate set changes the conditional classification task and need not preserve calibration.

The adapter parameters are learned from source development data and frozen before external evaluation. The implemented instance uses bounded contiguous signal patches and masked-mean pooling; the proposed general interface allows other compatible signal encoders provided their information access is specified and matched. Token-budget control does not justify undocumented downsampling, aliasing, fabricated sample rates, or discarding transients without evaluation. HSE supplies relevant prior input-alignment work, not a new component claimed here [@hse2025]. The first implementation must document patch duration, channel handling, amplitude normalization, and any anti-alias filtering. These settings have not been selected or executed for the proposed method. The six pilot statistics remain a distinct baseline, not a substitute for that signal-interface validation.

### 4.2 Supervision at the recorded label granularity

For a justified deterministic coarsening, the probability of observing $y_d$ is the sum over its preimage, giving

$$
\ell_d(\Theta;x,y_d)=
-\log\sum_{k\in S_d(y_d)}p_\Theta(k\mid x,m,\mathcal C).
$$

For exact labels this is ordinary cross-entropy. For sources with $N_d$ observations and predeclared nonnegative within-source weights $\omega_{di}$ summing to one, use

$$
\mathcal L(\Theta)=\frac1D\sum_{d=1}^{D}\sum_{i=1}^{N_d}\omega_{di}\ell_d(\Theta;x_{di},y_{di}).
$$

Here $D$ is the number of training sources, not a new neural component. Weights are specified from the sampling units before fitting, not chosen from favorable outcomes; they prevent overlapping-window counts from silently setting a source's training mass. Regularization and backbone adaptation are held comparable across controls. No extra contrastive, distillation, or reinforcement-learning objective is required by the formulation.

The objective is established coarse-label likelihood, not a new generic learning guarantee [@cour2011; @fotakis2021]. The informative contrast is compatible-event versus exact-label-only or common-coarse supervision using a common readout and adequate tuning. An otherwise identical semantic matcher with the same valid supervision is the same method, not an independent novelty baseline. A common-coarse predictor can be compared on aggregated event loss, but cannot be assigned fine-label NLL by inventing probabilities within its groups. If all sources have equivalent exact labels, as in the approved initial PU–MFPT evaluation task, the method reduces to conventional semantic classification and cannot support an annotation-granularity contribution. A mixed/dominant-fault label cannot be replaced by a union of pure defects unless the source semantics actually make that union correct.

The proposed Jev-PHM framework is summarized in the schematic below. Its inference pathway is separate from source-only fitting of exact or deterministically coarsened annotations. The diagram specifies a candidate method; neither the earlier six-statistic PU pilot nor the fixed-feature simulated conditional-preservation experiment is an end-to-end implementation of this semantic model.

![Jev-PHM conceptual framework, not an experimental result. (A) Measured vibration and available acquisition metadata are mapped into bounded signal tokens, represented by a compatible backbone, and compared with encoded fault descriptions using shared candidate scoring. The complete categorical distribution supports a typed diagnosis or deferral without generating answer tokens. (B) Source-only training interprets observed exact or verified coarsened labels through a deterministic annotation map and fits the corresponding event likelihood, with all selection performed on source development units. The 46-feature simulation evaluated an optional conditional KL penalty but selected zero strength and no additional update; it does not validate this full framework or a latency advantage.](../assets/figures/jev_phm_framework.svg)

### 4.3 A concrete identifiability boundary

Suppose the only labels are healthy and faulty, while the desired fine categories are healthy, inner-ring fault, and outer-ring fault. At a fixed observation let the fault-event probability be $\rho\in(0,1)$. For every $\alpha\in[0,1]$,

$$
p_\alpha=(1-\rho,\rho\alpha,\rho(1-\alpha))
$$

gives exactly the same healthy/faulty probabilities. Thus coarse supervision does not identify $\alpha$, even with unlimited repeated windows. More generally, when a partition matrix $A$ maps fine probabilities to observed event probabilities, any interior $p$ and nonzero $v$ satisfying $Av=0$ and $\mathbf1^\top v=0$ admit sufficiently small positive and negative perturbations $p+\epsilon v$ with identical observations. This follows by substitution and is an identifiability diagnostic, not a new theorem.

Additional exact labels or informative partitions may resolve an ambiguity within a shared population, as studied in coarse-label learning [@fotakis2021]. When labels and measurements both change across datasets, that transfer additionally needs shared predictive structure; concatenating source partition matrices cannot assume all sources have the same conditional posterior. Semantic names can supply an inductive bias, but do not prove the missing distinction. The proposed experiment must therefore distinguish real fine-label transfer from merely fitting the observed coarse groups.

There is a separate *observation* identifiability condition. Let $o=(x,m_{\mathrm{available}})$ denote exactly the input visible to a decision rule; the underlying physical state and simulator parameters are not part of $o$. If two equally weighted fault states produce the same $o$, every deterministic probabilistic classifier must assign the same conditional fault probability $r\in(0,1)$ to both. Its paired conditional NLL is

$
-\tfrac12\log r-\tfrac12\log(1-r)\ \ge\ \log 2,
$

because $r(1-r)\le 1/4$, with equality at $r=1/2$. This is a standard identifiability limit for an indistinguishable balanced pair, not a novel generalization theorem. It concerns the declared information set: if a tachometer or another channel differs across the two fault states and is available to the model, the inputs no longer coincide. Section 7.5 reports an explicit vibration-waveform collision under a restricted matched simulator condition.

### 4.4 Inference cost and decision policy

After candidate precomputation, dot-product scoring costs $O(\lvert\mathcal C\rvert d_h)$ for the declared candidates in addition to signal processing and one backbone encoding. There are zero generated response tokens. These are properties of the specified computation, not an end-to-end speedup result. Signal acquisition duration, preprocessing, warm inference, candidate compilation, and any downstream review are distinct costs. Candidate-dependent cross-encoding of the full observation would be a different algorithm with different complexity.

A necessary control is exact single-token label scoring. At the specified response boundary, let distinct tokenizer-verified tokens $v_k$ encode the declared classes and let $\ell_{v_k}$ be their next-token logits after encoding the complete prompt and signal. The restricted class distribution is

$$
p_{\mathrm{token}}(k\mid x,m,\mathcal C)
=\frac{\exp(\ell_{v_k})}{\sum_{j\in\mathcal C}\exp(\ell_{v_j})}.
$$

Every candidate logit is read, including low-scoring candidates absent from a returned top-token list. This computes a decision from one prompt encoding without a sequential response-decoding loop. It conditions on the allowed token set, not on all possible generated strings, and supplies no calibration guarantee. Multi-token labels require their stated sequence-scoring rule; shared first pieces cannot be treated as distinct class probabilities. The source-chosen prompt specifies the same class meanings for both readouts, without target-driven selection of label codes.

Consequently, removing response decoding alone cannot establish a speed advantage over this control. Compare the same eligible backbone and signal information using exact token scoring, shortest valid label generation, semantic readout, and a compact non-generative signal classifier. Vary $M$ and the candidate count $\lvert\mathcal C\rvert$ using source-declared task settings; report error or NLL jointly with measured p50/p95 latency, memory, precision, batch size, and acceptance coverage. Fix class definitions and independent test units within each comparison. No generative-versus-direct latency result currently exists in this study.

Any post-hoc temperature or acceptance threshold is fitted on source development units for the target-free experiment. A different candidate set or target distribution does not inherit a calibration guarantee. Backbone independence is a method contract; empirical portability requires separately trained instances from distinct compatible families and a comparison against each family's controls. Neither portability nor cross-dataset superiority follows from a successful import or forward pass.

## 5. Completed Pilot: Experimental Protocol

The pilot tests whether changing and composing diagnostic questions helps a frozen compact-statistic predictor. It does not train a signal–description encoder, use coarse-source supervision, compare backbone families, or exclude an entire target dataset. Its detailed prediction and calibration equations are retained in Appendix C; the six fitted arm names below remain unchanged for reproducibility. Generic method statements in Section 4 must not be inferred from these arm names.

### 5.1 Dataset and physical grouping

We use PHM-Vibench's PU collection `RM_027_PU`, upstream dataset identifier 20. The existing H5 and canonical metadata contain 2,559 indexed recordings from 32 physical bearings: six healthy, 13 inner-ring-dominant, and 13 outer-ring-dominant. These are bearing counts, not file counts. All available indexed operating conditions and repetitions are retained.

The vibration sensor is upstream channel index 2. Metadata gives a nominal sampling rate of 64 kHz. Windows use actual stored sample counts without resampling; we do not independently establish each raw time field's association with the stored vibration channel. Durations below are nominal. No frequency-specific interpretation depends on an inferred exact timebase.

The source's dominant-category assignments include mixed damage in KB23, KB24, and KB27. They are retained with this limitation rather than relabeled as pure defects. The task is classification of the declared source categories, not multi-label localization. One raw record, `KA08/N15_M01_F10_KA08_2.mat`, is absent from the existing canonical metadata and H5 following an observed read failure. Its historical cause is unknown. No missing signal is synthesized and no user data are repaired in place.

### 5.2 Parameterized coverage and observation selection

Window length is selected from $W\in\{2048,4096,8192\}$. For maximum start interval $S$, with $1\le S\le W$, and longest selected recording $L_{\max}$, the shared number of windows per recording is

$$
N(W,S)=1+\left\lceil\frac{L_{\max}-W}{S}\right\rceil.
$$

Upstream evenly spaced starts run from zero to $L_j-W$ in each recording. The first and last windows cover both ends, adjacent windows leave no gaps, and starts remain unique. A smaller $S$ increases overlap. Default $S=W$ gives full coverage with the smallest shared count, replacing the fixed 64-position limit. Actual intervals, start gaps, and overlap are retained. A too-short record or failed coverage check causes an error instead of padding or silent exclusion.

For stored maximum length $L_{\max}=299038$, candidate geometry is:

| Window length | Nominal duration | Default maximum start gap | Windows per recording | Total windows |
|---:|---:|---:|---:|---:|
| 2,048 | 32 ms | 2,048 | 147 | 376,173 |
| 4,096 | 64 ms | 4,096 | 74 | 189,366 |
| 8,192 | 128 ms | 8,192 | 37 | 94,683 |

These totals follow from coverage and the 2,559 indexed recordings; they are not independent sample sizes. Full coverage increases observed signal extent and often creates overlap. It does not imply exhaustive independent evidence or uniform sampling in physical time.

Window selection uses only the current split's train and tune bearings. Numerical-flat on amplitude-preserving statistics jointly searches window lengths and $C_{\mathrm{reg}}\in\{0.01,0.1,1,10,100\}$ by tuning-bearing NLL. Exact ties prefer fewer windows, then smaller $C_{\mathrm{reg}}$. Every arm shares the selected protocol in that split, including the standardized observation comparison. This choice favors the numerical baseline's observation criterion rather than an observation claimed optimal for all arms. Calibration and test predictions never choose window length or overlap. The existing configuration key remains `C`; the notation distinguishes it from the action-cost matrix.

### 5.3 Main split and repeated held-out bearings

The main split sorts bearings within each class, shuffles using seed 20260922, and assigns roles as follows:

| Class | Train | Tune | Calibration | Test |
|---|---:|---:|---:|---:|
| Healthy | 2 | 1 | 1 | 2 |
| Inner-ring-dominant | 7 | 1 | 2 | 3 |
| Outer-ring-dominant | 7 | 1 | 2 | 3 |
| Total | 16 | 3 | 5 | 8 |

Recordings and windows inherit their bearing's role. PHMFactory's validation role covers tune and calibration; the adapter distinguishes their whole-bearing assignments. Healthy bearings support all four roles but remain sparse, limiting claims about healthy-equipment diversity.

Six additional rotations provide a descriptive check across the healthy bearings. Each is tested once, the next in the frozen circular order calibrates, the following tunes, and the other three train. Each fault class is divided into six frozen test buckets of sizes 3, 2, 2, 2, 2, and 2. In a rotation, two remaining fault bearings calibrate, the next tunes, and the rest train. Every rotation repeats its own window and head selection using development bearings.

The first rotation has 17 training and seven test bearings; rotations 2–6 have 19 training and five test bearings. Each has three tuning and five calibration bearings. These rotations reuse the collection with overlapping fitting assignments and different training sizes and held-out compositions. They are not independent confirmations or pure perturbations of a fixed test set. The protocol is not revised between rotations based on test outcomes. Foldwise differences and their range describe this combined split sensitivity rather than six independent training seeds.

### 5.4 Comparison arms and fitting

The feasibility comparison crosses Numerical-flat and the original non-question-specific Qwen representation with raw, pooled, and equal-bearing calibration. It is reported for both frozen preprocessing protocols and is not relabeled as decomposition evidence.

The decomposition comparison contains six arms:

| Arm | Representation and output | Purpose |
|---|---|---|
| Numerical-flat | Six measured values; one leaf head | Same-observation numerical reference |
| Numerical-factorized | Same values; presence and fault-only heads | Hierarchy without Qwen |
| Qwen-flat | Flat question A; one leaf head | Direct language reference |
| Qwen-shared-factorized | Flat A representation; presence and fault-only heads | Hierarchy on the direct language representation |
| Qwen-factorized | Presence and conditional-identity representations | Question-conditioned factorization |
| Qwen-two-question-direct | Flat A/B representations; two leaf heads | Additional-question control |

Flat A and B share all category definitions and the healthy/fault grouping supplied to conditional questions. The two-direct-question control averages uncalibrated probabilities with fixed weights 1/2 before final calibration; no learned or test-selected fusion is used. One shared $C_{\mathrm{reg}}$ is selected at workflow level for every two-head method. All arms receive the same labeled recordings, window weights, five-value regularization search, tuning objective, and calibration bearings.

Heads use deterministic `lbfgs`, at most 2,000 iterations, and tolerance $10^{-7}$, with convergence status retained. Each rotation refits training-only standardizers and heads, selects $C_{\mathrm{reg}}$, and fits temperature. Frozen, protocol-compatible hidden states can be reused across role assignments without fitting to their labels. States from a different observation, interval, question, or model revision are not interchangeable.

The Qwen representation has 1,024 coordinates. Fitted-head parameter counts, including intercepts, are 3,075 for Qwen-flat, 2,050 for each Qwen factorized workflow, and 6,150 for the two-direct-question control. Numerical-flat has 21 parameters and Numerical-factorized 14. These counts make the different capacities and binary reference-logit parameterizations explicit rather than asserting exact model equivalence.

Pooled final-distribution temperature calibration is primary in the decomposition comparison. Raw and equal-bearing-scaled outputs are secondary analyses. All controls supply complete normalized distributions and the same leaf-or-defer contract. Equal question counts do not imply equal token counts, floating-point operations (FLOPs), latency, or parameter counts. We record tokens, padded tokens, forward batches, cache-extraction elapsed time, and fitted-head parameters. FLOPs are not measured, and unequal extraction schedules and warmup conditions do not constitute a matched deployment-latency benchmark.

### 5.5 Metrics and uncertainty

The primary endpoint is the paired bearing-weighted NLL difference, Qwen-factorized minus Qwen-flat; negative values favor factorization. Comparisons with Qwen-shared-factorized and Qwen-two-question-direct distinguish question conditioning from hierarchy and added evaluations. Numerical-factorized minus Numerical-flat tests whether the numerical hierarchy explains the effect's direction.

Multiclass Brier sums over all three classes, $\sum_k(p_k-\mathbb1[y=k])^2$, rather than using a one-component binary convention. Accuracy and macro-averaged F1 (Macro-F1) use the same equal-bearing, equal-window weights as NLL. ECE uses maximum probability and ten bins: $[0,0.1),[0.1,0.2),\ldots,[0.9,1]$. It sums each bin's probability mass times the absolute difference between its weighted mean confidence and accuracy; empty bins contribute zero. ECE is descriptive and depends on binning.

Risk–coverage curves sort saved maximum probabilities with stable ties, and report weighted accepted-window error against weighted coverage. AURC is the sum of each prefix's risk times its coverage increment, using right endpoints; lower values indicate lower integrated selective error. Fixed-cost outputs include coverage, accepted-sample error when coverage is positive, and realized cost including every deferral's 0.25 charge. Lower selective error from greater rejection is not treated as improvement at matched coverage.

Always deferring is an analytical reference with zero coverage and cost 0.25. A model attaining that cost by rejecting every window has made no accepted diagnosis; cost and coverage must therefore be interpreted jointly.

For the main split, 2,000 paired bootstrap draws resample its eight test bearings using seed 20260922. Each draw uses the same bearings for the compared methods. Only Qwen-factorized minus Qwen-flat is the prespecified primary contrast. The other reported 95% intervals are descriptive, unadjusted for multiple comparisons. All intervals condition on the split and fitted models, omitting training and selection variation. Overlapped windows are never bootstrap units. Foldwise differences are separate because fitting data overlap across rotations. Failure to exclude zero is not evidence of equivalence.

### 5.6 Reproducibility and predeclared interpretation

Artifacts retain grouped roles, actual intervals, observations and weights, queries, dependency and model versions, fitted standardizers and heads, tuning values, temperatures, raw factors, final probabilities, per-bearing losses, and failures. Figures and additional cost analyses read saved predictions without rerunning Qwen. Encoding records sequence lengths, tokens, padded tokens, forward batches, precision, hardware, and cache-extraction elapsed time. These describe actual resource use without estimating FLOPs or establishing deployment latency; no proprietary Jev speed ratio or end-to-end maintenance speed claim is imported.

For the complete 376,173-window amplitude-preserving 2,048-point export, the direct flat query processes 64,870,462 non-padding tokens in 1,470 batches. The presence and identity queries together process 135,759,692 tokens in 2,940 batches; the two-direct-query control processes 131,621,789 tokens in 2,940 batches. Corresponding padded-token totals are 65,199,017, 136,416,802, and 132,278,899. Thus two questions do not imply identical token cost. Shared-representation factorization reuses the flat query without another encoder call. These complete-cohort cache totals include all roles and are reused across protocols selecting that geometry; they are not training-repeat counts or deployment latency measurements. Per-query elapsed extraction times and the other geometry's counts are retained in the [encoding table](../assets/tables/jev_encoding.csv).

A numerical hierarchy gain supports a hierarchy explanation. A gain retained by Qwen-shared-factorized does not establish that two diagnostic questions are necessary. A gain matched by two direct questions is compatible with extra calls or prompt diversity. A gain following amplitude restoration belongs to the observation revision unless within-protocol controls isolate an additional effect. Zero and negative results remain findings. Interface or infrastructure failure is separated from a completed comparison.

## 6. Completed Pilot: Results

### 6.1 Observation and coverage verification

The exported 2,048-point protocol covers every stored sample in all 2,559 recordings, yielding 376,173 windows. Actual adjacent start gaps range from 1,697 to 2,035 samples, corresponding to overlaps of 13–351 samples. In the per-window-standardized export, RMS and standard deviation both have mean 0.9999999972 and across-window standard deviation $6.58\times10^{-7}$, with values ranging from 0.99999442 to 1.00000485. They are nearly constant rather than exactly constant: training standardization rescales the remaining finite-precision variation, so these observations do not establish an exactly reduced-dimensional state. The amplitude-preserving export remains a separately frozen comparison. These are input and coverage measurements, not diagnostic-performance results.

### 6.2 Amplitude-preserving feasibility comparison

The completed amplitude-preserving feasibility run uses the selected 2,048-point window on the main split. It evaluates 94,080 test windows from eight bearings, with 11,760 windows per bearing. Its five calibration bearings provide 58,653 windows: four have 11,760 and KA08 has 11,613, reflecting the existing missing record. The numerical and legacy-Qwen heads both select $C_{\mathrm{reg}}=0.01$ on the tuning bearings. Qwen encoding uses frozen bfloat16 weights; this comparison retains the non-question-specific legacy representation rather than the fixed diagnostic questions.

| Representation | Calibration | Test NLL | Brier | ECE | Coverage (%) | Cost |
|---|---|---:|---:|---:|---:|---:|
| Numerical | None | 1.0708 | 0.6555 | 0.1208 | 13.5406 | 0.2850 |
| Numerical | Pooled | 1.0227 | 0.6216 | 0.1110 | 0.0744 | 0.2498 |
| Numerical | Equal-bearing | 1.0225 | 0.6215 | 0.1105 | 0.0776 | 0.2498 |
| Legacy Qwen | None | 1.2595 | 0.7145 | 0.2008 | 35.8578 | 0.3461 |
| Legacy Qwen | Pooled | 1.0527 | 0.6370 | 0.0818 | 0 | 0.2500 |
| Legacy Qwen | Equal-bearing | 1.0526 | 0.6369 | 0.0816 | 0 | 0.2500 |

All values use equal-bearing evaluation weights; Brier sums the three class components. Numerical accuracy is 45.7345% and Macro-F1 is 0.3484; legacy-Qwen accuracy is 47.4596% and Macro-F1 is 0.4086. Each pair of class metrics is unchanged across its temperature variants, as required by global positive scaling. The pooled temperatures are 4.4970 for numerical and 7.2199 for Qwen; equal-bearing temperatures are 4.4776 and 7.1820. Pooled calibration lowers observed test NLL in both representations, while the calibrated legacy-Qwen NLL remains higher than its numerical counterpart. This is representation/calibration evidence, not a decomposition result.

Equal-bearing minus pooled NLL is $-0.000171$ for numerical, with conditional paired 95% interval $[-0.000557,0.000270]$, and $-0.000162$ for legacy Qwen, with interval $[-0.000597,0.000319]$. Both intervals include zero; these data do not establish an advantage or equivalence. The calibration counts differ only by one recording, so this collection provides little contrast between the two weighting populations.

The uncalibrated accepted-window error is 50.8125% for numerical and 51.8127% for legacy Qwen, yielding costs above the always-defer reference. Pooled numerical calibration accepts only 70 of the 94,080 windows, with no observed errors among those accepted; equal-bearing calibration accepts 73, also without observed errors. Both calibrated Qwen variants defer every window. Their cost of 0.25 therefore reflects zero diagnostic coverage, and the numerical cost near 0.25 reflects almost complete deferral rather than broad reliable diagnosis.

### 6.3 Standardized-observation feasibility comparison

The completed per-window-standardized feasibility run uses the same window intervals, eight test bearings, 94,080 test windows, labels, and role assignments. Both heads again select $C_{\mathrm{reg}}=0.01$. It tests the original observation protocol, including the nearly constant RMS/standard-deviation coordinates and their rescaled finite-precision variation described above.

| Representation | Calibration | Test NLL | Brier | ECE | Coverage (%) | Cost |
|---|---|---:|---:|---:|---:|---:|
| Numerical | None | 1.2616 | 0.7118 | 0.1215 | 2.2704 | 0.2668 |
| Numerical | Pooled | 1.0961 | 0.6647 | 0.0791 | 0 | 0.2500 |
| Numerical | Equal-bearing | 1.0961 | 0.6647 | 0.0796 | 0 | 0.2500 |
| Legacy Qwen | None | 1.1656 | 0.6975 | 0.1349 | 0.0797 | 0.2504 |
| Legacy Qwen | Pooled | 1.0897 | 0.6618 | 0.0525 | 0 | 0.2500 |
| Legacy Qwen | Equal-bearing | 1.0898 | 0.6619 | 0.0528 | 0 | 0.2500 |

Numerical accuracy is 43.1186% and Macro-F1 is 0.3277; legacy-Qwen accuracy is 37.9188% and Macro-F1 is 0.2934. These class metrics remain invariant under temperature scaling. Numerical pooled and equal-bearing temperatures are 8.6600 and 8.5583; the corresponding Qwen temperatures are 1.9971 and 1.9916. Both calibrated representations defer all test windows, attaining the always-defer cost and leaving accepted-sample error undefined.

Equal-bearing minus pooled NLL is $0.000012$ for numerical, with conditional paired 95% interval $[-0.000605,0.000751]$, and $0.000108$ for legacy Qwen, with interval $[-0.000329,0.000581]$. These small, uncertain differences do not establish a weighting advantage. Across the separately frozen feasibility protocols, preserving amplitudes reduces pooled test NLL by 0.0734 for numerical and 0.0370 for legacy Qwen. This observed preprocessing contrast supports retaining the distinction between observation and classifier effects; it is not a finding about question decomposition or broader industrial generalization.

### 6.4 Primary decomposition comparison

All six decomposition arms completed under the frozen amplitude-preserving observation and the same 94,080 test windows from eight held-out bearings. Every workflow selects $C_{\mathrm{reg}}=0.01$; calibration acts on each final three-class distribution. The numerical-flat result is the protocol-compatible feasibility result, retained as a shared comparison rather than independent evidence.

| Method | NLL | Brier | Accuracy (%) | Macro-F1 | ECE | AURC |
|---|---:|---:|---:|---:|---:|---:|
| Numerical-flat | 1.0227 | 0.6216 | 45.7345 | 0.3484 | 0.1110 | 0.4721 |
| Numerical-factorized | 1.0183 | 0.6190 | 45.9067 | 0.3496 | 0.1055 | 0.4642 |
| Qwen-flat | 1.0553 | 0.6386 | 46.4605 | 0.3884 | 0.0760 | 0.4934 |
| Qwen-shared-factorized | 1.0463 | 0.6334 | 46.3520 | 0.3859 | 0.0639 | 0.4915 |
| Qwen-factorized | 1.0542 | 0.6387 | 45.9460 | 0.3782 | 0.0650 | 0.5399 |
| Qwen-two-question-direct | 1.0568 | 0.6397 | 47.2523 | 0.3905 | 0.0854 | 0.5089 |

The primary Qwen-factorized-minus-Qwen-flat NLL difference is −0.001102, with conditional paired 95% interval [−0.008577, 0.005163]. Against the shared-representation factorization, the difference is +0.007866 [−0.001909, 0.018887]; against two direct questions it is −0.002561 [−0.009060, 0.003682]. These intervals do not establish an additional question-conditioning benefit or equivalence. The shared-representation hierarchy has lower point NLL than the question-conditioned model without a second encoding.

Numerical factorization has the lowest primary NLL. Its paired difference from Numerical-flat is −0.004344 [−0.007547, −0.000973]. This secondary, unadjusted conditional interval excludes zero, but describes an ordinary hierarchy on this split and fit; it is neither a language-model contribution nor independent confirmation across populations. The shared-Qwen-hierarchy-minus-flat difference is −0.008968 [−0.021275, 0.002886], retaining uncertainty about that descriptive contrast.

Probability and selection metrics disagree in an informative way. Qwen-factorized has ECE 0.0650 versus Qwen-flat's 0.0760, yet AURC is higher, 0.5399 versus 0.4934. Its Brier and class metrics also do not improve over Qwen-flat. Thus the lower ECE does not establish better confidence ordering or overall diagnostic utility. At the fixed cost threshold, all four calibrated Qwen methods have zero coverage and cost 0.25. Numerical-flat accepts 70 windows and Numerical-factorized 25, both without observed errors among those accepted; their costs are 0.249814 and 0.249934. These almost-complete deferrals are interpreted relative to always deferring, rather than as useful broad diagnostic coverage.

Before final calibration, Qwen-factorized NLL is 1.2429 versus 1.2237 for Qwen-flat, and its cost is 0.3511 versus 0.3311. The tiny favorable calibrated primary point difference therefore does not persist in the uncalibrated comparison. Raw and equal-bearing-calibrated probabilities, class metrics, and risk–coverage curves are retained in the complete result tables rather than used to select a favorable calibration regime.

![Completed pilot only: main-split paired bearing NLL differences and pooled-calibration risk–coverage curves from saved predictions. Only Qwen-factorized minus Qwen-flat is the prespecified primary contrast; other conditional 95% intervals are descriptive and unadjusted. These are not tests of semantic candidate learning or cross-dataset generalization.](../assets/figures/jev_primary_results.svg)

### 6.5 Split-sensitive attribution

All six predeclared rotations completed. Rotation 5 selects 2,048-point windows; the other five select 8,192 points using their own development bearings. Every method within a rotation shares its selected observation. The paired bearing-weighted NLL differences below are descriptive because fitting and evaluation populations overlap across rotations. Negative values favor question-conditioned factorization.

| Rotation | Window length | Against Qwen-flat | Against shared-representation factorization | Against two direct questions |
|---|---:|---:|---:|---:|
| 1 | 8,192 | −0.0231 | −0.0227 | −0.0075 |
| 2 | 8,192 | +0.0667 | +0.0669 | +0.0560 |
| 3 | 8,192 | −0.0123 | −0.0176 | +0.0173 |
| 4 | 8,192 | +0.0106 | +0.0067 | +0.0256 |
| 5 | 2,048 | +0.0532 | +0.0347 | −0.0149 |
| 6 | 8,192 | −0.0219 | −0.0211 | −0.0056 |

Each contrast favors factorization in three rotations and disfavors it in three. Against Qwen-flat, differences range from −0.0231 to +0.0667; against two direct questions, from −0.0149 to +0.0560. In rotation 3, factorization improves NLL relative to the flat question while two direct questions perform better than factorization. In rotation 5, factorization improves on two direct questions while being worse than the flat and shared-representation hierarchy. These reversals do not support a stable question-conditioning increment.

Per-bearing values reveal heterogeneous changes. In rotation 2, healthy bearing K002's factorized-minus-flat difference is +0.4008, contributing +0.0802 to the five-bearing mean of +0.0667. In rotation 5, K004's difference is +0.2535, contributing +0.0507 to the mean of +0.0532. In rotation 4, KA01's difference is +0.1381 despite three other bearings having negative differences. These are descriptions of error concentration, not evidence for a physical mechanism or independent replication.

Higher acceptance also need not improve the fixed-cost decision. In rotation 5, Qwen-factorized has 5.1395% coverage and cost 0.2808; Qwen-flat has 4.1446% coverage and cost 0.2739. Both exceed the 0.25 always-defer reference. Thus accepting more windows does not by itself demonstrate useful diagnosis. All rotation probabilities and metrics, including negative results and the numerical hierarchy controls, are retained in the complete saved tables.

![Completed pilot only: question-conditioned factorization minus three Qwen controls in six predeclared bearing rotations. Each rotation shares its development-selected window within the comparison. Direction changes describe sensitivity within one collection, not independent replications or target-dataset transfer.](../assets/figures/jev_rotation_contrasts.svg)

## 7. Discussion and Required Validation

### 7.1 What the pilot establishes

The pilot does not establish an additional question-conditioning benefit. The shared-representation hierarchy has a better primary point NLL, and the question-conditioned contrasts change direction across refits. Because observations and model-selection opportunities are matched within a comparison, a favorable split cannot be selected as a general benefit. These conclusions apply to the frozen compact-statistic model, not to the proposed signal–semantic learner. No new model inference or cross-dataset result is introduced by the formulation.

Six temporal statistics discard waveform order and frequency structure. A pretrained encoder can impose a useful prior or reparameterize these statistics, but cannot retrieve the particular signal information that was not supplied. The pilot does not determine whether richer measured inputs or a trained signal adapter would improve diagnosis. Its few healthy bearings, source-dominant categories including mixed damage, and overlapping rotations limit external validity. Full signal coverage does not resolve limited equipment diversity.

The observed concentration of errors on particular bearings is descriptive, not a physical mechanism. The original protocol was frozen before these test observations; neither its questions nor feature definitions were selected to repair those outcomes. Hypotheses motivated by these errors are exploratory and require evaluation on untouched data. The new generalization questions arose after the pilot and are not retrospectively described as its preregistered objectives.

### 7.2 Validation of annotation-compatible transfer

The first pending comparison holds observations, backbone, descriptions, and source model-selection opportunity fixed while changing valid supervision. Compare compatible-event training with exact-label-only training and a justified common-coarse-label reference. Fine-label evaluation uses independently available ground truth. The coarse reference is assessed on common aggregated events; fine-state scores require an explicitly justified fine-state predictor. An inner-versus-outer task conditional on a known fault is reported separately from a task including healthy bearings. Neither conditional task selection nor coarser requested answers may give the proposed method an easier test population.

Exclude whole target datasets and dependent physical units from task-specific fitting and calibration, and document their membership in any domain-pretraining corpus. Target-adapted models and source-only models occupy separately identified protocols. Audit native class definitions before constructing $S_d$; do not turn run-to-failure timestamps into fault-onset labels without supporting annotations. Sources whose dominant or mixed faults cannot share the proposed partition are reported as incompatible rather than discarded after inspecting test performance. Controlled coarsening of source labels can diagnose sensitivity, but is not a substitute for demonstrating that the native annotations present the claimed problem. Compare results per target dataset, retaining negative transfer and uncertainty at the physical-unit level.

A conventional semantic matcher receiving identical supervision, architecture and optimization is an implementation-equivalence reference: no incremental algorithm has then been specified. The supervision contrast and comparison with a mapped shared classifier test different changes. A compact signal-only model tests whether the pretrained backbone adds value. Practical comparisons allow each method reasonable source-only tuning; mechanism comparisons hold the relevant inputs, supervision and adaptation opportunities fixed. A single checkpoint is shared within each source-trained model instance; a different excluded-target fold generally trains a different checkpoint. Two compatible backbone families test method portability without implying universal performance.

The authorized first real-data protocol uses the Paderborn single-ring subset as a source and MFPT as an entirely excluded target; it does not contain native coarse-label supervision. Before external predictions may be opened, original-record correspondence, the true vibration channel and time/scale conversion, and MFPT specimen identity must be established for the retained cohort. Available MFPT file names and processed H5 shapes do not determine independent physical bearings. At the present stage that eligibility check remains unresolved. Paderborn internal comparisons have already influenced development and are exploratory, regardless of any new physical-bearing split. The explicitly excluded CWRU dataset is not used.

### 7.3 Probability estimates, rejection, and runtime

The pilot's lower ECE but higher AURC distinguishes calibration from selective ordering. Complete deferral attains the always-defer cost but produces no accepted diagnosis. Calibration bearings are few, and nearly equal counts make the two calibration-weighting objectives very similar; their uncertain differences establish neither a weighting advantage nor equivalence. The normalized deferral cost does not model expert accuracy, delay, downtime, or physical intervention.

The second pending comparison measures diagnostic error or NLL jointly with latency while varying the declared signal-token budget and candidate count. Use the same hardware, precision, batching, warmup and input duration, and include device synchronization. Include the exact label-token control of Section 4.4, not only minimal-output generation, and compare an efficient numerical classifier. Separate prompt construction, backbone encoding, output projection, candidate compilation and device transfer when explaining a timing difference; zero decoded response tokens is not an adequate explanation by itself. Report cold candidate compilation separately from repeated warm decisions, and retain failures and format errors in the generated-output denominator. A method that trades away accuracy or rejects everything cannot be declared uniformly faster at the same diagnostic service level.

Acquisition duration is not inference latency, candidate order is not class semantics, and a source-calibrated maximum probability is not an engineering interlock. These distinctions constrain interpretation but do not guarantee the proposed method's outcome. The study contains no matched latency experiment or real target-dataset exclusion experiment for the semantic method. Section 7.4 reports a separate simulated coarse-label learning comparison, not evidence of native industrial coarsening or measured deployment speed.

### 7.4 Simulation qualification of conditional preservation

A separate, completed simulation uses 46 fixed signal features: six temporal statistics and 40 squared-envelope order-energy bins. Three independent cohorts contain 270 target virtual units, each contributing four windows. Fine-source labels distinguish healthy, inner-ring and outer-ring states; coarse-source labels distinguish healthy from faulty. Fixed impulse-response, modulation, noise and transfer conditions generate the waveforms. This is a controlled coarse-supervision experiment, not measured equipment data or an implementation of the semantic encoder in Section 4. Its complete equations, design parameters, source-only search and failures are retained in the [simulation report](../experiments/jev_simulation_poc.md), with [computed results](../assets/tables/jev_simulation_poc.csv).

The soft control penalizes divergence from the initialized within-fault conditional distribution; zero penalty is eligible. A hard control instead adds the same residual to the two fault logits, retaining the initial group-mass offset and exactly preserving their log probability ratio. Ordinary joint event learning and the released weighted Libra loss provide matched-input references; a random forest trained only on fine-source labels is a different-information practical reference. All selected free-head extensions choose update step zero, and the soft control chooses zero penalty. Consequently, fine-only, joint-event, weighted-Libra and soft-anchor workflows have identical selected probabilities: mean target NLL 0.002140 and accuracy 99.907%. Their zero paired difference is caused by selection of the same predictor, not equivalence of active algorithms. Hard freezing yields NLL 0.001321 with unchanged accuracy; its smaller parameterization and penalty geometry are not matched to the full heads.

Positive-penalty trial paths reduce conditional drift, but those checkpoints are not selected. The source-side selection criterion is already near saturation at initialization, leaving the proposed soft-preservation benefit unresolved. Under the predeclared combined speed, slip and transfer mismatch, the common selected linear predictor falls to 40.370% accuracy; raw-spectrum-magnitude-preserving phase randomization yields 33.333%. These jointly specified stresses do not isolate an individual failure cause. The results motivate a source-only diagnostic of which fine distinctions remain unlearned, rather than changing target conditions to favor the intervention. No runtime advantage, real-domain transfer, or novel conditional-regularization principle follows from this simulation.

### 7.5 Ground-truth-controlled observation limits: numerical references only

An independently executed reduced vibration simulation isolates the information available to diagnostic predictors before attributing any gain to a semantic model. A local-defect impulse sequence excites a causal damped resonance with load modulation, background vibration and coloured measurement noise, following established bearing-diagnostic modeling principles [@mcfadden1984; @randall2011]. The generating process records fault events, structural response, background and sensor noise separately. Three fault states (healthy, inner-ring and outer-ring) are paired within each of 24 independent simulated nuisance families at evaluation; each family contributes three counterfactual records and two windows per state. The source fitting and tuning families are disjoint (24 and 12, respectively). Source-side tuning searches logistic regularization $C_{\mathrm{reg}}\in\{0.01,0.1,1,10\}$ and random-forest depth/leaf settings; no target measurement selects a model. Disturbance realizations are matched across states, the sensor-noise RMS is independent of fault class, and the target perturbations were defined before model selection. These are virtual parameter families, not measured bearings.

We evaluated a root-mean-square-only logistic control and logistic-regression/random-forest classifiers using either fixed-Hz or order-normalized squared-envelope features. The latter two representations use the **same noisy simulated tachometer measurement**; true rotation speed, generated fault times, resonance, state and perturbation parameters do not enter the predictors. These comparisons concern known signal representations, not the full Jev-PHM backbone or candidate scorer. The compact [computed metric table](../assets/tables/jev_ground_truth_controls.csv) reports all 30 model-by-condition results.

**Table 2. Source-fitted numerical-reference NLL (nats) under controlled simulated observation changes; not Jev-PHM performance.** The equal-family estimate averages record/window losses within each of 24 independent simulated nuisance families.

| Controlled target change | Fixed-Hz features + tachometer, logistic | Order features + tachometer, logistic | Order features + tachometer, random forest |
|---|---:|---:|---:|
| None | 0.001913 | 0.000306 | 0.013744 |
| Shaft speed only | 0.611435 | 0.000331 | 0.014914 |
| Resonance path only | 0.003926 | 0.000318 | 0.050201 |
| Shaft speed and resonance path | 0.682596 | 0.000190 | 0.006938 |
| Sensor-noise RMS ×4 | 0.021963 | 0.004079 | 0.196628 |
| Observable fault-path gain set to zero | 5.777052 | 6.392829 | 13.583630 |

Under simultaneous speed and resonance changes, the order-normalized logistic predictor reaches 100% accuracy with NLL 0.000190, whereas the otherwise matched fixed-Hz logistic predictor reaches 70.14% accuracy and NLL 0.682596. Their paired difference is −0.682406 nats with a *conditional*, 24-family bootstrap interval [−0.834627, −0.530851]. The strong established order-based reference therefore leaves little room for attributing additional signal discrimination to a more complex semantic readout in this particular generator. It does not show that all industrial shifts are solved by order tracking.

Two constructed boundary cases make the distinction between generating truth and observable fault identity explicit. First, setting the observable fault-path gain to zero retains latent defect events but yields identical input waveforms for all three states within a family; the reported numerical classifiers attain 33.33% accuracy, with differing and poorly calibrated NLL. Random-forest probabilities are clipped at $10^{-12}$ solely when calculating finite NLL, not recalibrated or presented as a comparable probabilistic optimum. Second, a restricted waveform-only comparison has the same fault repetition rate for an inner-ring fault at 20 Hz shaft rate and an outer-ring fault at 30 Hz: the declared orders are 4.8 and 3.2, giving 96 Hz in both cases. With matched amplitudes, transfer and noise and with shaft harmonics and inner-ring modulation suppressed, 12 paired virtual families yield exactly matching stored waveforms. Their balanced binary conditional NLL cannot beat $\log 2$ without additional distinguishing measurements. This boundary **does not apply when different shaft rates are supplied as measured metadata**, nor is it asserted for every real bearing/load configuration.

The generator, settings and numerical comparisons are an additional controlled *observability diagnostic*, not a third independently confirmed Jev-PHM intervention. The simulation code and large native truth arrays remain in the separately delivered local research package; the repository contains the compact numerical outcomes, and full-code/native-array integration remains a reproduction prerequisite. No semantic-backbone fit, cross-dataset Jev-PHM effect, inference latency or industrial fault advantage is established by these results.

## 8. Conclusion

Jev-PHM is a proposed, backbone-independent interface for non-generative fault decisions against declared semantic candidates. Its scientific value cannot be inferred from the presence of a language backbone or a direct probability head: both are established techniques. External evidence must show that correct fault-description correspondence adds diagnostic value beyond matched indexed-readout and strong physical-signal baselines, with any cost improvement assessed against exact label-token scoring.

Two necessary restrictions are now explicit. Paderborn mixed/dominant damage cannot be represented by an unjustified pure-fault partition; and even a correct candidate description cannot separate fault states that produce indistinguishable available measurements. The latter condition is illustrated by a separately executed, truth-controlled simulator and the strong results of a simple, established order-normalized classifier under paired speed/transfer changes. These calculations are neither new order-tracking theory nor evidence of Jev-PHM superiority. The completed six-statistic pilot finds no stable question-conditioning increment, while the 46-feature conditional-preservation simulation selects an inactive soft constraint. Their adverse and null results are preserved without pooling incompatible protocols.

The full pretrained signal–semantic model has not been shown to improve real excluded-dataset NLL, useful acceptance or inference latency. The approved source-to-target evaluation remains blocked on physical-unit and measurement-provenance requirements. The manuscript therefore establishes a precise candidate method, valid comparison conditions and bounded negative/intermediate results—not a validated general-purpose industrial diagnostic model.

## Appendix A. Fixed pilot question meanings

Every decomposition prompt begins with this identical definition, including the actual label strings and mixed-damage qualification:

> The mutually exclusive classes are healthy, inner-race-dominant, outer-race-dominant. The healthy class is healthy; the defined fault classes are inner-race-dominant, outer-race-dominant. A dominant fault label identifies the predominant annotated damage location and may include mixed damage; it does not assert a pure isolated fault.

The fixed questions below precede the identical state:

| Query | Question |
|---|---|
| Flat A | Which one of the defined classes describes this vibration window? |
| Flat B | Choose the most appropriate class from the complete defined diagnosis classes for this vibration window. |
| Presence | Does this vibration window belong to any of the defined fault classes rather than the healthy class? |
| Conditional identity | Conditional on this vibration window belonging to one of the defined fault classes, which fault class describes it? |

The feasibility legacy query is “Represent this vibration window for fault diagnosis.” It lacks the category-conditioned diagnostic question and is analyzed only as the retained feasibility representation. No prompt contains a sample's answer or identifier. Complete templates, tokenization settings, and class order are saved. These literal prompt strings are preserved; their use of “race” does not relabel the source categories or imply an additional damage class.

## Appendix B. Data and software availability

The implementation and compact pilot summaries are maintained at [PT01-jev-phm](https://github.com/liq22/PT01-jev-phm/tree/dev). PHMFactory/PHM-Vibench is fixed to commit `cdc0669167ca25ac848e34c880875ca39c6aed89` [@phmvibench2026]. Paderborn source data remain subject to the [provider's CC BY-NC 4.0 terms](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter), distinct from the original paper and code licenses. Large observations and hidden states are stored separately from version-controlled code.

The successful pilot run is `run-20261013-pu-001`. It contains two feasibility evaluations, the primary decomposition comparison, and six bearing rotations under `src/outputs/jev-phm/`. Each evaluation retains `metrics.json`, `predictions.npz`, configuration, and provenance. The [compact study summary](../assets/tables/jev_study_summary.json) retains all evaluation metrics and fitted configuration, while [per-bearing NLL values](../assets/tables/jev_unit_nll.csv) support numerical reconstruction of paired contrasts. Figures are generated by `src/S03_Scripts/plot_jev_results.py` from saved predictions. The [repository reproduction instructions](https://github.com/liq22/PT01-jev-phm/tree/dev#reproduce-the-pu-study) provide the pilot preparation, export, window selection, encoding, and evaluation commands using `src/S03_Scripts/run_jev_study.py`. They are not an executable implementation of Section 4. Reproduction uses a new output directory and retains failures.

The simulation discussed in Section 7.4 is the repository study `outputs/run-sim-002`, specified by `src/S02_Configs/jev_simulation_poc.json` and implemented by `src/S01_Package/jev_simulation.py`. Its committed report and compact CSV are linked above; saved native outputs and search histories belong to that run. A distinct earlier 16-feature simulation package is not pooled with this 46-feature experiment or the PU pilot. CWRU is excluded from all new experimental work in this project.

The separate truth-controlled vibration study (run `ground-truth-20261018-001`, configuration schema `ground_truth_control_v1`, executed 18 October 2026) is distinct from `run-sim-002`. Its 30 computed numerical-control rows are archived in [`jev_ground_truth_controls.csv`](../assets/tables/jev_ground_truth_controls.csv). The full generator, fit configuration, independent-unit predictions, truth-component archives, and tests were delivered separately as a local research package; they have **not** yet been integrated and validated in this Git repository. The result therefore does not constitute a complete repository-contained reproduction of the new PoC. All results are labelled model-based simulation; none constitutes a Jev-PHM backbone evaluation. The full source and native outputs must be integrated and checked before submission or independent reproduction claims. The authorized Paderborn-to-MFPT evaluation is still blocked on physical-unit and original-measurement provenance. No CWRU dataset was used.

The sole authoritative manuscript is Markdown. `src/S03_Scripts/build_jev_paper.py` builds derived offline HTML and PDF previews without maintaining a second text source. Author identities, funding, conflicts, target journal, and submission declarations remain unconfirmed. The complete previous candidate and its provenance are retained in Git at `ff9db6e4e43c8da2fc66990a03dcc78ed1fd2afb`.

## Appendix C. Completed pilot prediction and estimation details

### C.1 Scope, features, and frozen encoder

The pilot has $\mathcal C=\{H,I,O\}$: healthy, inner-ring-dominant, and outer-ring-dominant states. Every recording from a bearing remains in one fitting or evaluation role. Labels, bearing/file IDs, operating-condition IDs and role assignments are excluded from inputs. PHMFactory owns reading, preprocessing, window construction, and grouped assignments; validation bearings are divided into tuning and calibration roles without changing the original data.

For a univariate window of length $W$, the feature vector is

$$
s(x)=\left(\sqrt{\overline{x^2}},\sqrt{\overline{(x-\bar x)^2}},\overline{|x|},\max x-\min x,
\frac{\max|x|}{\sqrt{\overline{x^2}}},\overline{\left(\frac{x-\bar x}{\operatorname{std}(x)}\right)^4}\right).
$$

These are root mean square (RMS) amplitude, population standard deviation, mean absolute amplitude, peak-to-peak amplitude, crest factor, and ordinary kurtosis rather than excess kurtosis. A zero denominator gives the relevant ratio a declared zero convention. Values use fixed order and eight significant digits. The model does not calculate signal quantities. The two frozen preprocessing conditions are upstream per-window standardization and no signal normalization. They are shared by all arms within a comparison; the decomposition study uses amplitude-preserving observations. These choices were fixed before test-performance inspection, not selected to improve a test result.

Qwen/Qwen3-0.6B is the pilot instance, fixed to revision `c1899de289a04d12100db370d81485cdf75e47ca` [@qwen2025]. A non-thinking chat template supplies the common class definitions, a fixed question $q$, and the serialized statistics. The last non-padding hidden state $h_q(s)\in\mathbb R^{1024}$ is extracted without answer-token generation; the backbone is frozen. Training-only weighted standardization precedes each logistic head. The conditional-identity standardizer uses faulty training observations only. Numerical heads use the same six values without the encoder. These are supervised probes, not zero-shot diagnostic models.

### C.2 Conditional heads and output probabilities

Let $F=(I,O)$. The pilot's presence probability is denoted $r(s)$. With sigmoid $\sigma$, learned coefficient vectors $\beta_F,\beta_L\in\mathbb R^{1024}$ and scalar intercepts $b_F,b_L$,

$$
r(s)=\sigma(\beta_F^\top h_{q_F}(s)+b_F),\qquad
u(s)=\operatorname{softmax}\begin{pmatrix}0\\\beta_L^\top h_{q_L}(s)+b_L\end{pmatrix}.
$$

$I$ is the identity reference class and $O$ the learned non-reference class. Each binary branch therefore has one learned coefficient vector and intercept, whereas the flat head learns a three-row multinomial matrix. Their regularization geometry differs even for equal $C_{\mathrm{reg}}$. Code forms

$$
p(H\mid s)=1-r(s),\qquad p(k\mid s)=r(s)u_k(s),\quad k\in F.
$$

Presence targets are $\mathbb1[y\in F]$; identity targets exist only for faulty training observations. Both are derived from the same diagnostic label, not additional physical-symptom annotations. Every test observation executes both question branches; neither answer nor a test label is used to select the other branch. The figure's labels $r$ and $u$ denote these pilot factors, not the proposed semantic scorer.

![Completed pilot only: independent fault-presence and conditional fault-identity questions read the same six measured statistics. Diagnostic labels supervise the heads; code composes leaf probabilities, calibrates them, and applies a diagnosis/deferral rule. The figure notation r denotes the pilot fault probability. This diagram does not depict the new method in Section 4.](../assets/figures/jev_method.svg)

### C.3 Training weights, loss, and regularization

A recording $j$ with $m_j$ exported windows assigns scalar training weight $w_{ji}=64/m_j$. Its total training mass is always 64, preserving the earlier probe's loss scale rather than defining prevalence or an independent sample count. Fault-only heads retain the original weights without renormalizing them to the presence branch. The objective is

$$
\mathcal L_{C_{\mathrm{reg}}}=
\sum_{i\in\mathrm{train}}w_i\ell_{F,i}
+\sum_{i\in\mathrm{train}:y_i\in F}w_i\ell_{L,i}
+\frac{\|\beta_F\|^2+\|\beta_L\|^2}{2C_{\mathrm{reg}}}.
$$

Intercepts are unpenalized and class balancing is disabled. The data-fit term equals weighted leaf NLL because

$$
-\log p(y_i\mid s_i)=
\begin{cases}
-\log(1-r_i),&y_i=H,\\
-\log r_i-\log u_{i,y_i},&y_i\in F.
\end{cases}
$$

Separately averaging and equally adding the two losses would generally change faulty-example weight and is not the implemented objective. Flat heads use corresponding weighted categorical NLL with L2 regularization. One shared $C_{\mathrm{reg}}$ is selected for each pair, not a Cartesian product of penalties. The regularization parameter is denoted $C_{\mathrm{reg}}$ to distinguish it from the action-cost matrix $\Lambda$.

### C.4 Calibration population and temperature

Each method first forms its complete three-class distribution. A scalar temperature is fitted only on calibration bearings:

$$
\widetilde p_T(k\mid s)=\frac{p(k\mid s)^{1/T}}{\sum_{j\in\{H,I,O\}}p(j\mid s)^{1/T}}.
$$

The search interval is $[0.05,20]$; $T=1$ is retained if the fitted candidate does not improve the selected calibration objective. Boundary solutions are reported. No extra node-specific calibration is used. Calibrated factors, if exposed, are reconstructed as $\widetilde r=1-\widetilde p_T(H)$ and $\widetilde u_k=\widetilde p_T(k)/\widetilde r$, separately from raw factors.

For calibration bearing $b$, let $L_b(T)$ be mean window NLL and $n_b$ its window count. Pooled calibration minimizes $\sum_b n_bL_b(T)/\sum_b n_b$, while equal-bearing calibration minimizes $B^{-1}\sum_bL_b(T)$. With covariance computed using divisor $B$ and mean count $\bar n$,

$$
R_{\mathrm{window}}(T)-R_{\mathrm{unit}}(T)=\frac{\operatorname{Cov}_b(n_b,L_b(T))}{\bar n}.
$$

This elementary identity does not predict held-out benefit. Equal counts make the objectives equal, and uniformly replicating all windows of one bearing leaves its equal-bearing objective unchanged. The weighting defines a population, not an arbitrary-distribution calibration guarantee.

Positive global temperature preserves class argmax and therefore each arm's accuracy and Macro-F1. In binary classification, maximum probability is $\sigma(|z_1-z_0|/T)$, so confidence ordering and risk at matched coverage are unchanged under a fixed tie convention. No corresponding unconditional ordering claim holds for multiclass maximum probability. NLL, fixed-threshold acceptance and realized scenario cost can still change.

## References
