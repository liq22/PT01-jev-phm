# Question-Conditioned Probability Composition for Bearing Fault Diagnosis: A Controlled Attribution Study

## Abstract

Typed outputs and coherent probabilities do not establish useful fault diagnosis. We test whether independently posed fault-presence and conditional fault-identity questions improve closed-set probability estimation beyond direct classification, ordinary hierarchy, and two direct questions. A frozen Qwen3-0.6B encoder receives the same six vibration statistics as numerical controls. The study retains 2,559 recordings from 32 Paderborn bearings, covers each recording with parameterized windows, and separates physical bearings across fitting and evaluation roles. On the eight-bearing primary test, question-conditioned factorization has NLL 1.0542 versus 1.0553 for direct Qwen, a paired difference of −0.0011 with conditional 95% interval [−0.0086, 0.0052]. Factorization on the direct Qwen representation has lower point NLL, 1.0463; numerical factorization has the lowest, 1.0183. Lower ECE for the question-conditioned model coexists with higher area under the risk–coverage curve, and all calibrated Qwen methods defer every primary test window at error cost 1 and deferral cost 0.25. Six predeclared bearing rotations show changes of direction rather than a stable increment. These results establish a bounded negative attribution finding: question conditioning has not demonstrated additional diagnostic probability or selective-decision value in this frozen, compact-state setting. They do not prove equivalence, rule out other observations or training regimes, or establish practical superiority over full-signal PHM models.

## 1. Introduction

### 1.1 From diagnostic evidence to bounded decisions

Fault diagnosis turns sensor observations into claims about equipment condition. A maintenance application may then accept a diagnosis or request further examination. These operations require different semantics: a measured signal statistic is evidence, a class probability is an estimate about condition, and deferral is an action under a declared cost. A fluent explanation does not establish any of these quantities. This distinction matters when adapting language models to industrial diagnosis, where observations may be indirect, faults may be heterogeneous, and confidence can fail on equipment absent from training.

Vision-language systems such as FaultGPT connect diagnostic questions to time-frequency representations [@chen2025faultgpt]. Such systems broaden the diagnostic interface, but their question-answering output and input evidence differ from a fixed-ontology probability estimator. For a bounded diagnosis, software can expose a complete distribution over allowed fault categories and use an explicit action rule. A distribution can be internally valid while its predictions remain inaccurate or poorly calibrated; evaluation must address both properties separately.

Jev's public interface motivates state-conditioned typed decisions and code-owned composition [@typesafe2026jev; @typesafe_skill2026]. NanoJev already realizes small language-model decision heads and local judgments followed by code planning [@nanojev2026]. We adopt this interface idea, using the open Qwen3-0.6B backbone [@qwen2025], without reproducing proprietary Jev weights, its parallel sampler, or Reinforcement Learning for Calibrated Decisions. The research question concerns a measured PHM increment rather than the novelty of a softmax head or typed output.

### 1.2 Why decomposition requires attribution

Consider three allowed outcomes: healthy, inner-ring-dominant damage, and outer-ring-dominant damage. One classifier can estimate their probabilities directly. Another can ask whether a fault is present and, conditional on a fault, which fault category applies. The second construction is a standard conditional probability factorization. It supplies no additional signal observation and no additional physical annotation. Its possible benefit must arise from the representation induced by the questions, the statistical parameterization, or the resulting learning bias.

These explanations are easy to confound. A second model evaluation can help through ensembling. A hierarchy can help without a language model. Restoring amplitudes lost to preprocessing can improve diagnosis without changing the questions. More windows can improve measurement coverage without adding independent equipment. Post-hoc calibration can change probability quality and fixed-cost acceptance while leaving the predicted class unchanged. Comparing only a composed language predictor with a single uncalibrated numerical classifier cannot distinguish these mechanisms.

We ask: **at a fixed observation, label budget, class ontology, and evaluation population, does question-conditioned fault decomposition improve probability quality or selective diagnosis beyond direct prediction, a shared-representation hierarchy, and two direct questions?** The experiment is a controlled attribution study of a frozen language representation. It does not test a fully trained signal model's performance ceiling.

### 1.3 Findings and scope

The contribution is an empirical attribution result, rather than a new probability factorization or output type. Under the controlled observation and supervised frozen-backbone setting, question-conditioned composition does not demonstrate an increment beyond direct language prediction, ordinary hierarchy on the same representation, and a fixed two-direct-question ensemble. The primary paired interval includes zero, and the descriptive bearing rotations change direction. This leaves a useful boundary on the transfer of atomic decision interfaces to compact vibration observations.

The controlled comparisons expose two distinctions relevant to interpreting that result. First, preserving amplitude improves the observed feasibility probability scores independently of question decomposition. Second, a lower ECE can coexist with worse confidence ordering, while calibrated low-cost outputs can simply defer everything. The study provides the corresponding implementation and saved bearing-level outputs so that these effects can be separated without new model inference. Conditional heads, temperature scaling, typed validity, and rejection remain established foundations rather than individual novelty claims.

The scope is single-channel, supervised, closed-set classification on the Paderborn bearing collection. Mixed-damage bearings retain the source's dominant-category assignments; the task does not claim pure damage localization. It excludes remaining useful life, previously unseen fault recognition, causal symptom inference, continual learning, and actual maintenance execution. The negative finding concerns this observation, backbone, fitting regime, and ontology; it is not a claim that diagnostic decomposition is universally ineffective.

## 2. Related Work

### 2.1 Language and probability models for industrial diagnosis

FaultGPT combines vision-language modeling and diagnostic instruction supervision over time-frequency evidence [@chen2025faultgpt]. Wu's HCAA combines a preliminary probabilistic engine, numerical features and diagnostic images with a Qwen2.5-VL-32B arbitrator, followed by calibration [@wu2025trustworthy]. The arbitrator receives the preliminary diagnosis: this sequential review differs from two questions that never consume one another's answers. HCAA is a close probabilistic/LLM diagnostic reference, but its experimental provenance and evidence conditions do not support a matched comparison with our bearing study.

CLSGen trains classification and explanation-generation heads jointly, using LoRA adaptation of Llama-3.1-8B and generated explanation supervision [@yoon2026clsgen]. Our backbone remains frozen and produces no explanation tokens. Adding a discriminative head to a language model is therefore established, while preserving generative behavior is outside our experiment. Published classification or ranking results under different inputs, supervision, and splits are not treated as baselines here.

### 2.2 Decision interfaces and question-conditioned representations

The official TypeSafe guidance assigns semantic judgments to the model and calculations, composition, and execution to code [@typesafe_skill2026]. NanoJev uses Qwen3-0.6B, candidate-conditioned heads, and complete decision distributions. Its maze interface combines independent local traversability questions with a code planner [@nanojev2026]. Our fixed PHM ontology and supervised fault heads do not establish a new general-purpose candidate interface. Batching separate sequences also does not reproduce a shared-prefix parallel sampler.

Question text changes the contextual representation even when the numerical state is unchanged. This makes a direct language predictor, a hierarchy on that same representation, and a two-direct-question ensemble necessary controls. We estimate whether the particular diagnostic question conditioning adds value; we do not presume that linguistic decomposition universally recovers missing signal information.

### 2.3 Hierarchical estimation, calibration, and rejection

Conditional softmax probabilities and their pathwise negative log likelihood are established hierarchical parameterizations [@valmadre2022]. Valmadre also reports conditions in which flat softmax outperforms top-down classifiers, motivating a competitive flat control. Goren et al. calibrate leaf probabilities, aggregate them up a hierarchy, and select predictions of varying specificity [@goren2024]. All our methods estimate the same leaf ontology and return a leaf label or deferral. Coarse-node answers do not count as successful leaf diagnoses.

Temperature scaling is standard post-hoc calibration [@guo2017]. CAT applies calibration to a 1D-CNN teacher-student domain-adaptation workflow and uses calibrated probabilities in target pseudo-labeling [@forest2024cat]. Our calibration holds supervised logits fixed, fits one scalar on separate bearings, and compares pooled-window with equal-bearing objectives as a secondary population-weighting question. It does not claim adaptation to an unlabeled target domain.

Cost-based rejection and risk–coverage evaluation are established [@chow1970; @geifman2017]. Probability calibration and confidence ranking have different objectives [@cattelan2024]. Lower expected calibration error does not itself prove better accepted-sample risk or safe operation. Likewise, the two label-derived branches are not independently annotated physical concepts comparable to a concept bottleneck [@koh2020].

### 2.4 Evaluation on independent bearings

The Paderborn release documents physical bearings, damage characteristics, operating conditions, and sensor measurements [@lessmeier2016]. Bearing-level separation is essential when the intended test population contains bearings absent from model fitting. Vieira et al. show how leakage through repeated measurements and limited bearing diversity can distort fault-diagnosis evaluation [@vieira2026]. We preserve the physical bearing across every recording and operating condition, use separate bearings for tuning and calibration, and treat windows as repeated measurements rather than independent replications.

## 3. Problem Formulation

### 3.1 Observations, labels, and independent units

Let bearing $b$ provide recordings indexed by $j$, with vibration windows $x_{bji}$ and a label $y_{bji}$ in the mutually exclusive ontology $\mathcal Y=\{H,I,O\}$. Here $H$ is healthy, while $I$ and $O$ are the source's inner-ring-dominant and outer-ring-dominant categories. All recordings from a bearing belong to one of four roles: training, tuning, calibration, or test. No bearing crosses roles within an experiment.

The observed state $s(x)\in\mathbb R^6$ is computed from one vibration channel. Every comparison receives the same state in its protocol. Global category definitions are allowed context; a sample's label, bearing or file identifier, operating-condition identifier, and partition role are excluded. The output is

$$
p(x)\in\Delta^2,\qquad a(x)\in\{H,I,O,\bot\},
$$

where $\bot$ means defer. Maximum class probability is a predictive score whose quality is evaluated, not a verified safety confidence.

### 3.2 Probability quality and evaluation population

The primary population gives equal weight to test bearings and then equal weight to their exported windows. For $B$ test bearings with $n_b$ windows each,

$$
R_{\mathrm{unit}}(p)=\frac1B\sum_{b=1}^{B}\frac1{n_b}\sum_{i=1}^{n_b}-\log p(y_{bi}\mid x_{bi}).
$$

This estimand is a mean window loss within a mean bearing. It does not imply independent overlapped windows or a measured industrial deployment prevalence. Log loss and multiclass Brier are proper scoring rules [@gneiting2007], but finite-sample optimization does not guarantee calibration or shifted-domain reliability.

For calibration bearing $b$, let $L_b(T)$ be its mean NLL under temperature $T$. Pooled-window calibration minimizes $\sum_b n_bL_b(T)/\sum_b n_b$, whereas equal-bearing calibration minimizes $B^{-1}\sum_bL_b(T)$. With covariance using divisor $B$,

$$
R_{\mathrm{window}}(T)-R_{\mathrm{unit}}(T)
=\frac{\operatorname{Cov}_b(n_b,L_b(T))}{\bar n}.
$$

This elementary identity describes different weighting targets. Equal window counts make the objectives identical; replicating all windows within one bearing leaves its equal-bearing objective unchanged. Neither property predicts held-out improvement. Pooled calibration is the primary common calibration opportunity in the decomposition experiment; equal-bearing calibration remains a separate secondary question.

### 3.3 Diagnosis and deferral costs

For a declared action-cost matrix $C$, the model-relative action minimizes expected cost:

$$
a^*(x)=\arg\min_{a\in\mathcal Y\cup\{\bot\}}\sum_{k\in\mathcal Y}C_{ak}p_k(x).
$$

We use zero cost for a correct diagnosis, one for an error, and $c=0.25$ for deferral. The action accepts the maximum-probability class when $1-\max_kp_k(x)<c$ and defers on equality. These normalized costs define an experiment, not measured maintenance economics. Deferral does not add a fault category, assume an infallible expert, or implement a physical intervention.

## 4. Method

### 4.1 Reproducible vibration states

PHMFactory owns the reader, preprocessing, window construction, and grouped train/validation/test assignment. A narrow adapter selects the verified vibration channel and consumes returned windows with their source intervals. Validation bearings are divided into separate tuning and calibration roles. Raw data and canonical metadata remain unchanged.

For a finite window $x=(x_1,\ldots,x_W)$, the state contains RMS amplitude, population standard deviation, mean absolute amplitude, peak-to-peak amplitude, crest factor, and standardized fourth moment:

$$
s(x)=\left(\sqrt{\overline{x^2}},\sqrt{\overline{(x-\bar x)^2}},\overline{|x|},\max x-\min x,
\frac{\max|x|}{\sqrt{\overline{x^2}}},\overline{\left(\frac{x-\bar x}{\operatorname{std}(x)}\right)^4}\right).
$$

The corresponding ratio is set to zero for zero RMS or zero standard deviation. The fourth moment is ordinary kurtosis, not excess kurtosis. Text serialization uses a fixed feature order and eight significant digits. The language model calculates no signal value. These temporal statistics are a feasibility representation; they do not encode a spectrum, defect frequency, or sufficient statistic for damage.

Two preprocessing protocols are frozen before testing. The original feasibility protocol applies upstream per-window standardization before extracting the six values. The revised observation protocol uses no signal normalization and preserves measured amplitude. Window standardization can make RMS and standard deviation nearly constant; its effects are measured rather than repaired after observing test errors. Every arm receives the same preprocessing within its comparison. The decomposition experiment uses the amplitude-preserving protocol.

### 4.2 Frozen question-conditioned encoding

Qwen/Qwen3-0.6B is fixed to revision `c1899de289a04d12100db370d81485cdf75e47ca`. A non-thinking chat template includes the common class definitions, a fixed question, and the serialized state. No answer tokens are generated. The last non-padding hidden state defines $h_q(s)$; the backbone remains frozen.

The flat question asks which complete class describes the window. The presence question asks whether any defined fault applies rather than health. The identity question asks which fault applies **conditional on membership in the defined fault set**. Both latter questions read the same observation and neither sees the other's answer. They are evaluated for every test window, including healthy ones. Independent evaluation is an interface constraint, not a statistical independence assumption.

Each representation is standardized using its training observations and window weights only. The conditional-identity standardizer uses faulty training observations, matching that head's fit population. A regularized logistic head supplies class scores. Numerical controls standardize the same six values without Qwen. The direct language model is a supervised frozen-encoder probe, not zero-shot diagnosis or end-to-end language-model adaptation.

### 4.3 Presence, conditional identity, and leaf probabilities

Let the ordered fault set be $F=(I,O)$. A binary presence head estimates $r(s)$ and a fault-only head estimates $u(s)$:

$$
r(s)=\sigma(w_r^\top h_{q_r}(s)+b_r),\qquad
u(s)=\operatorname{softmax}\!\begin{pmatrix}0\\w_u^\top h_{q_u}(s)+b_u\end{pmatrix}.
$$

Both branches use the implemented binary logistic parameterization: one learned coefficient vector and intercept, with a zero reference logit. For identity, $I$ is the reference class and $O$ the non-reference class. Equivalently, its coefficient matrix has a zero row and $w_u^\top$ as the learned row; its penalty includes only $w_u$. The three-class flat head instead learns a three-row multinomial coefficient matrix. These parameterizations have different regularization geometry even when they share $C$.

Code forms the complete distribution:

$$
p(H\mid s)=1-r(s),\qquad p(k\mid s)=r(s)u_k(s),\quad k\in F,\qquad\sum_{k\in F}u_k(s)=1.
$$

The components sum to one by construction. This is conditional factorization, not multiplication of unrelated marginal probabilities. Coherence alone does not establish correctness.

![Independent fault-presence and conditional fault-identity questions consume the same six measured vibration statistics. Diagnosis-derived supervision trains the heads; code composes a coherent leaf distribution, calibrates it, and applies an explicit diagnosis/deferral cost rule. The branches are not independently measured physical concepts.](../assets/figures/jev_method.svg)

Presence supervision is $\mathbb 1[y\in F]$. Identity supervision uses only faulty training examples and their original fault labels. Healthy examples have no fabricated identity label. Both targets are transformations of the same diagnosis; no physical concept supervision is added.

### 4.4 Training mass and loss semantics

Full coverage changes the number of windows. To avoid changing each recording's total training mass at the same time, a recording with $m_j$ exported windows assigns

$$
w_{ji}=\frac{64}{m_j}.
$$

Each recording's total weight is 64 for every window candidate. The factor preserves the original 64-window probe's loss scale; it is not a prevalence or independence count. All arms use the same weights. The fault-only head retains the original faulty-sample weights and does not rescale them to match the presence branch.

The factorized objective is

$$
\mathcal L_C=
\sum_{i\in\mathrm{train}}w_i\ell_{\mathrm{presence},i}
+\sum_{i\in\mathrm{train}:y_i\in F}w_i\ell_{\mathrm{identity},i}
+\frac{\lVert w_r\rVert^2+\lVert w_u\rVert^2}{2C}.
$$

Intercepts are unpenalized and class balancing is disabled. The data-fit part equals weighted leaf NLL because

$$
-\log p(y_i\mid s_i)=
\begin{cases}
-\log(1-r_i),&y_i=H,\\
-\log r_i-\log u_{i,y_i},&y_i\in F.
\end{cases}
$$

Separately averaging and equally adding the branch losses would generally change the relative contribution of faulty examples. That alternative is not used. Flat heads use the corresponding weighted categorical NLL and L2 regularization. One shared $C$ is selected for each pair of heads, rather than a Cartesian product of branch penalties. Shared $C$ does not make parameter counts or regularization geometry identical; those differences belong to the tested parameterization.

### 4.5 Calibration of the final distribution

Every comparison first constructs a complete leaf distribution. A single positive temperature is fitted on calibration bearings. For direct logits this is standard softmax temperature scaling; for an assembled or averaged distribution it is equivalently

$$
\widetilde p_T(k\mid s)=\frac{p(k\mid s)^{1/T}}{\sum_{j\in\mathcal Y}p(j\mid s)^{1/T}}.
$$

The search uses $T\in[0.05,20]$ and retains $T=1$ if the candidate does not improve its calibration objective. Boundary solutions are reported. No method receives extra node-specific calibration. If calibrated presence or identity is exposed, it is reconstructed from $\widetilde p_T$: $\widetilde r=1-\widetilde p_T(H)$ and $\widetilde u_k=\widetilde p_T(k)/\widetilde r$. Raw factors remain distinct from these calibrated quantities.

Positive global temperature preserves class argmax, so accuracy and Macro-F1 must agree across raw and calibrated variants sharing a head. For binary logits, maximum probability is $\sigma(|z_1-z_0|/T)$ and its ordering is preserved; with the same tie convention, temperature cannot improve risk at fixed coverage. Multiclass maximum-probability ordering does not have this unconditional invariance. Calibration may still change NLL and acceptance at a fixed cost threshold.

## 5. Experimental Protocol

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

Window selection uses only the current split's train and tune bearings. Numerical-flat on amplitude-preserving statistics jointly searches window lengths and $C\in\{0.01,0.1,1,10,100\}$ by tuning-bearing NLL. Exact ties prefer fewer windows, then smaller $C$. Every arm shares the selected protocol in that split, including the standardized observation comparison. This choice favors the numerical baseline's observation criterion rather than an observation claimed optimal for all arms. Calibration and test predictions never choose window length or overlap.

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

Flat A and B share all category definitions and the healthy/fault grouping supplied to conditional questions. The two-direct-question control averages uncalibrated probabilities with fixed weights 1/2 before final calibration; no learned or test-selected fusion is used. One shared $C$ is selected at workflow level for every two-head method. All arms receive the same labeled recordings, window weights, five-value regularization search, tuning objective, and calibration bearings.

Heads use deterministic `lbfgs`, at most 2,000 iterations, and tolerance $10^{-7}$, with convergence status retained. Each rotation refits training-only standardizers and heads, selects $C$, and fits temperature. Frozen, protocol-compatible hidden states can be reused across role assignments without fitting to their labels. States from a different observation, interval, question, or model revision are not interchangeable.

The Qwen representation has 1,024 coordinates. Fitted-head parameter counts, including intercepts, are 3,075 for Qwen-flat, 2,050 for each Qwen factorized workflow, and 6,150 for the two-direct-question control. Numerical-flat has 21 parameters and Numerical-factorized 14. These counts make the different capacities and binary reference-logit parameterizations explicit rather than asserting exact model equivalence.

Pooled final-distribution temperature calibration is primary in the decomposition comparison. Raw and equal-bearing-scaled outputs are secondary analyses. All controls supply complete normalized distributions and the same leaf-or-defer contract. Equal question counts do not imply equal token counts, FLOPs, latency, or parameter counts. We record tokens, padded tokens, forward batches, cache-extraction elapsed time, and fitted-head parameters. FLOPs are not measured, and unequal extraction schedules and warmup conditions do not constitute a matched deployment-latency benchmark.

### 5.5 Metrics and uncertainty

The primary endpoint is the paired bearing-weighted NLL difference, Qwen-factorized minus Qwen-flat; negative values favor factorization. Comparisons with Qwen-shared-factorized and Qwen-two-question-direct distinguish question conditioning from hierarchy and added evaluations. Numerical-factorized minus Numerical-flat tests whether the numerical hierarchy explains the effect's direction.

Multiclass Brier sums over all three classes, $\sum_k(p_k-\mathbb1[y=k])^2$, rather than using a one-component binary convention. Accuracy and Macro-F1 use the same equal-bearing, equal-window weights as NLL. ECE uses maximum probability and ten bins: $[0,0.1),[0.1,0.2),\ldots,[0.9,1]$. It sums each bin's probability mass times the absolute difference between its weighted mean confidence and accuracy; empty bins contribute zero. ECE is descriptive and depends on binning.

Risk–coverage curves sort saved maximum probabilities with stable ties, and report weighted accepted-window error against weighted coverage. AURC is the sum of each prefix's risk times its coverage increment, using right endpoints; lower values indicate lower integrated selective error. Fixed-cost outputs include coverage, accepted-sample error when coverage is positive, and realized cost including every deferral's 0.25 charge. Lower selective error from greater rejection is not treated as improvement at matched coverage.

Always deferring is an analytical reference with zero coverage and cost 0.25. A model attaining that cost by rejecting every window has made no accepted diagnosis; cost and coverage must therefore be interpreted jointly.

For the main split, 2,000 paired bootstrap draws resample its eight test bearings using seed 20260922. Each draw uses the same bearings for the compared methods. Only Qwen-factorized minus Qwen-flat is the prespecified primary contrast. The other reported 95% intervals are descriptive, unadjusted for multiple comparisons. All intervals condition on the split and fitted models, omitting training and selection variation. Overlapped windows are never bootstrap units. Foldwise differences are separate because fitting data overlap across rotations. Failure to exclude zero is not evidence of equivalence.

### 5.6 Reproducibility and predeclared interpretation

Artifacts retain grouped roles, actual intervals, observations and weights, queries, dependency and model versions, fitted standardizers and heads, tuning values, temperatures, raw factors, final probabilities, per-bearing losses, and failures. Figures and additional cost analyses read saved predictions without rerunning Qwen. Encoding records sequence lengths, tokens, padded tokens, forward batches, precision, hardware, and cache-extraction elapsed time. These describe actual resource use without estimating FLOPs or establishing deployment latency; no proprietary Jev speed ratio or end-to-end maintenance speed claim is imported.

A numerical hierarchy gain supports a hierarchy explanation. A gain retained by Qwen-shared-factorized does not establish that two diagnostic questions are necessary. A gain matched by two direct questions is compatible with extra calls or prompt diversity. A gain following amplitude restoration belongs to the observation revision unless within-protocol controls isolate an additional effect. Zero and negative results remain findings. Interface or infrastructure failure is separated from a completed comparison.

## 6. Results

### 6.1 Observation and coverage verification

The exported 2,048-point protocol covers every stored sample in all 2,559 recordings, yielding 376,173 windows. Actual adjacent start gaps range from 1,697 to 2,035 samples, corresponding to overlaps of 13–351 samples. In the per-window-standardized export, RMS and standard deviation both have mean 0.9999999972 and across-window standard deviation $6.58\times10^{-7}$, with values ranging from 0.99999442 to 1.00000485. They are nearly constant rather than exactly constant: training standardization rescales the remaining finite-precision variation, so these observations do not establish an exactly reduced-dimensional state. The amplitude-preserving export remains a separately frozen comparison. These are input and coverage measurements, not diagnostic-performance results.

### 6.2 Amplitude-preserving feasibility comparison

The completed amplitude-preserving feasibility run uses the selected 2,048-point window on the main split. It evaluates 94,080 test windows from eight bearings, with 11,760 windows per bearing. Its five calibration bearings provide 58,653 windows: four have 11,760 and KA08 has 11,613, reflecting the existing missing record. The numerical and legacy-Qwen heads both select $C=0.01$ on the tuning bearings. Qwen encoding uses frozen bfloat16 weights; this comparison retains the non-question-specific legacy representation rather than the fixed diagnostic questions.

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

The completed per-window-standardized feasibility run uses the same window intervals, eight test bearings, 94,080 test windows, labels, and role assignments. Both heads again select $C=0.01$. It tests the original observation protocol, including the nearly constant RMS/standard-deviation coordinates and their rescaled finite-precision variation described above.

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

All six decomposition arms completed under the frozen amplitude-preserving observation and the same 94,080 test windows from eight held-out bearings. Every workflow selects $C=0.01$; calibration acts on each final three-class distribution. The numerical-flat result is the protocol-compatible feasibility result, retained as a shared comparison rather than independent evidence.

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

![Main-split paired bearing NLL differences and pooled-calibration risk–coverage curves from saved predictions. Only Qwen-factorized minus Qwen-flat is the prespecified primary contrast; the other conditional 95% intervals are descriptive and unadjusted for multiplicity. Risk–coverage curves sweep confidence acceptance rather than using the single fixed-cost operating point.](../assets/figures/jev_primary_results.svg)

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

![Question-conditioned factorization minus the three Qwen controls in six predeclared bearing rotations. Each rotation shares its development-selected window within the comparison; widths are annotated. Changes of direction describe sensitivity to fitting and held-out populations, not independent repetitions or a pooled significance test.](../assets/figures/jev_rotation_contrasts.svg)

## 7. Discussion

### 7.1 A bounded negative attribution finding

The primary result does not demonstrate a question-conditioning increment. The proposed model's calibrated NLL is close to Qwen-flat with an interval including zero, while the hierarchy on the single flat representation has a better point score. Descriptive rotations change direction against both that hierarchy and two direct questions. Their windows, observations, and selection opportunities are matched within each comparison; a favorable individual split therefore cannot be selected as evidence of a general benefit.

This finding identifies a boundary of the interface transfer. Asking two semantically explicit questions can change the learned representation without improving its diagnostic information or decision behavior. The conditional chain supplies coherent outputs and interpretable event definitions, but these structural properties do not predict generalization to a new bearing. Ordinary numerical hierarchy attains the best primary probability score, and its conditional secondary improvement does not require a language-specific explanation. The experiment leaves open whether other observations, training regimes, or independently annotated physical concepts would change the conclusion.

The bearing-level error concentrations suggest split sensitivity as an important empirical limitation, not a discovered physical mechanism. Healthy-bearing count is small, training populations vary by rotation, and individual bearing losses can determine a fold's mean. The protocol was frozen before these observations; neither feature definitions nor questions were revised to improve these test results. A future explanatory hypothesis based on the observed bearing patterns would be exploratory and would require a separate test.

### 7.2 Limits of the observation and collection

Six temporal statistics bound the attribution question. They discard waveform order and frequency structure and may not resolve damage whose distinction depends on those properties. A frozen language encoder can reparameterize textual context but cannot recover missing observation information. Linear numerical controls do not represent the ceiling of tuned nonlinear PHM models using complete signals. Practical superiority over mature signal diagnosis is not established by this comparison.

Few independent healthy bearings and a fixed source taxonomy including mixed damage limit the collection. Bearing isolation reduces leakage but does not establish external validity across installations, sensors, datasets, or unseen damage. The six rotations assess the same collection. Full coverage and many correlated windows do not resolve limited bearing diversity. Nominal sampling metadata supports indexing this experiment; it does not validate a raw timebase or defect-frequency interpretation.

### 7.3 Probability validity and operational meaning

Typed categories and normalized probabilities make outputs consumable but do not provide calibrated risk guarantees. The lower ECE and higher AURC of Qwen-factorized illustrate the difference between probability calibration and rejection ranking. Both calibrated Qwen and normalized-observation predictors can attain cost 0.25 by rejecting all observations. Decisions consequently require coverage and realized error alongside probability metrics.

Calibration bearings are few, and their composition affects the fitted temperature. The mean-bearing target is declared rather than estimated from deployment prevalence. The pooled/equal-bearing comparison shows tiny differences with intervals including zero, under almost equal recording counts. It supplies neither a weighting advantage nor an equivalence result. A meaningful comparison of these targets in a different deployment population would require corresponding independent-unit sampling, not artificial duplication to create a favorable effect.

The rejection cost defines a diagnostic scenario, not a maintenance policy validated with experts or downtime. Deferral is charged without modeling expert accuracy, delay, or intervention. A probability gate is not an engineering interlock. Fault-presence and fault-identity outputs are label-derived judgments, not verified physical explanations.

## 8. Conclusion

Using 32 physical Paderborn bearings, the study tests whether independent fault-presence and conditional fault-identity questions add probability or selective-decision value to a frozen Qwen3-0.6B representation of six vibration statistics. The primary paired NLL interval includes zero, an ordinary hierarchy on the direct representation has a better point score, and the six predeclared rotations reverse direction. These results do not support a stable question-conditioning increment in the stated setting.

The useful findings are the attribution boundary and the separation of observation, probability quality, and action utility. Preserving amplitude improves the observed feasibility probability scores, lower ECE does not ensure better selective ordering, and low fixed-cost outputs can reflect complete deferral. The code and saved outputs preserve these distinctions for reproduction. Stronger practical or mechanistic claims require richer validated signal evidence and broader independent equipment; they do not follow from typed validity or coherent probability composition.

## Appendix A. Fixed question meanings

Every decomposition prompt begins with this identical definition, including the actual label strings and mixed-damage qualification:

> The mutually exclusive classes are healthy, inner-race-dominant, outer-race-dominant. The healthy class is healthy; the defined fault classes are inner-race-dominant, outer-race-dominant. A dominant fault label identifies the predominant annotated damage location and may include mixed damage; it does not assert a pure isolated fault.

The fixed questions below precede the identical state:

| Query | Question |
|---|---|
| Flat A | Which one of the defined classes describes this vibration window? |
| Flat B | Choose the most appropriate class from the complete defined diagnosis classes for this vibration window. |
| Presence | Does this vibration window belong to any of the defined fault classes rather than the healthy class? |
| Conditional identity | Conditional on this vibration window belonging to one of the defined fault classes, which fault class describes it? |

The feasibility legacy query is “Represent this vibration window for fault diagnosis.” It lacks the category-conditioned diagnostic question and is analyzed only as the retained feasibility representation. No prompt contains a sample's answer or identifier. Complete templates, tokenization settings, and class order are saved.

## Appendix B. Data and software availability

The implementation and compact summaries are maintained at [PT01-jev-phm](https://github.com/liq22/PT01-jev-phm). PHMFactory/PHM-Vibench is fixed to commit `cdc0669167ca25ac848e34c880875ca39c6aed89` [@phmvibench2026]. Paderborn source data remain subject to the [provider's CC BY-NC 4.0 terms](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter), distinct from the original paper and code licenses. Large observations and hidden states are stored separately from version-controlled code.

The successful run is `run-20261013-pu-001`. It contains two feasibility evaluations, the primary decomposition comparison, and six bearing rotations under `src/outputs/jev-phm/`. Each evaluation retains `metrics.json`, `predictions.npz`, configuration, and provenance. The [compact study summary](../assets/tables/jev_study_summary.json) retains all evaluation metrics and fitted configuration, while [per-bearing NLL values](../assets/tables/jev_unit_nll.csv) support numerical reconstruction of the paired contrasts. Editable figures are generated by `src/S03_Scripts/plot_jev_results.py` from saved predictions. The [repository reproduction instructions](https://github.com/liq22/PT01-jev-phm#reproduce-the-pu-study) give complete preparation, export, window-selection, encoding, and evaluation commands using `src/S03_Scripts/run_jev_study.py`. Reproduction uses a new output directory, retaining failures and previous results.

The sole authoritative manuscript is Markdown. `src/S03_Scripts/build_jev_paper.py` builds offline HTML and PDF previews without maintaining a second text source. Author identities, funding, conflicts, target journal, and submission declarations require author confirmation and are not inferred from the repository.

## References
