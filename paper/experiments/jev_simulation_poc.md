# Jev-PHM: vibration-simulation proof of concept

Date: 2026-10-17. Base: `dev@541d52d184fa053146f60574114ccb82ba9890f5`. Completed output directory in the delivered artifact package: `outputs/run-sim-002`. **Computed model-based simulation, not measured machine data. CWRU is excluded; no real-dataset signal was read.** Existing PU results are unchanged.

## 1. Question and minimal refinement

### 1.1 Scientific question

Does preserving the fine conditional distribution while learning a coarse-labelled second source improve held-out diagnostic NLL beyond ordinary event likelihood and simple conditional freezing?

The earlier hard projection can block useful updates and is a restricted conditional-freezing parameterization in its fixed linear setting. This experiment therefore tests a source-selected soft conditional KL penalty, including zero, rather than declaring projection the final method. Conditional regularization is not claimed as a new generic algorithm. The experiment isolates the update mechanism using physical signal features and a small direct head; it does not execute Qwen, train text embeddings, reproduce proprietary Jev, establish backbone portability or measure acceleration.

### 1.2 Objectives and mathematical boundary

States are healthy, inner-ring fault and outer-ring fault. The fine source exposes all three labels; the coarse source exposes healthy or the union F of the two mutually exclusive fault states. Let v contain the common 46 signal features and an intercept, with source-training-only standardization. A linear head W gives a three-class softmax. Source expectations are equal-unit, then equal-window averages:

$$
\mathcal L_0(W)=\tfrac12\mathbb E_f[-\log p_W(y\mid x)]+\tfrac12\mathbb E_c[-\log\sum_{k\in S(y)}p_W(k\mid x)]+\tfrac\gamma2\|W_{1:46,:}\|_F^2.
$$

A fine-source model supplies a detached conditional distribution q0 among the two faults. The candidate objective is

$$
\mathcal L_\lambda(W)=\mathcal L_0(W)+\tfrac\lambda2\mathbb E_c[\mathbf1\{S(y)=F\}D_{\mathrm{KL}}(q_0(\cdot\mid F,x)\|q_W(\cdot\mid F,x))].
$$

The anchor is a prediction, not the hidden fine label. Eligible examples retain their actual source mass. Lambda zero is the same-code ordinary update at matched settings. Positive lambda discourages sampled-input changes but does not prove preservation on unseen inputs or lower target risk. An infinite finite-sample penalty is not automatically global functional freezing.

The simple hard control adds one linear residual to both fault logits:

$$
z^+(x)=z^0(x)+(0,\delta(x),\delta(x)).
$$

Its inner/outer logit difference, and therefore log probability ratio, is unchanged for every input. The coarse fault odds are multiplied by exp(delta), retaining the initial log-mass offsets. This proves preservation, not generalization. Its free coefficients and penalty differ from a full head.

The coarse Libra branch reproduces the released `log_prp_Loss(use_weighting=True)` formula with allowed-count factor s and detached coefficient:

$$
\ell_L=\operatorname{stopgrad}(1-m)[s\log\max(1-m,10^{-5})-\sum_k a_k\log(p_k+10^{-5})],\quad m=\sum_k a_kp_k.
$$

Fine-source learning remains cross-entropy. The authors' implementation was read directly at `agapiR/partial-label-learning/utils/utils_loss.py`, blob `480ea9c5793c51f6a9d7e8e4069d2511807e0c07`. This is a controlled use of that loss, not replication of its entire paper benchmark.

## 2. Physical model and frozen protocol

### 2.1 Signal generation

The reduced model combines jittered periodic fault impacts, inner-ring load modulation, a causal underdamped transfer response, class-independent random impacts, shaft harmonics and coloured noise:

$$
x(t)=\sum_j A_jh(t-t_j)+r(t)+\epsilon(t),\quad
h(t)=e^{-2\pi\zeta f_nt}\sin(2\pi f_n\sqrt{1-\zeta^2}t)\mathbf1_{t\ge0}.
$$

Ideal repetition orders are N(1+chi cos(alpha))/2 for inner-ring and N(1-chi cos(alpha))/2 for outer-ring. Declared fictitious geometry N=8, diameter ratio 0.2, angle zero gives 4.8 and 3.2; it was not taken from prohibited data. Consecutive intervals are jittered in shaft-cycle coordinates; fractional impulse times are deposited on adjacent samples. Inner-ring amplitudes use modulation 1+0.6 cos(shaft phase).

| Quantity | Design value |
|---|---|
| Sampling / windows | 16 kHz; 8192 samples (0.512 s); four nonoverlapping windows per virtual unit; 0.2 s guards |
| Fine-source speed / resonance | 20–24 Hz / 2.4–2.8 kHz |
| Coarse-source speed / resonance | 25–29 Hz / 3.0–3.4 kHz |
| Target speed / resonance | 34–38 Hz / 4.0–4.4 kHz |
| Damping / gain / impact amplitude | Uniform 0.025–0.045 / 0.7–1.3 / 0.6–1.4, identical distributions across classes |
| Interval jitter / nuisance | 1% relative interval SD; class-independent Poisson impacts 8/s, amplitude SD 0.3 |
| Coloured noise | AR(1) coefficient 0.5 |
| Source / target reference SNR | 6 dB / -10, 0, +10 dB |
| Combined mismatch | At 0 dB: +/-20% speed ramp, 8% interval jitter, second 6.5-kHz mode with weight 0.7 |

Noise is scaled by a class-independent nominal impulse-response energy, including healthy units. Reference SNR is not the measured per-waveform fault/noise ratio; amplitudes are arbitrary model units. The model follows the impulse/transfer/modulation principles of McFadden and Smith (1984) and the nonstationary impulse considerations of He et al. (2016), not their fitted parameters or a calibrated contact-dynamics model. All numerical ranges are declared simulation choices.

Every method sees the existing six temporal statistics plus 40 fixed squared-envelope order-energy bins, using a 1.5–5.5-kHz bandpass and observed mean shaft rate. No defect label, event times, true noise level, resonance, unit ID or domain ID enters the classifier. The fixed order grid is not tuned to the true class. Mean-rate normalization does not remove intra-window speed changes; the fixed band can omit the added mismatch resonance.

### 2.2 Information, tuning and statistical units

Protocol/configuration were written before fitting or target generation. Three independently generated cohorts use seeds 101, 202, 303. Each has 72 fine-training, 144 coarse-training, 36 fine-tuning, 36 coarse-tuning and 90 target units. Total: 1134 virtual units, including 270 target units. Windows and perturbations are not extra units. Fine target truth is withheld from selection; coarse training/tuning tensors expose only the coarse mask.

Fine-only initialization is selected on fine-source tuning NLL. All other heads start from the same selected weights; model selection averages fine-tuning and coarse-event tuning NLL. The scaler uses the two training sources only. No calibration or target-based threshold choice is performed.

Each linear method has six declared configurations, 400 full-batch Adam steps, learning rate 0.03, validation every 20 steps including step zero. Ordinary ridge values are 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03. The soft grid crosses ridge 0.001/0.01 with lambda 0/0.3/3. These are bounded searches, not exhaustive optimality. RF uses 150 trees, depth 4/unlimited and minimum leaf 1/4/12, trains on fine-source labels only, and uses the same source validation events. It is an information-limited practical reference, not an identical-supervision control.

There are 108 distinct configured fits (90 linear, 18 RF). Tool timeouts caused the first 72 to be repeated unchanged; they are not counted as extra evidence or an expanded search. A resume bookkeeping error was corrected with the exception retained; intact search records were recovered only after corresponding saved head arrays were verified identical. Models and predictions were then resumed. No physics, cohort seed, hyperparameter grid or selection rule changed after target inspection. Full failures and outputs are preserved in the delivered artifact package; final-resume time is not total training or deployment latency.

Primary comparison: soft-anchor minus joint-event NLL at reference SNR 0 dB. SD is between three simulated cohorts. Paired virtual-unit bootstrap intervals are conditional on those selected models; secondary effects are descriptive and unadjusted. Paired secondary conditions are noise, combined mismatch, phase randomization preserving raw Fourier magnitudes, and circular shift. The last is approximately invariant because of filter boundaries; phase randomization is a destructive control, not a new physical fault class.

## 3. Actual results and falsification

### 3.1 Main comparison

| Method | NLL mean +/- cohort SD | Accuracy (%) mean +/- cohort SD | Selected extra-update steps |
|---|---|---|---|
| Fine-only | 0.002140 +/- 0.002273 | 99.907 +/- 0.160 | None |
| Joint event | 0.002140 +/- 0.002273 | 99.907 +/- 0.160 | 0 / 0 / 0 |
| Weighted Libra | 0.002140 +/- 0.002273 | 99.907 +/- 0.160 | 0 / 0 / 0 |
| Soft conditional anchor | 0.002140 +/- 0.002273 | 99.907 +/- 0.160 | 0 / 0 / 0; lambda zero |
| Hard conditional freeze | 0.001321 +/- 0.001659 | 99.907 +/- 0.160 | 20 / 100 / 60 |
| RF fine-source reference | 0.163611 +/- 0.021413 | 99.537 +/- 0.321 | Source-selected tree configuration |

The first four selected predictors have identical saved probabilities. Primary delta and its empirical conditional interval are exactly zero because the same classifier was selected, not because equivalence of algorithms was established. Source joint validation already lay between 7.34e-6 and 1.06e-5 at initialization. The experiment therefore did not activate an incremental soft-preservation treatment in the selected model.

Hard freezing changes probability estimates slightly: mean delta versus joint event -0.000819, descriptive conditional 95% interval [-0.001936,-0.000142], without primary accuracy change. This is not an industrially meaningful advantage or a new algorithm contribution. RF probabilities were not calibrated; its higher NLL despite similar accuracy should not be misrepresented as a fully optimized probability comparison.

### 3.2 Mechanism and failure conditions

At fixed ridge 0.001, the executed trial's final mean absolute inner/outer log-odds drift on the coarse-tuning pool is 2.5270, 2.3652 and 1.7178 for lambda 0, 0.3 and 3. This demonstrates a diagnostic effect of the penalty, not a target-risk improvement: all final selected soft heads return to step zero. The histories include joint fine/coarse updates and regularization, not isolated coarse gradients. Hard-freezing ratios remain unchanged within 7.22e-15 in saved predictions.

| Target condition | Common selected linear NLL / accuracy | Hard freeze NLL / accuracy | RF NLL / accuracy |
|---|---|---|---|
| +10-dB reference SNR | 0.000738 / 99.907% | 0.000347 / 100.000% | 0.023888 / 99.907% |
| Primary 0 dB | 0.002140 / 99.907% | 0.001321 / 99.907% | 0.163611 / 99.537% |
| -10 dB | 3.580656 / 54.537% | 3.358207 / 55.556% | 3.774960 / 33.519% |
| Combined mismatch | 3.507442 / 40.370% | 3.029552 / 43.796% | 1.172086 / 35.463% |
| Phase randomization | 8.789076 / 33.333% | 8.429064 / 33.333% | 6.925614 / 33.333% |
| Circular shift | 0.005516 / 99.907% | 0.003247 / 99.907% | 0.207287 / 98.519% |

Full compact values: [simulation table](../assets/tables/jev_simulation_poc.csv). Fourier magnitudes survive the phase control but temporal organization does not; its accuracy collapse is consistent with dependence on impulse-envelope structure, not proof that the soft constraint discovered physics. Circular shifting preserves linear accuracy but not exact NLL. Combined mismatch is deliberately one stress case; no individual attribution to speed, slip or transfer is made.

### 3.3 Judgment and next decision

- Declared simulator mechanics and order extraction: SUPPORTED within the reduced model, not validated against a real machine.
- Penalty reduces trial-path conditional drift: SUPPORTED as a matched source-side diagnostic.
- Soft anchoring improves source-selected target diagnosis: INCONCLUSIVE / no incremental support; the chosen treatment is inactive.
- The selected model remains accurate under the declared combined mismatch: REFUTED in this simulated case.
- Real industrial transfer, semantic pretrained backbones and speed: UNTESTED.

Do not adjust simulator parameters to force the candidate to win. The next minimal diagnostic is source-only: establish from the saved source observations and simulator evaluation truth whether the initial predictor is missing any identifiable fine distinction in the coarse source. If it is already saturated, stop this candidate in the current task rather than adding complexity. Any subsequent simulation revision must have an independent physical/application justification and new confirmation data.

CWRU is prohibited and its former pending qualification is obsolete. Future industrial work requires an allowed dataset, verified physical-unit identities, measurement provenance and a real compatible annotation relation. Synthetic label coarsening is not evidence that an industrial source has native coarse labels. The old PU mixed/dominant-fault counterexample remains relevant.

## 4. Reproduction and actual verification

### 4.1 Commands

From a complete existing checkout, leaving `jev_phm.py` intact:

```bash
python -m unittest discover -s src/S04_Tests -p test_jev_simulation.py -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python src/S01_Package/jev_simulation.py \
  --config src/S02_Configs/jev_simulation_poc.json --out src/outputs/jev-phm/sim-poc-NEW
python src/S03_Scripts/check_jev_simulation_results.py --run src/outputs/jev-phm/sim-poc-NEW
python src/S03_Scripts/plot_jev_simulation.py --run src/outputs/jev-phm/sim-poc-NEW \
  --out paper/assets/figures/jev-simulation-NEW
```

Use a new output directory; `--resume` only accepts the same saved scientific configuration after an actual interruption. Plotting/checking read saved outputs without model calls. Figures are waveforms, raw spectra, squared-envelope orders, primary NLL, executed-trial drift and noise sensitivity, in editable-text SVG and PNG. Full predictions, source features, models, configuration, search histories and failure logs are included in the conversation's downloadable run package rather than large Git blobs.

### 4.2 Verification boundary

Twelve targeted tests passed. Recomputing 108 saved prediction/metric rows agreed within 1.78e-15; NLL decomposition and disjoint source/target units were checked. Observed equality of selected models is recorded, not required by the result validator. The fixed simulator and scientific configuration did not change after target inspection.

A full Git clone was unavailable in the authoring container. Local execution imported a separately labelled exact excerpt of the two retrieved existing helpers (`signal_features`, `unit_weights`), not a mock reader or model. That excerpt is not committed and must never overwrite the full module. Full repository integration/suite and remote CI remain unchecked; the previous 87-test result is not a current check. No PHMFactory export or language-model forward pass was claimed. Figure/report builds and saved-result checks are separate from machine-validity evidence.

## 5. Method clarification: what the conditional anchor does and does not learn

This analysis was added after `run-sim-002`, against `dev@193127a909a4fb9abf8e4539247fe6b30a8bcbe6`. It clarifies the implemented objective; it is not a new training protocol, a replacement result, or a claim to have invented distillation. Sections 1–4 and every saved simulation result remain the historical execution record. The scalar identities below use exact real arithmetic; general proofs, assumptions and implementation consequences are stated explicitly.

### 5.1 Definitions, assumptions and input permissions

The operational question remains whether limiting fine-conditional changes helps diagnosis when an added source exposes only a valid coarse fault event. The minimal evaluated model takes a vibration-derived vector `v` of 46 source-standardized features plus an intercept, so `v` has dimension 47. It uses `W` of dimension 47 by 3 and dimensionless logits `z = W^T v`, with class order H/I/O. The input features use only the declared waveform and observed shaft rate, not fault identity, source identity, noise truth or test labels. Their physical assumptions remain those in Section 2. No language backbone or semantic candidate encoder was executed in this instance.

For the analysis, extend the output only in notation to a finite set of K mutually exclusive diagnostic states. Let S be a nonempty, verified preimage of an observed annotation. For finite real logits, define

$$
p_j=\frac{e^{z_j}}{\sum_{l=1}^{K}e^{z_l}},\qquad
m_S=\sum_{j\in S}p_j>0,\qquad
q_j=\frac{\mathbf1\{j\in S\}p_j}{m_S}.
$$

A fixed initial predictor gives a normalized conditional distribution `q^0` on S; extend it by zero outside S. Its parameters and predictions are detached. It is a model prediction, not a new fine label. In the implemented three-class case the anchor is evaluated only for coarse-source observations with S={I,O}; health remains ordinary singleton supervision. The notation-level generalization does not add support for arbitrary partitions to the code.

Use lambda >= 0, fixed finite teacher logits, positive score temperature when present, and natural logarithms. All logarithm arguments on S are positive; zero terms outside S are omitted. The probabilities, losses and lambda are dimensionless, with loss reported in nats. If logits are defined as raw similarities divided by tau, gradients with respect to those similarities acquire the factor 1/tau. None of these quantities is a physical vibration amplitude or inference time.

### 5.2 Exact loss interpretation and gradient

For an eligible observation, the implemented coarse objective is

$$
\ell_\lambda=-\log m_S+\lambda D_{\mathrm{KL}}(q^0\|q).
$$

Let `CE(q^0,p)=-sum_{j in S}q^0_j log p_j` and `H(q^0)=-sum_{j in S}q^0_j log q^0_j`. Then

$$
\ell_\lambda=(1-\lambda)(-\log m_S)
+\lambda\operatorname{CE}(q^0,p)-\lambda H(q^0),
\qquad
\frac{\partial\ell_\lambda}{\partial z_j}
=p_j+(\lambda-1)q_j-\lambda q^0_j.
$$

**Proof.** Substitute `q_j=p_j/m_S` in the conditional KL and use `sum_{j in S}q^0_j=1`. Its expansion is `-H(q^0)+CE(q^0,p)+log m_S`, giving the first identity. For finite softmax logits, `partial(-log m_S)/partial z_j=p_j-q_j`, while `partial CE(q^0,p)/partial z_j=p_j-q^0_j`. Combine these derivatives; the fixed entropy has zero derivative. This proves the gradient for every class, including outside S, where both extended conditional probabilities are zero. QED.

At lambda=0 this is the ordinary event likelihood. At lambda=1 it is exactly full-class soft-target cross-entropy with zero target mass outside S, minus a teacher-dependent constant. For 0<=lambda<=1 it is a convex combination of those *loss terms* up to a constant; this is not a claim that the loss is convex in arbitrary model parameters. For lambda>1 the same algebra holds, but the coefficients are not a convex mixture; the original event-plus-KL expression remains nonnegative. No negative coefficient is a reason to silently cap the existing lambda=3 experiment.

Thus the anchor adds a fixed conditional preference, not observed fine-state information. LwF already uses frozen old predictions as distillation targets; the present conditioning on a known event and replay protocol must be distinguished from that earlier setup, not advertised as a new distillation principle. Libra's probability-ratio property concerns its stated parameterization; it is not a guarantee for this KL objective or an arbitrary semantic head.

At the common initialization, `q=q^0`, so the KL derivative is zero. With the same ridge, data, weights, optimizer state and arithmetic, the first parameter update equals the lambda=0 update for every lambda. Subsequent updates need not coincide. Selecting step zero makes the returned predictor the initializer regardless of how later trajectories behaved. This explains what an inactive selected treatment means; it does not explain *why* a particular source criterion selected it, prove algorithm equivalence, or establish that hidden coarse-source fine labels were already learned.

### 5.3 A population counterexample to unconditional benefit

The next statement is an idealized counterexample boundary, not a statistical guarantee for the finite linear experiment. Fix an observation x. Assume the fine and coarse sources share the same conditional state distribution at x, their local objective weights are `w_f>0` and `w_c>=0`, all true and teacher conditional probabilities are positive, and `0<m^*<1` is the true fault-event probability. The student can choose its event mass m and fault conditional q freely. There is no parameter penalty or cross-input tying in this pointwise calculation. The teacher `q^0` is a fixed possibly inaccurate finite-data predictor; it is not assumed to be the population-optimal fine classifier.

The pointwise population objective decomposes as

$$
\mathcal R_\lambda(m,q)
=(w_f+w_c)\operatorname{CE}(\operatorname{Bern}(m^*),\operatorname{Bern}(m))
+m^*\left[w_f\operatorname{CE}(q^*,q)+w_c\lambda D_{\mathrm{KL}}(q^0\|q)\right].
$$

Its unique interior minimizer is

$$
m_\lambda=m^*,\qquad
q_\lambda=\frac{w_f q^*+w_c\lambda q^0}{w_f+w_c\lambda}.
$$

**Proof.** The Bernoulli cross-entropy is uniquely minimized at the true event probability. In the conditional term, collect the coefficients of `-log q_j`; they are `w_f q^*_j+w_c lambda q^0_j` and sum to `w_f+w_c lambda>0`. Therefore that term is a positive multiple of cross-entropy with target `q_lambda`, plus a constant. By nonnegativity of KL, its unique minimum is `q=q_lambda`. Both solutions lie in their original open-simplex domains; substitution makes the original stationarity equations zero. QED.

Relative to the unrestricted population solution at lambda=0, the fine-label risk excess is exactly

$$
R^*(p_\lambda)-R^*(p_0)
=m^*D_{\mathrm{KL}}(q^*\|q_\lambda).
$$

It is strictly positive whenever `w_c lambda>0` and `q^0 != q^*`; it is zero when the teacher is correct or the anchor has zero weight. For an explicit interior counterexample, take `w_f=w_c=lambda=1`, `m^*=1/2`, `q^*=(3/4,1/4)` and `q^0=(1/4,3/4)`. Then `q_lambda=(1/2,1/2)` and

$$
R^*(p_\lambda)-R^*(p_0)=\frac18\log\frac{27}{16}>0.
$$

This exact analytical quantity is not a simulation score or an industrial effect size. It rules out an unconditional improvement guarantee even without measurement shift. In finite data, regularization may still reduce estimation variance or help optimization; neither the sign nor size of that net effect follows from this population calculation. The current simulation's result is not attributed retrospectively to teacher error.

With `w_c>0`, `lambda -> infinity` gives `q_lambda -> q^0` in this pointwise problem. With no coarse weight, lambda has no effect. If `w_f=0` and `w_c lambda>0`, the conditional solution merely copies the teacher; if also `w_c lambda=0`, the conditional is unidentified. These zero-weight branches must not be lost by cancelling the denominator. A source with only singleton annotations supplies ordinary CE and no nontrivial conditional penalty. If S contains every class, event likelihood supplies no label information and the penalty only matches the teacher.

Finite-sample large-lambda fitting does not establish global functional freezing. For example, a linear I/O logit difference can change by `c v_2` while all eligible training features have `v_2=0`. Training conditional KL then stays exactly zero for any c, but a new input with `v_2=1` changes its I/O log-odds by c. The hard residual construction in Section 1.2 instead preserves the ratio algebraically at every input. Those two statements must not be interchanged.

### 5.4 Minimal algorithm and implementation constraints

Keep the current measured-feature extractor, one direct head and the detached fine-source anchor. No projector, teacher refresh, new semantic module or extra inference branch is justified by this derivation. The ordinary event update remains the reference, and a nonzero conditional penalty remains a candidate regularizer rather than a demonstrated contribution.

The existing implementation follows this sequence: fit the source-only standardizer and fine head; freeze the teacher; build coarse allowed masks without hidden fine truth; calculate fine CE, coarse event NLL and the eligible conditional KL with their original unit/source weights; optimize only the declared student parameters; select by fine-tuning NLL plus coarse-tuning event NLL, including step zero; return the selected head for direct probability prediction. The source validation criterion does not include the penalty itself or the coarse source's hidden fine labels. Inference needs no teacher evaluation. All algorithmic selections stay source-side.

Use logsumexp for event likelihood and within-S log_softmax for the conditional; do not divide two underflowed class probabilities. Teacher probabilities must remain normalized and detached. An empty S is invalid, singleton S needs no penalty, and eligibility must not be renormalized to silently upweight faulty observations. The current numerical floor in the teacher-log term changes only its finite-precision constant when the teacher is fixed; underflow or loss-value checks must still be disclosed rather than presented as exact symbolic arithmetic.

A same-code zero-penalty control means identical ridge and all other choices. The archived grids contain six event ridge values but only two ridge values crossed with the three anchor strengths. They are not nested candidate sets; including lambda=0 alone does not guarantee that the anchor search can return every event-search winner. Do not retrospectively enlarge the grid or alter saved selections. For a future mechanism comparison, use matched existing ridge settings or explicitly retain the already fitted event winner as a no-extra-fit selection candidate; report that protocol change separately. Candidate-set inclusion can guarantee no worse *selected source criterion*, not lower target risk.

| Necessary object | Simplest alternative | Distinguishing prediction or removal consequence |
|---|---|---|
| Verified coarse event likelihood | Exact labels or one declared common coarse task | Exact labels reduce to CE; invalid events invalidate the stated likelihood |
| Fixed conditional anchor | lambda=0; lambda=1 soft-target CE | Zero is event likelihood; one is the exact distillation control, not a separate new mechanism |
| Soft preservation | Offset hard residual | Soft acts only through sampled loss; hard preserves all-input ratios but cannot repair their errors |
| Source-only selection | Prespecified fixed step, reported separately | Step-zero selection provides no incremental deployed treatment; hidden fine error can remain unknown |

### 5.5 Symbolic verification and the first empirical check

The user-specified SymPy skill was read at commit `a1043ef544e3a100c28e70a736b6cf32554d0677`, including its assumptions and core-calculus guidance. The working runtime was Python 3.13.5 and SymPy 1.14.0; no installation, data read, waveform generation, model forward pass, fitting or target reanalysis was performed. Exact checks covered the three-class loss identity, all logit derivatives, the zero initial penalty gradient, the lambda=1 identity, population stationarity by substitution, the open-simplex Hessian expression, boundary limits and the rational counterexample. A first generic-real-symbol ratio-log simplification remained unresolved; the check was corrected to the expanded real-log expression valid on the explicitly stated positive domain, without forced complex-log identities. That check failure did not change a model result.

The preceding proofs, not a successful computer simplification, establish the stated conditions. The symbolic checks do not certify arbitrary neural optimization or generalization. A self-contained record of the exact expressions is supplied with this revision's downloadable `symbolic_checks.py`; SymPy is not made a dependency of the experimental runtime or existing CI.

The first empirical action remains the STATUS source-only information diagnostic on saved `run-sim-002` source inputs, selected heads and virtual-unit truth: separate coarse-event NLL from fault-conditional NLL and error, with one observation weight per declared unit. Hidden fine truth is used by the evaluator only. Near-zero coarse-event NLL is not a substitute for this check. If the source conditional is already learned, stop seeking a preservation gain in that task; if it is wrong, establish whether the error is identifiable from permitted supervision before testing another constraint. This diagnostic has not been run in the present method-only revision. No target-driven simulator revision or restoration of CWRU is authorized.

Relevant prior methods are Zombori et al., JMLR 2024, Section 4.3 and Appendix B, and Li and Hoiem, *Learning without Forgetting*, TPAMI 2018, Section 3 of the accessible author version (arXiv:1606.09282v3). The references already exist in the manuscript bibliography. These elementary identities and the constructed boundary constrain the candidate; they are not claimed as a new general distillation or coarse-label theorem.

## References

1. McFadden and Smith. Model for the vibration produced by a single point defect in a rolling element bearing. JSV 96(1),69–82,1984. DOI:10.1016/0022-460X(84)90595-9. Original publisher abstract inspected: https://www.sciencedirect.com/science/article/abs/pii/0022460X84905959.
2. He, Ding and Lin. Fault feature extraction of rolling element bearings using sparse representation. JSV 366,514–527,2016. Original publisher methods/simulation excerpts inspected: https://www.sciencedirect.com/science/article/abs/pii/S0022460X15010068. Numerical parameters here are declared choices, not attributed to the paper.
3. Zombori et al. Towards Unbiased Exploration in Partial Label Learning. JMLR25(412),1–56,2024. https://jmlr.org/papers/v25/23-0868.html. Relevant loss text and official `log_prp_Loss` source were inspected; allowed-count scaling and detached weighting are retained.
4. Li and Hoiem. Learning without Forgetting. TPAMI, 2018. Accessible author version: https://arxiv.org/abs/1606.09282v3, Section 3. Frozen prediction targets are prior work; this revision does not claim a new distillation principle.
