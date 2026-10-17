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

## References

1. McFadden and Smith. Model for the vibration produced by a single point defect in a rolling element bearing. JSV 96(1),69–82,1984. DOI:10.1016/0022-460X(84)90595-9. Original publisher abstract inspected: https://www.sciencedirect.com/science/article/abs/pii/0022460X84905959.
2. He, Ding and Lin. Fault feature extraction of rolling element bearings using sparse representation. JSV 366,514–527,2016. Original publisher methods/simulation excerpts inspected: https://www.sciencedirect.com/science/article/abs/pii/S0022460X15010068. Numerical parameters here are declared choices, not attributed to the paper.
3. Zombori et al. Towards Unbiased Exploration in Partial Label Learning. JMLR25(412),1–56,2024. https://jmlr.org/papers/v25/23-0868.html. Relevant loss text and official `log_prp_Loss` source were inspected; allowed-count scaling and detached weighting are retained.
