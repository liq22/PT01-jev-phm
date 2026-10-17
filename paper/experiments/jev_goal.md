# Goal: Real PHM Attribution Study and Complete Manuscript

## Research question and stopping condition

At matched observations, taxonomy, supervision and final calibration, does a frozen Qwen3-0.6B question-conditioned health/fault factorization improve probability quality or selective decisions beyond flat prediction, ordinary hierarchy and two direct model calls? Unit-weighted calibration is a secondary estimation question. Types, heads, conditional factorization, temperature scaling and rejection are established foundations.

E0/E1 are milestones. Continue through E2, six predeclared bearing rotations, analysis, the complete English Markdown manuscript, editable figures and a readable PDF preview. Retain useful null or negative findings. Run completion, merge or build success alone does not establish paper completion. If evidence remains insufficient, identify the minimum necessary gap.

## Frozen inputs and prospective revisions: 13 October 2026

- PHMFactory submodule: external/PHM-Vibench at cdc0669167ca25ac848e34c880875ca39c6aed89. Its configuration resolver, data_factory, readers, datasets and native grouped partitioning own those operations.
- Data: local RM_027_PU.h5 and metadata.xlsx; Name=RM_027_PU, actual Dataset_id=20. There are 2,559 indexed recordings and 32 documented physical bearings. Label_Description is unique and constant per bearing; files, conditions and windows are dependent observations.
- Retained raw taxonomy: healthy (6 bearings), inner-dominant (13), outer-dominant (13). KB23/KB24 and KB27 include mixed damage. These mutually exclusive diagnosis categories do not assert mutually exclusive physical lesions.
- Select vibration channel 2 explicitly from the verified current/current/vibration order. The nominal rate is 64 kHz; analysis operates on actual sample indices without resampling or defect-frequency assumptions. Raw clock association remains independently unverified.
- KA08/N15_M01_F10_KA08_2.mat is unreadable in the two inspected SciPy versions and absent from the supplied metadata/H5. Disclose the unavailable recording; do not fill it or infer historical corruption origin. Raw data and metadata remain unchanged. Original PU data terms are CC BY-NC 4.0, distinct from wrapper code terms.
- Use a writable independent cache copy outside the original data directory and explicit use_cache=true. Cache writes must not reach raw files through links.
- Frozen non-thinking Qwen/Qwen3-0.6B, revision c1899de289a04d12100db370d81485cdf75e47ca; no answer generation. Public download and isolated dependency installation are authorized.
- Never use physical GPU 2. Bind visibility to the UUID of a permitted device before CUDA initialization and verify physical/logical mapping. Default physical GPU 0. No paid compute, paid APIs or private-data egress.

### Full-coverage windows and development-only selection

The original fixed 64 positions leave gaps in 82 indexed recordings. Retain upstream evenly_spaced windowing. Candidate width W is 2048, 4096 or 8192; maximum allowed start separation S defaults to W, with 1 <= S <= W. H5 lengths range from 249,940 to 299,038; actual shapes determine legal intervals and differences from metadata nominal lengths are recorded.

$$
N=1+\left\lceil\frac{L_{\max}-W}{S}\right\rceil.
$$

Check actual upstream intervals: first start 0, final stop L, width W, unique increasing starts, separation at most S and 100% union coverage. Short records fail explicitly; do not pad, drop, deduplicate silently or retain a 64-window cap. Default candidate counts are 147/74/37 windows per recording, totaling 376,173/189,366/94,683. Actual overlap varies by recording; this is not fixed-stride sampling. Greater overlap can be explicitly configured, but only W is searched here.

Within EACH protocol, Numerical-flat on the amplitude-preserving state selects W and C from the declared grids using training and tuning bearings only, by tuning-bearing-balanced uncalibrated NLL. Exact ties favor fewer windows, then smaller C. All methods share that protocol's selected observations and intervals; no input is asserted optimal for every method. Freeze all protocols, prompts and selection rules before test outcomes. Each rotation repeats selection within its own train/tune roles.

### Physical-bearing roles

Shuffle sorted documented bearing codes within each class with seed 20260922. Primary train/tune/cal/test allocations: healthy 2/1/1/2; each fault class 7/1/2/3; totals 16/3/5/8. Every available recording and operating condition follows its bearing.

Freeze six rotations. Healthy k is test, k+1 calibration, k+2 tuning cyclically, with three other healthy bearings training. Split each shuffled 13-bearing fault list into test buckets of sizes 3/2/2/2/2/2; from the remaining ordered bearings take two calibration, one tuning and the rest training. Use native frozen grouped partitions; upstream validation covers the disjoint tune/cal bearing sets.

The pinned public config schema does not expose native read-only partition fields. Resolve supported base settings with analyze_config, then visibly serialize the exact native manifest_mode/partition_map fields before the programmatic build_data call. This narrow, explicit bridge uses the existing native frozen-partition implementation; it does not claim the full public CLI validates those fields, alter the upstream pin or replace the requested split. Native required metadata verification is not extended with any custom hash system.

### Weighted supervised objective

Each window of recording j receives weight w=64/m_j, preserving total training weight 64 per source recording as coverage changes. The conditional identity head uses faulty training samples with their ORIGINAL weights, without renormalizing the subset:

$$
\sum_{\mathrm{all}}w_i\ell_{\mathrm{presence},i}
+\sum_{\mathrm{fault}}w_i\ell_{\mathrm{identity},i}
+\frac{\|W_p\|^2+\|W_t\|^2}{2C}.
$$

Use unpenalized intercepts, no class weighting, lbfgs, tolerance 1e-7, maximum 2,000 iterations and C grid 0.01/0.1/1/10/100. Record installed software versions. Paired heads share C selected on complete workflow tuning NLL. Convergence failures are failures, not permission to skip candidates. Refit every learned scaler, head, C and temperature within each rotation.

Presence and conditional-identity questions receive identical observations and full common taxonomy definitions, never each other's answers. Both execute on all test samples; only faulty training labels supervise identity. Assemble p(H)=1-r and p(k)=r*u_k. This is conditional factorization, without an output-independence assumption or new physical-concept targets.

## Experiments and predeclared interpretation

E0 verifies actual factory export, channel/labels/roles, interval coverage, normalization effects, local Qwen weights and a real forward/evaluation smoke. Keep smoke separate; prediction quality does not select the protocol.

E1 preserves numerical/Qwen crossed with raw/pooled/unit calibration. Evaluate both per-window-standardized and prospectively declared normalization=none amplitude-preserving states at the primary selected geometry. Shared observation restoration is not decomposition evidence. Nearly equal recording counts make pooled-versus-unit calibration an expected near-null; do not manufacture imbalance.

E2 uses the raw state and six arms: Numerical-flat; Numerical-factorized; Qwen-flat (A); Qwen-factorized (presence/conditional identity); Qwen-two-question-direct (A/B, fixed 50/50 probability mean); Qwen-shared-factorized (ordinary hierarchy on the saved flat-A representation, no extra encoding).

Assemble final leaf distributions before calibration. Pooled temperature is the fixed primary E2 regime; raw and unit temperature are secondary. All arms use the same scalar-temperature bounds 0.05 to 20 and calibration roles. Preserve raw factors; recompute any exposed calibrated factors from final leaf probabilities.

Primary contrast: Qwen-factorized minus Qwen-flat bearing-balanced test NLL. Required attribution contrasts use shared-Qwen hierarchy and the two-direct-question control. Numerical hierarchy gains support hierarchy; matched-call gains support added computation; restoration gains support information recovery. Favorable results still concern these selected prompts/backbone/observations/categories, not semantic causation or universal superiority. Nonsignificance is not equivalence. Preserve valid null/adverse outcomes and distinguish infrastructure failure.

## Estimand, uncertainty and saved evidence

The task predicts windows. Average window losses within each physical bearing, then give bearings equal weight:

$$
L=\frac1B\sum_b\frac1{n_b}\sum_{i\in b}-\log p_i(y_i).
$$

This is not the NLL of an aggregated bearing diagnosis. Tuning/test weights are bearing-balanced; primary pooled temperature uses uniform calibration-window weights. Report that population distinction.

Report NLL, multiclass-summed Brier, ten equal-width-bin ECE, accuracy, weighted-confusion Macro-F1, stable-tie risk/coverage and cost-sensitive decisions. Error cost is 1 and deferral cost 0.25, with deferral on a tie; these are scenario costs. Positive scalar temperature preserves argmax and binary confidence ordering; multiclass ordering can change.

Use 2,000 paired bearing-bootstrap draws, seed 20260922, for fixed-split conditional NLL intervals; they omit training/split variation. Six overlapping rotations are descriptive sensitivity evidence: show each contrast, direction and range rather than six independent replications or a naive pooled bootstrap. Two healthy primary test bearings limit inference.

Retain attempts, environment, code version, commands/configs, unit/label maps, IDs/intervals, weights and fits, pinned representations, tokens/forwards/time, all-role logits, temperatures and prediction/metric files. Reuse only compatible artifacts. Figures read saved results. Compact summaries and editable figures enter Git; large artifacts remain in src/outputs with local references.

## Full manuscript and final verification

Complete the unique English paper/draft/main.md; finalize Abstract/contributions after actual analysis. Update the existing evidence matrix and paper.yaml. Keep author/venue/funding/compliance unknowns pending. Markdown remains authoritative; HTML/PDF are previews. No official submission is authorized.

Targeted checks: actual coverage, recording weight sums, weighted leaf-loss identity, healthy exclusion, both-branch inference, no answer leakage, fixed fusion, role separation, probability coherence and temperature invariants. Recompute metrics from saved logits. Conduct one concentrated scientific/manuscript review when results and full draft exist, then fix material gaps.

Final PR checks: validate_project.py, validate_agent_skills.py --skills-dir .agent/skills, validate_skill_evals.py, and unittest discovery under src/S04_Tests. Check references/figures/preview, actual CI and merge state; merge this work only into dev, without force-push or master changes.

Bounded budget: three numerical candidate exports; up to three selected raw geometries; one normalized primary geometry; one deterministic head search per arm/protocol; seven predeclared bearing protocols. Record real smoke timing before full encoder extraction. Stop extra experimentation when it cannot change attribution or the claim boundary.

### Executed E0 and extraction budget

All three full upstream exports passed actual interval coverage (376,173 / 189,366 / 94,683 windows). The real 128-window, 32-bearing smoke passed frozen Qwen forwards and both E1/E2 evaluation interfaces; its scores are not paper efficacy evidence. Its five-query BF16 extraction used batch 128 and took 5.24 seconds including a 2.90-second model load. Query timings include unequal first-forward warmup and are not deployment latency benchmarks.

The preliminary four-window-per-record smoke subset retained the full 37-window export's weights, giving each included recording mass 6.918918919 rather than 64. Its original `smoke/data.npz` is retained. The final smoke uses the separate `smoke/data-recordbalanced.npz`, with weights 16 per included window and mass 64 per recording. Neither version supplies efficacy evidence; this smoke correction did not alter or replace any full-study observation or result.

The primary development-only selector chose 2,048 points. Parameter-free encoding of that already selected geometry may proceed while other development selectors finish; no test score is inspected until all selections are frozen. Full extraction uses BF16 and batch 256. To reduce wall time without changing observations or learned objectives, independent query/geometry cache jobs may use physical GPUs 0, 1, 3, 4, 5, 6 and 7, each explicitly restricted to its own verified UUID. Physical GPU 2 remains forbidden. Per-job device, batch, tokens and time are retained; parallel cache construction is not claimed as a deployment speedup.

The first full Qwen head candidate converged in 681 iterations and 155.83 seconds (188,160 training windows, 1,024 dimensions). Before any main test score was inspected, CPU execution was amended to allow three independent C candidates in isolated processes per protocol, with four BLAS/OpenMP threads per process. Each candidate retains the same initialization, data, weights, objective, solver, tolerance and iteration limit. Ordered candidate results and the fixed tie rule remain unchanged; any convergence failure still aborts the comparison. Up to seven independent protocols may run concurrently within the observed 144 logical CPUs and available memory. This operational parallelism is recorded and supplies no efficiency claim.

One additional training-only timing probe inspected single-thread execution after concurrent fitting became slow. On the unchanged rotation-1 8,192-point flat-Qwen training state, C=0.01 converged in 747 iterations and 873.19 seconds (881.43 seconds including the weighted scaler). It computed no tuning/test predictions or scores. Concurrent load and different scheduling prevent a fair speed-ratio interpretation; the probe supplied no basis to replace the four-thread execution budget. Its script and log remain in the run directory and are excluded from efficacy evidence.

### Completed evidence and scientific decision

Both E1 and all seven E2 evaluations completed with the declared grid and convergence contract. Primary pooled question-factorized-minus-flat NLL is -0.001102028, with conditional paired interval [-0.008577209, 0.005163370]; shared-Qwen hierarchy has lower point NLL than the question-factorized workflow. All four primary calibrated Qwen arms defer every window at the fixed cost. Each comparison with flat, shared hierarchy and two direct questions changes sign across the six refitted rotations (three lower / three higher NLL).

The stable question-conditioned increment is not established, and equivalence is not asserted. Preserve the bounded attribution result, the calibration/coverage mismatch and the separately observed input effect as the manuscript's empirical findings. No further within-test development is authorized by these outcomes. A broader advantage claim needs a prospectively frozen evaluation on additional independent equipment and appropriate application baselines; it is not supplied by more correlated PU windows. The current stopping decision is to complete the evidence-grounded candidate, readable preview and final PR checks, retaining author/venue/submission confirmations as pending.
