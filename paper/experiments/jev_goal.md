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

## Approved P1 continuation: PU single-ring source to MFPT external target

### Authorization and scientific scope — 17 October 2026

The author explicitly replied “确认并继续” to the 17 October P1 protocol and its complete table/figure templates. This approves the PU single-ring-component population, MFPT known-label test-rig target, the stated execution matrix, and a ceiling of **24 GPU-hours plus 12 CPU wall-hours**. Do not request the same authorization again. Approval does not waive data, implementation or information-access prerequisites. The preceding E0–E2 record remains historical and completed; its old parallelism/window/weighting rules do not override P1.

P1 tests the current manuscript Section 4 framework: signal adapter, compatible frozen first-instance backbone and shared fault-description scores, against an indexed head on the same backbone. Conditional KL and projection are not required components. A correct-description versus fixed mismatched-description control tests semantic correspondence. C4's native coarse-label subclaim is not tested by these two exact-label datasets. C5 concerns measured diagnostic error and latency, not a construction-only speed claim.

CWRU data and explicitly CWRU-trained industrial weights remain excluded. Physical GPU 2 is forbidden. At most one GPU job and eight CPU threads are active. No paid service, new cluster allocation or private-data egress is authorized. A second backbone, multi-source training with a third target, a replacement dataset, or a changed diagnostic population is outside P1.

### G0 executed: fixed PU population and roles

The new [P1 bearing assignments](../assets/tables/jev_p1_pu_unit_roles.csv) were generated from the 32 `primary` rows of the existing [PU role table](../assets/tables/jev_unit_roles.csv) at `810cc01fccd0c34539b03e8f3f8009e5d841f344`. Published single-ring categories use the existing `PUB_SITES` interpretation in `analyze_jev_annotation_compatibility.py`, not identifier-prefix fault inference. Remove KB23/KB24/KB27 from the NEW P1 population only; preserve all 240 associated records and the historical 32-bearing experiment unchanged. The remaining categories are healthy, inner-ring damage and outer-ring damage; microscopic single-defect purity is not claimed.

Sort physical bearing codes within each class; use one Python-standard-library `random.Random(20261017)` instance, shuffling classes in H/I/O order. Consume each shuffled class in train/tune/cal/test order. This fixes an implementation detail left open in the accepted plan before any new signal/model result. Original source label integers and record counts remain in the CSV; no raw metadata is changed. The declared map is consumed by PHMFactory's existing frozen grouped-partition path, not by a new data loader.

| Class | Train | Tune | Calibration | Internal test | Indexed records |
|---|---:|---:|---:|---:|---:|
| Healthy | 3 | 1 | 1 | 1 | 480 |
| Inner-ring damage | 6 | 2 | 1 | 2 | 880 |
| Outer-ring damage | 7 | 2 | 1 | 2 | 959 |
| Total | 16 | 5 | 3 | 5 | 2,319 |

Indexed records by role are 1,279 / 400 / 240 / 400. KA08 retains 79 indexed records; its missing historical recording is not filled. Checks performed on this metadata projection: 32 unique original bearings and 2,559 records; exactly the three approved exclusions and 240 retained-outside-P1 records; 29 unique P1 bearings; per-class allocations; disjoint roles; complete accounting of 2,319 records. This establishes the roster, NOT raw H5 availability, timebase/units, spectral validity, model efficacy or an independent new PU confirmation.

### G0 executed: MFPT publication and remaining unit boundary

The MathWorks publisher mirror [README](https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data/blob/master/README.md) and [license](https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data/blob/master/license.txt) explicitly identify the original data owner and **CC BY-NC-SA 4.0**. The permission boundary was verified for that distributed version; it is not a certification of the user's processed H5 provenance, and a repository wrapper's software license is not substituted for data terms. No raw data is redistributed in this change.

The complete public tree `d3efefb6ce84fa1ee6c0311f80f7c89cf903ad1d` contains 20 MAT files under publisher train/test folders. Together with the [publisher processing documentation](https://www.mathworks.com/help/predmaint/ug/Rolling-Element-Bearing-Fault-Diagnosis.html), this verifies the distributed known-label inventory: 3 baseline, 10 outer-ring and 7 inner-ring recordings. These are recordings, NOT 20 verified bearings. Both folders belong entirely to the P1 external target; their example train/test split supplies no permissions to tune on MFPT.

No physical-specimen join was supplied by the inspected README, license or file tree. Actual `bearing` payload fields and the local `RM_007_MFPT.h5` correspondence remain UNKNOWN; filename spelling is not used to certify their contents. The old MFPT provider URL now redirects to the ASNT affiliate page; the original data-acoustics page could not be retrieved. These source-access limitations do not prove that a unit map does not exist elsewhere.

The authoring container does not mount `/home/user/data/PHMbenchdata/PHM-Vibench`, its `metadata.xlsx`, either real H5, or the user's GPU. Git clone also fails at DNS resolution. A remote-execution integration search returned no usable connected SSH/terminal action. The earlier Library notes describe the user's data root but are not current bytes or a remote execution session. No target waveform, class-conditioned target statistic or target prediction was read. G0 remains BLOCKED for the approved device-balanced external evaluation, not failed as a method test.

### Approved input and evaluation contract

Train only on the new PU train role, select only on PU tune, use PU cal only for declared post-hoc sensitivity, and freeze every selected instance before external prediction. PU internal evaluation remains exploratory because earlier PU results influenced research design. The same checkpoint is used on PU internal and all eligible MFPT records; do not add reverse-direction training and call it the same instance.

Approved initial observation: one documented acceleration channel, 25.6 kHz after qualified anti-alias processing, 8,192-point (0.32 s) nonoverlapping windows, no crossing original records, and no padding of incomplete tails. Record discarded tail duration. This is not the older full-coverage-window protocol. Nominal sampling and empty units cannot certify the new physical transform; source/target time and channel provenance are hard prerequisites. Transform settings and units must be resolved without optimizing on target spectra. All model families receive the same observation. Use neither original rate as a text class cue nor filenames, bearing/source ID, original length or target truth as model inputs.

Initial semantic interface: 64 patches of 128 samples, a source-trained signal projection, frozen first backbone, declared pooling and one shared scoring module. Candidate embeddings are frozen. Qwen3-0.6B at the earlier pinned revision is only the first eligible instance, conditional on a real verified cache and signal interface. The complete interface/runner remains unverified; neither an invented command nor the old PU-specific `prepare` command authorizes it.

Main contrast is MFPT JEV-minus-B_IDX raw NLL. For each verified physical unit, average windows within each original recording, then records within the unit, then give units equal weight. This differs from the old direct unit-window estimator; do not pool results. Macro-F1 comes from the resulting weighted confusion matrix, not a mean of per-bearing Macro-F1. Three neural fits use seeds 101/202/303; search runs on 101, then two refits use the selected setting and the same source stopping rule. No seed selection. Use a paired physical-unit interval conditional on these fits where estimable, and report fit variability separately. Unknown or singleton-per-class specimen structure cannot be rescued by file/window bootstrap. No numerical superiority or no-loss-acceleration threshold has been established; report effect estimates and trade-offs without inventing such a threshold.

### Approved finite run matrix

| Job | Information | Fits / search | Ceiling and stopping rule |
|---|---|---|---|
| G0b, real-data qualification | Metadata/provenance; no classifier outcomes | 0 | CPU 30 minutes per source; unresolved required fields stop admission |
| G1, real interface smoke | PU train only | Not an efficacy comparison | 1 GPU-hour total; actual data and every model input must be consumed |
| G2a, B_RF | PU train/tune | 6 | 300 trees; depth 8/None x leaf 1/4/12; 20 CPU minutes per fit |
| G2b, B_SIG/B_IDX/JEV/JEV-PERM/B_TOK | PU train/tune | 5 x 6 = 30 | LR 1e-4/3e-4/1e-3 x decay 0/1e-2; 30 min AND 2,000 updates per fit |
| G3, selected neural settings | Same PU roles | 5 x 2 = 10 additional fits | Same per-fit cap; never read MFPT to choose a setting |
| G4/G5, calibration sensitivity, frozen evaluation and timing | PU cal; then sealed tests | 0 new fitting families | 1 GPU-hour shared allowance; timing 30 warmups then 200 measurements per readout in five interleaved blocks |
| Technical recovery | Unchanged scientific config | At most one retry per failed job, two globally | Within 2 GPU-hour reserve and total CPU cap; keep failed attempts |

Total approval is 46 fitting runs: 40 neural and 6 RF. GPU allowance is 20 hours fitting + 1 interface + 1 evaluation/timing + 2 recovery = 24 GPU-hours. CPU wall time, including preparation/checking/RF/analysis/recovery, is at most 12 hours with eight threads, not 12 core-hours. A cap hit before qualification is uncertainty, not permission to weaken a baseline. This G0 execution used zero GPU hours and 0/46 fits; CPU metadata operation time was not separately instrumented, so it is not reported as zero.

For neural methods use AdamW, validation every 100 updates including the initializer, and stop after five consecutive nonimprovements or the stated hard cap. Source NLL chooses the model; exact ties favor fewer trainable parameters when applicable, then lower compute and earlier fixed trial order. B_SIG must be an actually available registered signal model, not a fabricated replacement. JEV-PERM uses H-to-I-description, I-to-O-description, O-to-H-description throughout fitting/evaluation while output labels retain their physical meaning. Token-score and shortest-generation controls share B_TOK weights, prompt and verified distinct label codes. No separate generation training family and no long-answer strawman.

Optional scalar calibration is source-only, T in [0.05,20], keeping T=1 when no source-calibration improvement. Primary results remain raw probabilities. Preserve original raw logits/probabilities, failures and all approved fit outputs. Keep accepted error undefined at zero coverage. C4 native-coarsening and nonzero preservation hypotheses remain UNTESTED; do not insert their losses into this exact-label matrix.

### Current handoff and one next action

**Authorization is complete; G0 admission is not.** Resume this same G0 on a session that can read the user's approved raw/processed PU and MFPT files. Verify original file to processed Id, channel and physical time/scale, and the MFPT physical-specimen-to-record relation. The generated P1 CSV fixes roles but does not certify those remaining fields. Keep MFPT target scores sealed. Do not silently substitute a record-balanced estimand, another dataset, synthetic waves or a default geometry.

If G0 passes, continue directly to authorized minimal interface implementation/G1 and the bounded matrix; no repeated budget/population confirmation is needed. If a required join is still UNKNOWN, stop with the exact missing original source. If only a small verified specimen set exists, keep the fixed-archive limitation and do not manufacture a population-level interval. A different estimand or expanded population needs a scientific amendment, not merely another run.

The earlier `run-sim-002` source-information diagnostic remains unfinished and separate; this author approval does not mark it complete. The mounted simulation archive contains `sim-20261017-001`, a different protocol, so it is not substituted. It becomes a prerequisite only for reviving the preservation-mechanism claim, not an excuse to relabel P1 results. All old PU and simulation outputs, PR #16, the manuscript and the figure sources are untouched by this execution record.

### P1 implementation binding and real-input check — 17 October 2026

The local candidate has been integrated at `run_jev_study.py semantic` without
replacing the pilot or simulation commands. The six `src/S02_Configs/real/`
configurations instantiate the approved 46-fit matrix. The fixed registered
signal comparator is MobileNet1D (width multiplier 0.5, dropout 0.2); the semantic
first instance uses the already pinned local Qwen3-0.6B snapshot. These are
implementation bindings, not measured baseline qualification or model benefits.

The original 17 October implementation required a separately supplied
PU/MFPT-only metadata CSV. On 18 October this artificial input requirement was
removed: the native `metadata.xlsx` is the source. A bounded worksheet stream
decodes only the header and PU/MFPT rows, then selects the fixed approved
population; it never loads the mixed workbook as a full table. A preprojected
CSV remains supported. The G0 qualification map must establish native PU
physical identity for every selected record, not only counts and labels. PU
uses raw 0/1/2 for H/I/O, whereas MFPT uses raw 0/1/2 for H/O/I; the latter must
map to shared indices 0/2/1. The map and selected role/metadata contents are
frozen in the consumed run contract. The
PHMFactory pin, reader, grouped partition and sequential window implementation
remain upstream; caches are independent copies. Per-record resampled lengths,
retained intervals, tail duration and actual transformation time are saved.

Source interface acceptance now covers all five neural families and RF's fixed
features before the search. An isolated technical update and state roundtrip in
G1 is not a new efficacy family. `verify-source` loads every selected model twice,
checks source inference, freezes source-only temperature and checks fixed
candidate-vector contracts before target export. A file-presence marker cannot
release MFPT. All source verification, external evaluation and timing share the
original one GPU-hour allowance; no reserve is silently reassigned. Manual
technical recovery remains within the original per-job/global and total caps,
with no automatic retry or optimizer continuation.

The descriptive sensitivity is fixed before target access: `race_wording` uses
“A rolling bearing without diagnosed ring damage.”, “A rolling bearing with
inner-race damage.”, and “A rolling bearing with outer-race damage.” in H/I/O
order. Its vectors are compiled and saved on the source side, then reused with
the selected adapter. It is raw-probability sensitivity, not another fit or a
separate selected temperature. The canonical descriptions and fixed permutation
stay unchanged. This one wording set cannot establish general language
robustness or semantic understanding.

Saved-output analysis uses the approved record-within-unit estimand. Brier sums
three squared class errors; ECE has ten equal-width bins. The main paired effect
averages the three seed effects per physical unit, then uses 10,000 paired unit
bootstrap draws with seed 20261017, conditional on those fits. Unknown units or
fewer than two units in any class give NA intervals. Fit SD is separate and NA
for a single fit. Fixed acceptance and fault-probability thresholds 0,0.1,...,1
are sensitivity scenarios; .5 is the displayed alarm point, not a target-selected
or industrially validated optimum. Undefined costs and zero-coverage accepted
error stay NA. Generated outputs have their own measured errors. Saved results
alone produce three figure groups; no plotting call reruns a model.

Actual CPU-only `semantic check` on this machine returned **exit 2**, wall time
**0.79 seconds**, with precisely the unbound qualification map and missing
approved-only metadata projection. PU/MFPT H5 and weight paths and fixed package
versions are locatable, but their measurement chain and full runtime contract
were not inspected by that check. Its local output is
`src/outputs/jev-phm/g0-20261017-integration/preflight.txt`. No actual metadata,
H5 signal, target score, pretrained forward or CUDA job was read/run. G0 remains
BLOCKED and efficacy fitting remains **0/46**. Code fixtures and the native
adapter numerical compatibility test do not replace G0. The next action is the
same missing record/specimen/measurement binding, not another dataset or plan.

On 18 October, the corrected native-workbook check actually recovered the fixed
2,339-row P1 population: PU H/I/O counts 480/880/959 across 29 units; MFPT
H/I/O counts 3/7/10 across 20 records, with physical specimen count UNKNOWN.
MFPT nominal rates are 97,656 Hz (six records) and 48,828 Hz (fourteen); all
declare one channel. These are metadata facts, not verified acquisition units.
The first real check exposed an incorrect space-free description literal; the
expected native strings were corrected to `outer race fault` and
`inner race fault`, preserving that failed attempt. The separate corrected run
reported metadata and then stopped at the missing qualification map. Both runs
returned exit 2 and took 2.05 seconds each, with separate outputs under
`src/outputs/jev-phm/g0-20261018-native-metadata/`. No waveform, model, GPU or
target outcome was accessed; G0 remains BLOCKED, C4/C5 UNTESTED, and fits 0/46.

### G0 follow-up: original signal semantics and local headers — 18 October 2026

The P1 thesis, six comparison families, source selection and budget remain
fixed. The dominant transfer hypothesis and C5 readout comparison are UNTESTED;
the native-granularity hypothesis is outside this exact-label P1. This follow-up
asks only whether original documentation or existing storage metadata can
resolve physical-unit and measurement qualification before G1.

The [MathWorks original processing example](https://www.mathworks.com/help/predmaint/ug/Rolling-Element-Bearing-Fault-Diagnosis.html),
under “Envelope Spectrum Analysis for Bearing Diagnosis”, uses `bearing.gs`
directly, forms seconds from sample indices divided by `bearing.sr`, and labels
the acceleration in g. This supports those fields' published interpretation;
it does not establish the local H5 conversion or identify physical specimens.
The [publisher README](https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data/blob/master/README.md)
explicitly describes reorganizing the original first four folders into example
train/test directories. Those directories are not a specimen mapping or P1
target-training permission.

An additional [2025 paper involving data provider Eric Bechhoefer](https://journals.sagepub.com/doi/10.1177/14759217251350962),
in its MFPT subsection, describes ten outer-race recordings and their sampling,
duration and loading conditions. The [authors' MFPT code](https://github.com/omriMatania/novel_approaches_of_synchronous_averaging_of_gear_and_bearing_vibrations/blob/main/main_MFPT.m)
uses bearing specifications for fault-frequency calculations. Neither inspected
material supplies the file-to-physical-specimen join. Bearing geometry, load,
file suffixes and the word “bearings” cannot supply it. The original MFPT data
page and original envelope-PDF URL redirected to ASNT; the data-acoustics origin
was unavailable. These retrieval limits do not prove universal absence of a map.

The following **executed read-only header command** returned exit 0. It opened
only the approved MFPT H5, never a signal array, other dataset, model or CUDA.
Its captured stdout is archived locally as
`src/outputs/jev-phm/g0-20261018-mfpt-qualification/mfpt-headers.jsonl`:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import json
from pathlib import Path
import h5py

path = Path('/home/user/data/PHMbenchdata/PHM-Vibench/RM_007_MFPT.h5')
with h5py.File(path, 'r') as handle:
    print(json.dumps({'file': str(path), 'root_attribute_names': list(handle.attrs.keys())}))
    for identity in range(46282, 46302):
        key = str(identity)
        if key not in handle:
            print(json.dumps({'Id': key, 'status': 'MISSING'}))
            continue
        item = handle[key]
        record = {'Id': key, 'kind': type(item).__name__, 'attribute_names': list(item.attrs.keys())}
        if isinstance(item, h5py.Dataset):
            record.update(shape=item.shape, dtype=str(item.dtype))
        else:
            record['child_names'] = list(item.keys())
        print(json.dumps(record))
PY
```

All twenty IDs were present as float64 Datasets. IDs 46282–46287 had shape
(585936,1,1); IDs 46288–46301 had shape (146484,1,1). The root and every selected
entry had no attributes. This verifies only the inspected keys and storage
layout. It supplies no specimen IDs, repeated-acquisition join, channel names,
scale units or timebase metadata, and does not verify waveform values.

**Decision:** retain G0-BLOCKED and 0/46 fits. Stop repeated metadata/header
checks and keep G1–G5 unopened. The missing evidence comprises the MFPT original
specimen-to-file/repeated-acquisition mapping **and measurement provenance for
both PU and MFPT**: original-file to local-H5 correspondence, sampling clock/rate,
vibration-channel identity and acceleration scale. PU's raw clock association
remains independently unverified under the earlier source contract; its known
bearing roster and channel index alone cannot qualify its resampling. Receiving
an MFPT specimen map does not authorize G1 while PU provenance remains unresolved.
Inspect new source material when available and bind the existing qualification
map only after every source and target requirement passes. MFPT material paths
or public links have been requested; no replacement dataset, file-based
estimand, invented unit map or artificial coarsening is adopted. This is a
qualification decision, not evidence that the method succeeds or fails.

### Authorized native-value execution revision — 19 October 2026

Before any P1 model result, the user stated that the H5 files only change the
original storage format, that acquisition information is in metadata, and that
one experiment should default to one bearing. Accept format-only preservation
as **user-provided provenance**, rather than claiming an independent conversion
audit. Accept the experiment-as-bearing convention where specimen membership
was unresolved. These statements supersede the previous wait for a separate
conversion description and MFPT specimen join; no renewed budget approval is
needed.

Keep the documented PU roster and all its dependent recordings together:
29 bearings, 2,319 records, unchanged train/tune/cal/test roles. For MFPT only,
use its twenty known-label native records as twenty **assumed experiment units**,
with identity from the metadata record/File. Do not rename these as independently
verified physical specimens. Paired unit intervals are conditional on this
user-authorized independence convention and the fitted models; they do not
establish machine-population generalization. Preserve this basis in the consumed
qualification CSV, saved definitions, effects and figures.

Retain native signal amplitudes, with an explicit identity `scale_multiplier=1`
for both datasets. The inspected PU MAT's vibration Unit field is empty; the
sensor sensitivity alone does not identify the saved amplitude unit. Therefore
this revision withdraws the earlier common-g input interpretation: PU amplitudes
remain in unspecified native units, while the published MFPT gs convention is g.
Do not invent a PU conversion or infer it from target spectra. This is a
prospective input-protocol revision, not a post-result preprocessing choice.
All six families receive the same native-value observation, documented channel
(PU 2, MFPT 0), metadata nominal rate, whole-record anti-alias resampling to
25.6 kHz, and existing source-train-only mean/standard-deviation scaling. Unknown
cross-source amplitude calibration is an explicit transfer limitation.

The existing entry consumes
`paper/assets/tables/jev_p1_record_bindings.csv`; its rows preserve native Id,
File, raw label, shared H/I/O label, fixed role, rate, channel, value-unit scope
and evidence basis. It is a derived runtime input, not an edit to native metadata
or raw H5. The six-family 46-fit maximum, search, seeds, stopping rules, target
seal and resource ceilings remain unchanged. Inspect actual common inputs,
prepare source windows and run the real source-interface smoke before fitting.
MFPT waveforms remain sealed until every selected source model has passed the
existing verification/calibration gate. CWRU and physical GPU 2 remain forbidden.

Execution directory: `src/outputs/jev-phm/run-real-20261019-001`. Reproduce with
`JEV_RUN=src/outputs/jev-phm/run-real-20261019-001 bash run_all.sh`; do not reuse an
incomplete run or reset the shared approved-study budget. Technical failures and
cap-reached fits remain explicit, with no automatic retry or target release.

#### First real acceptance and technical recovery

Source preparation completed in 283 CPU wall seconds and exported train/tune/
cal/test arrays of 15,377 / 4,813 / 2,884 / 4,815 windows. Their record counts
are 1,279 / 400 / 240 / 400 and declared PU units 16 / 5 / 3 / 5. Every window
has 8,192 samples and one channel; tails are recorded. All five real neural
interfaces and RF features passed the source-only smoke in 94 GPU-lane seconds,
including actual Qwen weights, gradients and checkpoint roundtrips on physical
GPU 0. These are interface observations, not diagnostic efficacy evidence.

The first RF child failed before fitting: the config handler parsed a saved
JSON file as YAML, turning `1e-12` into a string and correctly triggering the
shared-contract check. Repair only format dispatch (JSON parser for `.json`,
YAML parser for YAML); do not coerce the scientific value or relax equality.
This is manual technical recovery 1 of the existing maximum two globally,
not a new family or extra efficacy fit. Preserve its five consumed CPU seconds,
stage log and the original child log/failure under
`fits/rf/technical_recovery_01/`. Reuse unchanged source exports and smoke; keep
MFPT sealed. Do not present this failed infrastructure attempt as a PHM result.
