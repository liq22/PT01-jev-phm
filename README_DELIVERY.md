# Jev-PHM P1 implementation and acceptance

The implementation is integrated into the existing study CLI. On 19 October the
author confirmed that H5 preserves original values and authorized one experiment
as one bearing where physical membership is unresolved. The current prospective
execution revision is recorded at the end of the existing Goal. It retains PU's
documented groups and uses twenty assumed MFPT experiment units. Earlier G0
failures below remain historical; actual real-source smoke and fitting outcomes
are reported in STATUS. Code fixtures are not PHM results.

**Current outcome:** real source export and all six source interfaces passed;
28 efficacy fits completed. The lowest-NLL semantic candidate (8.73848 on PU
tune) hit the 2,000-update cap and remains source-unqualified. One already-started
refit was stopped, with its original logs retained. MFPT waveforms, calibration,
external predictions and timing remain sealed/unrun. This is an executable,
partially exercised study, not a completed or validated external experiment.
See the [source result table](paper/assets/tables/jev_p1_source_fit_results.csv)
and [current stop decision](paper/STATUS.md#e-唯一下一步). No further metadata
is currently requested. Do not restart the incomplete fit or reset its budget.

The authority for population, claims and resources is the existing
[approved P1 protocol](paper/experiments/jev_goal.md#approved-p1-continuation-pu-single-ring-source-to-mfpt-external-target).
CWRU and explicitly CWRU-trained industrial weights are excluded. Physical GPU 2
is forbidden. No new dataset, backbone or budget is introduced.

## Native metadata and real input requirements

The earlier input check returned exit code 2 on 17 October 2026:

```text
BLOCKED:
data.qualification_file is unbound: physical units, record correspondence, clock/channel/scale need G0
Missing required input: /home/user/data/PHMbenchdata/PHM-Vibench/metadata_pu_mfpt.csv
```

The local log is `src/outputs/jev-phm/g0-20261017-integration/preflight.txt`;
`preflight-time.txt` records 0.79 wall seconds. Only file/directory presence and
installed versions were inspected. No real metadata, H5 payload, model weights,
target prediction or CUDA workload was read or executed. The authoring package's
older environment failures remain in the original Obsidian candidate directory;
they do not describe this machine's dependency availability.

After committing the implementation, the public `bash run_all.sh --check` entry
also returned exit 2 for the same G0 requirements, in 0.94 seconds. Its saved
output is `shell-preflight.txt` alongside the direct CLI check. Combined G0
preflight wall time is 1.73 seconds, not total development/checking time.

The separately supplied CSV requirement was an implementation error: the native
metadata already exists in `metadata.xlsx`. `data.metadata` now points to that
workbook. The check streams worksheet rows, decodes only the header and rows
whose Name matches PU or MFPT, and selects the already approved population.
Excluded-source rows are not parsed into records; their H5 and waveforms are not
opened. This bounded projection does not change the workbook or add a waveform
reader. A preprojected CSV remains usable, but is not an additional original
input the user must provide.

The native labels require an explicit conversion to the shared H/I/O order:

| Source | Native labels | Shared class indices |
|---|---|---|
| PU | 0 healthy, 1 inner-ring, 2 outer-ring | 0, 1, 2 |
| MFPT | 0 baseline, 1 outer-race, 2 inner-race | 0, 2, 1 |

The qualification map must agree with both the native descriptions and this
conversion. It remains configured separately as `data.qualification_file`:

```text
Id,Name,File,unit_id,role,class_index,raw_label,sample_rate_hz,channel_index,scale_multiplier,physical_source,measurement_source,license_source
```

The population is exactly 2,319 approved PU source records and 20 known-label
MFPT target records. PU records must match their native physical bearing and
[fixed roles](paper/assets/tables/jev_p1_pu_unit_roles.csv), raw label and original
file. MFPT is entirely external. Its physical specimens and repeated recordings,
and both sources' real sampling clock, acceleration channel and scale, need
original evidence; nonempty citation fields alone do not authenticate it.
The workbook supplies record IDs, original paths, fault labels, nominal sampling
rates and channel counts. These facts do not identify MFPT physical specimens or
establish repeated-record relationships, acceleration units or the local H5
measurement chain. Unknown required facts stop admission; no file is promoted to
a physical unit. The historical PU mixed-bearing counterexample and full
32-bearing pilot remain unchanged.

On 18 October the real CPU check read **2,339 approved rows** directly from the
workbook: PU 29 units / 2,319 records (H/I/O: 480/880/959), and MFPT 20 records
(H/I/O: 3/7/10). MFPT records declare one channel, six at 97,656 Hz and fourteen
at 48,828 Hz. Its description field supplies categories, not physical specimen
IDs. The first check failed on an incorrect space-free description literal;
after correcting it to the observed native text, a separate check reported the
metadata and stopped on missing physical qualification. Both returned exit 2
in 2.05 seconds each. Logs are retained separately as `check.txt` and
`check-corrected.txt` under
`src/outputs/jev-phm/g0-20261018-native-metadata/`. No H5, model or CUDA execution
was involved, and no experiment run directory was created.
After committing the implementation, `bash run_all.sh --check` reproduced the
same metadata observations and physical-qualification stop (exit 2, 2.18 seconds).
Its separate outputs are `shell-check.txt` and `shell-check-time.txt` in that
directory.

A subsequent bounded G0 inspection opened **headers and attributes only** for
the twenty approved MFPT H5 entries. All were present as float64 Datasets: six
of shape (585936,1,1) and fourteen of shape (146484,1,1). The root and all twenty
entries had empty attributes. No signal array was read. The exact executed
command and source checks are in the existing
[G0 follow-up](paper/experiments/jev_goal.md#g0-follow-up-original-signal-semantics-and-local-headers--18-october-2026).

The [publisher's original processing documentation](https://www.mathworks.com/help/predmaint/ug/Rolling-Element-Bearing-Fault-Diagnosis.html)
supports interpreting the released `gs` as acceleration in g and `sr` as Hz.
Local H5 preservation of those semantics and physical specimen membership
remain unverified. Required evidence is the original MFPT specimen/record join
and **both PU and MFPT** raw-to-H5 measurement provenance: sample clock/rate,
vibration channel and acceleration scale. PU's raw clock association is still
unverified; its known bearing roles and channel index do not establish the new
resampling contract. An MFPT map alone cannot release G1. Only after all source
and target requirements pass may the existing qualification map be bound.
This was the 18 October stopping decision. It was superseded operationally by
the author's 19 October provenance statement and explicit unit convention,
without turning those statements into independently verified specimen evidence.
The derived binding is now `paper/assets/tables/jev_p1_record_bindings.csv`, with
additional `unit_basis` and `value_unit` columns. PU keeps its 29 known bearing
groups; each of the twenty MFPT records is an assumed experiment unit. Saved
definitions, effects and plots state that assumption. Both use an identity
`scale_multiplier=1`: PU's saved amplitude unit is unspecified, so no common-g
calibration is claimed. All methods share unchanged native values, common-rate
resampling and source-train-only scaling. This revision predates any target score.

## Commands and execution order

Use the repository's pinned environment and PHMFactory revision
`cdc0669167ca25ac848e34c880875ca39c6aed89`. No package, weight or data download is
performed by the runner. All operations use the existing entry:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python src/S03_Scripts/run_jev_study.py semantic --help
bash run_all.sh --check
bash run_all.sh --status
```

For the actual stopped study, inspect without restarting fits:

```bash
JEV_RUN=src/outputs/jev-phm/run-real-20261019-001 bash run_all.sh --status
CUDA_VISIBLE_DEVICES='' .venv/bin/python src/S03_Scripts/run_jev_study.py semantic fit \
  --config src/S02_Configs/real/semantic.yaml \
  --run-root src/outputs/jev-phm/run-real-20261019-001
```

The second command reuses the six completed grid results and exits nonzero at
the existing source-qualification gate before loading a model or starting a
refit. It does not open target data. The original pipeline commands below remain
documented for the protocol; they do not authorize overriding this scientific
stop. Small configuration/result JSONs and diagnostics are retained with the
run; waveform caches and checkpoints remain in local study storage.

`--check` is CPU-only and returns a nonzero status while G0 is unbound. Commit
reviewed scientific code/configuration before calling the Shell runner. Once
G0 actually passes, execute:

```bash
bash run_all.sh
# Reuse only complete, unchanged stages; no interrupted optimizer continuation:
bash run_all.sh --resume
```

The sequence is `check → prepare → smoke → fit → verify-source → infer → timing
→ summarize`. The six YAML configurations under `src/S02_Configs/real/` select
RF, native MobileNet1D, indexed head, semantic scores, fixed mismatched semantic
assignment and exact candidate-token scores. G1 checks all five neural interfaces
and RF features on source observations only. The one technical update per neural
interface is not an efficacy fit. `verify-source` actually reloads every selected
checkpoint twice, checks source predictions and freezes calibration and the
predeclared wording-variant vectors before any MFPT prediction.

P1 uses 25.6 kHz, 8,192-point nonoverlapping windows and no incomplete-tail
padding. The registered adapter performs whole-record channel/scale conversion
and polyphase resampling before native window generation. It writes only to
independent native caches. There is no alternative raw reader or split routine.

The budget is 40 neural fits plus 6 RF fits; at most one GPU process and eight
CPU threads, 24 GPU hours and 12 CPU wall hours. Within that ceiling, ordinary
fitting has 20 GPU hours, source interface checks one hour, frozen verification /
evaluation / timing one shared hour, and technical recovery two hours. A changed
run directory cannot reset the approved study's consumed time. The shared
`src/outputs/jev-phm/p1-approved-20261017/execution/usage.tsv` retains stage use.
A failed or interrupted job is retained and blocks implicit retry. Manual
technical recovery remains limited to one per job and two globally within the
same original allowance; no optimizer-resume or automated recovery is claimed.

The Shell validates the configured GPU UUID against physical indices, rejects
GPU 2 and restricts visibility before starting any CUDA command. Python checks
again. Source input mapping, roles and native metadata contents are retained and
compared on resume, rather than treating unchanged paths as unchanged inputs.

## Saved outputs and scientific limits

Local verification used Python 3.10.0, PyTorch 2.6.0, Transformers 4.51.3 and
the pinned project environment. On 17 October all 163 repository unit tests passed, including
64 new P1 tests. The native adapter compatibility test used the actual pinned
PHMFactory with numerical arrays only. Skill, wrapper, case-definition and Core
backend checks passed. Project validation still fails on seven pre-existing
evidence-ID/enum errors, reproduced on an unchanged extraction of `25ec6781`;
the validator and evidence-strength fields were not relaxed. These software
checks do not establish real G0, model or CUDA acceptance. Detailed local outputs
are alongside `preflight.txt`; remote CI must be reported from its actual run.
The subsequent PR review found a stale figure filename in the resume guard. The
guard now checks the actual three SVG/PDF pairs; 18 affected execution tests pass,
including two new complete-resume/missing-PDF cases. The full suite was not
repeated after this narrow correction.

The 18 October native-metadata change passed the complete **176-test** suite,
both skill-definition checks, Shell syntax and the diff check. Project validation
still exits 1 with the same seven evidence-ID/enum findings, compared against the
preserved unchanged-baseline output; no new project-validator finding appeared.
The validator and scientific evidence grades were not changed. Current outputs
are `unittest.txt` and `validate-*.txt` in the 18 October G0 directory above.

The run defaults to `src/outputs/jev-phm/run-real-20261017-001`:

| Output | Meaning |
|---|---|
| `study.json`, `source_scale.json` | Consumed common inputs and source-only scaling |
| `exports/<role>/` | Native windows, record/unit references, resampled intervals and per-record tails/transform times |
| `smoke/<family>/`, `smoke/result.json` | Actual source interface check, only when G0 permits it |
| `fits/<family>/` | All trials, failures, stopping reasons and source-selected states |
| `source_verification/<family>/<seed>/` | Reloaded state, frozen source temperature and variant vectors |
| `evaluations/<family>/<seed>/` | Raw logits/probabilities, actual generated labels and wording sensitivity |
| `summary.csv`, `paired_unit_effects.csv`, `reporting/` | Recomputed metrics, fit SD, paired unit effects, curves and fixed-threshold scenarios |
| `timing/`, `figures/` | Measured output/timing observations and saved-result SVG/PDF plots |

Primary NLL averages windows within a record, records within a physical unit,
then units equally. Brier sums squared errors across all three classes. ECE uses
ten equal-width confidence bins. Fit SD and 10,000-draw paired physical-unit
intervals are distinct; unknown identities or fewer than two units in any class
produce an unavailable interval, not a file/window bootstrap substitute.

Raw and source-calibrated results stay separate. The fixed 0–1 threshold grid
in steps of 0.1 is a sensitivity analysis, not a target-optimized deployment
policy. FPR/FNR include rejected observations. Accepted error at zero coverage
and undefined scenario costs remain NA. Correct versus fixed wrong descriptions
test this particular correspondence, not general semantic understanding. The
source-frozen `race_wording` variant is an auxiliary raw-probability comparison,
without another fitted family or variant-specific calibration.

Timing records actual one-token generation separately from exact-token scoring;
the former never borrows the latter's error rate. Thirty warmups and 200 measured
calls in five blocks use source windows, batch one and synchronization. The
shared-weight token readouts are interleaved; different trained model families
are measured separately, so cross-family comparisons do not remove all run-order
variation. Full-record channel/scale/FIR times from source export, initial
candidate compilation, and model loading are reported separately. H5 I/O and
acquisition are outside warm window latency; full-record preprocessing is not
added to per-window p95. No sensor-to-decision end-to-end speed claim follows.

Plots read saved results only. The three figure groups are diagnostic effects,
error versus measured window-processing cost, and selective/alarm behavior. No
real P1 plots or results are generated while G0 is blocked. The unique manuscript
and C1–C3 remain unchanged; C4/C5 efficacy stays UNTESTED.
