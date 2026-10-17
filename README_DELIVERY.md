# Jev-PHM P1 implementation and acceptance

The implementation is integrated into the existing study CLI. Real P1 data,
pretrained-weight and CUDA acceptance remains **BLOCKED**, with **0/46 efficacy
fits**. Code tests and numerical adapter fixtures are not PHM results.

The authority for population, claims and resources is the existing
[approved P1 protocol](paper/experiments/jev_goal.md#approved-p1-continuation-pu-single-ring-source-to-mfpt-external-target).
CWRU and explicitly CWRU-trained industrial weights are excluded. Physical GPU 2
is forbidden. No new dataset, backbone or budget is introduced.

## Real input requirement

The current input check returned exit code 2 on 17 October 2026:

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

G0 needs an already authorized **PU/MFPT-only CSV projection**, configured as
`data.metadata`, and a documented record map configured as
`data.qualification_file`. The shared Excel workbook is never loaded and then
filtered. Do not invent this projection or recover missing identities from
filenames. The map's existing columns are:

```text
Id,Name,File,unit_id,role,class_index,raw_label,sample_rate_hz,channel_index,scale_to_g,physical_source,measurement_source,license_source
```

The population is exactly 2,319 approved PU source records and 20 known-label
MFPT target records. PU records must match their native physical bearing and
[fixed roles](paper/assets/tables/jev_p1_pu_unit_roles.csv), raw label and original
file. MFPT is entirely external. Its physical specimens and repeated recordings,
and both sources' real sampling clock, acceleration channel and scale, need
original evidence; nonempty citation fields alone do not authenticate it.
Unknown required facts stop admission. The historical PU mixed-bearing
counterexample and full 32-bearing pilot remain unchanged.

## Commands and execution order

Use the repository's pinned environment and PHMFactory revision
`cdc0669167ca25ac848e34c880875ca39c6aed89`. No package, weight or data download is
performed by the runner. All operations use the existing entry:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python src/S03_Scripts/run_jev_study.py semantic --help
bash run_all.sh --check
bash run_all.sh --status
```

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
the pinned project environment. All 163 repository unit tests passed, including
64 new P1 tests. The native adapter compatibility test used the actual pinned
PHMFactory with numerical arrays only. Skill, wrapper, case-definition and Core
backend checks passed. Project validation still fails on seven pre-existing
evidence-ID/enum errors, reproduced on an unchanged extraction of `25ec6781`;
the validator and evidence-strength fields were not relaxed. These software
checks do not establish real G0, model or CUDA acceptance. Detailed local outputs
are alongside `preflight.txt`; remote CI must be reported from its actual run.

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
