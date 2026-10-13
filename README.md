# Jev-PHM

## Research question

### Decisions from verified evidence

Investigate Jev-inspired typed probabilistic decisions for PHM with **PHMFactory + Qwen/Qwen3-0.6B**. This project does not use proprietary Jev weights, its API, or a reproduced RLCD algorithm.

The research question is whether explicitly scoped diagnostic judgments and code-owned composition improve probability quality or selective decisions beyond competitive direct predictors. Typing, softmax heads, hierarchical classification and calibration are prior art; NanoJev already implements Qwen3-0.6B decision heads and atomic-judgment workflows. See the [source assessment](paper/refs/reading_matrix.md).

The completed PU attribution study uses 2,559 recordings from 32 physical bearings. The primary question-factorized-minus-flat Qwen NLL is −0.001102 (conditional paired 95% interval [−0.008577, 0.005163]); six refitted rotations give three lower and three higher losses. Every primary calibrated Qwen arm defers all windows at the declared cost. These results support a bounded negative attribution finding, without establishing equivalence or industrial superiority. Read the [English candidate](paper/draft/main.md), [derived PDF preview](paper/preview/jev-phm-preview.pdf) and [saved result summary](paper/assets/tables/jev_study_summary.json).

## Implementation

### Shared observations and probability heads

```text
PHMFactory windows -> six deterministic features -> frozen Qwen representation
-> tuned logistic head -> raw / pooled / unit calibration -> diagnosis or deferral
```

The numerical control bypasses Qwen and shares the downstream implementation. E1 compares numerical and Qwen representations with raw, pooled and unit-weighted calibration. Positive scalar temperature cannot change ordinary class argmax. Unit-weighted calibration remains a secondary estimation question.

E2 implements two questions over the same state: whether a fault is present, and which fault fits conditional on belonging to a defined fault class. The conditional head uses fault training samples only; every test observation executes both branches. The leaf probabilities are `p(healthy) = 1-r` and `p(fault k) = r*u_k`. The comparison includes flat numerical/Qwen heads, numerical factorization, shared-representation Qwen factorization, question-conditioned Qwen factorization and a fixed equal-weight fusion of two direct questions. Conditional factorization, typing and calibration are existing techniques; their empirical value and attribution are determined by saved experiments.

## Files and execution

### Primary products

| Product | Location |
|---|---|
| Unique manuscript | [paper/draft/main.md](paper/draft/main.md) |
| Current research state | [paper/paper.yaml](paper/paper.yaml) |
| Contribution/evidence mapping | [paper/experiments/evidence_matrix.md](paper/experiments/evidence_matrix.md) |
| Frozen protocol and execution decisions | [paper/experiments/jev_goal.md](paper/experiments/jev_goal.md) |
| Sole current implementation | [src/S01_Package/jev_phm.py](src/S01_Package/jev_phm.py) |
| Study entrypoint | [src/S03_Scripts/run_jev_study.py](src/S03_Scripts/run_jev_study.py) |
| Fixed first-run settings | [src/S02_Configs/jev_probe.json](src/S02_Configs/jev_probe.json) |
| Public frozen cohort, roles and selected windows | [paper/assets/tables/README.md](paper/assets/tables/README.md) |

### Environment, data and model

Run commands from the repository root. The PHM-Vibench submodule must remain at `cdc0669167ca25ac848e34c880875ca39c6aed89`; the driver rejects a different version. Initialize only this execution dependency:

```bash
git submodule update --init external/PHM-Vibench
git -C external/PHM-Vibench rev-parse HEAD

python3.10 -m venv .venv
.venv/bin/python -m pip install -r src/S02_Configs/jev_requirements.txt
```

The requirements pin the selected experiment, configuration, figure and preview packages. The current machine reuses PyTorch 2.6.0/CUDA 12.4 through a documented `--system-site-packages` environment; a new checkout can install the same requirements into the clean environment above. Select a PyTorch wheel compatible with the host driver. Local PDF rendering additionally requires a Chromium binary; the existing machine has `/home/user/.cache/ms-playwright/chromium-1208/chrome-linux64/chrome`.

Preparation's original `inputs/environment.txt` records PyYAML 6.0.1. The isolated environment was subsequently aligned to PyYAML 6.0.3, matching both current requirement files, and the public configuration resolver was rechecked. Retain the original preparation record; use the current pinned requirements for reproduction. A complete clean-environment installation has not been independently exercised.

The immutable local source is `/home/user/data/PHMbenchdata/PHM-Vibench`, containing `metadata.xlsx` and `RM_027_PU.h5`. The study uses PU (`Dataset_id=20`), the documented vibration channel at index 2, and physical-bearing groups from `Label_Description`. The three labels are healthy, inner-race-dominant and outer-race-dominant; mixed damage remains disclosed in the manuscript. The factory writes only to an independent H5 copy inside the run cache, never to the original data.

The public [cohort](paper/assets/tables/jev_cohort.csv), [bearing-role assignments](paper/assets/tables/jev_unit_roles.csv) and [protocol snapshot](paper/assets/tables/jev_protocol_20261013.json) identify the exact 2,559-record population, seven splits, label order, window plans, development-only selections and factory contract. A Git checkout contains these compact references, not the processed H5, Excel metadata, weights or saved prediction arrays. Full reproduction requires authorized access to the supplied processed data; a separately downloaded or reprocessed PU copy is not established as numerically identical.

Download the public Qwen weights once into the normal Hugging Face cache, with the fixed revision and no credential token:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_DISABLE_IMPLICIT_TOKEN=1 .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
    token=False,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "README.md", "LICENSE"],
))
PY
```

The fixed snapshot is already available on this machine at `/home/user/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`. Set `HF_HUB_OFFLINE=1` after the entire snapshot is available. An incomplete download is an execution failure, not a method result.

### Reproduce the PU study

The actual execution directory is `src/outputs/jev-phm/run-20261013-pu-001`. The commands below use a fresh directory so they preserve existing artifacts. Preparation, export, selection, embedding and evaluation refuse to overwrite their outputs. Change the run name before beginning another complete run.

```bash
JEV_RUN=src/outputs/jev-phm/run-20261013-pu-reproduce-001
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

.venv/bin/python src/S03_Scripts/run_jev_study.py prepare \
  --run-root "$JEV_RUN" \
  --data-root /home/user/data/PHMbenchdata/PHM-Vibench \
  --factory-config src/S02_Configs/jev_factory.yaml \
  --probe-config src/S02_Configs/jev_probe.json \
  --window-sizes 2048 4096 8192

.venv/bin/python src/S03_Scripts/run_jev_study.py export \
  --run-root "$JEV_RUN" --normalization none
.venv/bin/python src/S03_Scripts/run_jev_study.py select --run-root "$JEV_RUN"
.venv/bin/python src/S03_Scripts/run_jev_study.py export \
  --run-root "$JEV_RUN" --normalization per_window_standardization
```

`select` uses each protocol's train/tune bearings and the numerical flat control to freeze a common window length. It does not use calibration or test labels to choose the window. The primary split and six predeclared bearing rotations share frozen source observations; each protocol independently fits its preprocessing, heads, regularization and temperatures.

`prepare` writes the native manifests, explicit tune/cal roles, validated public factory configuration and the required native split adapter into `inputs/`; no manual field construction is needed. The published protocol snapshot is a reference for the frozen run. Reproduction recomputes selection from train/tune data through `select`, rather than copying a chosen window into a new run after inspecting results.

**Physical GPU 2 is forbidden for every process, including subprocesses.** Inspect physical indices and UUIDs before CUDA execution. On this machine the UUID below identifies physical GPU 0; explicitly restrict visibility to that single permitted device. The implementation checks that it becomes logical `cuda:0` and rejects physical GPU 2.

```bash
nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader
JEV_GPU_UUID=GPU-d77295d4-8e79-f0fa-8d82-791c244a2148
export CUDA_VISIBLE_DEVICES="$JEV_GPU_UUID"

.venv/bin/python src/S03_Scripts/run_jev_study.py embed \
  --run-root "$JEV_RUN" --normalization none \
  --device cuda --gpu-uuid "$JEV_GPU_UUID" --dtype bfloat16 --batch-size 256
.venv/bin/python src/S03_Scripts/run_jev_study.py embed \
  --run-root "$JEV_RUN" --normalization per_window_standardization \
  --device cuda --gpu-uuid "$JEV_GPU_UUID" --dtype bfloat16 --batch-size 256

export CUDA_VISIBLE_DEVICES=''
.venv/bin/python src/S03_Scripts/run_jev_study.py evaluate \
  --run-root "$JEV_RUN" --experiment e1 --normalization none --protocols primary
.venv/bin/python src/S03_Scripts/run_jev_study.py evaluate \
  --run-root "$JEV_RUN" --experiment e1 \
  --normalization per_window_standardization --protocols primary --head-workers 3
.venv/bin/python src/S03_Scripts/run_jev_study.py evaluate \
  --run-root "$JEV_RUN" --experiment e2 --normalization none --head-workers 3

.venv/bin/python src/S03_Scripts/plot_jev_results.py --run-root "$JEV_RUN"
.venv/bin/python src/S03_Scripts/build_jev_paper.py \
  --chromium /home/user/.cache/ms-playwright/chromium-1208/chrome-linux64/chrome
```

The serial embedding command processes every distinct selected geometry and its five named questions. It uses the same source observations, prompts, precision and batch size as separate per-query cache jobs; those jobs may record different elapsed times and hardware use. E1's standardized observation comparison is primary-split only. E2 evaluation runs the primary protocol and all six rotations. Plotting reads saved predictions, checks their metrics and creates editable SVG/PDF figures and CSV/JSON tables; it does not rerun a model. Paper building renders the sole Markdown manuscript into offline HTML/PDF previews under `paper/build`.

`--head-workers 3` fits independent regularization candidates in three isolated CPU processes, preserving their objective and fixed selection rule. Set `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4` for the recorded per-worker thread budget. The default is serial fitting; any convergence failure aborts either mode.

### Full coverage and overlap controls

`window_candidates` and `coverage_max_stride` in [jev_probe.json](src/S02_Configs/jev_probe.json) can be overridden at preparation with `--window-sizes` and `--max-stride`. A null maximum stride means `S=W` separately for each candidate. With actual maximum recording length `L_max`, the driver computes the upstream evenly-spaced count as `N = 1 + ceil((L_max-W)/S)`. It validates the actual exported intervals: first start 0, last stop at the record end, no duplicate starts and no gap larger than `S`. Window counts are derived from complete coverage rather than capped at 64.

For example, prepare a **new** run with a 4,096-point window and at least 50% overlap:

```bash
.venv/bin/python src/S03_Scripts/run_jev_study.py prepare \
  --run-root src/outputs/jev-phm/run-pu-w4096-overlap-001 \
  --window-sizes 4096 --max-stride 2048
```

An explicit stride applies to all window candidates and must satisfy `1 <= S <= W` for each. The actual evenly-spaced stride varies with recording length. Short records, uncovered intervals and repeated starts fail explicitly. Overlap does not create independent bearings: uncertainty resamples paired physical bearings. Training assigns each recording total weight 64, divided among its windows, so increasing coverage does not silently change its total loss weight.

### Artifacts and checks

Large data copies, model weights, feature/embedding arrays, fitted heads and per-window predictions stay in machine-local storage; `src/outputs/` and the Hugging Face cache are not committed to Git. Keep each run's inputs, commands, environment, logs and saved artifacts together. The repository carries code, configurations, the manuscript, publication figures and compact summary tables.

| Artifact | Location |
|---|---|
| Frozen inventory, configuration, bearing roles and window selection | `<run-root>/inputs/` |
| Independent PHMFactory cache copy | `<run-root>/cache/` |
| States, labels and actual source intervals | `<run-root>/exports/` |
| Frozen Qwen representations, token counts and encoding metadata | `<run-root>/embeddings/` |
| Fitted heads, probabilities and evaluation summaries | `<run-root>/evaluations/` |
| Editable result figures and compact tables | `paper/assets/figures/`, `paper/assets/tables/` |
| Derived manuscript previews | `paper/preview/jev-phm-preview.html`, `paper/preview/jev-phm-preview.pdf` |

For the closest code checks and command reference:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s src/S04_Tests -p test_jev.py -v
.venv/bin/python src/S03_Scripts/run_jev_study.py --help
.venv/bin/python src/S03_Scripts/run_jev_study.py embed --help
```

Synthetic tests establish code properties, not PHM efficacy. The target venue and author approvals remain to be confirmed. Continue on a work branch through a PR into `dev`, preserving unrelated work and the unique Markdown manuscript.
