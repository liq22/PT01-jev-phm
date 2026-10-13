# Experiment Run Ledger

只记录真正执行过或已明确批准的运行。主要产物是实验结果；本表仅保留理解和复现实验所需的信息。

| Run ID | Date | Code version | Config | Data/version | Seed | Budget/hardware | Primary metric | Result | Status | Output path | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PU-20261013-E0 | 2026-10-13 | 3137920; export fixes through 6a838c8 | Saved factory configs and window plans | PHM-Vibench cdc0669; canonical PU 2,559 recordings / 32 bearings | 20260922 roles | CPU export; permitted single-UUID RTX 4090 forwards | Actual recording coverage | 100% for W=2048/4096/8192; real 128-window Qwen/E1/E2 smoke passed | completed | src/outputs/jev-phm/run-20261013-pu-001/exports; smoke | Smoke scores are excluded from efficacy evidence; nominal Fs only |
| PU-20261013-selection | 2026-10-13 | 6a838c8 | inputs/probe.json, window_tuning.csv | Same canonical PU | 20260922 | CPU, five C values per candidate W | Tuning-bearing NLL | Primary and rotation 5 select 2048; rotations 1/2/3/4/6 select 8192 | completed | src/outputs/jev-phm/run-20261013-pu-001/inputs/selection.json | All selections frozen before test analysis |
| PU-20261013-E1-amplitude | 2026-10-13 | 19e1163 fitting; 6a838c8 encoding | Primary, amplitude-preserving, workers=1 | Same canonical PU; Qwen c1899de | 20260922 bootstrap | BF16 B256 encoder; CPU four-thread fits | Eight-bearing mean window NLL | Pooled numerical 1.022681590; legacy Qwen 1.052741468 | completed | src/outputs/jev-phm/run-20261013-pu-001/evaluations/primary_raw_e1 | Qwen pooled/unit defers all windows; no decomposition evidence |
| PU-20261013-E1-standardized | 2026-10-13 | 7c3e8e7 | Primary, per-window standardization, workers=3 | Same canonical PU; Qwen c1899de | 20260922 bootstrap | BF16 B256 encoder; isolated CPU candidates, four threads each | Eight-bearing mean window NLL | No final result yet | running | src/outputs/jev-phm/run-20261013-pu-001/evaluations/primary_normalized_e1 | Do not use unfinished candidate fits as results |
| PU-20261013-E2 | 2026-10-13 | 7c3e8e7; 922d121 (same fitting source) | Primary plus six frozen rotations, six arms, workers=3 | Same canonical PU; Qwen c1899de | 20260922 | Frozen cached encodings; independent CPU candidates | Paired bearing NLL, question-factorized minus flat | No final result yet | running | src/outputs/jev-phm/run-20261013-pu-001/evaluations/*_raw_e2 | Seven refitted protocols; rotations are descriptive, not independent repetitions |
| PU-20261013-CPU-probe | 2026-10-13 | 922d121 (same fitting source) | One train-only rotation-1 W8192 flat-Qwen C=0.01 fit | Same frozen training state | Deterministic fit | One CPU thread; no CUDA | Convergence and elapsed fit time only | 747 iterations; classifier 873.19 s | completed | src/outputs/jev-phm/run-20261013-pu-001/cpu-one-thread-probe.log | No tuning/test predictions; not efficacy or a fair speed comparison |

## Status

- `planned`
- `running`
- `completed`
- `failed`
- `timeout`
- `cancelled`
- `invalid`
- `to_verify`

## Rules

- 进程正常退出不自动等于实验完成；使用真实结果状态。
- `failed`、`invalid`、`planned` 和 `to_verify` 不得支持正向结论。
- 协议、代码行为、数据、配置或 seed 实质变化时创建新 run。
- `completed` 记录代码版本、配置、数据版本、seed、指标、结果和输出路径；不需要 hash 或 receipt。
- 负结果、null、unstable 和 contradictory outcome 若影响判断，必须保留。
