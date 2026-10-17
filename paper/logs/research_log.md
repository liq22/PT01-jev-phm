# Research Log

只记录能够改变研究判断的信息。Git 负责代码历史；本文件不记录 hash、文件清单、命令流水账或普通执行过程。

## 13 October 2026: completed attribution and stopping decision

The H0 question-conditioned increment was tested using the frozen primary split
and six refitted rotations. Primary pooled NLL is 1.054200184 for question
factorization and 1.055302213 for flat Qwen, giving -0.001102028 with conditional
paired 95% interval [-0.008577209, 0.005163370]. Shared-representation hierarchy
has the lower point NLL, 1.046334565. Each question-factorized contrast with flat,
shared hierarchy and two direct questions changes direction across rotations
(three lower / three higher NLL). This establishes neither a stable increment nor
equivalence. Rotations are descriptive within one collection.

All four primary calibrated Qwen arms defer every window at cost 0.25. The
question-factorized ECE is lower than flat's, while its Brier and AURC are higher.
Thus this instance separates probability validity, calibration summaries,
confidence ranking and useful decision coverage. The small primary numerical
hierarchy reduction has a conditional secondary interval below zero; it is not
an independently confirmed new method effect.

The H1 input revision retains a separate result: amplitude preservation lowers
pooled E1 NLL for both representations. All four E1 unit-minus-pooled intervals
include zero. Neither result is credited to question decomposition. Explanations
involving lost signal structure, calibration composition or semantic question
quality remain H3 unless directly tested. There is no H2 external confirmation.

Decision: retain the bounded negative attribution and decision-boundary findings,
complete the manuscript and reproducible artifacts, and stop within-test
development. A broader advantage claim would require a prospectively frozen
evaluation on additional independent equipment and appropriate application
baselines. More correlated windows do not supply that evidence.

## 13 October 2026: first real-data probability result

H0: question-conditioned factorization may improve bearing-level probability
quality beyond flat prediction, shared-representation hierarchy and two direct
questions. Those comparisons were fixed before PHM test analysis and remain in
progress.

H1: amplitude-preserving input was motivated by inspection of upstream
window standardization before diagnostic test results. The actual standardized
export confirms RMS and standard deviation near one, with across-window standard
deviation 6.58e-7. Their remaining finite-precision variation is rescaled by the
training standardizer; it is not treated as exactly zero.

The completed amplitude-preserving feasibility comparison gives pooled NLL
1.022681590 for the numerical head and 1.052741468 for the legacy Qwen head on
eight disjoint test bearings. Qwen has slightly higher accuracy but poorer NLL;
its calibrated fixed-cost policy defers every test window, incurring the
always-defer cost 0.25. Unit-minus-pooled NLL differences are about -0.00017 and
their fixed-fit paired bearing intervals include zero.

This result establishes neither a language probability-quality increment nor
useful diagnostic coverage for the legacy representation. It is not a test of
question decomposition. Continue the frozen attribution comparison; do not
change questions, classes, costs, input selection or controls to protect the
original narrative. Six rotations provide within-collection sensitivity rather
than independent external confirmation. Mechanistic explanations remain untested.

## Entry template

### Date

TODO

### Claim at risk

当前最需要验证、削弱或修正的科学主张。

### Key uncertainty

威胁该主张的最大未决不确定性。

### Competing explanations

- TODO

### Chosen action

能够最直接降低该不确定性的最小连贯动作。

### Possible outcomes and decisions

- Outcome A → TODO
- Outcome B → TODO
- Outcome C → TODO

### Observation

实际获得的文献、理论、代码、实验或统计结果。

### Interpretation

分别说明：

- Fact：数据或来源直接支持什么；
- Inference：基于哪些假设推出什么；
- Hypothesis：仍需什么新证据。

### Decision

保留、削弱、拒绝或修改主张；更新方法、边界、novelty 或下一步。

### Stop or next iteration

说明当前不确定性是否已足够降低；若继续，给出唯一最高价值动作。
