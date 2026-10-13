# Research Log

只记录能够改变研究判断的信息。Git 负责代码历史；本文件不记录 hash、文件清单、命令流水账或普通执行过程。

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
