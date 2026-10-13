# Jev-PHM｜骨干无关的非生成式跨数据集故障决策学习

## 1. 研究定位与已有结果

### 1.1 方法定位

拟题：**Jev-PHM: Backbone-Agnostic Non-Generative Decision Learning for Cross-Dataset Fault Diagnosis**。

研究对象是“异构振动观测到语义化故障候选的共享概率决策”，不是某个基础模型的故障诊断应用。方法定义、题目和贡献使用 signal encoder、pretrained backbone、semantic candidate 和 shared decision readout；Qwen3-0.6B 仅作为已经使用的起步实例，具体版本仍应保留在实验设置中。

这里有两个不同的“通用性”：骨干无关指方法可在满足接口的不同骨干上重新实例化；跨数据集复用指同一实例化模型的一个 checkpoint 跨多个数据集使用，不训练数据集专属分类头。前者不意味着不同架构共享同一套权重，后者不意味着任意新故障都可零样本识别。结构上可替换和实验上可迁移必须分别验证。

### 1.2 本文件与已完成候选稿的关系

本文件复用原有 `paper/kickstart/core_innovation.md`，维护下一阶段的前瞻方法设计，不是第二份完整正文。唯一候选稿仍为 `paper/draft/main.md`；它及其预览、结果与 C1–C3 保留原研究范围。

`dev@ff9db6e4e43c8da2fc66990a03dcc78ed1fd2afb` 已完成单一 PU 集合、六维时域状态、冻结编码器的受控归因研究。主要 NLL 差 −0.001102，条件 95% 区间 [−0.008577, 0.005163]；六轮换方向变化，主拆分校准后 Qwen 全部拒判。见 [原研究状态](../paper.yaml)、[结果说明](../assets/tables/README.md)和[贡献—证据表](../experiments/evidence_matrix.md)。

这些结果不能改写为本新方法的速度、跨数据集或骨干通用性证据；它们也没有检验经过信号—语义训练的新方法。E0/E1/E2 的完成状态不回退，不为新方向复写旧结果。当前修订不改变运行代码、参数、数据划分或既有负结果。

## 2. Motivation

### 2.1 具体技术矛盾

固定类别诊断并不总需要在线生成解释文本。逐 token 生成和解析可能增加决策开销，但小模型也不天然低精度，大模型也不在所有配置下都慢。应比较完整诊断任务、观测长度与输出契约，而不是用“参数大/小”替代研究问题。

需要解决的矛盾是：专用分类器通常把决策绑定在训练类别和输出参数上，而可用语义描述指定新候选的生成式模型又可能承担并非诊断必需的响应解码。不同数据集同时存在信号条件变化和标签语义/粒度差异。仅取消文本生成不能解决这两种变化；仅对齐信号也不能自动形成可复用的类别接口。

GLiClass 已研究轻量通用分类与动态候选；HSE、UniFault、UniTS 和 YOTOnet 已分别提供异构信号或跨数据集学习的基础。CLIP 与 FD-LLM 已提供语义对齐近邻。因此本文应检验的是：**能否通过故障语义兼容的多源训练，将可迁移诊断能力与低开销共享读出结合，而非把已有模块重新命名。**相关来源与实际发表状态见 [20 篇下载清单](../refs/jev_generalist_20.md)，尤其 02–06、10、16–20。

### 2.2 可进入下一版引言的英文段落

Fault recognition across datasets requires a model to accommodate both changes in the measured signal and differences in how faults are defined. A task-specific classifier can be efficient, but its output parameters may remain tied to a particular label vocabulary. Generative language interfaces allow diagnoses to be specified in natural language, yet a bounded recognition task need not incur response-token decoding and answer parsing. These considerations motivate a shared decision interface rather than a blanket claim that large models are too slow or small models are inaccurate.

We formulate cross-dataset fault recognition as non-generative prediction over a declared set of semantic fault candidates. The proposed learning procedure couples a bounded signal representation with a shared candidate scorer, while preserving the distinctions between exact, coarsened, and incompatible source annotations. The method is defined independently of a particular pretrained backbone. For each instantiated backbone, one trained checkpoint is intended to serve multiple datasets without dataset-specific classifier heads; replacing the backbone constitutes a separate implementation and training configuration.

The scientific question is whether this formulation preserves diagnostic quality while reducing decision overhead and improving transfer to datasets excluded from fitting. Neither a normalized probability vector nor a backbone-independent interface establishes these benefits. They must be evaluated against direct signal classifiers, conventional semantic matching, and the same backbone used for minimal-output generative classification.

## 3. 一个最小方法，而不是堆叠模块

### 3.1 Backbone-independent interface

Let \(x\) be a vibration window, \(m\) its available measurement metadata, and \(\mathcal C\) a declared finite set of semantically defined, mutually exclusive diagnostic candidates. The first scope assumes the true diagnostic state is represented in \(\mathcal C\). Multi-label faults and unknown-candidate detection require different supervision and are not silently handled by a softmax.

A signal encoder \(E_\phi\) produces at most \(M\) signal tokens. A small projection adapts their width to the chosen backbone \(B_\theta\); \(P\) denotes a specified pooling/projection operation. Candidate descriptions \(d_k\) are encoded by \(T_\psi\) into the same decision space:

$$
Z=E_\phi(x,m),\qquad
h=P\!\left(B_\theta(Z)\right),\qquad e_k=T_\psi(d_k).
$$

For a shared score \(s_\omega\),

$$
p_\Theta(k\mid x,m,\mathcal C)=
\frac{\exp(s_\omega(h,e_k)/\tau)}
{\sum_{j\in\mathcal C}\exp(s_\omega(h,e_j)/\tau)}.
$$

The backbone has no dataset-specific output rows. The same score applies to every candidate. Candidate order is not a label identifier: permuting candidates should permute the output vector. Aliases that denote the same event must be canonicalized rather than counted twice in the denominator.

Candidate embeddings can be computed once after model training and reused while the model and candidate definitions remain unchanged. One signal encoding supports all candidate scores; with a dot-product readout, candidate scoring costs \(O(|\mathcal C|d)\) after encoding. This is not an end-to-end latency guarantee. Repeated cross-encoding of the full signal with every candidate would be a different, more expensive implementation and must not be described as this shared-encoding design.

The interface does not require a particular decoder family. A compatible language or time-series backbone can be instantiated with explicit adapters and source-only training. Backbone swaps do not imply zero-cost adaptation or identical pretrained knowledge.

### 3.2 Preserve usable observations

PHMFactory remains the data entrypoint. The signal interface should preserve diagnosis-relevant time/frequency structure and physical duration subject to the declared token budget; six scalar statistics are retained as an ablation, not assumed sufficient. A fixed token count does not authorize aliasing, loss of transient events, fabricated sampling rates or implicit resampling.

Sampling rate, channel meaning, normalization and window duration must be known where used. Speed/geometry-dependent features require actual metadata. Dataset IDs, filenames, individual labels and held-out test information are not input shortcuts. HSE-style temporal alignment is an existing foundation, not a newly invented module.

### 3.3 Semantic and annotation-granularity compatibility

The main PHM-specific design candidate is a shared fault vocabulary whose supervision respects what each dataset actually annotates. A source's `class_1` is not automatically another source's `class_1`; “inner-race-dominant damage” is not automatically a pure inner-race fault. Sensor location, component, damage mode and annotation granularity must be checked before joining labels.

When a validated ontology supplies mutually exclusive states and a coarse source annotation logically entails membership in a set \(S_d(y)\), use its total probability rather than inventing a finer target:

$$
\ell(x,y,d)=-\log
\sum_{k\in S_d(y)}p_\Theta(k\mid x,m,\mathcal C).
$$

For a singleton, this reduces to ordinary cross-entropy. If a source annotation is equivalent to the union of admissible states, this is the likelihood of that coarsened event. If annotation selection depends on unmodeled factors, or the set merely supplies weak inclusion information, it is only a surrogate compatibility loss; the likelihood interpretation cannot be assumed.

Set-valued supervision and partial-label learning are established ideas; see Cour, Sapp and Taskar, [Learning from Partial Labels, JMLR 2011](https://jmlr.org/papers/v12/cour11a.html). The expression above is not claimed to reproduce their convex objective or inherit their guarantees. Its proposed contribution would be an effective, explicitly verified PHM treatment of cross-dataset label mismatches, not a new generic partial-label theorem.

Co-occurring faults are not uncertainty about which single class is true. Dominant/mixed-damage labels cannot be silently mapped to a set of pure mutually exclusive faults. Retain a justified broader/mixed event or declare the tasks incomparable for this experiment. Missing annotations are not negative labels. An ontology without useful common semantics is a scientific blocker, not a reason to fabricate mappings.

If all selected sources have identical exact labels, this component becomes conventional shared semantic classification. Do not retain a “granularity-aware innovation” claim when the evaluated data do not test it.

### 3.4 Minimal learning and decision policy

Start with one trainable signal adapter, a shared candidate scorer and a documented backbone adaptation policy. Train on multiple source datasets with weights that avoid domination by the dataset containing the most overlapping windows. Define those weights in the protocol. Candidate sampling must preserve known true and compatible labels; do not create false negatives from missing or ambiguous annotations.

Use exact or compatible label likelihood as appropriate. An additional contrastive loss, teacher distillation, MoE, RAG or reinforcement learning is not mandatory; add one only if a concrete failure requires it and the comparison can isolate its benefit.

Fit any calibration and decision threshold on source development units only for the target-free transfer experiment. Probabilities change when the candidate set changes. Calibration on one set cannot automatically be reused as a guarantee on another. A rejection action is not an unknown-fault detector unless its behavior has been separately validated.

## 4. 拟议贡献与最近邻差异

### 4.1 Framework contribution

We propose a backbone-agnostic non-generative decision-learning framework for cross-dataset fault recognition. A shared signal representation and semantic candidate scorer separate the diagnosis interface from a dataset-specific classifier and avoid online response-token generation.

This is a proposed framework contribution, not an established performance claim. Qwen is an implementation choice, not part of this definition. Typed outputs, direct probability heads, pretrained backbones and semantic similarity are existing components.

### 4.2 Method contribution to test

We develop a fault-semantics-compatible multi-source learning procedure that preserves verified differences in annotation granularity instead of forcing incompatible dataset labels into a shared one-hot vocabulary.

The core test is whether this treatment improves genuine cross-dataset transfer beyond an ordinary semantic dual encoder, a carefully mapped shared classifier and the same observation without a pretrained backbone. If the compared annotations are already equivalent or the ordinary mapping explains the outcome, narrow this claim. No claim to invent semantic alignment or the compatible-set objective is made.

### 4.3 Empirical contribution to establish

We evaluate the accuracy–latency–transfer trade-off under shared-checkpoint deployment and whole-dataset exclusion, separating the effect of response decoding from signal representation and label semantics. Demonstrating the method with distinct compatible backbones is required before claiming empirically established backbone portability.

The current completed PU study supplies none of these new effect estimates. Its valid negative results remain available as a pilot; they cannot be converted into supporting experiments for the new framework.

### 4.4 “First” is not a default contribution

The original [HVAC Jev study, arXiv:2610.09937](https://arxiv.org/abs/2610.09937), posted 7 October 2026, already uses Jev for fault diagnosis. It is a submitted preprint, but that does not erase its relevance as prior work. [NanoJev](https://github.com/TianyuCodings/NanoJev) and GLiClass already cover important small-model/direct-decision patterns.

Use “we propose Jev-PHM” rather than “the first Jev-PHM framework.” A narrower priority claim needs an explicit, fully checked novelty comparison; no such claim is authorized by this document. The general method must stand on a substantive difference from the strongest relevant work regardless of its publication status.

## 5. 最少但能区分贡献的验证

### 5.1 Three different meanings of generalization

**Multi-source deployment:** one checkpoint evaluates held-out units of source datasets without dataset-specific heads.

**Target-dataset transfer:** a whole target dataset is excluded from fitting, prompt/ontology refinement based on its samples, checkpoint selection, normalization fitting and calibration. Public task definitions can specify allowed labels, but target samples and labels cannot tune the system. A new exclusion fold trains a new checkpoint using only its permitted sources; do not claim one immutable checkpoint was used across mutually incompatible training folds.

**Few-shot adaptation:** target-labeled or target-unlabeled updates belong to separately named adaptation protocols. An unseen dataset with familiar fault semantics is not an unseen fault-class experiment.

### 5.2 Necessary comparisons

Compare the proposed shared readout with a mapped shared fixed head, an ordinary signal–text semantic matcher, and a compact signal-only model receiving the same signal information. Only compare exact-label and granularity-compatible supervision where verified annotation mismatches make the distinction real. Perform candidate-order and label-definition checks without using outcomes to select favorable test prompts.

To test backbone portability, use two genuinely distinct compatible compact backbones with stated training budgets and compare the method's increment relative to each backbone's controls. This tests a method contract; it does not require universal superiority on every backbone.

For speed, include the same backbone emitting the shortest valid label, not only a long generated explanation, plus a competitive non-generative signal classifier. Fix hardware, precision, batch size, candidate set, signal duration and preprocessing. Report warm inference, cold candidate compilation, peak memory and p50/p95 latency separately. Include signal processing and device synchronization. Longer signal acquisition time is not model inference time.

### 5.3 Endpoints, limits and next action

Report Macro-F1/balanced discrimination measures alongside NLL/Brier and actual coverage at source-selected policies. Treat independent physical units as the sampling units, not overlapping windows. Record per-dataset results rather than hiding negative transfers in pooled averages. A typed output does not establish calibration; a cheap model that rejects everything has not established useful diagnostic throughput.

No new experiment is run or reported by this revision. The next action is to read the most relevant full texts and inspect the available PHMFactory sources for a defensible common label schema, then update the existing Goal with the smallest testable implementation. Do not restart completed E0/E1/E2 under their old names. Keep future implementation in the established source/runner, not a second runtime.

There is no journal or author-approval update here. The completed manuscript, formalization gates and submission state retain their existing meanings.
