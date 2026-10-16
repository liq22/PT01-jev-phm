# Jev-PHM｜文献约束的最终实验结果蓝图

## 来源与用途

归档日期：2026-10-16。来源：用户于该日提供的完整消息正文；参考基线为 `dev@d7fec8090354e85577aff003bd76ba0f2362a58c`。消息中的 sandbox Markdown、HTML 和 ZIP 附件未在本次工作区取得，本文件不声称是附件的逐字副本。

本文件是**后续结果设计参考**。文献阅读、页码核验和原构建情况均按原材料的报告归档，本次整理没有重新核验原始论文。文献数字不是本项目复现结果；新实验数值全部为 `TBD`，M* 尚未确定。表图编号仅在本蓝图内使用，不重排正文已有表图。当前研究判断与唯一下一动作仍以 [STATUS](../STATUS.md) 为准；本文件不替代现有 Goal 或新增执行队列。

建议将最终结果体系收敛为：**4 张主表、1 张基线资格附表、3 张必需结果图，以及 1 张有条件保留的机制图**。主表回答实用竞争力，机制表回答增量来源，效率图回答相同诊断要求下的计算代价，不重复展示同一个平均分数。

通用语义方法尚未获得跨数据集或匹配时延结果，投影干预也没有被确定为最终方法。因此，下文以 **M\*** 表示待定干预，所有新实验数值均为 `TBD`，不沿用上一版 *Illustrative* 样稿中的人工数值。

## 1. Literature Benchmark Matrix｜文献实验与结果对标

### 1.1 原文与文件范围

原材料报告：该轮核对了仓库已收集文献的索引，并阅读对应固定版本的**原始 PDF 方法与实验部分**；其执行环境未能取得仓库二进制副本，因此没有声称与用户本地 PDF 逐字比对。消息提到原下载稿 §1.1 另有逐项仓库 PDF 路径、原始版本及页码表；该附件本次未取得，不补造未提供的详表。下表沿用归档编号。原材料列出的仓库路径前缀为：

```text
paper/library/jev-generalist-20/
```

页码均为 **PDF 文件页码，从第一页起计数**，不混用其他版本的页码。Libra 论文为补充阅读的 JMLR 原始 PDF，原材料所读归档索引未列其本地路径。原材料报告已查看主要结果表的原始页图；JMLR 页图获取失败，其 Table 6 按原 PDF 的可解析表格文本核验。以上是原材料的阅读范围，不作为本次新增的全文精读记录；实际文件位置与许可仍参见 [现有文献清单](../refs/jev_generalist_20.md)及[归档元数据](../library/jev-generalist-20/references.json)。

### 1.2 六篇最近邻的实验矩阵

**下表数字仅为原材料整理的文献报告值，不是本项目复现结果，也不能直接填入本文比较表。**

| 文献与定位 | 科学问题、实验设置与重复方式 | 核心比较与真实报告结果 | 对本文设计的直接约束 |
|---|---|---|---|
| **GLiClass，2025，v1**。归档 R02；§2.1、§2.4，pp.7–10；Tables 2–6、Fig.2 | 共享候选编码与评分能否兼顾通用分类和吞吐。文本任务；零样本与每类 8-shot 分开。A6000、batch=1；候选数及输入长度扫描，每配置 10 次前向求平均吞吐。 | 对照 DeBERTa、RoBERTa cross-encoder。Table 2 的 large 模型平均 F1 为 **0.7193**，平均吞吐 **25.22 examples/s**。 | 这是文本分类，不是振动迁移。联合编码与分离编码不能自动继承彼此结果；Tables 3/4 的任务列表不完全一致，不能直接拿平均行计算配对优势。10 次平均吞吐也不是稳定的 p95 估计。[arXiv](https://arxiv.org/pdf/2508.07662v1) |
| **UniFault，2025，v2**。归档 R03；pp.8–10，Table 3、Fig.2；p.13，Fig.5、Table 4 | 异构数据协调、预训练与少样本适配。IMS 使用 100 个目标标签样本，PU/CNC 使用 1%；三次重复，报告均值±SD。 | 对照 ROCKET、WDCNN、MOMENT 等。Table 3 的 PU Accuracy：ROCKET **80.1±0.2%**，UniFault-Base **79.3±0.5%**。 | 目标标签适配必须与 source-only 分开。**预训练接触范围表述需澄清**：p.8 强调排除微调样本，p.9 又称排除三个数据集的全部样本，不能仅凭其中一句断定协议。Table 4 是训练成本，不是推理时延。[arXiv](https://arxiv.org/pdf/2504.01373v2) |
| **YOTOnet，2026，v1**。归档 R04；pp.10–13，Tables 1、3、4、Fig.6；pp.15–16，Tables 5–7 | 五个轴承数据集，内圈／外圈二分类，统一至 25.6 kHz；30 种源／目标组合。所读实验设置未明确随机重复数及区间。 | 对照 DANN、CORAL、MixStyle。Table 3 的 CWRU F1：YOTOnet **0.3033**，MixStyle **0.4568**。Table 4 的 Full CWRU 为 **0.3582**。 | 不能预设最新方法在每个目标都最好；不能将两张表的 Full 直接相减作消融。原始二类设置不证明健康检测或混合损伤识别；256 个目标标签适配结果不能并入零目标数据比较。[arXiv](https://arxiv.org/pdf/2605.04528v1) |
| **Jev HVAC study，2026，v1**。归档 R01；§3.3–3.8，pp.16–20；p.24 Table 5；pp.25–26 Fig.3 | 表示、推理需求与分布变化下的决策模型诊断。60 分钟窗口；季节、系统及建筑变化；按天 bootstrap，另检查重复推理的非确定性。 | Jev、Qwen2.5、LightGBM、APAR。Table 5 的 ORNL R1：yes/no 检测率／误报率为 **95%／84%**；condition answer 为 **53%／8%**。 | 高检测率必须同时报告误报。已有单词输出及 next-token 概率对照，不能只拿长篇生成作慢基线；应比较目标绝对分数，而非只比较“下降幅度小”。[arXiv](https://arxiv.org/pdf/2610.09937v1) |
| **Towards Unbiased Exploration in Partial Label Learning，JMLR 2024**。§4.3；p.30 §6.3；pp.33–35 §6.5、Table 6 | 部分标签 NLL 的偏好放大及概率比保持。真实部分标签数据、线性模型与 MLP；10-fold，Accuracy 均值±SD。 | NLL、RC、LWS 等损失。Table 6 的 Lost 线性模型：Libra **69.8±3.1%**，NLL **61.6±3.2%**；优势不是所有数据和模型都成立。 | **后续实验使用带权 Libra**，权重为 `1−Σ yᵢpᵢ`，用于缓解过拟合。不能用裸损失冒充论文完整基线；具体反传语义仍须核官方实现。原参数化下的保持性质不能自动推广到语义候选头。[机器学习研究期刊](https://jmlr.org/papers/volume25/23-0868/23-0868.pdf) |
| **Towards a more realistic evaluation…，2026，v5**。归档 R20；p.12 §4.1.3；p.23 Table 10；p.24 Fig.13；p.36 Table 16 | 物理轴承分离、输入表征与评价协议。部件多标签 AUROC；调参分割后使用 100 个随机评价分割。 | RF、SVM、WDCNN。Table 10 的 CWRU Macro-AUROC：RF 时域＋包络 **85.06±8.92%**，WDCNN 包络 **74.51±9.43%**。 | 必须保留认真调优的信号基线。论文明确某些评价数据此前可出现在调参中，不能把“100 新分割”视为 100 次独立确认；多标签 AUROC、其 PU 队列均不能与本文原先导直接混用。[arXiv](https://arxiv.org/pdf/2509.22267v5) |

### 1.3 文献对标后的关键取舍

**最重要的不是把更多方法放进主表，而是区分三类参照：**

| 参照类型 | 本文需要的对象 | 回答什么 |
|---|---|---|
| 实用竞争力 | 认真调优的 RF／紧凑信号网络，以及任务兼容的 source-only 近邻 | 新方法是否值得使用？ |
| 机制归因 | 同读出的事件 NLL、论文版本的加权 Libra、必要的条件冻结 | 增量是否来自所提出的干预？ |
| 计算比较 | 精确单 token 标签评分、最短合法生成、语义读出 | 在线成本差异究竟来自哪里？ |

**当前不能把 M\* 与普通事件 NLL 预设为两个不同算法。**如果它们最终使用相同架构、监督和优化，应该合并成一行，而不是让同一方法同时扮演 baseline 与 proposed method。原材料指出，参考基线正文已经明确这一边界。

## 2. Claim–Experiment Matrix｜主张、实验与证伪条件

### 2.1 一个核心科学问题

> **在可核验的原生标注差异和测量变化下，保留源标注所支持的故障区分，何时能够降低完全留出数据集上的、按物理设备等权计算的诊断 NLL？**

沿用 **C4：诊断迁移增量**作为主问题；**C5：相同诊断要求下的计算代价**作为部署验证。保护组内区分的机制分析属于 C4 的条件检验，不新增一项平行贡献。

### 2.2 决定性实验链

| Claim → Hypothesis | Baseline | Protocol／固定条件 | Metric | 假设成立时应出现的模式 | 证伪或收缩条件 |
|---|---|---|---|---|---|
| **C4：实用迁移增量** | 合格信号基线、可比 source-only 近邻 | 同目标事件、同评价单位；各自合理源侧调优；目标不参与选择 | 设备等权 NLL、错误率、Macro-F1、Brier | 在声明的目标范围内，相对强参照有可重复、具有实际意义的负 ΔNLL | 简单方法解释全部收益；改善仅出现在参与开发的目标；区间过宽则不确定 |
| **C4：监督信息的作用** | 精确来源-only、共同粗标签、原生事件 NLL | 读出和表示一致；明确各组实际使用的精确／粗标签数量 | 共同粗事件 NLL；存在细真值时的细类 NLL | 保留原生信息的作用不能仅由降低任务粒度解释 | 全部来源只有同义精确标签，或所谓粗细关系不合法 |
| **C4：保护机制的独立作用，条件项** | 事件 NLL、加权 Libra、带偏置条件冻结 | 同一初始模型、合法固定分区、表示和更新信息 | 组内 log-odds 漂移、粗／条件／细 NLL | 减少有害漂移，同时保留粗事件学习，并产生超出简单方法的诊断增量 | 只减少漂移却伤害诊断；冻结对照足够；投影零有效秩 |
| **C5：误差—时延关系** | 精确标签评分、最短生成、紧凑信号模型 | 同硬件、精度、输入时长、候选集合和任务；先比较全覆盖率 | p50／p95、错误率、覆盖率、显存 | 在相同诊断要求下获得有价值的成本改善或明确权衡 | 优势依赖长回答、更少观测、较高错误率或全部拒判 |

**预期优势应是有条件的：**只有真实粗标签提供了有用信息、共享测量关系能够迁移、且简单方法不能充分解释增量时，C4 才值得作为方法贡献。不能从“语义接口通用”直接推出“跨数据集精度更高”。

### 2.3 统一统计对象

主要差值建议继续采用设备配对的 NLL：

$$
R_b^{(m)}
=
\frac{1}{n_b}
\sum_i-\log p_m(y_{bi}\mid x_{bi}),
$$

$$
\Delta R
=
\frac{1}{B}
\sum_b
\left(R_b^{(M^*)}-R_b^{(M_0)}\right).
$$

这里是**设备等权的窗口损失**，不是先输出一个设备诊断再计算损失。

“均值±SD”用于实际重复拟合之间的变异；物理设备配对区间用于评价单位不确定性，二者分列。一次拟合的 SD 应为 `NA`。不预先填写 p 值或排名；目标数量少、事件定义不同的结果不做平均排名或临界差异图。实际意义界值目前仍为 `TBD`，不能任意规定“提升 1% 即有价值”。

## 3. Final Tables｜完整论文结果表模板

### Table 1. Native fault semantics, independent units, and information access

**论文作用：**定义哪些结果具有可比性，而不是只列数据集名字。

| Role / Dataset | Native labels / diagnostic events | Exact or coarsened annotation / mixed damage | Channel / rate / window duration | Independent units: train / tune / cal / test | Pretraining exposure | Status |
|---|---|---|---|---|---|---|
| Fine-label source `S_f` / TBD | TBD | Exact；原始依据 TBD | TBD | TBD / TBD / TBD / NA | TBD | 未准入 |
| Coarse-label source `S_c` / TBD | TBD | 合法粗化映射 TBD | TBD | TBD / TBD / TBD / NA | TBD | 未准入 |
| Excluded target `T1` / TBD | TBD | 可评估细真值／粗事件 TBD | TBD | NA / NA / NA / TBD | 不允许任务相关接触；否则 UNKNOWN | 未准入 |
| Additional target `T2` / 条件项 | TBD | TBD | TBD | NA / NA / NA / TBD | 同上 | 仅范围主张需要时增加 |

**英文表注：**

> Files and windows inherit the identity of their physical acquisition unit. Public class definitions specify the task but do not authorize target-sample access. A shared checkpoint refers to one source-trained model instance; independently retrained leave-one-dataset-out folds have distinct checkpoints.

CWRU 和 PU 只是待核验候选，不能自动填成“精确来源”和“粗标签来源”。现有 PU 混合／主导标签问题也不能通过名称映射消除。两个集合本身不自动构成“两个来源＋独立目标”的完整设计。

### Table 2. Diagnostic performance on entirely excluded target datasets

**论文作用：**回答各自合理调优后的实用竞争力。

每个目标单独一个 panel，表头必须给出：

```text
Target / event vocabulary / source set / backbone / physical units /
number of fits / checkpoint(s)
```

| Method | Supervision / comparison role | NLL ↓, mean ± SD | Error (%) ↓ | Macro-F1 ↑ | Brier ↓ | ΔNLL vs M0 [95% CI] | Adjusted p / rank* |
|---|---|---|---|---|---|---|---|
| RF, time + envelope features | 精确标签子集；实用强参照 | TBD | TBD | TBD | TBD | TBD | TBD |
| Selected compact signal classifier | 同一合法监督；实用参照 | TBD | TBD | TBD | TBD | TBD | TBD |
| Closest qualified source-only method | 可比 YOTOnet／MixStyle 等，版本 TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Semantic scorer, exact sources only | 信息量参照 | TBD | TBD | TBD | TBD | TBD | TBD |
| Semantic scorer + event NLL **M0** | 同读出的主要机制参照 | TBD | TBD | TBD | TBD | Reference | NA |
| Semantic scorer + paper-faithful weighted Libra | 现有损失修正 | TBD | TBD | TBD | TBD | TBD | TBD |
| Jev-PHM intervention **M\*** | 具体增量／实现 TBD | TBD | TBD | TBD | TBD | TBD | TBD |

\* 只有在比较族、检验和排名范围合理时保留该列，否则删除，不为完整性强行计算。

**英文表注：**

> All entries are computed under the stated target-exclusion protocol. NLL and Brier use the same event space and unit weights. Fit-to-fit standard deviations and paired physical-unit confidence intervals quantify different sources of variation. Published scores from unmatched protocols are not inserted into this table.

必须保留的公平性条件：

- RF 若不能利用粗标签，就与语义模型的 **exact-only** 组对应，不冒称获得相同的全部监督。
- YOTOnet 原版只支持所读论文的内／外圈任务时，应放入独立的匹配任务面板；扩展输出空间必须标明是修改后的实现。
- **M\* 仅胜过 exact-only，而不胜过 M0，不足以证明新的更新机制。**
- ROCKET 等方法可在源侧资格判断后替代某个参照，不自动把所有方法加入训练队列。

### Table 3. Supervision effects and conditional-update diagnostics

**论文作用：**把“信息增加”与“学习机制改变”分开。

#### Panel A：相同事件上的监督对照

| Training scheme | Fine / coarse labelled units | Common-event NLL ↓ | Fine-label NLL ↓ | Conditional NLL ↓ | ΔFine NLL vs M0 [95% CI] |
|---|---|---|---|---|---|
| Exact sources only | TBD / 未使用粗标签 | TBD | TBD | TBD | TBD |
| Common-coarse supervision only | TBD | TBD | **NA** | **NA** | **NA** |
| Native event supervision **M0** | TBD / TBD | TBD | TBD | TBD | Reference |
| **M\*** with the same annotation information | 与 M0 相同 | TBD | TBD | TBD | TBD |

不能给共同粗标签模型任意分配组内概率，再用细类 NLL 判它较差。该限制与参考基线正文的概率对象一致。

#### Panel B：固定粗来源更新块，仅在保护机制被保留时运行

| Update rule | Effective rank / applicability | Observed log-odds drift ↓ | Coarse NLL after update ↓ | Conditional NLL after update ↓ | Fine NLL after update ↓ | 理论预期，非测量 |
|---|---|---|---|---|---|---|
| Pre-update checkpoint | TBD | Reference | TBD | TBD | TBD | 共同起点 |
| Ordinary event NLL | TBD | TBD | TBD | TBD | TBD | 不预设漂移有害 |
| Paper-faithful weighted Libra | TBD | TBD | TBD | TBD | TBD | 原定理条件须匹配 |
| Frozen conditionals + offset coarse residual | TBD | TBD | TBD | TBD | TBD | 固定条件下组内比值不变 |
| Projected adapter，条件项 | TBD；零秩不训练 | TBD | TBD | TBD | TBD | 固定条件下保持；零秩整体不变 |

**英文表注：**

> All post-update quantities refer to the same diagnostic observations and initial predictor. The offset-head control retains the initial log block masses. Analytical invariance is stated separately from numerical drift. Under a valid partition and identical evaluation weights, fine NLL equals coarse-event NLL plus within-event conditional NLL.

漂移必须在预先固定的观测池和类别对上计算，使用绝对 log-odds 变化，避免正负抵消。全部标签精确、没有组内类别对时，该量为 `NA`，不是自动测得零。

### Table 4. Diagnostic error and computational cost

**论文作用：**展示可用诊断与计算成本，而不是只展示“加速倍数”。

#### Panel A：全覆盖率与处理时延

| Backbone / model | Readout | Training / weight comparison | Error at 100% coverage ↓ | Candidate NLL ↓ | Warm p50 / p95, ms ↓ | Cold candidate encoding, ms | Peak memory, GB |
|---|---|---|---|---|---|---|---|
| Compact signal model | 原生分类头 | 各自调优的实用参照 | TBD | TBD | TBD / TBD | NA | TBD |
| Backbone A | 精确单 token 评分 | 同已训练权重的读出比较，条件 TBD | TBD | TBD | TBD / TBD | NA | TBD |
| Backbone A | 最短合法标签生成 | 同上；格式失败计入错误分母 | TBD | TBD／NA | TBD / TBD | NA | TBD |
| Backbone A | 普通语义读出 M0 | 训练条件明示 | TBD | TBD | TBD / TBD | TBD | TBD |
| Backbone A + M\* | 相同语义读出 | 实际训练干预 TBD | TBD | TBD | TBD / TBD | TBD | TBD |
| Backbone B，条件项 | 重复必要读出比较 | 重新实例化，不与 A 共用权重 | TBD | TBD | TBD / TBD | TBD | TBD |

#### Panel B：源侧固定阈值后的目标表现

| Target / method | Source-fixed acceptance / alarm thresholds | Target coverage ↑ | Accepted error ↓ | False-alarm rate ↓ | Missed-fault rate ↓ | Scenario cost* ↓ |
|---|---|---|---|---|---|---|
| T1 / strongest qualified comparator | TBD / TBD | TBD | TBD | TBD／NA | TBD／NA | TBD／NA |
| T1 / M0 | TBD / TBD | TBD | TBD | TBD／NA | TBD／NA | TBD／NA |
| T1 / M\* | TBD / TBD | TBD | TBD | TBD／NA | TBD／NA | TBD／NA |

\* 只有成本定义有清楚场景依据时保留。

**接受阈值与报警阈值是两个对象。**前者作用于诊断置信分数，后者可作用于完整类别分布导出的故障概率。误报、漏报在全部合规观测上计算，不删除拒判样本，也不将拒判自动算作正确处理。没有健康类的内／外圈条件分类不填写这些检测指标。

**英文表注：**

> Acquisition duration is reported separately from processing latency. Timings include the implemented preprocessing, encoder and readout, with device synchronization. Cold candidate compilation is reported separately. Source-fixed deployment coverage is distinct from retrospective risk–coverage analysis; accepted error is undefined at zero coverage.

### Table S1. Baseline qualification and model-selection budget

| Baseline | 需核验的关键配置 | 最大 trials／计算预算 | 选择与停止规则 | 实际消耗／收敛 | 资格 |
|---|---|---|---|---|---|
| RF／紧凑信号网络 | 输入表征、树／叶限制或学习率／正则 | TBD | 源侧有效风险；规则 TBD | TBD | 待核验 |
| Source-only 最近邻 | 原版／修改版、目标数据权限 | TBD | 源侧选择，不接触目标 | TBD | 待核验 |
| Semantic M0／exact-only | 相同候选、源监督、适配预算 | TBD | 同指标、明确 tie rule | TBD | 待核验 |
| Weighted Libra | 原文加权、反传语义、停止条件 | TBD | 不用错误版本弱化对照 | TBD | 待核验 |
| Conditional freeze／M\* | 初始偏置、更新空间和正则 | TBD | 机制匹配与各自调优分开 | TBD | 待核验 |

## 4. Final Figures｜最终结果图设计

### Figure 1｜目标效应与物理设备异质性

**目的：**Table 2 给绝对分数，这张图给配对效应和设备间差异，避免重复画平均柱状图。

| 子图 | 布局与轴 | 关键内容 |
|---|---|---|
| **(a)** | 左侧约 55%；X＝配对 ΔNLL；Y＝目标×方法 | 零线、真实点估计、设备级区间；不同标签空间不合并 |
| **(b)** | 右侧约 45%；X＝物理设备；Y＝该设备的 M\*−M0 NLL | 一点一个独立单元，检查平均改善是否完全由单个设备主导 |

**English caption**

> **Paired diagnostic effects on entirely excluded target datasets.** (a) Unit-balanced NLL differences relative to the prespecified reference. (b) Per-unit differences between the selected intervention and that reference. Intervals resample physical evaluation units and state whether training variability is included. Negative differences favor the intervention. Targets with different event vocabularies are not pooled.

理想模式是有辨识度的负差值，同时存在可解释的设备差异；目标间反转或个别轴承主导必须保留，不能只显示平均改善。

### Figure 2｜保护性质与学习能力，条件保留

**仅当保护干预进入最终方法时保留。**

| 子图 | 布局与轴 | 关键对照 |
|---|---|---|
| **(a)** | 左；X＝实际更新步；Y＝固定观测池的平均绝对组内 log-odds 漂移 | 事件 NLL、加权 Libra、带偏置冻结、候选干预 |
| **(b)** | 右；X＝粗事件 NLL 变化；Y＝组内条件 NLL 变化 | 同一起点到终点的变化；识别“粗类学好了但细类被破坏” |

**English caption**

> **Conditional preservation versus coarse-event learning.** (a) Within-block log-odds drift during a fixed coarse-supervision update block. (b) Changes in coarse-event and within-event conditional NLL from the same initial predictor. All methods receive the same update information. Preservation is a design property rather than evidence of improved target diagnosis; offset-head and loss-based controls test simpler explanations.

理想模式不是“漂移曲线最低”本身，而是**减少有害漂移、没有阻断粗事件学习，并在独立目标上产生额外诊断收益**。若简单冻结足够，就收缩投影贡献。解析零秩反例只能单独标为解析诊断，不能混入 PHM 实测点。

### Figure 3｜误差—时延与输入规模

| 子图 | 布局与轴 | 关键对照 |
|---|---|---|
| **(a)** | 左；X＝warm p95 时延；Y＝100% 覆盖率错误率 | 精确 token、最短生成、语义 M0、M\*、紧凑信号模型 |
| **(b)** | 右；X＝信号 token 预算；Y＝p95 时延 | 保持物理观测时长和候选定义不变，比较读出方式 |

**English caption**

> **Diagnostic error and computational cost under matched conditions.** (a) Full-coverage error versus warm p95 processing latency. (b) Latency as signal-token budget varies at fixed candidate definitions and acquisition duration. Minimal label generation, exact label-token scoring and semantic readout use the same execution conditions. An independently tuned compact signal model is shown as a practical reference; cold compilation is reported separately.

不预设 M\* 比紧凑模型快。若改变的只是训练规则、推理图相同，M0 与 M\* 的延迟接近是合理结果。

候选数扫描只有在存在合法、不同且覆盖任务的候选定义时才添加；**不能复制别名凑 128 类，或删掉真实类别来制造可扩展性曲线。**

### Figure 4｜选择性诊断与误报—漏报

| 子图 | 布局与轴 | 关键内容 |
|---|---|---|
| **(a)** | 左；X＝接受覆盖率；Y＝接受样本错误率 | 从保存概率计算真实风险—覆盖率曲线，不强制平滑或单调 |
| **(b)** | 右；X＝目标／方法；Y＝误报率、漏报率 | 源侧固定报警阈值，成对展示，不只选有利指标 |

**English caption**

> **Selective diagnosis and source-fixed alarm behavior.** (a) Empirical risk–coverage curves computed from saved class distributions. (b) False-alarm and missed-fault rates at thresholds selected using source data only. Full-coverage error and achieved deployment coverage are reported alongside the curves. Detection metrics require a defined healthy state; reduced coverage is not treated as an improvement by itself.

ECE 图暂放附录，除非最终确实提出独立的校准贡献。当前先导已经出现较低 ECE、较高 AURC 和全部拒判并存，不能只选“校准更好看”的图。

## 5. Expected vs. Falsifying Outcomes｜支持与推翻模式

| 最终观察到的结果 | 可以支持什么 | 必须放弃或限制什么 |
|---|---|---|
| M\* 优于 exact-only，但未优于同架构事件 NLL | 增加合法监督可能有价值 | 不能归因新的更新规则 |
| 相对 M0 与合格实用参照均有足够精度的负 ΔNLL | 声明范围内的迁移增量 | 不外推为任意设备／故障通用 |
| 漂移减少，但目标 NLL 增加 | 实现了保护性质，可能保存错误判断 | 否定该条件下的有益保护假设 |
| 带偏置冻结或加权 Libra 解释全部可辨识效果 | 现有简单方法已经足够 | 不通过新名称保留算法贡献 |
| 投影有效秩为零 | 命中已有退化条件 | 不继续用更多训练预算掩盖 |
| 精确 token 评分与语义读出速度接近 | 直接分类本身已经高效 | 不再将避免长生成当作独有加速 |
| 检测率很高但误报也很高，或低成本仅因全拒判 | 尚未建立有用诊断 | 不称可靠、安全或工业优势 |
| 原生标签全是同义精确标签 | 可研究共享标签诊断 | 不宣称验证了原生粗标签机制 |
| 区间跨零且较宽 | 当前证据不足 | 不写成等效、无效或普遍失败 |

**没有必要提前规定一组必须出现的“全优趋势”。**本文最低值得形成独立知识的结果应是：在明确标注和测量条件下，识别何时存在增量、何时被简单方法解释、何时约束反而阻断学习。该判断仍需要真实比较，不由模板本身保证。

## 6. Experimental Execution Priority｜最小执行顺序

本节保留原材料建议的条件顺序，作为后续结果组织参考。实际当前动作、预算及准入判断以 [STATUS §E](../STATUS.md#e-唯一下一步) 为准。

| 优先级 | 动作 | 验收产物 | 停止条件 |
|---|---|---|---|
| **P0：数据资格** | 完成既有 CWRU／PU 文件、标签、通道和独立单元对应 | Table 1 可以据实填写；合法事件覆盖无冲突 | 无可访问输入则 BLOCKED；只有精确同义标签则收缩粒度主张 |
| **P1：定义增量与基线资格** | 明确 M\* 相对 M0 实际改变什么；核对 Libra、条件冻结和信号参照 | Table S1、源侧选择与收敛记录 | 相同学习过程不重复命名；基线不合格不扩大主比较 |
| **P2：低成本机制筛选** | 仅当保护机制仍保留，在预留源侧诊断池做固定更新块 | Table 3B、Figure 2 | 零有效秩、简单冻结足够或约束伤害学习时停止该候选 |
| **P3：锁定目标的核心比较** | 方法和源侧选择固定后，评价一个未参与开发的合法目标 | Table 2、Table 3A、Figure 1 | 负结果保留；若据此改方法，必须寻找新的独立确认 |
| **P4：效率与适用范围** | 对有实际识别能力的实例测读出成本；按主张需要再加第二目标／骨干 | Table 4、Figures 3–4 | 不以全部拒判、长答案或不匹配信息制造优势 |
| **P5：成稿** | 从原生产物替换 TBD，更新结论与限制 | 唯一正文、表图和结果一致 | 只补影响核心结论的缺口 |

当前立即动作仍是原 STATUS 中的 **CWRU 数据资格核验**，而不是启动新的大模型训练。本蓝图不替换该动作，也不把先前的投影推导当成新实验准入。

原材料报告其对应工作已完成文献对标、结果表模板及图形规格，Markdown／HTML 构建无警告；同时说明未修改仓库、实现、数据、已有结果或实验完成状态，也未运行模型实验。这是原材料的历史说明，不作为本次归档的验证结论。本次只归档上述消息内容并接入现有参考入口。
