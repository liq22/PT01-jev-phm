# Jev-PHM｜20 篇相关文献：发表状态与下载入口

## 1. 核验口径

### 1.1 状态与文件版本分开

核验日期：2026-10-13。保留上一版 01–20 编号，便于对应下载文件。分组依据为出版社、会议论文集、arXiv 作者声明或作者/实验室主页；arXiv 是文件托管渠道，不等于未录用。未检索到录用记录只表示本轮未核验，不等于作者未获录用。

A 组 13 项有正式期刊/会议记录或明确会议录用声明；B 组 1 项由作者公开列为期刊论文，但正式出版信息仍待核；C 组 6 项仅核验到预印本或技术报告。第 09 项改为优先阅读同团队的相关期刊论文；旧预印本不是自动认定的期刊终稿。版本关系不能靠相同模型名推断。

本轮核验的是题录、发表状态、原始摘要及可获得的方法说明，不声称逐篇完整阅读、下载 PDF 或复现结果。某些出版社完整页面访问失败，使用了其公开索引和作者一手页面；相应限制列于备注。

## 2. A｜已发表或会议录用明确（13 项）

### 2.1 PHM 与跨数据集诊断

| 原编号 | 优先阅读的正式文献 | 已核验状态/来源 | 下载入口与版本说明 | 对新方法的用途 |
|---|---|---|---|---|
| 05 | Li et al. **HSE: A plug-and-play module for unified fault diagnosis foundation models** | Information Fusion 123, 103277 (2025)；[出版社](https://www.sciencedirect.com/science/article/pii/S1566253525003501) | [DOI / PDF入口](https://doi.org/10.1016/j.inffus.2025.103277)；校内访问可能必需 | 异构采样、长度和通道的输入统一；作为基础，不重复声称提出统一嵌入。 |
| 06 | Ren et al. **PHM–GPT: A Large Language Model for Prognostics and Health Management** | Engineering 63, 156–172 (2026)；[期刊官网](https://www.engineering.org.cn/engi/EN/) | [DOI / 原文入口](https://doi.org/10.1016/j.eng.2025.11.001)；DOI 中的 2025 不是最终卷期年 | 多任务 PHM 与信号—语言接口；核查其目标数据适配条件，而不直接继承“通用”结论。 |
| 07 | Li et al. **VSLLaVA: A pipeline for Large Multimodal Foundation Models in industrial vibration signal analysis** | Advanced Engineering Informatics 76, Part B, 105023 (2026)；[出版社记录](https://www.sciencedirect.com/science/article/pii/S1474034626007159) | [正式版](https://doi.org/10.1016/j.aei.2026.105023)；[公开预印本 PDF](https://arxiv.org/pdf/2409.07482)。网页已公开，标注卷期为 2026 年 11 月；不能说该卷期已经过去 | 领域对齐与生成式分析近邻；预印本与期刊题名/内容可能不同，引用正式题录。 |
| 09 | Qaid et al. **Large language models for explainable fault diagnosis of machines** | Engineering Applications of Artificial Intelligence 163, Part 4, 113131 (2026)；[出版社](https://www.sciencedirect.com/science/article/abs/pii/S0952197625031628) | [正式版](https://doi.org/10.1016/j.engappai.2025.113131)。原清单的 [FD-LLM 预印本](https://arxiv.org/abs/2412.01218)为相关早期工作；未完全核实版本沿革，不把该 PDF 标为期刊接受稿 | 更新为较新的分类与谱—语言建模工作；与第 10 项作者和论文不同。 |
| 10 | Lin et al. **FD-LLM: Large language model for fault diagnosis of complex equipment** | Advanced Engineering Informatics (2025), 103208；[出版社/DOI](https://doi.org/10.1016/j.aei.2025.103208) | [正式原文/PDF入口](https://doi.org/10.1016/j.aei.2025.103208) | 时间序列—文本对齐近邻；不能把已有对齐模块本身当成新贡献。 |
| 11 | Wang et al. **PGA-LLM: A Probability-Guided Alignment Large Language Model Framework for Fault Diagnosis** | Technologies 14(8), 494 (2026)，2026-08-06；[出版社](https://www.mdpi.com/2227-7080/14/8/494) | [开放原文/PDF入口](https://www.mdpi.com/2227-7080/14/8/494)；DOI 10.3390/technologies14080494 | 区分信号侧分类和语言侧报告；不能将信号侧分类分数归为 LLM 生成收益。 |
| 12 | Zhang et al. **A Multi-Stage Post-Training Framework for Domain-Specific Language Models in Fault Diagnosis** | AI 7(9), 383 (2026)；[出版社](https://www.mdpi.com/2673-2688/7/9/383) | [开放原文/PDF入口](https://www.mdpi.com/2673-2688/7/9/383)；DOI 10.3390/ai7090383 | 领域后训练参考；任务是 HVAC 维护文本，不是振动跨数据集验证。 |
| 20 | Vieira et al. **Towards a more realistic evaluation of machine learning models for bearing fault diagnosis** | Mechanical Systems and Signal Processing 258, 114640 (2026)；[正式题录](https://doi.org/10.1016/j.ymssp.2026.114640) | [正式版](https://doi.org/10.1016/j.ymssp.2026.114640)；[公开预印本 PDF](https://arxiv.org/pdf/2509.22267)。以正式版的任务和数据集描述为准 | 物理轴承隔离、来源与故障共现；窗口数不能替代独立设备数。 |

### 2.2 语义分类、通用表示与效率对照

| 原编号 | 文献 | 已核验状态/来源 | 下载入口与版本说明 | 对新方法的用途 |
|---|---|---|---|---|
| 15 | BehnamGhader et al. **LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders** | COLM 2024；[arXiv 明确写 Accepted to COLM 2024](https://arxiv.org/abs/2404.05961)，[作者实验室](https://mcgill-nlp.github.io/publications/llm2vec/) | [公开 PDF](https://arxiv.org/pdf/2404.05961)。不是“仅预印本”；不假定托管 PDF 与会议终稿逐字相同 | 骨干转编码器的一般方法；最后 token 池化不是其完整复现。 |
| 16 | Radford et al. **Learning Transferable Visual Models From Natural Language Supervision** | ICML 2021，PMLR 139；[论文集](https://proceedings.mlr.press/v139/radford21a.html) | [正式原文/PDF入口](https://proceedings.mlr.press/v139/radford21a.html) | 共享语义候选与对比学习已有基础；不能把简单信号版 CLIP 拼接视为充分创新。 |
| 17 | Gao et al. **UniTS: A Unified Multi-Task Time Series Model** | NeurIPS 2024；[论文集](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fe248e22b241ae5a9adf11493c8c12bc-Abstract-Conference.html) | [公开 PDF](https://arxiv.org/pdf/2403.00131)；[正式记录](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fe248e22b241ae5a9adf11493c8c12bc-Abstract-Conference.html) | 同一模型跨任务复用的参照；共享代码与共享 checkpoint 不同。 |
| 18 | Hsieh et al. **Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes** | Findings of ACL 2023，8003–8017；[论文集](https://aclanthology.org/2023.findings-acl.507/) | [正式 PDF](https://aclanthology.org/2023.findings-acl.507.pdf) | 大模型训练监督与小模型部署解耦；蒸馏不是当前必加模块。 |
| 19 | Tan et al. **Are Language Models Actually Useful for Time Series Forecasting?** | NeurIPS 2024；[论文集](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ed5bf446f59e2c6646d23058c86424b-Abstract-Conference.html) | [公开 PDF](https://arxiv.org/pdf/2406.16964)；arXiv 明确注明 NeurIPS 2024 Spotlight | 对骨干必要性的反证对照；预测任务的反例不直接否定所有故障诊断方法。 |

## 3. B｜作者公开列为期刊论文，出版元数据仍待补齐（1 项）

### 3.1 不再归为“仅 arXiv”，也不编造 DOI

| 原编号 | 文献 | 可核验依据 | 下载入口 | 仍未核验 |
|---|---|---|---|---|
| 03 | Eldele et al. **UniFault: A Fault Diagnosis Foundation Model from Bearing Data** | [第一作者主页](https://emadeldeen24.github.io/)已列 IEEE Transactions on Industrial Informatics (TII), 2026；主页还报告 2026 年 9 月一篇 TII 录用。期刊归属采用作者报告层级，不等同出版社记录 | [作者链接的 arXiv PDF](https://arxiv.org/pdf/2504.01373)；[arXiv 题录](https://arxiv.org/abs/2504.01373) | IEEE 正式 DOI、卷期/页码、接受稿与预印本差异。本轮没有足够依据填写这些字段。 |

UniFault 应与没有录用线索的预印本分开。对其具体期刊状态，严谨写法是“作者公开列为 TII 2026；正式出版元数据待核”，而非“未发表”，也不写成“已核验 IEEE 终稿”。

## 4. C｜本轮仅核验到预印本或技术报告（6 项）

### 4.1 不把未找到录用记录写成未获录用

| 原编号 | 文献 | 状态及原始来源 | 下载入口 | 阅读重点 |
|---|---|---|---|---|
| 01 | Jung. **Where Can a Decision Model Diagnose HVAC Faults? Reasoning Demand, Physical Representation, and Robustness Under Shift** (2026) | [arXiv:2610.09937](https://arxiv.org/abs/2610.09937)，2026-10-07；作者标注 Submitted to Energy and Buildings，不是 Accepted | [PDF](https://arxiv.org/pdf/2610.09937) | 已直接用 Jev 做 HVAC 诊断，是“首次 Jev-PHM”宽泛主张的反例线索；预印本也属于应讨论的先行工作。 |
| 02 | Stepanov et al. **GLiClass: Generalist Lightweight Model for Sequence Classification Tasks** (2025) | [arXiv:2508.07662](https://arxiv.org/abs/2508.07662)；未找到可确认的正式录用记录 | [PDF](https://arxiv.org/pdf/2508.07662) | 非生成式、动态候选、轻量通用分类的直接方法近邻。 |
| 04 | Wang et al. **YOTOnet: Zero-Shot Cross-Domain Fault Diagnosis via Domain-Conditioned Mixture of Experts** (2026) | [arXiv:2605.04528](https://arxiv.org/abs/2605.04528)；未找到可确认的正式录用记录 | [PDF](https://arxiv.org/pdf/2605.04528) | 一次训练/跨数据集诊断；仔细检查目标集合是否完全留出。 |
| 08 | Chen et al. **FaultGPT: Industrial Fault Diagnosis Question Answering System by Vision Language Models** (2025) | [arXiv:2502.15481](https://arxiv.org/abs/2502.15481)；未找到可确认的正式录用记录 | [PDF](https://arxiv.org/pdf/2502.15481) | 生成式诊断问答近邻；不以未录用作为忽略相关方法的理由。 |
| 13 | Qwen Team. **Qwen3 Technical Report** (2025) | [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)；模型技术报告，未核验期刊/会议录用 | [PDF](https://arxiv.org/pdf/2505.09388) | 实例化/复现材料，不作为本文通用方法的命名或创新来源。 |
| 14 | Zhang et al. **Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models** (2025) | [arXiv:2506.05176](https://arxiv.org/abs/2506.05176)；技术报告，未核验期刊/会议录用 | [PDF](https://arxiv.org/pdf/2506.05176) | 可替换的语义编码实现之一；不把其 checkpoint 与普通 Qwen3 混用。 |

## 5. 对研究定位的直接影响

### 5.1 先读哪些全文

方法与创新先读 02、03、04、05、06、10、16、17；近邻排查保留 01，评价边界读 19、20。发表状态不是相关性排名：尚未正式发表的直接近邻仍需讨论。

Qwen 文献退到“实现实例与对照”的位置。下一步方法定义维护于 [现有创新设计文件](../kickstart/core_innovation.md)，拟题为 **Jev-PHM: Backbone-Agnostic Non-Generative Decision Learning for Cross-Dataset Fault Diagnosis**。

### 5.2 引用与下载规则

引用有正式记录的工作时优先正式题录和 DOI；公开 PDF 可以来自作者 arXiv。对第 09 项，期刊文章与旧预印本暂按相关工作沿革处理，不宣称逐字终稿对应。UniFault 保留作者报告的期刊归属和未核验字段。不将 20 项全部塞入当前已完成候选稿的 bibliography。

本文件是用户要求的聚焦下载清单，并非替代 50 篇 [reading_matrix.md](reading_matrix.md)。新建的理由是保留原 01–20 下载编号与逐项出版/文件版本状态，避免重排已有 50 篇学习记录。没有新实验、已验证性能或“首个”结论。
