# 方法、理论与最小验证

各任务独立交接；已确认方法从具体未决事项进入。论文规范与范围见 [使用说明](README.md)。

## 方法与理论推导

主 Skill：[method-design](../../.agent/skills/method-design/SKILL.md)。

```text
围绕已确认科学问题、最近邻原文和当前实现，优化最小必要方法。
先读取项目规则、真实方法定义、作者决定及已有证明；继承有效内容。

需要符号化简、求导、极限、方程或矩阵运算时，读取外部 sympy Skill：
https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/sympy/SKILL.md
核对现有运行环境后执行必要符号计算，保留定义域、非零条件和分支假设，
把解代回原式。数值根、有限采样或化简通过只能支持所检查的步骤，不能代替证明。

按定义 → 假设 → formulation → 机制 → 算法 → 必要实现细节展开。
明确输入输出、变量的物理/统计意义、学习设置和信息权限。
每个组件必须改变一个必要科学对象，并有简单替代及删除/替换后的可检验预测。

推导只承担有实际作用的责任：说明研究量、条件、命题、证明及对方法的约束。
区分定义/恒等式、完整理论保证、数值检验和经验规律。
审查符号、维度、单位、极端情形、假设是否使用及公式与算法的一致性。
已承诺的数学保证需要相应证明；若反例表明它不成立，明确修订或撤回该保证，
说明对贡献的影响，不能悄然改论文类型或换措辞后声称证明完成。
不添加无法解释、预测或约束方法的装饰性定理。

输出可实现的方法、必要完整推导、成立条件、最强反例方向及首项最小验证。
公式使用可复制的 $$ LaTeX $$。更新现有方法载体，保留未知与负结果。
新增代码或实验只在相应任务授权内执行；推导完成不等于实际效果已经成立。
```

## 针对关键命题寻找反例

主 Skill：[scientific-critical-thinking](../../.agent/skills/scientific-critical-thinking/SKILL.md)。在关键推导或机制存在实质疑点时选用。

```text
只读核查本次指定的命题、证明或机制解释。直接读取定义、假设、推导和
相关原始结果，不以作者或另一个 Agent 的“已证明”总结作为依据。

读取外部 scientific-critical-thinking，采用其证据评价与替代解释检查：
https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/scientific-critical-thinking/SKILL.md
应用到本次命题及其假设，给出可核验反例或缺失证据，不扩成全稿审稿。

寻找会改变结论的缺失条件、循环论证、量词错误、极端情形、反例和更简单解释。
分别判断数学证明、数值自检、经验支持与实现一致性，不用其中一项替代另一项。
输出具体问题位置、反例/推导依据、最小修订和修订后仍可成立的结论。
没有发现问题时说明已检查的范围；同模型的不同会话不称为跨模型独立验收。

本轮保持只读，不启动多轮审稿或按评分判定论文通过。
```

## 最小物理 PoC 设计

主 Skill：[experimental-design](../../.agent/skills/experimental-design/SKILL.md)。

```text
围绕当前方法最关键的可证伪假设，设计一个最小、受控且可解释的 PoC。
读取真实方法、最近邻原文、可用实现和资源限制，先明确什么结果会改变判断。

读取外部 scientific-critical-thinking，使用其替代解释与混杂检查能力：
https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/scientific-critical-thinking/SKILL.md
只对本次核心机制设计区分实验；具体物理模型与参数仍须有领域原文依据。

振动研究优先参考有原始文献依据的故障激励、周期冲击、调制、传递路径和
噪声模型；只保留区分该假设所需机制。参数有依据，已知真值明确。
Agent、RUL 或维护决策研究采用相应受控任务、退化过程或决策环境，
不为满足模板添加无关振动生成器。

选择最少且有作用的比较：最强最近邻、足以挑战复杂度的简单替代、
核心机制消融、主要混杂的匹配/负对照，以及相关的失配或失效边界。
统一输入、信息权限、评价协议和合理模型选择预算；不调生成参数制造优势。

每项明确：Claim → Controlled variable → Competing prediction → Metric
→ Support pattern → Falsification criterion。
给出独立单位、实际配置、运行顺序、单次与总预算、有限技术重试及停止条件。
输出最小表格/图设计：信号或环境、定量比较、机制响应、敏感性/失效条件。
计划值使用 TBD，真实结果另列；仿真支持仅限受控环境。
```

## 执行最小 PoC

主 Skill：[06-experiment-ops](../../.agent/skills/06-experiment-ops/SKILL.md)。缺少必要实现时先单独使用 `code-change` 完成 [统一实现任务](05_execution_and_claims.md)。

```text
执行当前已明确且已授权的最小 PoC，复用真实通用入口与配置。
先核验所需实现、实际输入、真值、对照、指标、信息权限和预算。
缺失语义或输入直接报告；不能悄悄替换数据、物理参数、设备或任务。

确需且已经启用 ARIS 执行支持时，经现有 adapter 解析 run-experiment：
https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/3e49e63aae6a653067f9e2101d50457f1f7d6a2f/skills/skills-codex/run-experiment/SKILL.md
按实际环境读取对应 Skill 并执行已授权配置；此 Codex 链接用于源定位，
不照搬示例服务器、部署或通知动作，不新增运行器或平行环境台账。

运行后检查真实产物和关键干预是否被接收、使用并产生预期作用。
区分实现错误导致的无效测量、有效的机制非使用、有效负结果与机制初证。
按协议计算必要效应和不确定性，保存实际结果和复现所需最少信息。

对当前假设给出 Supported / Refuted / Inconclusive 及限定范围。
不显著不等于被否定；未执行、信息不足或对照失效不能写为支持。
若不能区分解释，只提出最有信息量的一项补充；假设被有效反证时停止同类追加。
说明是否有依据进入真实数据验证，仿真不替代真实数据泛化结论。
```

## 方法 framework 图

主 Skill：[15-figure-table-design](../../.agent/skills/15-figure-table-design/SKILL.md)。

```text
依据正确且现用的 Method、Notation 与实现，制作论文 framework 图。
先核实输入、操作、表示、约束、监督来源、训练/推理差异和输出的真实关系。
每个框与箭头都应能对应方法定义，不能把仓库目录或工具日志画成科学机制。

读取 nature-figure 及与本图匹配的 AI 草图或 vector schematic 资源：
https://github.com/Yuan1z0825/nature-skills/blob/f8797feed6405035e7d57f59cb8ba393c6a98802/skills/nature-figure/SKILL.md
用其构图和可编辑对象指导完成本图，保持下述已有工具、科学边界与真实验收。

先通过可用的 GPT Image 形成构图草案，再依据 Method 修正科学语义。
图像工具不可用时交付精确生成 prompt 和构图规格，标记尚未生成。
需要投稿资产时，将确定内容重构为独立文字、矢量对象与有方向的连接；
不是把 PNG 嵌入 SVG。按现有 Nature/期刊图示规范调整层次与颜色。

交付实际图源、英文 caption 和直接视觉检查结果。
示意图与真实结果分开；不生成精确数据曲线、伪公式或未验证收益。
Visio/Illustrator 的原生可编辑性只按实际打开、编辑和重开结果报告。
```
