# 分阶段科研 Prompt

这组 Prompt 用于把一个研究问题推进到可复现结果和投稿材料。复制对应任务小节的“主 Skill”说明与代码块，附上实际材料后交给 GPT Pro；也可以让本地 Codex 读取对应文件并执行其中指定的一步。已有项目从尚未解决的事项进入，继承有效文献、结果和作者决定。

它是可选任务文本，不是新 Router、安装脚本或强制工作流。PaperTrace 的执行入口仍是 `00-router`；内部任务映射以 [ROUTING_MATRIX.md](../../.agent/skills/ROUTING_MATRIX.md) 为准。每个代码块对应一个主 Skill，后续代码块是下一项独立任务，按需顺序交接。

## 选用入口

| 文件 | 可独立复制的任务 | 对应主 Skill |
|---|---|---|
| [01_idea.md](01_idea.md) | 初步 idea；正式假设；motivation 草图与选题确认 | `scientific-brainstorming`；`hypothesis-generation`；`15-figure-table-design` |
| [02_literature.md](02_literature.md) | 50 篇调研；全文获取；科学比较矩阵 | `03-literature-deep-research`；外部 `project-paper-curator`；`03-literature-deep-research` |
| [03_method_and_poc.md](03_method_and_poc.md) | 方法与推导；反例核查；最小 PoC 设计与执行；framework 图 | `method-design`；`scientific-critical-thinking`；`experimental-design`；`06-experiment-ops`；`15-figure-table-design` |
| [04_experiment_blueprint.md](04_experiment_blueprint.md) | 真实数据协议；最终图表模板与实验确认 | `experimental-design`；`15-figure-table-design` |
| [05_execution_and_claims.md](05_execution_and_claims.md) | 通用实现与 Shell；完整实验；统计；创新主张判断 | `code-change`；`06-experiment-ops`；`statistical-analysis`；`05-claim-evidence` |
| [06_manuscript_and_submission.md](06_manuscript_and_submission.md) | 全文；图表定稿；De-AI；引文；复审；真实审稿回复；TeX；投稿包 | `08-markdown-draft`；`15-figure-table-design`；`10-language-polish`；`11-reference-audit`；`peer-review`；`13-reviewer-response`；`09-tex-freeze-formalize`；`12-submission-pack` |

方向未清楚时先用初步 idea；完成文献比较后再正式化假设。不要先确认一个方法名称，再用检索和实验替它寻找优势。

`project-paper-curator` 的已有来源是 [LQQL_OS canonical Skill](https://github.com/LQQL1997/LQQL_OS/blob/6ad5308eee13fcc29be4e02a6bba3fb11a89a012/.agents/skills/project-paper-curator/SKILL.md)（2026-10-17 核验）。全文归档任务在具有该 Skill 的环境执行；它不是 PaperTrace 新增的内部主 Skill。先核验实际可读路径与能力，不能把链接存在当作本地已安装。

## 何时使用专项支持

按 [任务、输入、能力、产物与验证](../../.agent/references/task_specific_skill_selection.md) 判断本轮是否需要专项 Skill。尊重用户显式选择，复用适合且可用的项目本地能力；每次保留一个主 Skill，至多增加一个必要支持。不会根据名称或热度批量导入外部 Skill。

各阶段的可复制代码块已写入外部 `SKILL.md` 的固定版本链接、采用范围及实际产物要求。Agent 应打开对应原文并把相关能力用于本步工作；正文、结果与进展继续写回项目已有位置。

| 外部仓库 | 直接 Skill 入口示例 | 使用阶段 |
|---|---|---|
| [K-Dense Scientific Agent Skills](https://github.com/K-Dense-AI/scientific-agent-skills) | [brainstorming](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/scientific-brainstorming/SKILL.md)、[SymPy](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/sympy/SKILL.md)、[statistics](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/statistical-analysis/SKILL.md)、[writing](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/scientific-writing/SKILL.md) | idea、假设、文献、理论、统计、写作与引文；完整 8 项入口见专项选择表 |
| [Nature Skills](https://github.com/Yuan1z0825/nature-skills) | [nature-figure](https://github.com/Yuan1z0825/nature-skills/blob/f8797feed6405035e7d57f59cb8ba393c6a98802/skills/nature-figure/SKILL.md)、[nature-reviewer](https://github.com/Yuan1z0825/nature-skills/blob/f8797feed6405035e7d57f59cb8ba393c6a98802/skills/nature-reviewer/SKILL.md)、[nature-response](https://github.com/Yuan1z0825/nature-skills/blob/f8797feed6405035e7d57f59cb8ba393c6a98802/skills/nature-response/SKILL.md) | motivation/framework、图表模板与定稿、科学复审、真实审稿回复 |
| [ARIS](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep) | [run-experiment](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/3e49e63aae6a653067f9e2101d50457f1f7d6a2f/skills/skills-codex/run-experiment/SKILL.md)、[analyze-results](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/3e49e63aae6a653067f9e2101d50457f1f7d6a2f/skills/skills-codex/analyze-results/SKILL.md) | 经已启用 adapter 完成检索、运行、分析或编译；实际宿主路径由 adapter 解析 |
| [LQQL_OS](https://github.com/LQQL1997/LQQL_OS) | [project-paper-curator](https://github.com/LQQL1997/LQQL_OS/blob/6ad5308eee13fcc29be4e02a6bba3fb11a89a012/.agents/skills/project-paper-curator/SKILL.md) | 获取、核验与增量归档真实全文 |

上游完整 Skill 仍在原仓维护，使用前读取完整入口和本步所需依赖。可读的规范能够参与分析和编辑；脚本、软件与 API 操作需要真实可用的运行能力，未运行的检查不能报告通过。按本项目主任务限定采用范围，沿用已有授权；缺少外部能力时完成可行部分并说明具体缺口。模拟审稿与真实回复是独立任务，不自动串成返修流程。

## 使用前给出什么

在所选 Prompt 前提供实际项目/材料位置和本轮目标即可。可查明的信息由 Agent 自行读取，不要求重新填写完整 intake。

- 本地 Codex：先读适用的 README、AGENTS、所选 Skill 与相关原始材料；论文工作同时读 [Paper Skill Core](../../.agent/PAPER_SKILL_CORE.md)。
- GPT Pro：附上相关正文、原文、实际结果或公开可读链接。没有仓库、下载、执行或绘图权限时，交付本阶段的可用论证和精确交接内容，并说明依赖缺口；不声称访问了未提供文件或自动调用了另一个产品。
- 显式只读、计划或审阅范围继续有效。文件可写与 Git 发布是不同授权；沿用当前任务已经授予的范围。

## 科研与运行边界

以下约束应用于本组任务；具体研究对象、理论责任与数据协议仍由论文项目定义。

1. 一个核心问题、1–3 项不重复且可证伪的主张。区分观察、假设、推断、探索发现与独立确认。
2. 原文、证明和真实运行结果分别承担相应责任。评分、脚本存在、退出成功或 PR 合并均不证明创新成立。
3. 本组经验研究路径的数量目标是 50 篇独立相关工作、至少两个独立真实数据集。它们来自本次研究需求，不强加给所有论文：综述转为原文综合与分析框架，纯理论转为命题与证明，Agent/RUL 采用相应受控环境。
4. 预期图表使用 `TBD`；定量结果来自实际测量。有效负结果保留，缺失输入不以默认物理参数、替换数据或虚构结果补齐。
5. Python 用于必要算法、数值计算、统计、绘图和验证；实验差异由配置表达，Shell 组织真实入口。不用 Python 生成文风分数或平行状态报告。
6. 自动迭代开始前明确允许改动范围、开发信号、单次与总预算、有限技术重试和停止条件。最终测试不能成为调参反馈。
7. 获得足以改变判断的结果、达到预算、必要预测被有效反证、持续没有新增区分信息，或缺少必要输入时，停止相应循环。记录下一项具体判断，不无期限运行。
8. 可选 ARIS 只经现有 adapter 使用固定版本的允许能力；不启用上游高层研究/审稿循环，不安装第二套控制入口。

## 三个作者决定

| 决定 | 先完成可审阅材料 | 已有决定怎样处理 |
|---|---|---|
| 选题 | 最近邻差异、最小机制、竞争解释、区别性预测、反证条件和 motivation 草图 | 同一方向已有真实批准时继承；Agent 推荐不等于作者选择 |
| 完整实验 | 数据与协议、强基线、完整图表模板、配置矩阵和预算 | 在相同范围与预算内持续执行；实质变化只确认新增决定 |
| 投稿 | 最终稿、图表与补充材料、投稿信和真实作者声明 | 正式提交沿用有效授权；只有得到回执才报告已投稿 |

Markdown 转为 TeX 仍遵守项目已有的作者确认要求，放在投稿材料准备中处理；它不成为日常小修订的新批准流程。没有相应批准时，先把可审阅材料和转换范围准备具体。

## 文件与进展

正文、配置、结果、图源和引用更新在项目既有位置。不要为每个 Prompt 新建一套 PLAN、REPORT、STATUS 或运行台账。

本项目的 [STATUS.md](../STATUS.md) 是当前科学判断与唯一下一步的入口；[paper.yaml](../paper.yaml) 保留项目元数据、主张索引和作者决定，并引用该入口。先修正实际产物和受影响的既有权威，不新增第二份状态。唯一正文继续为 `paper/draft/main.md`，继承当前数据访问排除、负结果与实验停止决定。

每轮简洁说明：实际改动、新增认识、直接验证、仍不能下的结论及下一项最有信息量的行动。没有访问或执行的部分保持未完成。
