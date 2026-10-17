# 文献：调研、全文与科学定位

三个任务分别验收题录、全文和论断。数量目标按 [使用说明](README.md) 的适用范围执行。

## GPT Pro 调研 50 篇相关工作

主 Skill：[03-literature-deep-research](../../.agent/skills/03-literature-deep-research/SKILL.md)。GPT Pro 是本次指定的调研执行面；本地 Agent 可提供材料并接回结果。

```text
围绕当前论文的核心科学问题，结合已有参考文献，检索并核验 50 篇独立相关工作。
读取能访问的项目规则、正文与文献索引；已有合格条目计入本轮目标，
除非作者明确要求新增 50 篇，不重复收集或重新编号。

读取外部 literature-review Skill，用于检索边界、原文提取与跨研究综合：
https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/literature-review/SKILL.md
实际检索使用 GPT Pro 当前可用渠道。若本地已启用 ARIS 且本步需要 OpenAlex，
改用 adapter 解析的 openalex 作为本次一个支持，先读对应上游入口：
https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/3e49e63aae6a653067f9e2101d50457f1f7d6a2f/skills/skills-codex/openalex/SKILL.md
该链接是 Codex 版；当前环境的实际路径由 adapter 决定。按需择一，不假定
结果整理脚本已访问数据库，也不因链接存在就启用未配置的付费检索渠道。

优先近两年的最近邻、强竞争方法和关键理论；保留决定问题与协议的经典来源。
检索同一问题的等价术语、邻近领域机制、简单替代、失败报告和相反结果。
期刊档次、引用数和题名相似度不能代替技术相关性。

每条给出题名、作者、年份、出版状态、DOI/arXiv ID、权威来源链接、
与本研究的关系、已有收录情况及全文状态；记录检索日期和主要检索边界。
同一工作的预印本、作者稿和出版稿按版本关系维护，只计一项独立工作。

标出 3–5 项真正最近邻以及供深入比较的 8–12 项代表工作。
说明它们各自解决的问题、可能推翻本文新颖性的重叠点和仍需全文验证的判断。
只读摘要或检索片段时不能断言某机制、实验或能力缺失。

交付可供 project-paper-curator 接续的题录清单与优先阅读顺序。
不足 50 篇时报告实际已核验数量、未解决检索范围和原因，不用弱相关作品补数。
本地没有可用的 Pro 路由时给出材料与本 prompt 的交接，不能声称自动调用成功。
```

## 用 project-paper-curator 获取全文

主 Skill：[project-paper-curator](https://github.com/LQQL1997/LQQL_OS/blob/6ad5308eee13fcc29be4e02a6bba3fb11a89a012/.agents/skills/project-paper-curator/SKILL.md)，在实际具备该外部 Skill 的环境执行。

```text
使用现有 project-paper-curator，获取并核验本次 50 篇目标工作对应的全文。
外部 canonical Skill 的已核验入口：
https://github.com/LQQL1997/LQQL_OS/blob/6ad5308eee13fcc29be4e02a6bba3fb11a89a012/.agents/skills/project-paper-curator/SKILL.md
先定位并完整读取该 Skill、目标论文清单和项目已有归档记录；继承同一会话
已经授予的公开或订阅来源权限。未安装或不可读取时明确缺口，不能假装运行。

1. 对照既有题录与 PDF，核验身份、版本关系和项目编号，复用合格文件。
2. 按实际可用且已授权的渠道逐篇获取出版稿、作者公开稿或预印本，
   记录所取得版本。缺少全文仍保留题录、权威补全入口与原因。
3. 使用 Skill 的既有脚本与记录格式完成结构检查、身份检查和归档预检。
   它的本地 verify/archive 脚本不等于搜索、登录或下载器。
4. 核对首页题名、作者、DOI/版本关系；解析成功、HTTP 200、扩展名和
   参考文献中出现目标题名均不能证明下载正确。复杂首页按需渲染核查。
5. 保留旧编号、有效副本和负面核验结果，不用无关文献或 HTML 补齐 PDF 数量。

分别报告：已核验独立题录数、身份正确的独立全文数、实际深读数、
项目副本数及缺失/待复核条目；没有统计的量写未统计。
下载和深读不是同一完成状态。关键最近邻全文缺失时，保留依赖其全文的判断，
并继续其他独立工作。沿用 curator 自带记录，不建立第二套文献台账或下载器。
```

## Scientific Comparison Matrix

主 Skill：[03-literature-deep-research](../../.agent/skills/03-literature-deep-research/SKILL.md)。以原始全文和当前论文为输入。

```text
阅读当前论文、已有研究成果和本地原始 PDF，形成可投稿的英文文献比较矩阵，
支撑 Research Gap、Motivation 与 Contributions。先遵守项目规则与唯一正文，
只有实际读取的来源才能用于全文级判断。

读取外部 literature-review Skill，使用其原文提取、主题综合和缺口分析能力：
https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/literature-review/SKILL.md
本轮输出下面的科学比较矩阵，保留本项目的比较层级、逐格证据与引用格式。

按 Scientific Question → Closest Related Work → Comparison Dimensions
→ Research Gap → Verifiable Novelty 组织论证。

从已核验文献中选择 8–12 项代表工作，其中明确 3–5 项最近邻。
研究论文比较具体方法，综述比较已有综述的范围、组织原则和分析深度。
从科学问题、方法假设、机制、适用条件和验证范围导出 5–7 个非重复维度，
不以组件数量或让 Ours 多打勾为目标。

输出英文三线表：Work/Year 为行，Ours 最后加粗并以横线分隔。
符号定义：✓ 已充分处理；○ 部分处理并写明条件；✗ 有原文依据证明缺席；
— 当前信息不能建立结论。设计能力、实现能力和已验证能力应可区分。
未报告不等于不支持，Ours 未验证的能力同样不能打 ✓。
有争议的判断用简短具体描述，不强行二值化。

每项判断能够回查 PDF 路径、页码、章节/公式/图表及支撑原文。
保留已有工作的真实优势，不从表格符号直接推出创新。
区分已解决问题、未解决问题、本文的实质变化、可检验增量及其适用边界。

交付：英文比较表、适配现有模板的 booktabs LaTeX、维度与逐格依据、
可用于 Introduction/Related Work 的英文 gap 段落，以及贡献对齐修订。
有正文修改授权时更新唯一有效正文和既有 BibTeX；只读任务直接返回稿件。
缺少关键原文时标明尚不能作出的结论，不虚构引用或另开冗余论文稿。
```
