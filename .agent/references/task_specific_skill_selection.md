# 按任务选择专项 Skill

本文件说明何时需要现有项目 Skill 或外部支持，供 `00-router` 和所选主 Skill 按需读取。
任务到主 Skill 的唯一映射仍在 [ROUTING_MATRIX.md](../skills/ROUTING_MATRIX.md)。
执行遵守项目 AGENTS；科学语义遵守 [Paper Skill Core](../PAPER_SKILL_CORE.md) 与论文本地约定。
这里不建立新的 Router、注册格式、安装器或任务状态。

## 选择依据

1. 先确定本轮产物和范围：是在检索、获取全文、设计、执行、分析、绘图、评审，还是回复真实意见。
   不因出现 review、Nature、Agent 等词就调用一整套流程。
2. 读取当前项目已经声明且与本轮有关的 Skill、配置和实现入口。
   用户显式选择优先；在该选择范围内，优先复用合适且可用的本地工具。
   不为查找“最强工具”遍历、排名或安装全部外部 Skill。
3. 用下表检查所选能力是否适配。名称相似、热度或仓库规模不能替代检查。

| 检查 | 要回答的问题 |
|---|---|
| 任务 | 它解决的具体问题是否就是本轮缺口？ |
| 输入证据 | 需要的正文、原文、数据、评论或图源是否实际可读？ |
| 执行能力 | 对应运行时、脚本、软件和已授权渠道是否可用？ |
| 输出 | 能否直接产生所需正文、图表、结果或判断，并写回现有位置？ |
| 验证 | 能否打开实际产物或检查关键计算，而非只返回自述？ |

4. 保留一个主 Skill；只有存在具体缺口时增加一个必要支持，并限定本次采用部分。
   新问题需要不同主任务时顺序交接，不同时加载多个控制流程。
5. 使用外部支持前完整读取其实际 `SKILL.md`，再读本次采用部分需要的 manifest、core、引用或脚本。
   可读的 Skill 与参考即可用于推理支持；实际执行脚本或操作软件时再核实所需运行能力。
   本地副本与本文指定版本不同时说明实际版本，不把混合读取称为固定版本执行。
6. 外部默认输出、绘图后端或审批要求不能替换当前用户指令和有效项目约定。
   缺少输入或能力时完成不依赖它的已授权部分，准确说明受阻动作；不模拟调用成功。
7. 最终回复以产物、结论和实质缺口为主。仅在影响结果理解时简要说明所用支持及边界，
   不向普通用户输出冗长路由决策或完整 Skill 清单。

## 什么时候使用什么能力

下表细化已有主任务的支持时机，不增加主 Skill。已有项目从当前未决问题进入。

| 当前任务 | 内部主 Skill | 外部或本地支持的具体入口与采用时机 | 必要输入与实际动作/产物 |
|---|---|---|---|
| 初步 idea | `scientific-brainstorming` | [K-Dense scientific-brainstorming](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/scientific-brainstorming/SKILL.md)：需要拓展候选机制、识别锚定与遗漏时 | 真实现象、约束与最近邻；生成不同解释、检索后重开候选，写明假设和未知，不用评分选赢家 |
| 正式假设与 Pro 讨论 | `hypothesis-generation` | [K-Dense hypothesis-generation](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/hypothesis-generation/SKILL.md)：需要把观察变成区别性预测时 | 观察来源、主张类型、竞争解释与已看过的数据；形成条件—预测—反证—不可判定结果，不记作已获作者批准 |
| 文献检索与定位 | `03-literature-deep-research` | 在用户指定 Pro 中采用 [K-Dense literature-review](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/literature-review/SKILL.md) 组织检索与筛选；单纯检索能力缺口可改选 [ARIS openalex](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/3e49e63aae6a653067f9e2101d50457f1f7d6a2f/skills/skills-codex/openalex/SKILL.md) | 问题、既有索引、纳排标准；执行真实检索、去重与题录核验，保存检索边界；50 篇是本次经验研究目标，不声称穷尽 |
| 定位所需全文获取 | `03-literature-deep-research` | [LQQL_OS project-paper-curator](https://github.com/LQQL1997/LQQL_OS/blob/6ad5308eee13fcc29be4e02a6bba3fb11a89a012/.agents/skills/project-paper-curator/SKILL.md)：需要补齐和核验 PDF 时 | 明确清单、合法来源和现有归档；实际获取并打开全文，核对版本身份、归档路径与缺失条目 |
| 文献比较矩阵与创新定位 | `03-literature-deep-research` | [K-Dense literature-review](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/literature-review/SKILL.md)：需要全文提取和跨研究综合时 | 真实 PDF 与页码；提取机制、协议和边界，形成公平维度及依据，未知不标为明确缺失 |
| 方法与推导 | `method-design` | 本地领域/理论 Skill 优先；符号推导采用 [K-Dense sympy](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/sympy/SKILL.md)，若缺口是推断条件则改选 [scientific-critical-thinking](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/scientific-critical-thinking/SKILL.md) | 真实定义、精确数域、假设、公式与实现；执行必要符号运算和代入核验，检查反例；数值根、化简通过或批判性分析不替代完整证明 |
| 最小物理 PoC 设计 | `experimental-design` | 本地物理仿真能力优先；需排除混淆或替代解释时采用 [K-Dense scientific-critical-thinking](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/scientific-critical-thinking/SKILL.md) | 机制、已知真值、控制变量、预算；设计强基线、负对照和失效条件，形成可执行协议与配置 |
| 真实数据协议与图表蓝图 | `experimental-design` | [K-Dense statistical-analysis](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/statistical-analysis/SKILL.md)：需要估计目标、配对/聚类或精度设计时 | 至少两个独立真实数据集（本次经验研究范围）、采样单位与划分；明确估计量、区间、多重比较和停止规则，图表数值保持 TBD |
| 实验执行 | `06-experiment-ops` | 已有通用入口优先；实际需要且已启用时采用 [ARIS run-experiment](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/3e49e63aae6a653067f9e2101d50457f1f7d6a2f/skills/skills-codex/run-experiment/SKILL.md) | 已授权协议、真实输入、配置矩阵与预算；运行既有 Shell/runner，核验干预生效、结果文件和失败状态 |
| 实现或指标错误 | `code-change` | 项目局部实现 Skill 能完成相关修复时 | 目标源码、配置和最近测试；交付真实最小补丁，不用绘图或审稿掩盖错误 |
| 统计分析 | `statistical-analysis` | [K-Dense statistical-analysis](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/statistical-analysis/SKILL.md) 处理实际统计问题；若缺口是批量结果提取与比较，则改选 [ARIS analyze-results](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/3e49e63aae6a653067f9e2101d50457f1f7d6a2f/skills/skills-codex/analyze-results/SKILL.md) | 实际结果、单位、配对/聚类和预定分析；执行有效数值分析、核对汇总与区间；不依诊断 p 值自动换检验 |
| 创新主张判断 | `05-claim-evidence` | [K-Dense scientific-critical-thinking](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/scientific-critical-thinking/SKILL.md)：需要检验证据是否跨越推断边界时 | 完整证明、有效结果与替代解释；逐项判断 Supported / Refuted / Inconclusive，写清条件与下一项动作 |
| Motivation、framework 或图表模板 | `15-figure-table-design` | [nature-figure](https://github.com/Yuan1z0825/nature-skills/blob/f8797feed6405035e7d57f59cb8ba393c6a98802/skills/nature-figure/SKILL.md)：需要科学构图、面板分工或图例规范时 | 正确方法/已确认协议；实际生成概念草图或 TBD 规格，不能混入定量证据 |
| 真实结果图、流程图与拼图 | `15-figure-table-design` | [nature-figure](https://github.com/Yuan1z0825/nature-skills/blob/f8797feed6405035e7d57f59cb8ba393c6a98802/skills/nature-figure/SKILL.md)：需要多面板结构、矢量布局或出版级检查时 | 真实数据/图源、单位、后端与格式；用实际工具生成并打开资产，核对数据、拓扑和图注 |
| 全文与论证意群 | `08-markdown-draft` | [K-Dense scientific-writing](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/scientific-writing/SKILL.md)：需要从已核验事实组织全文时 | 正文、证据、方法与真实结果；写出段落目的、证据顺序和有边界结论，直接更新唯一正文 |
| 稳定稿件表达 | `10-language-polish` | 去防御性表达用已有 `anti-defensive-writing`；需段落论证指导时改选 [K-Dense scientific-writing](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/scientific-writing/SKILL.md) 的写作部分 | 原文与术语/边界；直接改进表达，不用 prose linter 得分验收，不自动增加 Nature 审稿 |
| 终稿或指定段落的科学复审 | `peer-review` | [nature-reviewer](https://github.com/Yuan1z0825/nature-skills/blob/f8797feed6405035e7d57f59cb8ba393c6a98802/skills/nature-reviewer/SKILL.md)：用户指定，或需要相应科学审查维度时 | 可读稿件与关键原文/结果；完成一次集中、有依据的发现与最小修订建议 |
| 真实 reviewer response | `13-reviewer-response` | [nature-response](https://github.com/Yuan1z0825/nature-skills/blob/f8797feed6405035e7d57f59cb8ba393c6a98802/skills/nature-response/SKILL.md)：真实意见需要逐点沟通或稿件—回复一致性处理时 | 原始意见、正文、实际改动与编辑要求；完成真实回复及已授权的必要稿件修订 |
| 引文核验 | `11-reference-audit` | [K-Dense citation-management](https://github.com/K-Dense-AI/scientific-agent-skills/blob/a1043ef544e3a100c28e70a736b6cf32554d0677/skills/citation-management/SKILL.md)：需要权威题录提取、去重或 BibTeX 修复时 | 精确论断、当前引用和原始来源；执行所需题录检索/修复，语义核验回到原文，保留现有键与维护位置 |
| TeX 编译或投稿包 | `09-tex-freeze-formalize` 或 `12-submission-pack`，按任务择一 | 已有编译能力优先；需要且已启用时采用 [ARIS paper-compile](https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep/blob/3e49e63aae6a653067f9e2101d50457f1f7d6a2f/skills/skills-codex/paper-compile/SKILL.md) | 官方要求、有效 TeX/图源、真实作者信息；编译实际文件并打开 PDF，投稿仍需有效授权与回执 |

纯全文归档仍可在 LQQL_OS 的既有 `project-paper-curator` 环境独立执行；上表不把它注册为 PaperTrace 新入口。
ARIS 继续采用 [现有 adapter 与 profile](../../integrations/aris/README.md)；表中 Codex 链接为准确来源入口，实际按宿主解析 Claude/Codex Skill 路径，不强切宿主或启用上游高层流程。
只采用当前 adapter 允许的本次检索、运行、计算或编译动作。代码、协议、独立单位与官方期刊要求优先，不照搬通用 Shell、环境 hash/ledger、通知、自动 Git/远端部署或第二状态。
结果不机械套用“相对提升 + seed 标准差”；编译成功、固定页数或 PDF 大小阈值不能替代科学验证与当前期刊文件要求。
选择 GPT Pro 是用户指定的执行面；没有真实调用通道时提供材料交接，不能声称自动调用。

## K-Dense 支持的适配边界

表中 K-Dense 入口已于 2026-10-17 完整读取，固定版本为 `a1043ef544e3a100c28e70a736b6cf32554d0677`。
旧仓库已重定向为 `K-Dense-AI/scientific-agent-skills`。外部 Skill 可实际参与本步推理、编辑、检索或数值执行；先读本次必要参考与脚本，再核实输入和运行能力。
采用 brainstorm 的候选与反例，不导入评分选优或 session 登记 CLI；采用 hypothesis 的预测和竞争解释，不新增平行证据账本或声明扫描。
Literature 的 `search_databases.py` 处理已有检索记录，不能冒充完成数据库检索；Parallel、OpenRouter 示意图和 PDF 导出只在本次确需且可用时采用，不是调研的必需前提。
Critical-thinking 的 GRADE/RoB 等框架只用于合适研究类型，不把临床清单套给物理方法；统计按真实独立单位与模型计算，沿用可用项目环境，不机械安装上游整套版本或把 APA 模板当期刊规定。
SymPy 运算须显式定义数域和假设，核对解的条件与原式代入；版本服从项目实际兼容检查，不为采用外部 Skill 全局安装。
Writing 采用证据组织与准确表达，继续维护现有正文；不强制新建上游 E/C/N 等 registry、逐行 hash、重复人工核验批准、`submission_ready` 或 prose linter。
Citation 采用真实题录查询与所需格式修复；不为本领域强制 PubMed、不全局重建 citation key、不把未分配卷期/DOI 当虚假，题录存在不能证明论断获支持。
上游的工具论文自引要求不能自动改写当前论文；实际有贡献时按项目与目标期刊要求核验来源并准确署明，不能生成未核验引用或把工具论文充作科学发现依据。

## Nature 支持的具体范围

完整读取入口后，先读下列依赖，再选择本轮需要的部分。路径相对于各 Skill 目录，版本同文末固定来源。

| 支持 | 先读的依赖与用途 |
|---|---|
| `nature-reviewer` | `references/source-basis.md`：区分 Nature 本地归档标准与该仓库实现规则；目标期刊当前政策仍查官方。 |
| `nature-figure` | `manifest.yaml` 及其 `always_load`：`static/core/contract.md`、`static/core/stance.md`；再读当前图件需要的资源。 |
| `nature-response` | `manifest.yaml` 及其 `always_load`：`static/core/stance.md`、`static/core/workflow.md`；再读当前回复需要的资源。 |

上游正文中的 `core/...` 是简写，以上两项实际 core 路径位于 `static/core/`。
读取这些规范即可支持只读分析；运行上游脚本或软件操作才需要实际环境和真实输出。

### `nature-reviewer`：帮助一次集中科学复审

读取上游完整 Skill 后，按当前主张采用原创性、科学意义、技术成立性和读者理解等相关维度。
与目标期刊无关的跨学科读者要求不能成为 TII、MSSP 等项目的自动否决条件。
每项实质发现指向真实陈述和原文/结果，说明为什么改变结论及最小解决检验。
区分 Major/Minor 与是否阻碍中心结论；不设问题数量、字词或评分配额。

上游默认有三份互盲报告、综合以及单独 forensic consistency audit；这不是 PaperTrace 的常规交付。
本地 `peer-review` 限定采用上述有依据的维度和指向规则，完成一次 focused review。
只读审查保持只读，不新增报告包、审计脚本运行或第二轮全文审阅，不称作完整上游流程已执行。
涉及数值矛盾时在本次检查中核实相关数字与汇总层级；异常外观不能直接判为伪造。

只有用户明确要求多份独立报告，才按要求使用真实隔离上下文和同一原始材料。
各报告完成前不互传结论，之后才综合；同一上下文扮演多角色不算互盲，
独立上下文也不自动等于不同模型家族。不能满足隔离时说明限制。

### `nature-figure`：从具体图件选择所需部分

图表规划/只读检查只需要科学问题、面板作用和缺失数据，不启动绘图后端选择。
真实绘图沿用项目已确认的 Python、R、MATLAB 或其他工具及唯一图源。
需要 Nature 的 Python/R 脚本时先核验其实际可用性和依赖；不擅自切换后端、
新增 OpenRouter/API 调用、写全局偏好文件或加入已有授权之外的批准步骤。
必要时只借鉴其构图、面板和图例指导，由当前可用绘图工具完成产物。

上游图件检查包括若干脚本与输出记录。只读取规范不等于安装或运行这些脚本。
采用实际脚本时检查它所需输入和真实输出，只有运行成功且结果被检查后才报告相应通过。
未使用的上游 audit 不写成已通过，也不为普通图件机械生成所有上游记录。
定量图仍来自真实数据；生成图像仅作明确的概念草稿，科学语义以 Method 为准。
实际软件不可用时，不声称 Visio/Illustrator 原生对象或往返编辑已经验收。

### `nature-response`：处理已有真实意见

输入必须包含真实评论、编辑信或已有回复；模拟审稿结束不自动生成“真实返修信”。
采用与本轮相关的逐点回应、完成/待办区分、精确稿件位置和稿件—回复一致性检查。
先完成已授权科学改动，再写已完成措辞；缺实验、来源或作者信息时明确保留缺口。
回复正文、清稿/标注稿、编号和格式复用项目当前版本及编辑实际要求。

不强制建立上游 master tracker、第二 readiness、平行回复正文或自动标红全部修改。
只在真实审稿可见性或编辑要求需要时按 reviewer 分件，不臆造互盲关系。
不能从意见数量或语气猜测 Major/Minor 决定；缺该事实只影响依赖它的包级策略。
本地小段回复修订仍可完成。需要进展更新时复用既有 STATUS 视图和状态权威。

## 来源与维护

核验日期：2026-10-17。Nature Skills 是第三方项目，其示例不代替目标期刊当前官方要求。
本次引用固定版本 `f8797feed6405035e7d57f59cb8ba393c6a98802`，源码和完整 Skill 继续在上游维护：

- [nature-reviewer/SKILL.md](https://github.com/Yuan1z0825/nature-skills/blob/f8797feed6405035e7d57f59cb8ba393c6a98802/skills/nature-reviewer/SKILL.md)
- [nature-figure/SKILL.md](https://github.com/Yuan1z0825/nature-skills/blob/f8797feed6405035e7d57f59cb8ba393c6a98802/skills/nature-figure/SKILL.md)
- [nature-response/SKILL.md](https://github.com/Yuan1z0825/nature-skills/blob/f8797feed6405035e7d57f59cb8ba393c6a98802/skills/nature-response/SKILL.md)
- [project-paper-curator/SKILL.md](https://github.com/LQQL1997/LQQL_OS/blob/6ad5308eee13fcc29be4e02a6bba3fb11a89a012/.agents/skills/project-paper-curator/SKILL.md)

本文件只记录采用边界，不复制上游 Skill 树，也不新增同步状态或自动更新器。
变更版本时核查实际需要的入口与依赖，不把未读取版本称为兼容。
本地优先与显式选择都不能掩盖能力缺失：报告实际读取、生成、执行和验证的范围。
