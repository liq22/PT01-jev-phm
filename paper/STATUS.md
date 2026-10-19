# Paper Status

核验日期：2026-10-19。续接基线：`dev@733341c1e8ca7d64ca1d48129370297508c6950e`（PR #21 已合并）。工作分支：`work/jev-p1-real-execution-20261019`。唯一正文 `paper/draft/main.md`；原实现 `jev_phm.py`、`jev_simulation.py` 和 PHMFactory pin 不改。P1 使用既有 `run_jev_study.py semantic`，不替换先导实验。

**作者已确认 P1：PU 单环部件子群＋MFPT 已知标签试验台记录；24 GPU 小时＋12 CPU 墙钟小时；最多8 CPU线程、一个GPU任务。**这是对上一条完整协议及图表模板的“确认并继续”，不再请求同一授权。执行以[既有 Goal 的 Approved P1 continuation 段](experiments/jev_goal.md#approved-p1-continuation-pu-single-ring-source-to-mfpt-external-target)为准；授权不等于数据或实现已准入。

**CWRU：EXCLUDED BY USER；物理 GPU 2 禁用。**不读取该数据或使用明确以其预训练的工业权重；不恢复历史资格任务。旧PU/仿真记录和未合并PR #16保持原状。

前次交接在另一容器完成 PU 角色计数与 MFPT 发布材料核验，其未挂载数据／GPU及网络失败记录保留为历史环境观察。本机现可定位 PU/MFPT H5、指定权重目录、固定 PHMFactory 和项目依赖。17日预检在缺资格映射及另行提供的 CSV 处停止；PR #20 已取消人为 CSV 前置，接入原生工作簿的获准行投影，并显式核对跨库标签顺序。该PR未打开H5；本次接续只读检查了20个获准MFPT条目的header和属性，没有读取波形、目标分数或权重。元数据及存储布局可定位仍不证明物理试件或本地测量链准入；`data.qualification_file` 未绑定，0 次模型／CUDA调用、0/46 拟合。

**19日新增确认：**作者明确H5仅改变原始存储格式，其他采集信息以metadata为准，并授权“一次实验一个轴承”的默认约定。按此提供者陈述建立[运行绑定](assets/tables/jev_p1_record_bindings.csv)：PU保持已知29轴承及既有角色，MFPT按20条原生记录作为20个假定实验单元。保留原始幅值、identity multiplier=1，不为PU空缺的幅值单位编造g换算；所有对照共享该事先修订。此陈述解除前次等待额外来源文件的执行阻塞，不构成独立试件或物理幅值标定证据。实际`semantic check`已退出0、绑定2,339记录；源侧准备／真实模型smoke接续执行，效果尚未建立。

历史 CWRU 检查只保留为 **EXCLUDED BY USER** 的会话历史，不作为 P1 结果、选择依据或待恢复动作；本轮没有重新读取或分析 CWRU。

## A. 科学主线与当前判断

- **论文类型／读者：**真实来源间故障诊断的方法与经验研究；目标期刊、作者／投稿声明仍待确认。
- **一个核心问题：**在相同原生振动数值、源侧监督和合理开发预算下，故障描述构成的共享输出空间能否比同骨干固定类别读出降低未参与拟合的MFPT实验单元等权NLL？MFPT“一实验一轴承”是作者授权的分组假设；P1不检验原生粗标签／条件保持。
- **最近邻→缺口：**GLiClass已有共享语义读出，UniFault/YOTOnet/Jev–HVAC具有不同目标信息和任务边界；不能仅由框架名称、直接输出或现有组件组合宣称创新。[现有逐项原文依据](assets/tables/related_work_comparison_notes.md)。
- **核心干预／固定变量：**P1的JEV与B_IDX共享波形、源单元和骨干；JEV-PERM保持候选向量集合与容量但固定错误对应。B_RF/B_SIG检验实用竞争力，B_TOK及同权重最短生成检验计算解释。条件KL／投影不加入首批。候选描述、实际语义接口与所有预处理须在目标揭盲前固定。
- **默认主张：**C4框架迁移；C4中正确语义对应的增量；C5同诊断要求下的计算代价。C4原生粒度子主张仍UNTESTED，不伪造粗标签来源。
- **与旧研究边界：**旧32轴承PU研究已经参与开发；新29轴承内部测试是探索性。`run-sim-002`是46特征仿真，不是语义基础模型；`sim-20261017-001`是不同协议，不替换前者。
- **最强可成立结论：**PU的29轴承／2,319记录及16/5/3/5角色固定；MFPT20条已知标签记录可定位。H5原值保留及metadata采集信息按作者提供者陈述采用；MFPT单元独立性按新约定采用。PU源MAT的vibration_1字段及缓存列2已核对，但Unit为空，故只保留native数值，不宣称两来源共同g标定。统计区间必须条件于已声明分组假设。
- **尚不能成立：**P1数据完全准入、真实接口smoke通过、基线已调优、语义迁移／时延优势。实现已接入、数值测试或仓库合并均不证明这些效果。正确／固定错误对应的差值只检验这组描述对应的迁移作用，不能排除候选几何与优化解释，也不能单独证明一般语义理解。
- **最低独立知识／停止条件：**在有效外部比较中识别共享语义相对同骨干与简单强方法的增量或失效范围。无效映射、未知物理单位、参数／目标越权必须阻止主试验；简单方法解释全部增量则收缩创新。不得按MFPT结果换数据或删类。
- **路线：CONTINUE。**19日按新确认的输入与单元约定，G0运行绑定检查通过；依次执行源侧导出、真实预训练模型smoke和既定搜索。目标波形仍须等全部源侧模型选择、重载及校准冻结后开放。技术失败、计算上限或接口不一致仍明确停止，不以假设代替模型验收。

## B. Claim—Proof—Evidence—Manuscript

| Claim与条件 | 本轮增量 | 理论／证明 | 产物 | 有效性／推断 | 正文及缺口 |
|---|---|---|---|---|---|
| C4的PU人口与分组前置 | 由候选29轴承落实为固定角色与真实索引计数 | 复用已核验损伤位置，不新造理论 | [P1 PU roles](assets/tables/jev_p1_pu_unit_roles.csv)；[Goal](experiments/jev_goal.md) | 元数据变换COMPLETED/VALID；H5与物理时钟UNVERIFIED | 旧结果不改；新人口不声称全波形已核验 |
| C4的MFPT外部准入 | 原生20条记录及本地H5 header已定位；发布版gs/sr语义有原始文档依据 | 文件不等于物理单元，发布单位不自动等于本地H5单位 | Goal中的原始文档、header结果及现有G0入口 | 元数据／存储布局PARTIAL；物理unit与本地生成链UNKNOWN | 不执行设备等权最终测试或伪造bootstrap |
| C4框架与语义增量；C5错误—时延 | 作者批准独立P1执行范围，不是效果证据 | 未承诺一般优势 | 本轮拟合0/46；没有预测／计时 | UNTESTED | main §4设计与framework图保持；结果数值TBD |
| C4原生标注／固定投影边界 | 无新增证明，历史结论保留 | [映射、条件冻结与有效秩证明](method/native_annotation_compatibility.md) | 既有PU反例 | 既有范围PROVED/COUNTEREXAMPLE；本轮未重验 | 禁用决定覆盖方法补充中的旧CWRU待访问语句 |
| 已完成模拟条件保持 | 本轮没有新增或混合实验 | [模拟协议](experiments/jev_simulation_poc.md) | `outputs/run-sim-002`；[CSV](assets/tables/jev_simulation_poc.csv) | 历史VALID；选中lambda=0/step=0，增量INCONCLUSIVE | 低粗事件损失不证明隐藏细类已学会；不将此当P1结果 |

原C1–C3的[贡献索引](experiments/evidence_matrix.md)、PU指标、失败记录和[旧结果蓝图](experiments/expected_results_blueprint.md)不覆盖。作者2026-10-17确认的新完整图表设计由会话附件提供；执行参数已收敛到本Goal的P1段，不把旧蓝图的投影候选重新加入P1。PR #16未随本轮合并，其方法推导不被冒充为dev已接受内容。

## C. Baseline 竞争力

| Baseline及必要性 | 已批准搜索／实际预算 | 选择规则／证据 | 资格 | 未决项 |
|---|---|---|---|---|
| B_RF信号强参照 | 300树；depth 8/None x leaf 1/4/12；本轮0fit | PU tune raw NLL | 待核验 | 特征/频带须源侧定义；原始概率与校准分开 |
| B_SIG紧凑信号模型 | 6配置＋2额外初始化；本轮0fit | 源侧NLL；实际注册模型待确认 | 待核验 | 不另造弱模型或默认示例替代 |
| B_IDX／JEV／JEV-PERM／B_TOK | 各6配置＋2初始化；本轮0fit | 同一源信息、适当参数披露、目标密封 | 待核验 | 完整语义接口尚未验证；不能用仅有六统计量的旧入口冒充 |
| 原PU／run-sim-002基线 | 历史搜索与结果保留 | 原协议内判断不变 | 仅限旧协议；本轮未重新认证 | 不借旧87测试或模拟高分认证新基线 |

P1总授权：40次神经拟合＋6次RF；24GPUh/12CPU墙钟h上限；最多8线程/单GPU。当前GPU消耗0h、拟合0/46，CPU元数据处理时间未单独计时，不填0。首次G0不消耗未来拟合名额。曾经拟合的108组仿真及其72组超时重复属于旧预算和原日志，不混入P1。

## D. 本轮真实变化

- **19日执行起点：**已生成2,339条完整绑定，明确`unit_basis`及`value_unit`；`scale_to_g`改为不带物理单位承诺的`scale_multiplier`。G0实际退出0（`src/outputs/jev-phm/g0-20261019-authorized/check.txt`）；MFPT波形未读取。首次全仓177项检查因旧测试仍要求资格文件为null失败1项，已更新为验证绑定文件存在，同时保留缺绑定必须BLOCKED的检查；失败日志保留，修后检查另报。执行目录`src/outputs/jev-phm/run-real-20261019-001`，GPU仅允许已核对物理0的UUID；禁止GPU2。
- **授权：**从作者待确认变为P1及预算已确认；这是执行条件，不是方法效果。
- **已执行元数据动作：**仅取版本化PU role表的32条primary行；使用既有公开部件映射排除新人口中的KB23/KB24/KB27，保留历史记录；标准库Random(20261017)，H/I/O顺序、类内bearing code排序后shuffle。生成29行固定角色。train/tune/cal/test记录数为1279/400/240/400；KA08的79记录原样保留。
- **MFPT来源：**读取MathWorks README、license与完整非截断tree（20个MAT条目，未取波形）。许可文本明确CC BY-NC-SA 4.0；原始数据所有者与重组范围可定位。publisher train/test目录全部作为P1外部目标，不继承示例训练权限。未找到物理试件join；不把20文件计成20轴承。
- **17日实现与预检保留：**已接入六配置、共同拟合、原生注册适配器、全部选定模型的源侧验收、冻结校准、目标访问门槛及保存结果统计／图表链。当时在未绑定资格映射及缺独立CSV处停止；日志为 `src/outputs/jev-phm/g0-20261017-integration/preflight.txt`，退出码 2，墙钟 0.79 秒。该预检未读取原表，不是表中缺少记录的证据。
- **18日原生元数据修正：**`data.metadata` 改为现有 `metadata.xlsx`。按Name路由worksheet原始行，只解码表头和PU/MFPT行，再取固定P1人口；不把混合工作簿整表读入后过滤。原始表和H5不改，未解析排除来源记录。PU原始0/1/2表示H/I/O，MFPT原始0/1/2表示H/O/I；资格映射须分别对应共享索引0/1/2及0/2/1。此差异在目标模型访问前固定，不是根据模型表现重标注。
- **18日真实CPU结果：**现有CLI已从原生表取得2,339条获准记录。PU为29单元／2,319记录（H/I/O记录数480/880/959），名义64,000Hz、3通道；MFPT为20记录（H/I/O为3/7/10），名义97,656Hz的6条及48,828Hz的14条、均标注1通道。MFPT描述列只有`baseline`、`outer race fault`、`inner race fault`，未提供物理试件标识。首轮描述校验误将原生文本去空格而退出2；按实际原表修正后独立重跑，报告元数据事实，再因资格映射未绑定退出2。两次均耗时2.05秒，首轮及修复日志分别保留在 `src/outputs/jev-phm/g0-20261018-native-metadata/check.txt`、`check-corrected.txt`，时间见同名`*-time.txt`。没有创建实验run目录，未打开H5或模型；这是资格结果，不是性能结果。
- **资格边界：**表中的Id、File、Domain_id、标称采样率及通道数量不能代替物理试件身份、重复采集关系或幅值单位。已查本地MFPT随附包络分析PDF未给出逐记录试件join；这不证明其他原始资料不存在。未知项仍阻止G1，不按文件或工况编造独立设备。
- **本次MFPT header结果：**只访问`RM_007_MFPT.h5`的46282–46301条目及属性，未读取数组。20个键均存在、均为float64 Dataset；前6条shape=(585936,1,1)，后14条=(146484,1,1)。根属性和20个条目属性全部为空，未提供物理试件、重复采集或单位／时钟映射。该结果确认局部可访问性和布局，不证明payload数值或独立统计单元。原始命令及观察见既有Goal的[本次资格段](experiments/jev_goal.md#g0-follow-up-original-signal-semantics-and-local-headers--18-october-2026)。
- **原始来源缩小未知范围：**[MathWorks正式文档](https://www.mathworks.com/help/predmaint/ug/Rolling-Element-Bearing-Fault-Diagnosis.html)直接使用`bearing.gs`、`bearing.sr`构造秒时间轴并标注Acceleration(g)；[发行说明](https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data/blob/master/README.md)明确示例目录经过重组划分。查阅的发行材料及数据提供者参与的2025年论文／作者代码未给出逐记录试件对应；未采纳第三方“每文件独立”或具体轴承数说法。这不是任何地方都不存在对应资料的证明。
- **前次 G0 验证保留：**实际检查原人口32/2559、排除3/240、保留29/2319、类计数6/11/12、角色16/5/3/5、无重复物理单元及完备记账。该次只生成 metadata 角色，不是神经模型 smoke；本次实现检查另行报告，不能倒填为前次已执行。
- **17日软件检查保留：**项目固定环境实际运行 163 项单元测试全部通过，包含 64 项新增模型／数据／执行／统计测试；固定 PHMFactory 注册适配器的数值兼容测试在本机实际运行，未用真实波形。技能、wrapper、案例定义和 Core backend 检查通过；Shell 语法、CLI help 与差异检查通过。项目验证仍 FAILED：7 项原有 evidence ID／枚举错误在未修改的 `25ec6781` 提取副本上逐项复现，没有放宽校验器或改写科学证据等级来消除失败。
- **18日最终精简检查：**固定环境完整单元测试176项通过；技能定义和案例定义检查通过；Shell语法及`git diff --check`通过。项目验证仍退出1，同7项历史ID／枚举错误与保留的未修改基线输出逐条排序比较一致，无新增该类错误。输出分别为本轮G0目录下`unittest.txt`、`validate-project.txt`、`validate-skills.txt`、`validate-skill-evals.txt`。不把定义检查或软件测试当作真实数据准入、行为测试或远端CI通过。
- **本次接续检查：**仅修订现有STATUS、Goal及接入说明，未改实现、配置、正文、PU结果或科学证据等级。最终PR检查实际运行176项单元测试及两项技能定义检查，均通过；项目验证仍为相同7项既有错误，与PR #20保留输出逐条比较一致，整体仍FAILED。新输出保存于 `src/outputs/jev-phm/g0-20261018-mfpt-qualification/`，没有覆盖前次检查。
- **PR #21实质修正：**自动评审指出接续条件只列MFPT资料，可能遗漏仍未核验的PU原始时钟／通道／尺度依据。已在STATUS E、Goal和接入说明中明确两来源测量链都是G0前置，MFPT资料单独到位仍不能放行G1。只重读受影响条款及相邻原协议，`git diff --check`通过；未重跑模型或重复全仓测试。
- **18日公开入口：**提交实现后实际执行`bash run_all.sh --check`，2.18秒输出同一2,339条元数据事实后在未绑定物理资格处退出2；输出`src/outputs/jev-phm/g0-20261018-native-metadata/shell-check.txt`及`shell-check-time.txt`。没有打开H5、权重或启动GPU任务。
- **公开 Shell 入口验收：**实现提交后实际执行 `bash run_all.sh --check`，同样以退出码 2 在 G0 停止，耗时 0.94 秒；日志 `src/outputs/jev-phm/g0-20261017-integration/shell-preflight.txt`。两次 G0 预检共 1.73 秒，不是全部开发／检查耗时。没有因此启动任何模型、GPU、目标读取或拟合。
- **集中 PR 复查：**PR #19 自动评审指出恢复检查仍引用旧图名，导致完成的运行重复汇总并消耗 CPU 预算。已改为实际三组 SVG/PDF 的完整输出检查，并新增完成恢复／缺失 PDF 两项行为测试。只重查受影响的执行测试：18 项通过，Shell 语法与 diff 检查通过；未重复全仓测试或扩展研究实验。
- **未完成动作保留：**run-sim-002保存源侧信息诊断未完成；现有挂载包是另一个sim-20261017-001，不替代它。该动作在恢复条件保持子主张前仍必要，但不阻止作者已批准的无保持项框架P1先执行G0。

## E. 唯一下一步

- **当前不确定性：**真实PHMFactory导出、固定Qwen权重和CUDA能否在同一已声明输入下完成全部源侧接口与拟合？
- **主动作：**`JEV_RUN=src/outputs/jev-phm/run-real-20261019-001 bash run_all.sh`。G0已绑定；先执行真实源侧smoke，再运行原46次拟合上限。所有方法及seed完成并通过原验收门槛后才开放目标；保存全部失败、上限及负结果。
- **资源：**已授权24GPUh／12CPU墙钟h、最多8线程及单GPU；绑定物理0 UUID，GPU2禁用。19日作者确认输入及单元约定，无需再次请求相同人口、预算或数据批准。
- **准入边界：**PU按已知轴承分组；MFPT按声明的实验独立假设分组。原始数值、名义采样率和H5原值保留采用提供者陈述；PU不宣称g幅值。保存结果和图表必须披露该范围，不能把条件区间改写为已验证的设备总体推断。
- **停止条件：**输入矛盾、真实接口失败、拟合／预算上限或目标门槛未通过即保留产物并明确停止；不自动换模型、数据、预处理或seed，不将基础设施失败解释为方法失败。

## 按需使用的研究工具

[分步 Prompt](prompts/README.md) · [本地与外部专项 Skill](../.agent/references/task_specific_skill_selection.md)。这些入口不替代数据、运行或证明，不改变本页授权边界。
