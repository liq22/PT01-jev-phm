# Native annotation compatibility: PU and the CWRU candidate task

## 1. Scope and primary sources

### 1.1 The decision examined

Before fitting the proposed semantic model, determine whether the existing PU labels and the CWRU fault vocabulary can be treated as deterministic coarsenings of one diagnostic state space. This is a qualification of manuscript Sections 3.2 and 4.2, not another diagnosis experiment. No source or target waveform, model prediction, calibration outcome, or accuracy was used to select the mapping. The completed PU pilot and all its observations remain unchanged.

The evidence consists of the [versioned PU bearing-role table](../assets/tables/jev_unit_roles.csv), the [pinned PU reader](https://github.com/PHMbench/PHM-Vibench/blob/cdc0669167ca25ac848e34c880875ca39c6aed89/src/data_factory/reader/RM_027_PU.py), the [pinned CWRU reader](https://github.com/PHMbench/PHM-Vibench/blob/cdc0669167ca25ac848e34c880875ca39c6aed89/src/data_factory/reader/RM_001_CWRU.py), and original dataset documentation. These readers select signal channels; neither establishes cross-dataset label equivalence.

The [Lessmeier et al. original paper](https://mb.uni-paderborn.de/fileadmin-mb/kat/PDF/Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf) was inspected at Tables 4, 5 and 7 (printed pages 5, 6 and 8), and Section 7.4 (page 13). Table 5 and Section 7.4 were visually checked: KB23 and KB24 have inner- and outer-ring damage with inner-ring dominance; KB27 also has both, with outer-ring dominance. The original class numbers differ from the processed metadata numbers; the join uses bearing identity and class meaning, not numeric equality.

The [CWRU provider overview](https://engineering.case.edu/bearingdatacenter/welcome) and [fault specifications](https://engineering.case.edu/bearingdatacenter/fault-specifications) describe separately seeded inner-raceway, outer-raceway and rolling-element faults. This verifies the published fault vocabulary, not the contents or independence of the user's processed CWRU cohort. Its local metadata, H5 arrays, selected files and physical-unit map are unavailable in this environment and remain UNKNOWN.

## 2. Reproducible metadata result

### 2.1 Reported damage locations and stored labels

The [analysis output](../assets/tables/jev_native_annotation_compatibility.json) counts only `protocol == primary`, so repeated rotation assignments are not counted as new units. It joins explicit original-paper bearing codes, rather than interpreting identifier prefixes as fault labels.

| Published component state | Physical bearings | Source recordings | Stored pilot category |
|---|---:|---:|---|
| Healthy | 6 | 480 | Healthy |
| Inner-ring damage only | 11 | 880 | Inner-race-dominant |
| Outer-ring damage only | 12 | 959 | Outer-race-dominant |
| Both rings damaged, inner dominant: KB23, KB24 | 2 | 160 | Inner-race-dominant |
| Both rings damaged, outer dominant: KB27 | 1 | 80 | Outer-race-dominant |
| Total | 32 | 2,559 | Existing three-class pilot |

“Only” refers to the component locations reported in the inspected tables; it does not assert one point defect, one damage mechanism, or exhaustive inspection of every microscopic lesion. In particular, multiple lesions on one ring are not the same as lesions on both rings. The existing 79-record KA08 inventory is retained, not filled to 80.

Thus 240 records from three bearings cannot be interpreted as pure inner-only or outer-only faults. Their published dominant-location labels are nevertheless correct for the pilot's stated task. The result does not invalidate that pilot and does not authorize removing these records from any existing run. CWRU's rolling-element category also cannot be inferred from the letters `KB` in PU bearing identifiers.

### 2.2 A native counterexample to a too-coarse state space

Suppose the proposed canonical state space describes only lesion location:

$$
\mathcal Y_{\mathrm{site}}=\{H,I,O,J\},
$$

where $I$ and $O$ are inner-only and outer-only damage, and $J$ denotes simultaneous damage at both rings. The existing PU annotation assigns the same site state $J$ to two distinct observed categories: inner-dominant for KB23/KB24 and outer-dominant for KB27. Consequently, there is no deterministic map from this four-state ontology to the stored PU categories.

Omitting $J$ instead fails the closed-set population assumption. Assigning overlapping compatibility sets $S(I_{\mathrm{dom}})=\{I,J\}$ and $S(O_{\mathrm{dom}})=\{O,J\}$ does not repair a deterministic-partition likelihood. Together with $S(H)=\{H\}$, their predicted annotation probabilities sum to

$$
p_H+(p_I+p_J)+(p_O+p_J)=1+p_J,
$$

which exceeds one whenever $p_J>0$. An additional annotation model could be studied, but it is not the deterministic map claimed in the current method. Merely renaming logits or temperature-scaling them does not resolve this mismatch.

## 3. Exact condition and complete proof

### 3.1 When does a deterministic annotation map exist?

Let $\Omega$ be a set of admissible physical states, $\pi:\Omega\to\mathcal Y$ their representation in a proposed canonical ontology, and $a:\Omega\to\mathcal L$ the deterministic annotation rule. A map $g:\pi(\Omega)\to\mathcal L$ satisfying $a=g\circ\pi$ exists **if and only if** $a$ is constant on every fiber of $\pi$:

$$
\pi(\omega_1)=\pi(\omega_2)
\quad\Longrightarrow\quad
 a(\omega_1)=a(\omega_2).
$$

**Proof.** If $a=g\circ\pi$ and the two canonical states agree, applying $g$ gives equal annotations, proving necessity. Conversely, assume the displayed implication. For each $z\in\pi(\Omega)$ choose any physical state $\omega$ with $\pi(\omega)=z$ and define $g(z)=a(\omega)$. The implication makes this definition independent of which preimage is chosen. It then satisfies $g(\pi(\omega))=a(\omega)$ for every $\omega$, proving sufficiency. Values outside $\pi(\Omega)$ have no support in the stated population and are not identified by this argument. QED.

This elementary factorization criterion is not a new statistical-learning theorem. It supplies a necessary check for the manuscript's observed-event likelihood. With an actually valid map, disjoint preimages yield $P(a=y\mid x)=\sum_{k:g(k)=y}P(k\mid x)$ by partitioning the event. It provides neither a bound on target-domain error nor neural optimization consistency. Relevant prior foundations are [Cour et al. (2011)](https://jmlr.org/papers/v12/cour11a.html) and [Fotakis et al. (COLT 2021)](https://proceedings.mlr.press/v134/fotakis21a.html); their partial-label/coarse-label assumptions and guarantees are not inherited here.

### 3.2 Scope of the counterexample

KB23/KB24 and KB27 supply observed violations of fiber constancy when $\pi$ retains component location but discards dominance. They refute **that canonicalization**, not every possible PU/CWRU mapping or the general semantic-learning method.

Refining $J$ into inner-dominant and outer-dominant mixed states can remove this particular contradiction, but the target semantics and distinguishability must then be established. Alternatively, a declared dominant-location task can use the existing categories; on comparable exact annotations it is ordinary shared-label learning, not evidence of a coarse-label advantage. A prospective single-ring subset would contain 29 PU bearings and 2,319 recordings, but it is a changed population requiring a separate declaration; no such exclusion is performed here. It must not be selected for favorable prediction results, and it does not remove CWRU's separate rolling-element state automatically.

## 4. Consequences and reproduction

### 4.1 Current decision

C4 remains UNTESTED as a performance claim. Two shortcuts are now specifically REFUTED: covering all PU observations with pure single-ring states, and treating one undifferentiated mixed state as sufficient for a deterministic dominant-label map. The procedure's empirical comparison must wait until its actual annotation model and target population are fixed. C5 and the completed pilot's C1--C3 are unchanged.

The next unfinished part of the same qualification action is the user's actual CWRU metadata: associate processed labels and selected channels with provider files, retain rolling-element observations explicitly, and determine repeated acquisitions/physical-unit grouping. A reader name or a four-state public vocabulary is not enough. No CWRU export or training command has been validated; the existing `run_jev_study.py prepare` is PU-specific and must not be repurposed by changing a dataset name alone.

### 4.2 Reproduction without model calls

From a complete checkout of this revision:

```bash
python src/S03_Scripts/analyze_jev_annotation_compatibility.py \
  --roles paper/assets/tables/jev_unit_roles.csv \
  --out /tmp/jev-native-annotation-check.json
```

Use a fresh output path; the script refuses overwrite. It uses the Python standard library only, does not read waveforms, and produces no trained classifier or efficacy result. The local authoring execution used the exact primary-row extract returned by the GitHub connector because the checkout and remote data were unavailable. The analysis selects those same primary rows from the full repository table, so the other 192 rotation assignments are intentionally irrelevant. A transcribed KA07 role field was corrected against the source before accepting the final output; the rejected intermediate file was retained locally, and no model result or scientific count changed.

## 5. Conditional-freezing equivalence of the proposed update restriction

### 5.1 Scope, prior work, and assumptions

This is the single analytical follow-up to the proposed annotation-constrained update, performed while the existing CWRU metadata qualification is blocked. It screens a candidate intervention for C4; it does not adopt a new runtime, add a performance claim, or replace the native-data requirement. Sections 1–4 and the completed PU pilot remain unchanged.

Zombori et al., *Towards Unbiased Exploration in Partial Label Learning*, JMLR 25(412):1–56 (2024), define probability-ratio preservation as an architecture-dependent property and establish their Libra-loss result for softmax regression (Section 4.3, Definitions 5–7 and Theorem 8; [published text](https://jmlr.org/papers/volume25/23-0868/23-0868.pdf)). Farajtabar et al., *Orthogonal Gradient Descent for Continual Learning*, AISTATS 2020, derive projections from previous-task output gradients (Section 3, Algorithm 1; [published text](https://proceedings.mlr.press/v108/farajtabar20a/farajtabar20a.pdf)). Neither ratio preservation nor gradient projection is claimed here as a new principle. The derivation below checks the specific fixed semantic-head candidate, not those papers' algorithms. Only these relevant sections were reread; the OGD page was visually checked, whereas the JMLR screenshot request failed and its method was read as parsed text.

Fix one valid annotation partition throughout a block of updates. Let its nonempty blocks be $C_1,\ldots,C_Q$, covering $K$ mutually exclusive classes. Fix the signal features $v(x)\in\mathbb R^{d_f}$, candidate vectors $e_k\in\mathbb R^{d_h}$, and temperature $\tau>0$. The only trainable object is $U\in\mathbb R^{d_h\times d_f}$, with finite logits $z_k=e_k^\top Uv/\tau$ and probabilities $p_U=\operatorname{softmax}(z)$; there is no normalization after $Uv$ or additional trainable bias.

The suggested restriction forms

$$
\mathcal V_d=\operatorname{span}\{e_i-e_j:i,j\in C_b\text{ for some }b\},\qquad
\Pi_d=\operatorname{proj}_{\mathcal V_d^\perp},\qquad
\Delta U=\Pi_d\Delta U.
$$

Projected SGD is one way to satisfy this condition. The actual total parameter increment must satisfy it, including any momentum, adaptive preconditioning or decay. Invariance of the raw gradient alone is insufficient. Updates with different source partitions are not one fixed-partition block.

### 5.2 Exact equivalence to a frozen conditional predictor

**Proposition.** For any initial $U_0$ and any allowed finite update $U_+=U_0+\Delta U$, the predictor has the form of fixed within-block conditional distributions and updated block masses. Specifically, choose any representative $k_b\in C_b$ and define

$$
\bar e_b=\Pi_d e_{k_b},\qquad
q_b^0(x)=\sum_{k\in C_b}p_{U_0}(k\mid x),\qquad
\delta_b(x)=\bar e_b^\top\Delta Uv(x)/\tau.
$$

Then for $k\in C_b$,

$$
q_b^+(x)=\frac{q_b^0(x)\exp(\delta_b(x))}{\sum_c q_c^0(x)\exp(\delta_c(x))},\qquad
p_{U_+}(k\mid x)=\frac{p_{U_0}(k\mid x)}{q_b^0(x)}q_b^+(x).
$$

**Proof.** Within a block, $e_i-e_j\in\mathcal V_d$, hence $\Pi_de_i=\Pi_de_j$. Because $\Delta U=\Pi_d\Delta U$ and the orthogonal projector is symmetric, $e_k^\top\Delta Uv/\tau=\bar e_b^\top\Delta Uv/\tau=\delta_b$ for every $k\in C_b$. Thus each logit in block $b$ receives the same increment. Substitution into the softmax gives

$$
p_{U_+}(k\mid x)=\frac{p_{U_0}(k\mid x)\exp(\delta_b)}{\sum_c q_c^0\exp(\delta_c)}.
$$

Summing within the block proves the formula for $q_b^+$, and dividing proves the conditional factorization. All masses are positive for finite logits. The sum of updates satisfying the same projector also satisfies it, so the result applies to a whole fixed-partition block. QED.

The explicit simple alternative is a frozen fine conditional predictor plus a coarse residual head with the inherited offsets $\log q_b^0(x)$ and learned linear increments $\delta_b(x)$. A fresh coarse softmax head **without** those offsets is not necessarily the same comparator. The projected family is contained in this offset-head family; the next result characterizes equality. This is an equivalence of predictors, not of arbitrary optimization trajectories, regularizers, parameter counts or generalization errors.

### 5.3 The remaining ability to learn coarse distinctions

Define the $(Q-1)\times d_h$ matrix

$$
A_d=\frac1\tau\begin{bmatrix}(\bar e_1-\bar e_Q)^\top\\\vdots\\(\bar e_{Q-1}-\bar e_Q)^\top\end{bmatrix}.
$$

**Proposition.** Changes in coarse log-odds relative to block $Q$ are exactly $A_d\Delta Uv$. Every linear coarse log-odds increment on the full feature space is representable if and only if $\operatorname{rank}(A_d)=Q-1$. If $\operatorname{rank}(A_d)=0$, every allowed update leaves the entire class distribution unchanged, even if $\mathcal V_d^\perp$ has nonzero dimension.

**Proof.** Taking the log ratio of $q_b^+/q_Q^+$ and subtracting its initial value yields $\delta_b-\delta_Q$, namely row $b$ of $A_d\Delta Uv$. Every row of $A_d$ lies in $\mathcal V_d^\perp$. At full row rank, for any desired coefficient matrix $R\in\mathbb R^{(Q-1)\times d_f}$, set $\Delta U=A_d^\top(A_dA_d^\top)^{-1}R$. Its columns lie in $\mathcal V_d^\perp$ and $A_d\Delta U=R$. Conversely, rank deficiency supplies nonzero $a$ with $a^\top A_d=0$. Every attainable coefficient then satisfies $a^\top A_d\Delta U=0$, whereas some $R$ violates $a^\top R=0$. Thus not every linear increment is attainable. At rank zero, all block increments are identical and cancel in normalization; the first proposition then fixes every fine probability as well. QED.

The rank criterion describes representability, not successful optimization. Full row rank does not imply equal training outcomes under different coordinate systems or penalties, and ill-conditioning can matter. On a restricted feature support, only the corresponding restricted functions need be compared. When all annotations are exact, the within-block difference space is zero and the restriction reduces to the original unconstrained adapter.

### 5.4 An explicit counterexample: preservation can prevent coarse learning

Take four distinct **unit** candidates in two dimensions,

$$
e_1=(1,0)^\top,\quad e_2=(-1,0)^\top,\quad
e_3=(0,1)^\top,\quad e_4=(0,-1)^\top,
\qquad C_1=\{1,2\},\quad C_2=\{3,4\}.
$$

Their within-block differences span $\mathbb R^2$, so $\Pi_d=0$. With $v=1$, $\tau=1$, and $U_0=(1,0)^\top$, the coarse mass is

$$
q_1^0=\frac{\cosh(1)}{\cosh(1)+1}.
$$

For a sample labelled $C_1$, every projected update is zero regardless of the loss or step size. Yet the same unprojected semantic model can improve the event loss: along $U=(u,0)^\top$,

$$
\frac{d[-\log q_1(u)]}{du}
=-\frac{\sinh u}{\cosh u(\cosh u+1)}<0\qquad(u>0).
$$

Even preserving both within-block conditionals does not inherently prevent improvement: the simple offset-head alternative can use $\delta_1=t$, $\delta_2=0$, giving $q_1(t)=e^t\cosh(1)/(e^t\cosh(1)+1)$ and loss derivative $-(1-q_1(t))<0$. It obeys the intended conditional-preservation rule but cannot be represented by this projected fixed head. The obstruction is therefore the tied candidate geometry plus the restriction, not an absence of a valid coarse target.

This is a constructed analytical counterexample, not an observed PHM failure or an experiment on pretrained embeddings. Appending zero coordinates gives the same counterexample in any larger dimension: ambient width alone does not restore diagnostic learning directions.

### 5.5 Decision, scope and implementation constraints

The proposed projection cannot be promoted as a generally new way to preserve fine distinctions in the **fixed-partition, frozen-feature linear** setting. It is a restricted coarse residual update with frozen within-block probabilities; at the stated full-rank condition the predictor families coincide. A potential optimization or inductive-bias difference would need its own matched test rather than being inferred from the preservation identity.

Before this candidate could enter a legitimate experiment, the valid annotation partition and actual candidate geometry would be required. Zero effective rank rules out improvement from this update; nonzero rank does not establish that the coarse gradient has a useful allowed component. Exactness also requires unchanged features/candidates/temperature and a constrained total update. Joint nonlinear representation learning or alternating non-nested partitions are outside this result; no conclusion about their usefulness is inferred.

The result narrows a proposed intervention under C4, not the established coarse-label likelihood, the existing pilot, or the entire Jev-PHM direction. No C6 is introduced, no corpus is relabelled, and no new training entrypoint is warranted. The unfinished CWRU file/label/channel/unit qualification remains the unique next action. Do not repeatedly expand this proof or substitute synthetic partitions for missing native data.
