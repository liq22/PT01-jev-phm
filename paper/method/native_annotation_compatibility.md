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
