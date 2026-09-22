# Research-First Paper Loop

This is the default reasoning contract for PaperTrace research, code, experiment,
figure, manuscript, review, and revision work. It combines the Research-First
Agent and Adaptive Research Paper Agent principles without turning them into a
fixed stage pipeline. Load only the section needed by the current task.

## Highest authority

At every iteration:

1. state the strongest current claim that matters to the paper;
2. identify the largest unresolved uncertainty that threatens it;
3. list the strongest competing explanations;
4. choose the smallest coherent action that most directly distinguishes them;
5. define before execution what each plausible outcome would mean;
6. obtain real evidence through literature, theory, code, experiment, statistics,
   figure inspection, or manuscript revision;
7. update the claim, mechanism, boundary, novelty judgment, or next decision;
8. stop when the action no longer has material scientific value.

Do not continue merely because a workflow has a next stage.

Before acting, answer:

```text
Q1. What is the key uncertainty blocking the research decision?
Q2. What is the smallest action that directly reduces it?
Q3. Could any plausible outcome change the scientific judgment or next action?
```

If Q3 is no, do not perform the action.

A useful decision heuristic is:

$$
a^* = \arg\max_{a\in\mathcal A}
\frac{\mathbb E[\Delta U_{\mathrm{sci}}(a)]}{C(a)},
$$

where $\Delta U_{\mathrm{sci}}$ is the expected reduction in the current key
scientific uncertainty and $C(a)$ is time, compute, code, and cognitive cost. The
formula guides judgment; it does not create a scoring bureaucracy.

## Adaptive research state

Maintain only the state needed to choose the next action:

```text
Central question:
Current claim:
Evidence supporting it:
Strongest competing explanation:
Largest unresolved uncertainty:
Current novelty risk:
Most informative next action:
Stopping condition:
```

Do not write an Introduction merely because the state is incomplete. Return to the
smallest relevant literature, theory, code, experiment, or analysis action.

## Non-negotiable rules

1. Evidence before narrative.
2. Problem before method.
3. Failure before novelty.
4. Mechanism before module.
5. Theory must constrain, predict, explain, or delimit.
6. Every major claim needs theoretical, experimental, or verified-literature
   support.
7. Negative, null, unstable, and contradictory results remain evidence.
8. Exploratory findings are not confirmatory evidence.
9. Facts, inferences, and hypotheses must remain distinct.
10. Do not fabricate citations, results, statistics, novelty, author decisions, or
    implementation behavior.
11. Scientific semantics precede infrastructure.
12. Do not create rigor theatre through hashes, receipts, ledgers, exhaustive
    validators, reviewer quotas, or CI volume.

## Problem definition

A research question should make the boundary explicit:

```text
Q = (Object, Environment, Observation, Task, Failure, Desired understanding)
```

Ask:

- What object or relation is being studied?
- Under which environments and interventions?
- What is observed and at what independent unit?
- What task or decision is evaluated?
- What failure or unexplained behavior motivates the work?
- What new understanding should survive if the preferred method does not win?

## Scientific semantics and fail-fast

Prioritize the actual computation chain:

```text
data
-> metadata
-> split and protocol
-> transforms
-> objective
-> model inputs/outputs
-> metric
-> result
```

For dataset, label, metadata, sampling rate, window, stride, split, transform,
objective, metric, model, checkpoint, or protocol inconsistencies:

```text
detect semantic inconsistency
-> clear error
-> stop
```

Do not guess, silently repair, infer intent, switch tasks, or fall back to a
different scientific problem. Recover only from operational failures that leave
the experiment meaning unchanged.

## Literature and novelty falsification

The first purpose of literature search is to try to invalidate the novelty claim,
not to collect support for a preferred gap.

Search for:

- foundational and recent representative work;
- the closest direct prior art;
- equivalent, historical, and neighboring-field terminology;
- the same mechanism under another name;
- theoretical equivalents;
- conference, journal, preprint, benchmark, and reproduction versions;
- work reporting the same failure or boundary;
- evidence that directly weakens the proposed novelty.

For load-bearing sources, record only what changes the decision:

```text
Citation and verified source
Research question
Method and core assumption
Main result
Known limitation
Exact overlap with our work
Remaining distinction
Evidence that the distinction matters
```

The output should state:

```text
Strongest prior art against novelty:
Exact overlap:
Difference that remains:
Evidence that the difference matters:
Novelty verdict: invalid / weak / defensible / strong
Required repositioning:
```

Search snippets do not replace original-source verification. A low paper count or
missing exact keyword is not a scientific gap.

## Claim tree and hypothesis provenance

Prefer one central claim and the smallest claim tree needed by the paper:

```text
C1: the baseline exhibits a defined failure.
C2: the failure is associated with mechanism Z.
C3: intervention M changes mechanism Z.
C4: the expected observable behavior changes accordingly.
C5: the advantage weakens when assumption A is broken.
```

Include only claims the paper actually needs. Mark evidence state as:

```text
literature-supported
 theory-supported
 exploratory evidence
 independent confirmation
 post-hoc interpretation
 unsupported
```

For each major hypothesis record provenance:

```text
H0: proposed before observing the relevant result
H1: inspired by exploratory evidence
H2: independently confirmed by new evidence
H3: still a post-hoc interpretation
```

Do not rewrite H1 or H3 as if it had always been H0. Exploratory findings may be
reported honestly; a central claim that depends on them should receive an
independent confirmation test.

## Counterexample-first theory

Theory starts from the real experiment:

```text
E = (D, P, f_theta, L, M)
```

where `D` is data and environments, `P` is protocol/split/transform, `f_theta` is
the model, `L` is the objective, and `M` is the evaluation quantity.

Before strengthening a proposition:

1. actively search for a counterexample;
2. identify the minimum assumption needed to exclude it;
3. remove assumptions not used by the reasoning;
4. check whether the remaining assumptions hold in the experiment;
5. derive an observable prediction and failure condition.

The theory chain must close:

```text
Assumptions -> Claim -> Observable prediction -> Experiment
```

A theoretical contribution should provide a checkable definition, decomposition,
condition, bound, invariant, identifiability result, consistency result, failure
condition, or testable design principle. Otherwise call it analysis,
interpretation, hypothesis, or empirical observation. Do not label intuition as a
theorem.

## Failure-driven method loop

Do not ask what module can be added. Ask what failure must be explained or
changed.

```text
observation
-> competing mechanism hypotheses
-> minimal intervention
-> controlled comparison
-> mechanism check
-> keep, simplify, reject, or revise
```

For each iteration:

```text
Observed failure:
Competing explanations:
Proposed intervention:
Why it targets the mechanism:
Simplest alternative:
Expected signature:
Actual result:
Decision:
```

A component earns its place only when it has one scientific role, a simpler
plausible alternative, a deletion or replacement test, and an observable
signature. If a simpler version performs equivalently, prefer the simpler version
or reduce the contribution claim.

## Claim-driven experiment design

Every experiment must answer one claim or one major uncertainty:

```text
Claim:
Competing explanation:
Required evidence:
Experiment:
Possible outcomes:
Decision under each outcome:
```

Define the experimental unit before statistics:

```text
What is the experimental unit?
Which observations are dependent?
At what level is train/test separation defined?
At what level should uncertainty be estimated?
```

Windows, frames, patches, folds, and repeated measurements are not automatically
independent replicates.

Comparison fairness includes:

```text
Data access
Target information
Metadata access
Pretraining
Search/tuning budget
Compute budget
Evaluation protocol
```

Any asymmetry must be disclosed and cannot support an unconditional superiority
claim.

Use only the baseline set required by the question: naive, canonical, strong
recent, closest prior art, method without the key mechanism, and simplest valid
alternative. Do not add weak baselines merely to fill a table.

Ablation should answer whether the mechanism or complexity is necessary. Prefer:

```text
full method
vs key mechanism removed
vs simplest valid alternative
```

## Experiment stop rule

Do not run an experiment merely because papers usually contain it. Run it only
when at least one plausible outcome could change:

- claim confidence;
- mechanism discrimination;
- method choice;
- theoretical interpretation;
- applicability boundary;
- novelty judgment;
- a major reviewer-critical conclusion.

If all plausible outcomes leave the research decision unchanged, stop that
experiment.

## Statistics and interpretation

Statistics serve the estimand and independent unit. Use repeated runs,
uncertainty, confidence intervals, effect sizes, paired analysis, bootstrap,
non-parametric methods, multiplicity correction, Bayesian analysis, or power only
when they improve the actual inference.

Do not add a p-value because it looks rigorous. Do not choose the analysis after
seeing which method produces the preferred result.

Interpret results as:

```text
Observation
-> quantitative evidence
-> mechanism interpretation
-> relation to hypothesis
-> remaining uncertainty
```

Always separate:

```text
Fact: directly supported by data.
Inference: derived from facts and assumptions.
Hypothesis: requires new evidence.
```

## Code, abstraction, and validation

Research code must make it easy to answer:

```text
Where did the data come from?
What transformations were applied?
What model and objective ran?
What protocol and split were used?
What result was produced?
```

Before adding a validator, wrapper, fallback, cache, factory, registry, schema,
test, CI rule, or metadata field, ask:

1. Does it solve a real observed/current problem?
2. Could the current scientific conclusion be wrong without it?
3. Is there a simpler direct implementation?

If the first two answers are no, do not add it. If the third is yes, use the
simpler option. Abstract only after at least two real current use cases exist and
only when the abstraction does not hide data flow or mathematical semantics.

Python is for numerical work, data processing, signal processing, simulation,
training, inference, statistics, plotting, necessary file generation, and tests
of scientific semantics. It is not a generic wrapper, text checker, report
generator, silent input repair layer, or hypothetical-validator framework.

Do not proactively create custom hashes, checksums, digests, receipts, ledgers,
or integrity chains. An upstream official checksum may be used only to verify a
real download, or when actual corruption has occurred, or when the user explicitly
requests content verification. Git already records code history.

## Paper construction

Systematically organize the paper only after the central question, main claim,
and key evidence are sufficiently stable.

### Argument structure, not a research workflow

Use the smallest complete argument:

```text
scientific question -> proposed answer -> evidence -> bounded conclusion
```

WHY–WHAT–HOW helps explain a problem or method, but does not by itself establish
what was learned. Expand it when the argument needs the missing links:

```text
WHY -> GAP -> WHAT -> HOW -> EVIDENCE -> CLAIM
```

| Part | Reader's question | Required content |
|---|---|---|
| WHY | Why does this question matter? | Concrete phenomenon, need, failure, or scientific uncertainty. |
| GAP | What remains unresolved? | Verified limits of existing knowledge under explicit conditions; not merely an unused module combination. |
| WHAT | What answer or idea is proposed? | Specific question, hypothesis, explanation, method, or mathematical object, as appropriate to the study. |
| HOW | How is the idea made testable? | Mechanism, assumptions, formulation, implementation, and comparison. |
| EVIDENCE | What supports or challenges it? | Actual observations, proofs under stated assumptions, controlled comparisons, or verified literature. |
| CLAIM | What can now be concluded, and why does it matter? | Knowledge gained, scope, remaining alternatives, and scientific or engineering implication. |

This is reader-facing argument order, not the chronology of discovery or a fixed
execution pipeline. Preserve hypothesis provenance. A proposed explanation is not
an established answer, and a failed prediction can still yield useful knowledge.
Do not require a new theorem, mathematical object, mechanistic hypothesis, or
experiment in every paper. Use the evidence appropriate to the contribution.

### Section-specific logic

Do not repeat all six parts in every section or use WHY/WHAT/HOW as mandatory
manuscript headings. Apply only the structure that serves the section's job.

| Section | Argument order |
|---|---|
| Abstract | Problem and specific gap -> approach and scope -> actual principal finding -> bounded implication. |
| Introduction | Real problem -> existing capability and assumptions -> unresolved failure or contradiction -> scientific question -> key insight -> supported contributions. |
| Related Work | Compare relevant assumptions, mechanisms, information access, results, and limitations -> exact overlap -> meaningful remaining distinction. |
| Method | Local difficulty -> component's scientific role -> inputs, operations, outputs, objective, and essential implementation -> justified property or testable prediction. |
| Theory, when needed | Question -> definitions and assumptions -> statement -> proof -> observable consequence -> counterexample or validity boundary. |
| Experiments | Question and competing explanation -> quantity to estimate -> protocol and fair comparison -> actual result -> conclusion supported. |
| Results | Main observation -> quantitative comparison and uncertainty -> interpretation -> corroborating or conflicting result -> remaining uncertainty. |
| Discussion | Explain or qualify the findings -> consider alternative explanations and prior work -> identify failure conditions and separate method/data/experiment/theory/scope limitations -> implication. |
| Conclusion | Answer the original question with the knowledge actually gained; do not add untested superiority, causality, or universality claims. |

Results and Discussion may be combined when appropriate; do not duplicate the
same table description or explanation merely to fill both sections. Before results
exist, keep missing findings explicit in the working draft, not as invented
numbers or expected outcomes written in the past tense.

### Paragraph and method logic

Give each paragraph one intellectual job. State its point early, provide the
reasoning or source/result that supports it, and explain the consequence when
needed. This is not a fixed sentence count. Preserve unique definitions,
mechanisms, examples, equations, citations, negative findings, and boundaries.

For a method component, ask internally:

```text
WHY: Which specific difficulty requires an intervention?
WHAT: What does this component change or preserve?
HOW: What are its inputs, operations, outputs, and mathematical objective?
PROPERTY: What follows by proof, what is intended, and what still needs testing?
```

An equation should define a computation or support an argument. Explain its
symbols, assumptions, and connection to the next operation; do not introduce
symbols simply to restate prose. Methods explain mechanism, not software
architecture. Factories, registries, caches, managers, hashes, manifests, and
logging stay out of the manuscript unless they are the research object.

For example, introducing an alignment loss does not establish that condition
information decreases while fault information is preserved. If that is only the
design intention, say so and identify the comparison that tests it. Do not present
$I(Z;C)\downarrow$ and $I(Z;Y)\uparrow$ as established properties without suitable
assumptions and proof or an appropriate empirical analysis. Accuracy improvement
or a visually separated embedding alone does not establish disentanglement or a
causal mechanism.

### Writing an experiment as an argument

Reuse the existing experiment notes; do not create a second registry or require
new metadata fields. E1, E2, and similar labels are optional reader-facing labels,
not a required experiment count. For each necessary experiment, make these
answers recoverable from the text, table, and caption:

```text
Purpose (WHY): Which question or competing explanation is being tested?
Target (WHAT): Which quantity, population/conditions, and comparison answer it?
Protocol (HOW): What information, split, controls, budget, and analysis are used?
Outcome meaning: What would improvement, no change, or deterioration imply?
Result: What was actually observed, with relevant uncertainty and result source?
Conclusion: What does it support, what remains unresolved, and under which scope?
```

The outcome meanings belong to the prospective design when specified before the
experiment. Do not retroactively label an exploratory analysis as predeclared.
Separate completed findings from planned experiments and post-hoc interpretation.

The estimand is the quantity the study intends to estimate, not merely a metric
name or the number returned by one run. For an illustrative fixed predictor $f$
and a specified held-out-condition distribution $P_{\mathrm u}$, distinguish:

$$
R_{\mathrm u}(f)
= \mathbb E_{(X,Y)\sim P_{\mathrm u}}[\ell(f(X),Y)],
$$

$$
G_{\mathrm{shift}}(f)
= R_{\mathrm u}(f)-R_{\mathrm s}(f),
\qquad
\Delta_{\mathrm u}
= R_{\mathrm u}(f_{\mathrm{base}})-R_{\mathrm u}(f_{\mathrm{new}}).
$$

Here $R_{\mathrm s}$ is risk on a specified held-out seen-condition distribution,
using the same task and loss. $R_{\mathrm u}$ measures unseen-condition predictive
risk; $G_{\mathrm{shift}}$ measures the seen-to-unseen risk difference for the same
predictor; $\Delta_{\mathrm u}$ compares two predictors on the same unseen target
and is positive when the new predictor has lower risk. A sample estimate
$\widehat R$ is not the population quantity $R$.

Do not substitute the shift difference for unseen-condition performance: a small
difference can coexist with poor performance in both settings. It is also not the
usual population-risk minus empirical-training-risk generalization gap. No
subtraction is required unless it answers the research question. Define domain
weighting and averaging over training randomness when relevant, and do not infer
performance on every unseen condition from one held-out suite.

For uncertainty, identify the independent unit and what the variation represents.
Random-seed dispersion does not by itself measure uncertainty across machines or
operating conditions; overlapping windows are not independent replicates. Select
task-relevant metrics rather than requiring Accuracy, F1, and AUROC for every task.
State the aggregation, units, and whether a reported improvement is a percentage
point difference or a relative percentage.

### Generalization and adaptation must remain separate

When the paper concerns unseen conditions, define what was unseen and when.
A source-only generalization experiment keeps the evaluated target conditions
unavailable for fitting, target-specific normalization, model selection, and
checkpoint selection; ordinary inference with a frozen source-fitted predictor
and preprocessing is allowed. Access to unlabeled target data for adaptation,
labeled target data for fine-tuning, or parameter/statistic updates at test time
changes the information-access protocol and must be reported separately.

For example, E1 may test source-only unseen-condition performance, E2 a matched
component deletion or replacement, and E3 a predicted failure boundary. Retain
only the experiments required by the actual claims. A target-adaptation study can
be a separately labeled extension; its results must not be pooled with source-only
results to claim better generalization without target access. Match information
access and relevant budgets, and disclose unavoidable asymmetries rather than
claiming that every comparison uses an identical architecture.

### Match the conclusion to the evidence

A main comparison supports predictive performance under its stated protocol.
An ablation supports the component's incremental contribution under that
comparison; it does not by itself prove the intended mechanism. A mechanism claim
requires observations or interventions that distinguish it from plausible
alternatives. An observed association is not automatically causal, and statistical
significance is not engineering importance.

Use the supported result, not a stronger story: distinguish “improves on these
held-out conditions,” “is consistent with the proposed mechanism,” and “is proved
under these assumptions.” Keep null, negative, and conflicting evidence in the
argument. Place a material limitation where it affects interpretation, without
repeating defensive disclaimers in every paragraph.

Before finishing, check the substantive alignment:

```text
gap -> idea -> method -> measured quantity -> experiment -> result -> conclusion
```

This is a reasoning check, not a new required graph, checklist file, or validation
system. Background, definitions, and implementation details may be necessary for
understanding or reproduction even when they do not supply a new claim. Remove
redundancy, not unique scientific content. Figures and tables must communicate
problem, mechanism, evidence, or boundary. Decorative AI artwork and panels
without a reader question are excluded.

## Adversarial review and stop rule

Review is a tool for discovering independent scientific problems, not a fixed
quota of personas or findings. Select only the lenses needed by the paper:
novelty, theory, methodology, experiment, statistics, generalization, domain
validity, reproducibility, positioning, writing, figures, or adversarial reject.

Prioritize:

```text
P0: invalidates the scientific conclusion
P1: seriously weakens the contribution
P2: important but non-fatal
P3: presentation
```

Each reviewer or review lens should report:

```text
Decision:
New independent P0/P1 issue:
Weakest claim:
Evidence missing:
Alternative explanation:
Required revision:
```

Stop expanding review when consecutive independent lenses no longer discover a
new P0/P1 issue. Repetition of an existing issue does not count as new information.
Do not manufacture reviewer theatre after the marginal scientific information has
fallen to zero.

Revision follows:

```text
review concern
-> root cause
-> required evidence
-> experiment, theory, method, analysis, or rewrite
-> observed result
-> claim update
```

When new evidence rejects the original claim, change the claim.

## Lightweight logs

Maintain only information that changes the research or paper decision.

`research_log.md`:

```text
Claim at risk
Key uncertainty
Action
Observation
Interpretation
Decision
```

`paper_log.md`:

```text
Claim changed
Hypothesis provenance
Evidence added
Novelty risk
Reviewer concern
Revision decision
Remaining blocker
```

Do not use logs to duplicate Git history or create an audit system.

## Completion

A research or paper action is complete when it changes a primary product and
resolves the intended decision:

- question, claim, or hypothesis;
- source-grounded novelty or literature conclusion;
- method or theory;
- executable code;
- actual experimental result;
- figure/table and caption;
- manuscript section;
- revised claim or reviewer response;
- uploadable submission files.

It is not complete because a plan, checklist, matrix, ledger, audit report,
validator, or CI job exists.

The optimization objective is:

```text
Research value
= scientific correctness × evidence × mechanism × clarity
  / unnecessary complexity
```
