# Paper Skill Core

Shared rules for scientific writing, revision, experimental design and review.
Read this file with the selected paper Skill and the repository-local scientific
instructions. It defines no model version, execution engine or project state.

These rules govern scientific content and evidence-bearing work. For code-only
tasks, preserve scientific semantics and reproducibility, but do not force
manuscript-style language onto source code or implementation documentation.
README.md routes task types and records the author-maintained README boundary;
AGENTS.md owns execution, Git and validation rules; this file owns shared scientific
writing and evidence norms.

## 1. Evidence first

Each important claim must have identifiable evidence, especially in the Abstract,
Introduction, Contributions, Results and Discussion. When support is missing,
obtain the minimum necessary evidence or narrow/remove the claim. Preserve a
substantive, defensible contribution rather than deleting it into emptiness.

Do not invent experiments, results, citations, SOTA, source content or manuscript
passages. Read the relevant original source; absence from an abstract does not
establish absence from the paper. Separate observation, inference, exploratory
finding, hypothesis and independent confirmation. Performance alone does not
establish a mechanism, causation or universal generalization.

## 2. Respect the research state

Identify the actual state: idea, formulation, method, implementation, experiment,
manuscript or revision. These descriptions are not sequential approval gates.
Implemented, executed, checked and scientifically supported are different states.
Before experiments exist, design the protocol or Results structure without
writing planned outcomes as findings. Keep unknowns explicit.

Preserve hypothesis provenance: H0 before the relevant result; H1 inspired by
exploratory evidence; H2 independently confirmed; H3 still post-hoc. Never rewrite
H1 or H3 as H0. Negative, null and contradictory evidence must be allowed to change
the direction; non-significance is not proof of no effect.

## 3. Scientific abstraction boundary

Keep three levels distinct:

- Repository operations: branches, PRs, commits, paths, scripts, commands, task
  records and local workflows belong to development, not the scientific argument.
- Implementation mechanics: APIs, wrappers, handlers, runners, registries,
  parsers, payloads, caches, callbacks and adapters belong in the paper only when
  needed for the algorithm definition, reproducibility or computational properties.
- Scientific methodology: problem, representation, objective, transformation,
  constraint, inference, optimization, decision rule, interaction and mechanism.

Necessary implementation and reproducibility details belong in Implementation
Details, an appendix, supplementary material or the repository, without hiding
information essential to interpreting the study.

## 4. Publication-language firewall

Describe scientific meaning, not the Agent's work log. For example, a verified
machine-disjoint split is described as keeping machine identities disjoint
between training and testing, not as a script processing a folder.

An actual evidence-conditioned diagnostic policy may be described through its
selection rule and declared insufficient-evidence behavior, rather than a tool
call trace. A wrapper is not automatically a unified scientific formulation:
claim such a formulation only when it is defined and supported. Rewording must
not invent a method, guarantee, capability or result.

## 5. Contributions are not software organization

A new API, class, wrapper, registry, configuration, automation script or refactor
is not by itself a scientific contribution. Ask what problem formulation,
methodological principle, measurable capability, experimental protocol or finding
remains when code organization is removed. A genuine tool, benchmark or resource
contribution is legitimate when its capability and evidence are specified.

## 6. Preserve legitimate technical terms

Do not use a word blacklist. Agent, tool, pipeline, module, interface, framework,
routing and workflow are appropriate when formally defined, algorithmically
meaningful, experimentally studied, or necessary to preserve scientific meaning.
Keep real uncertainty, qualifications, negative results and scope. Remove empty
self-protection, not evidence-bounded language.

Write for the scientific reader, not an imagined reviewer. Each sentence should
state a scientific fact, question, definition or method; report evidence; interpret
it; or give a boundary that changes the conclusion. Prefer what was done, observed
and supported over repeated statements of what the authors do not claim. Remove
apologies, promotional emphasis, empty self-explanation and narratives of checking
or constructing the manuscript. Actual reviewer responses remain a separate genre.

State a material limitation where it changes interpretation, normally once;
repeat only for a distinct scientific purpose. Preserve population and condition
scope, independent units, split separation, uncertainty, null/negative/failed
results, confounders, alternative explanations, assumptions and causal/transfer
boundaries. A limitation belongs because it affects the evidence, not because a
reviewer might ask. Positive wording must not turn association into causation,
an observed difference into general superiority, or a suggestion into a proof.

## 7. Scientific writing structure

Apply these requirements to original drafting as well as revision:

1. **One main line.** Formulate the core question and supported contribution in
   one sentence to guide the argument. Each paragraph advances it; remove digression
   and repetition, not necessary context. Missing evidence is a gap, not a stronger
   contribution to be supplied by wording.
2. **Consistent terminology.** Use accepted domain terms and conventional grammar.
   Introduce abbreviations as full term (abbreviation) at first use. Keep one term
   for one concept; do not rotate synonyms merely to avoid repetition.
3. **Precise descriptions.** Distinguish objects, variables, conditions, dimensions
   of change, observations and metrics. Specify what changes and relative to what;
   replace vague claims of performance, information gain or robustness with the
   supported quantity, comparison and conditions.
4. **No empty prose.** Remove uninformative modifiers, stock phrases, repetition
   and unnecessary syntactic complexity. Prefer the scientific subject and a
   direct operation/result statement; preserve conditions and qualifications.
5. **Reader order.** Assume domain knowledge, not knowledge of this study. Define
   concepts, symbols and parameters before using them; introduce the problem before
   the method and the result before its interpretation.
6. **Explicit logic.** Separate definition from identification/measurement,
   observation from mechanism, association from causation, assumptions from
   evidence/conclusions, and design from findings. Transitions must express an
   actual contrast, condition, progression or causal relation, not manufacture one.
7. **Evidence-bounded conclusions.** Support claims with data, derivation or
   reliable literature. Identify speculation and its basis, scope and alternatives.
   Use the strongest justified wording, never stronger; apply Sections 1 and 6.

Use problem -> gap -> scientific question -> method -> evidence -> interpretation
-> necessary boundary where it serves the argument, not as compulsory headings.
A paragraph normally gives its main point, evidence/mechanism and implication or
boundary. Explain why a design is needed and what scientific property it changes.
Preserve valid text and unique claims, definitions, equations, numbers, citations,
examples and limitations. Apply section roles without forcing every paper type
into an empirical template:

- **Abstract:** problem, gap, approach, main supported result or synthesis,
  implication and essential scope; not a compressed protocol or development status.
- **Introduction:** field need, relevant prior work, unresolved gap, question,
  approach and evidence-supported contributions.
- **Related Work:** compare questions, assumptions, methods and findings. Do not
  weaken prior studies to manufacture novelty or make unsupported first/all-prior
  claims; the gap must follow from verified comparison.
- **Method:** objects, inputs/outputs, assumptions, formulation, mechanism,
  operations and necessary protocol; give design reasons, not assurances of care.
- **Results:** actual comparisons, effect magnitudes, uncertainty/sensitivity and
  null/negative outcomes; report them directly rather than defend them.
- **Discussion:** supported mechanisms, helpful/failing conditions, alternative
  explanations, relation to prior work and interpretation-changing limitations.
- **Conclusion:** what was learned, its evidence and where it applies; no new
  unsupported claims, repeated disclaimers or defense of the paper's importance.

## 8. Methods explain semantics first

Prefer definition -> assumptions -> formulation -> mechanism -> algorithm ->
necessary implementation details. Separate the general principle, default
implementation and experimental configuration. Retain theory only when it
constrains, explains, predicts or delimits the method; inspect counterexamples and
unused assumptions before strengthening a theorem. Pure theoretical contributions
need appropriate proof, not invented experiments.

For empirical claims, specify the estimand, independent unit, comparison,
information access, tuning/compute budget, metric and inference boundary. Establish
competitive baselines. Check assigned, received, used and behaviorally effective
interventions where relevant. A valid non-use result differs from a broken run;
do not discard either to select favorable outcomes. Outcome, mechanism and boundary
evidence answer different questions; request only the kinds the claim needs.

## 9. Contribution check

Each contribution must answer: what scientific problem, what technical novelty,
and what supporting evidence? Statements centered on built, implemented,
integrated, wrapped, automated, called, configured or organized require a check
that they express more than development work. Stronger verbs cannot replace
missing evidence, and replacing a software noun cannot manufacture novelty.

## 10. Revision priority

Correct scientific errors, claim-evidence mismatch, implementation-as-contribution,
unclear nearest-neighbor distinctions, unfair comparison and cross-section
contradictions before ordinary polishing. Correct engineering-style prose and
reproducibility omissions where they affect understanding; a validity-threatening
omission is a scientific error, not a low-priority formatting issue.

## 11. One final reviewer QA

After a substantive draft or revision, make one focused pass: claim-evidence
check -> scientific-abstraction check -> reviewer check -> minimum correction.
Check that implementation was not promoted into novelty and operation records
were not promoted into manuscript content. Do not launch repeated reviewer loops,
fixed persona quotas or comprehensive audit packages. Recheck a corrected defect
only when a material change requires it. Review-only tasks remain read-only.

Within that pass, compare meaning before and after revision: estimands, comparison
groups, populations, causal direction, independent units, conditions, equations,
values, units, sample sizes, uncertainty, citations and failure cases must not
silently change. Ask what scientific information a deletion loses; retain content
whose removal strengthens a claim, hides a result or changes its scope.

Judge meaning, not lexical triggers. Do not use prose scores, hedge/banned-word
counts, keyword quotas, grep-based style gates or sentence-length thresholds.
May, could, however, although and not remain valid when scientifically needed.

## 12. Repository boundary

PaperTrace owns these generic norms. The consuming repository owns its problem,
theory, method, data, metrics, experimental protocol, contribution boundary,
venue requirements, approved decisions and active manuscript. Read and preserve
those local constraints; do not upload them into PaperTrace or replace them with
template defaults. Reuse existing state and evidence sources without creating a
second truth source. Execution rules are in the repository's AGENTS.md; a paper
Skill does not authorize Git publication, paid runs or data disclosure.
