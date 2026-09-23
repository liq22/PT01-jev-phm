# Jev-PHM

## Research question

### Decisions from verified evidence

Investigate Jev-inspired typed probabilistic decisions for PHM with **PHMFactory + Qwen/Qwen3-0.6B**. This project does not use proprietary Jev weights, its API, or a reproduced RLCD algorithm.

The research question is whether explicitly scoped diagnostic judgments and code-owned composition improve probability quality or selective decisions beyond competitive direct predictors. Typing, softmax heads, hierarchical classification and calibration are prior art; NanoJev already implements Qwen3-0.6B decision heads and atomic-judgment workflows. See the [source assessment](paper/refs/reading_matrix.md).

## Implementation and design

### Implemented starting point

```text
PHMFactory windows -> six deterministic features -> frozen Qwen representation
-> tuned logistic head -> raw / pooled / unit calibration -> diagnosis or deferral
```

The numerical control bypasses Qwen and shares the downstream implementation. This existing E0/E1 path is unchanged. Positive scalar temperature cannot change ordinary class argmax. Unit-weighted calibration is a secondary population-specific hypothesis, not Jev's defining mechanism or an established performance gain.

### Designed extension, not yet implemented

For a verified healthy-plus-fault taxonomy, ask two questions over the same state: whether a fault is present, and which fault fits under the explicit assumption that one is present. Code combines the resulting fault probability and conditional identity distribution into one leaf distribution. This is conventional conditional factorization applied to a question-conditioned PHM design, not a new probability theorem.

The current CLI does **not** implement these question-conditioned heads or the E2 comparisons. They are specified in [main Section 4.4](paper/draft/main.md) and the appended [E2 Goal design](paper/experiments/jev_goal.md). Do not report them as runnable, validated, or effective.

## Files and execution

### Primary products

| Product | Location |
|---|---|
| Unique manuscript | [paper/draft/main.md](paper/draft/main.md) |
| Current research state | [paper/paper.yaml](paper/paper.yaml) |
| Contribution/evidence mapping | [paper/experiments/evidence_matrix.md](paper/experiments/evidence_matrix.md) |
| Existing E0/E1 commands and design-only E2 | [paper/experiments/jev_goal.md](paper/experiments/jev_goal.md) |
| Sole current implementation | [src/S01_Package/jev_phm.py](src/S01_Package/jev_phm.py) |
| Fixed first-run settings | [src/S02_Configs/jev_probe.json](src/S02_Configs/jev_probe.json) |

### Existing lightweight checks

Use an isolated environment compatible with the PHMFactory installation:

```bash
python -m pip install -r src/S02_Configs/jev_requirements.txt
python -m unittest discover -s src/S04_Tests -p test_jev.py -v
python src/S01_Package/jev_phm.py --help
```

PR #1 reports nine synthetic numerical/interface tests. This documentation revision does not rerun or upgrade those checks into real PHM evidence. Follow the Goal to verify local data, labels, independent units, PHMFactory configuration and an immutable Qwen weight revision before actual inference.

### Manuscript preview

From the repository root with Pandoc installed:

```bash
pandoc paper/draft/main.md --citeproc --bibliography paper/refs/references.bib \
  --mathjax --standalone --metadata title="Jev-PHM draft" -o /tmp/jev-phm.html
```

The revised manuscript was built successfully with citeproc and MathJax during this revision. No Results section or empirical gain is asserted. The target venue and author approval gates remain unselected/unapproved. The original PaperTrace framework is retained; work through a branch and PR into `dev`, without modifying `master` or force-pushing.
