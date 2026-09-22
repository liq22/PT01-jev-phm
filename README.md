# Jev-PHM

## Research question

### Typed decisions, tested rather than assumed

Apply the public Jev idea of state-to-typed-probabilistic decisions to closed-set fault diagnosis. The first probe uses **PHMFactory + Qwen/Qwen3-0.6B**, not proprietary Jev weights, RLCD, or an external Jev API.

The candidate PHM adaptation fits temperature calibration with equal weight per verified independent unit. Compare it against ordinary pooled calibration **on identical logits**, and compare Qwen against a tuned classifier receiving the same six numeric features. These are hypotheses, not demonstrated improvements. CAT and CLSGen already establish important calibration and language-model-head prior art; see the [source assessment](paper/refs/reading_matrix.md).

## Current implementation

### One minimal path

```text
PHMFactory windows -> six deterministic features -> frozen Qwen3 embedding
-> tuned logistic head -> raw / pooled / unit temperature
-> categorical probabilities -> diagnosis or deferral
```

The numeric control bypasses Qwen and shares the downstream implementation. Positive scalar temperature cannot change class argmax. The first probe is single-channel, closed-set classification; it makes no RUL, open-set, causal explanation, or operational safety claim.

### Files to use

| Product | Location |
|---|---|
| Unique manuscript, including problem formulation and method | [paper/draft/main.md](paper/draft/main.md) |
| Existing research state | [paper/paper.yaml](paper/paper.yaml) |
| Existing contribution/evidence mapping | [paper/experiments/evidence_matrix.md](paper/experiments/evidence_matrix.md) |
| Local execution Goal | [paper/experiments/jev_goal.md](paper/experiments/jev_goal.md) |
| Sole probe implementation | [src/S01_Package/jev_phm.py](src/S01_Package/jev_phm.py) |
| Fixed first-run search and decision settings | [src/S02_Configs/jev_probe.json](src/S02_Configs/jev_probe.json) |

## Run and validate

### Lightweight checks

Use an isolated environment compatible with the existing PHMFactory installation:

```bash
python -m pip install -r src/S02_Configs/jev_requirements.txt
python -m unittest discover -s src/S04_Tests -p test_jev.py -v
python src/S01_Package/jev_phm.py --help
```

The nine numerical/interface tests passed locally using synthetic fixtures. They do **not** establish real Qwen inference, real PHMFactory data integration, or method effectiveness. The complete inherited PaperTrace suite has not been run in the local authoring environment; the PR's existing CI remains a separate check.

### Real experiment prerequisites

Follow the Goal to locate the actual PHMFactory configuration and metadata, verify labels and independent units, and resolve an immutable Qwen revision. No dataset path or label mapping is supplied as a guess. The adapter uses the inspected upstream `build_data(args_data, args_task)` at commit `cdc0669167ca25ac848e34c880875ca39c6aed89`; it does not duplicate a loader or silently change a split.

The commands are `export`, `embed`, and `run`. Their parsers and numerical path are tested; data export and model forward pass still require local end-to-end validation. Preserve failed runs, do not select experiments for positive outcomes, and do not commit large or sensitive raw artifacts.

## Manuscript and project boundary

### Evidence before conclusions

The draft contains Introduction, Related Work, Motivation, Contributions, formulation, method, and the minimal experimental protocol. It deliberately has no Results section. A manuscript preview can be built from the repository root with installed Pandoc:

```bash
pandoc paper/draft/main.md --citeproc --bibliography paper/refs/references.bib \
  --mathjax --standalone --metadata title="Jev-PHM draft" -o /tmp/jev-phm.html
```

The target venue is unselected; no venue was invented or downgraded. The original PaperTrace agent framework is retained; see [AGENTS.md](AGENTS.md) and [paper/README.md](paper/README.md). Work on a branch and use a PR into `dev`; do not modify `master` or force-push.
