---
name: seaborn
description: Render a specified statistical view from tidy data when seaborn's distribution, categorical, relational, regression, or faceting APIs materially simplify the figure. Use as a supporting renderer, not as an inference or figure-planning system.
---

# Seaborn

## Purpose

Create an actual statistical plot from supplied tidy data when seaborn reduces the
rendering code without obscuring the analysis. The selected primary Skill owns the
scientific question, estimand, comparison, and caption boundary.

## Workflow

1. Confirm the reader question, variables, groups, independent unit, and supplied
   data. Stop if rows, groups, repeats, or missing values have ambiguous meaning.
2. Select the simplest plot family that answers the question: distribution,
   categorical comparison, relationship, regression diagnostic, matrix, or
   faceted small multiples.
3. Set estimators and uncertainty explicitly when summaries are shown. Do not use
   a default interval whose statistical meaning differs from the analysis.
4. Preserve individual observations, pairing, clustering, or repeated structure
   when they matter. Do not let aggregation make dependent samples appear
   independent.
5. Render the requested asset and hand off to matplotlib only for local axes-level
   control; do not switch libraries silently because a call fails.
6. Open the result and check values, units, group labels, uncertainty meaning,
   overplotting, and whether the visual supports only the stated conclusion.

## Output Contract

Produce:

- the requested statistical figure;
- minimal plotting code when regeneration is part of the product;
- the explicit estimator and uncertainty definition, when used;
- one direct visual check and the interpretation boundary.

## Boundaries

- Do not use a plot as a substitute for statistical analysis or held-out
  evaluation.
- Do not create a confidence interval, regression line, or aggregate merely
  because the API provides a default.
- Do not fabricate data, hide null groups, or wait for approval before reporting
  an unfavorable pattern.
- Do not require deleted statistics, insight, change-log, or decision-log files.
- Do not impose fixed library versions, universal themes, palette rituals, or
  multi-format exports without a current need.
- Do not create wrappers, registries, style scorecards, hashes, checksums, or
  provenance packets around a plot.
