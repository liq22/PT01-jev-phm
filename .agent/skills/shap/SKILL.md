---
name: shap
description: Compute SHAP attributions for an already evaluated predictive model when the output scale, background data, and explanation question are defined. Use for bounded attribution analysis, not causal inference or a substitute for model evaluation.
---

# SHAP

## Purpose

Explain the predictions of a supplied, evaluated model through SHAP attribution.
This Skill provides the implementation and direct checks; the selected primary
Skill owns the scientific claim, model-validity judgment, and manuscript
interpretation.

## Workflow

1. Confirm the model, evaluation data, feature meanings, prediction output scale,
   explanation target, and background/reference data. Stop if probability,
   margin, log-odds, or regression-output semantics are unclear.
2. Select the simplest compatible explainer for the model family. Use a generic
   explainer only when no valid specialized option exists and the computational
   cost is acceptable.
3. Compute attributions on the stated evaluation cases. Preserve the sampling and
   cohort boundary; do not choose only examples that support the preferred story.
4. Check the applicable additivity or reconstruction relation and inspect whether
   the background choice materially changes the conclusion.
5. Produce the requested global or local table/figure and state the model output
   being explained.
6. Report leakage signals, unstable rankings, subgroup differences, null
   attributions, and contradictions directly. Distinguish model behavior from
   domain mechanism or causality.

## Output Contract

Produce:

- the requested attribution values, summary table, or figure;
- explainer, output scale, background-data definition, and evaluated sample
  boundary;
- the direct numerical/visual validation;
- a concise statement of what the attribution supports and does not support.

## Boundaries

- SHAP explains a model's behavior; it does not establish causality, physical
  mechanism, fairness, or scientific truth.
- Do not explain an undefined or unevaluated model as paper evidence.
- Do not train, tune, or modify the model inside an attribution task unless that
  change is separately requested.
- Do not fabricate, smooth, or selectively report attributions.
- Do not require deleted statistics, reproducibility, ablation, change-log,
  decision-log, insight, or reviewer-response files.
- Do not impose universal background sizes, package versions, plotting bundles,
  or output formats.
- Do not add an explanation registry, wrapper hierarchy, hash, checksum, or
  approval gate for unfavorable findings.
