---
name: umap-learn
description: Compute a bounded UMAP embedding or transform from supplied feature data when nonlinear neighborhood structure is relevant. Use as supporting exploratory or preprocessing work, never as proof of clusters, causality, or model quality.
---

# UMAP

## Purpose

Produce an actual UMAP embedding or transform for a defined exploratory,
visualization, clustering-preparation, or feature-engineering question. The
selected primary Skill owns the scientific claim and decides whether the result is
evidence, exploration, or a negative finding.

## Workflow

1. Confirm the role of the embedding, feature semantics, sample independence,
   train/validation/test boundary, preprocessing, distance metric, and output
   needed. Stop when any of these are ambiguous.
2. Fit only on the data allowed by the protocol. For downstream held-out
   evaluation, fit the reducer on training data and transform later data without
   exposing labels or test information.
3. Choose parameters from the current data scale and question rather than a fixed
   recipe. Record the material choices and a random seed.
4. Compute the embedding or transform. Add a stability, negative-control, or
   alternative view only when it can change the interpretation.
5. Return the actual coordinates, table, or figure input and state what the
   geometry does and does not establish.
6. Validate finite inputs, output shape, split isolation, and repeatability under
   the recorded seed. Report collapsed, fragmented, unstable, or null structure
   without seeking approval to preserve the preferred story.

## Output Contract

Produce:

- the requested embedding or transformed features;
- material preprocessing, metric, parameter, seed, and split information;
- the direct validation result;
- an explicit boundary separating visualization from cluster, performance, or
  causal claims.

## Boundaries

- Visual separation is not proof of natural classes, mechanism, or predictive
  value.
- Do not fit on combined train and test data when the embedding feeds a held-out
  claim.
- Do not silently fall back to PCA, t-SNE, another metric, or a different
  preprocessing rule.
- Do not impose universal neighbor counts, distances, component counts, package
  versions, or diagnostic thresholds.
- Do not require deleted insight, ablation, reproducibility, statistics,
  decision-log, or dead-end files.
- Do not create parameter registries, sweep frameworks, hashes, checksums, or
  approval gates for null or unfavorable results.
