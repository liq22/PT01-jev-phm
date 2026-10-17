---
name: 15-figure-table-design
description: Plan, generate, or revise one scientific figure or table by fixing the reader question, data comparison, message, and interpretation boundary before styling.
---

# 15 Figure and Table Design

## Purpose

Create a visual asset that helps the reader understand a scientific comparison or
mechanism. The actual figure/table and caption are the product.

## Workflow

1. State the reader question and one evidence-supported message.
2. Fix the source data, comparison, units, uncertainty, and interpretation
   boundary.
3. Choose the simplest encoding that answers the question.
4. For creation or revision, change the actual asset and caption. For explicit
   audit, inspect the supplied asset and return located findings without edits.
   Use requested `nature-figure` assistance through
   [task-specific selection](../../references/task_specific_skill_selection.md),
   retaining the existing source and plotting backend.
5. Open it and check values, labels, units, legend, readability, and whether the
   visual exaggerates the conclusion.
6. Stop when the scientific message is clear.

For a planning-only request, provide the panel structure and exact data needs. For
an explicit review, report only defects that change understanding or publication
quality.

## Output Contract

Produce one of:

- a bounded design plan;
- an actual editable figure/table plus export and caption;
- a revised asset with concrete corrections.
- located findings for an explicitly read-only figure audit.

GPT Image may supply a requested motivation/framework composition draft when
available; verify its topology before rebuilding precise editable objects.
Quantitative figures use actual data and plotting code. Check panel roles, shared
scales, final-size alignment and collisions as relevant. Verify Visio/Illustrator
format and editability in the actual available application; an SVG export alone
does not establish native-object round-trip editing. Report source reading,
generation and executed QA separately.

## Boundaries

- Do not replace generation with a manifest entry or design report.
- Do not tune color, spacing, font, borders, or decoration before data and message
  are correct.
- Do not use Python merely to scan visual formatting; use it when generating or
  analyzing the actual data/asset.
- Do not require hashes, prior-card quotas, visual audit matrices, or repeated
  screenshot comparisons.
- Do not add panels that do not answer a reader question.
