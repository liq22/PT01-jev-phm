---
name: matplotlib
description: Render a specified static scientific figure with matplotlib when low-level control over axes, annotations, layout, or export is required. Use only as a supporting renderer after the figure question and source data are defined.
---

# Matplotlib

## Purpose

Render an actual static figure from verified data or results. This Skill provides
low-level control over axes, marks, annotations, layout, and export; the selected
primary Skill remains responsible for the scientific question, figure message,
and caption boundary.

## Workflow

1. Read the figure question, source data or result, and requested output. Confirm
   variable meaning, units, aggregation, independent unit, and uncertainty. Stop
   when any of these would change the interpretation.
2. Choose the simplest faithful marks and axes. Use the object-oriented API so
   data flow and panel ownership remain explicit.
3. Plot the supplied values without smoothing, rescaling, filtering, or
   aggregation that was not part of the stated analysis. Distinguish observations,
   summaries, and uncertainty visually when they answer different questions.
4. Add only labels, legends, reference lines, and annotations needed to interpret
   the result. Use redundant encodings when color alone would be ambiguous.
5. Export only the format and size needed by the current product. Use a
   non-interactive backend in headless environments only when it preserves the
   requested file output, and report a rendering failure rather than changing
   libraries silently.
6. Open the exported asset and inspect data values, labels, units, clipping,
   readability, and correspondence with the source result.

## Output Contract

Produce:

- the requested figure asset;
- the minimal rendering source when the user or repository needs an editable
  regeneration path;
- a concise statement of the source data/result and the direct visual check;
- no additional manifest, style report, or process document.

## Boundaries

- Do not invent the scientific message, select a more favorable subset, or alter
  values to improve appearance.
- Do not infer causality, significance, or generality from the plot.
- Do not impose a journal style, color palette, DPI, or multi-format export unless
  the task or verified venue requirement needs it.
- Do not create run-ledger, evidence-matrix, change-log, hash, checksum, or receipt
  entries merely because a figure was rendered.
- Do not add a wrapper, plotting factory, theme registry, or fallback chain for a
  one-off figure.
- Preserve null, negative, unstable, and boundary results in the visual encoding.
