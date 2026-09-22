---
name: networkx
description: Build or analyze an explicitly defined graph with NetworkX using classical graph structures and algorithms. Use as a supporting implementation capability after node, edge, direction, weight, and analysis semantics are known.
---

# NetworkX

## Purpose

Construct a graph, compute requested classical graph quantities, or render a
bounded topology view from supplied data. This Skill does not define the research
question, claim, or causal interpretation.

## Workflow

1. Confirm what nodes and edges represent, whether direction, parallel edges,
   weights, time, or missing links matter, and which data source is authoritative.
   Stop when those semantics are ambiguous.
2. Choose the graph type that directly matches the relationship. Do not collapse
   direction, multiplicity, or weights for convenience.
3. Run only the algorithm needed for the current decision. State assumptions that
   affect shortest paths, centrality, communities, flow, connectivity, or a null
   graph.
4. Set and report a seed for stochastic generation, layout, or approximation.
   Keep exact and approximate results distinguishable.
5. Return the requested metrics, graph artifact, table, or topology figure. Empty,
   disconnected, unstable, or contradictory results remain valid results.
6. Validate with the smallest direct check: recompute a known small case, inspect
   graph counts and attributes, or reopen the serialized/visual artifact.

## Output Contract

Produce the requested graph product and include:

- node and edge semantics;
- algorithm and material options;
- seed or approximation boundary when relevant;
- the actual result and one direct validation;
- no extra workflow or logging packet.

## Boundaries

- Do not infer causality or mechanism from topology alone.
- Do not invent nodes, edges, weights, directions, or missing relationships.
- Do not silently substitute a different backend, approximation, sampling rule,
  or graph type when the requested computation is infeasible.
- Do not require deleted statistics, insight, ablation, change-log, or
  decision-log files.
- Do not require fixed NetworkX or Python versions unless the actual code uses a
  version-specific API.
- Do not create graph factories, backend registries, hashes, checksums, or a
  general graph pipeline for a bounded analysis.
