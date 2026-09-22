---
name: citation-management
description: Add, verify, repair, de-duplicate, and cross-check bibliography records for the current paper. Use as supporting bibliography maintenance; literature discovery and synthesis belong to the selected research Skill.
---

# Citation Management

## Purpose

Maintain an accurate bibliography that matches the manuscript and the original
publication records. The goal is correct, resolvable citations—not a reference
quota, venue score, or audit package.

## Workflow

1. Read the target bibliography entry, manuscript citation, or bounded set of
   records that needs work. Do not require a target venue or full reading matrix
   when the task is a specific citation fix.
2. Resolve available identifiers against the original publisher, DOI registry,
   discipline database, or official repository. Use title/author/year matching
   when a valid source has no DOI.
3. Preserve verified author order, title, year, venue, pages/article number,
   version, and identifier. Leave genuinely unavailable fields absent rather than
   inventing them or inserting a misleading note.
4. De-duplicate by stable identifiers and normalized bibliographic identity, then
   use one clear citation key per retained work.
5. Cross-check manuscript keys against the bibliography when manuscript text is
   available. Report unresolved and unused entries; remove or retain them according
   to the requested scope rather than a universal rule.
6. Write the corrected bibliography or entry and perform the smallest direct
   check: parse BibTeX, resolve load-bearing identifiers, and confirm changed keys
   match manuscript uses.

## Output Contract

Produce:

- the corrected bibliography or requested records;
- the source used to verify each materially changed entry;
- a concise list of unresolved identifiers, duplicates, or manuscript-key
  mismatches;
- one direct validation result.

## Boundaries

- Do not fabricate citations, metadata, identifiers, page ranges, or publication
  status.
- Do not pad a bibliography to meet a reference-count target or rank sources by
  venue prestige, citation count, or author reputation alone.
- Do not require every valid source to have a DOI, volume, or page range.
- Do not run broad literature searches or write Related Work inside a bibliography
  maintenance task.
- Do not create mandatory JSON reports, enrichment logs, change logs,
  decision logs, open-question files, hashes, checksums, or receipts.
- Do not store API keys or silently replace an unresolved source with a different
  paper.
