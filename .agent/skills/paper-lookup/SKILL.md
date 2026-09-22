---
name: paper-lookup
description: Retrieve verified scholarly metadata, identifiers, or open-access locations from an appropriate academic source. Use as a supporting lookup capability; literature synthesis, novelty judgment, and bibliography decisions remain with the selected primary Skill.
---

# Paper Lookup

## Purpose

Resolve a bounded scholarly lookup using an authoritative database or publisher
record. The product is a verified bibliographic record, identifier mapping, search
result set, or accessible full-text location—not a literature review or a raw-search
archive by default.

## Workflow

1. State the exact lookup need: known identifier, title, author, topic, date range,
   citation relation, or open-access location.
2. Select the smallest suitable source. Prefer the original publisher, DOI
   registry, discipline database, or official repository that directly answers the
   request; use multiple databases only when coverage disagreement matters.
3. Confirm identifier format and query scope before the network call. Read
   credentials only from the user-provided environment and never echo them.
4. Make the bounded request while respecting the source's current rate limits and
   terms. Do not automatically retry across a chain of databases; report a failure
   or choose one explicit alternative that can resolve the same object.
5. Verify load-bearing fields against the returned record: title, authors, year,
   venue, identifier, version, and access location as applicable.
6. Return zero results, conflicting records, unavailable full text, and uncertain
   metadata explicitly. Save raw responses only when the user requests them or the
   search protocol itself is part of the research method.

## Output Contract

Produce:

- the verified record or bounded result set;
- the source/database, query or identifier, and lookup date;
- any conflict, missing field, access boundary, or zero-result outcome;
- a handoff-ready record for literature synthesis or bibliography maintenance.

## Boundaries

- Do not synthesize themes, decide novelty, rank authors or venues, or draft
  manuscript prose.
- Do not infer full-text methods or results from metadata or search snippets.
- Do not require exhaustive multi-database searching for a bounded lookup.
- Do not save every response, create a search ledger, or write decision, dead-end,
  insight, open-question, reproducibility, hash, checksum, or receipt records.
- Do not invent or persist API keys, identifiers, metadata, or access links.
- Do not silently change the query, substitute a different research object, or
  bypass a paywall or source terms.
