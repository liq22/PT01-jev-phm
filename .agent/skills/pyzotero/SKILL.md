---
name: pyzotero
description: Read, search, export, or explicitly update a user-authorized Zotero library through pyzotero. Use as a supporting library bridge when the user supplies the library scope and credentials; do not use for de novo literature search or citation verification.
---

# Pyzotero

## Purpose

Move a clearly scoped set of records between an existing Zotero library and the
current paper workspace. Zotero is the source library; this Skill does not invent,
verify, or reinterpret bibliographic metadata.

## Workflow

1. Confirm the requested operation, library type, collection/search/tag scope,
   destination, and whether the action is read-only or writes remotely.
2. Read credentials only from the user-provided environment. Stop when the
   library ID, API key, local Zotero availability, or collection scope required by
   the operation is missing.
3. For reads, retrieve only the scoped items and preserve Zotero item keys and
   available metadata. Do not fill absent fields by guessing.
4. For export, write only the requested BibTeX, CSL-JSON, shortlist, or attachment
   set. Hand metadata repair and duplicate resolution to citation management.
5. For create/update/upload/delete operations, require that exact remote action to
   be explicitly requested. Show the intended scope before destructive or bulk
   writes and do not broaden it.
6. Validate the actual item count, target collection, output readability, and
   absence of credentials in generated files.

## Output Contract

Produce the requested Zotero-backed product, such as:

- a scoped bibliography export;
- a reading shortlist with Zotero item keys;
- explicitly requested attachments;
- the result of an authorized item create/update operation;
- a concise list of missing or ambiguous fields.

## Boundaries

- Do not perform de novo literature search, thematic synthesis, DOI verification,
  or venue-based citation selection.
- Do not hardcode, echo, log, or commit credentials or private library identifiers.
- Do not create, update, upload, delete, or trash remote items unless that exact
  operation is explicit.
- Do not require change logs, decision logs, open-question files, freeze packets,
  hashes, checksums, or receipts.
- Do not impose a fixed pyzotero or Zotero version unless the requested API call is
  version-specific.
- Do not silently switch between local and Web API modes or substitute another
  collection when the requested scope is missing.
