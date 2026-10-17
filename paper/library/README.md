# Jev-PHM literature library

The complete 20-item reading list is maintained in
[jev_generalist_20.md](../refs/jev_generalist_20.md). This directory contains
publicly redistributable PDFs and their source, version and license metadata.
The archive statistics below count this public subset, not all locally acquired
papers. The acquisition snapshot is [references.json](jev-generalist-20/references.json).

Restricted or unconfirmed-license PDFs remain in the ignored machine-local
archive at `.venv/paper-curator/library/jev-generalist-20/`. Institutional access
does not itself establish permission to publish a PDF here. Each included paper
retains its original copyright and license; the repository's code license does
not replace them. CC BY-NC copies are provided for noncommercial research, and
all PDFs are unmodified. Per-paper attribution and license sources are recorded
in the manifest and acquisition snapshot.

## Reproduce acquisition and archiving

Run from the repository root. Supply the original user-provided
`.venv/project-paper-curator.zip`; it is not redistributed with this repository.
The ZIP's unmodified `verify_pdf.py` and `archive_papers.py` are the actual
verification and archive tools. Their dependency versions are pinned separately
from the model-experiment environment.

```bash
python3.10 -m venv .venv/paper-curator/runtime
.venv/paper-curator/runtime/bin/python -m pip install \
  -r paper/library/jev-generalist-20/requirements.txt
bash src/S03_Scripts/curate_jev_library.sh --download
```

The download option uses only the stable public PDF URLs in the curated records;
it does not obtain institutional access, log in, or download replacement works.
Existing inputs are retained. Downloads go to
`.venv/paper-curator/downloads/R01.pdf` through `R20.pdf`; failed `.part` files
are never archived. HSE was reused from an existing local publisher copy and has
no public download URL in this batch. A clean checkout therefore needs that
authorized local input as well as any still-missing institutional papers.

To verify and archive existing inputs without downloading:

```bash
bash src/S03_Scripts/curate_jev_library.sh
```

The script runs the ZIP verifier for each actual input, then the original
archiver's dry run and archive commands. Each input must also match the PDF
SHA256 already returned by the skill for the reviewed version in
`references.json`; a different or previously unreviewed file stops archiving
until its version and license have been checked and its record updated. This
uses the existing PDF verification result, not a separate digest system.
Complete local results, including
`manifest.csv`, `missing.csv`, PDF checks and original-number links, stay under
`.venv/paper-curator/`. The public output includes only records whose exact
version has a verified sharing license. `references.json` describes the observed
2026-10-14 acquisition; a reproduction's actual outcomes are its generated
manifests, not that historical snapshot.

The original verifier checks signatures, strict parsing, independent PDFium
page counts/dimensions, and first-two-page title/author matching. Its title
threshold is a screening rule, and an invisible or mismatched DOI does not
automatically fail its text pass. This batch additionally checked the visible
authors and DOI or the official source-to-PDF relationship. New inputs require
that same review; never raise a score or assert a visual check to bypass a
conflict. Update actual version metadata before recovering a missing entry;
coordinate version replacement explicitly instead of silently upgrading an
already archived UID.

## Institutional recovery

R09 and R10 remain missing. Use normal authorized ScienceDirect access or a
verified author manuscript. R09's matching SSRN deposit is recorded as an
alternative, but its PDF has not been acquired; the older arXiv FD-LLM is not
treated as an established replacement.

The ZIP specifies the optional live-session fetcher commit
`0a24d96b0044e0be03efa2f1daafa84f81a9ebb9` from
[sciencedirect-live-session-fetcher](https://github.com/Given-Dream/sciencedirect-live-session-fetcher/tree/0a24d96b0044e0be03efa2f1daafa84f81a9ebb9).
Its prescribed script was checked against the checksum supplied in the skill.
It was not executed: this environment had no graphical login session and no
task-specific authenticated browser endpoint. No credentials or browser session
files are included. An authorized recovery uses a dedicated local browser,
normal user login, serial requests with a 6-second interval, and at most two
transient retries; authentication challenges pause that channel.

## Verification performed

All 18 available PDFs passed the original verifier and the independent PDFium
page checks. First-page/two-page identity and exact-version license review were
also performed; this is not evidence of full-paper reading. The ZIP's 18 local
helper regression tests passed. Font-encoding and nonzero-xref warnings were
retained; they did not cause a structure or identity failure. See the generated
manifests for per-file results and reasons.

<!-- project-paper-curator:start -->
## 文献归档更新

引用 10 条，独立论文 10 篇；取得 10 篇全文，归档 10 份项目 PDF；待补 0 篇、0 条引用。

[完整清单](manifest.csv) · [待补清单](missing.csv) · [项目入口](CURATION_OVERVIEW.md)。
<!-- project-paper-curator:end -->
