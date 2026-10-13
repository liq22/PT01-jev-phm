"""Build an offline HTML/PDF preview from the sole Markdown manuscript.

This produces derived previews, never a second editable manuscript or TeX source.
Pandoc resolves citations and embeds local figures; Chromium renders MathML on CPU.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]

CSS = """
@page { size: A4; margin: 19mm 19mm 19mm 21mm; }
html { color: #171717; font: 11pt/1.45 'Liberation Serif', 'Times New Roman', serif; }
body { margin: 0 auto; max-width: 175mm; }
h1 { font-size: 19pt; line-height: 1.18; margin: 0 0 6mm; }
h2 { font-size: 14pt; margin: 5mm 0 2.5mm; }
h3 { font-size: 12pt; margin: 4mm 0 2mm; }
h1,h2,h3,h4 { break-after: avoid; }
p { margin: 0 0 2.8mm; orphans: 3; widows: 3; }
a { color: #154c70; text-decoration: none; overflow-wrap: anywhere; }
figure { margin: 5mm 0; break-inside: avoid; }
img,svg { display: block; margin: 0 auto; max-width: 100%; height: auto; }
figcaption { font-size: 9.5pt; line-height: 1.3; margin-top: 2mm; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 4mm 0; }
thead { display: table-header-group; }
th,td { padding: 1.7mm 1.5mm; border-bottom: 0.3pt solid #b7b7b7; vertical-align: top; }
th { border-top: 0.6pt solid #171717; border-bottom: 0.6pt solid #171717; }
tr { break-inside: avoid; }
math[display='block'] { margin: 3.5mm 0; overflow: visible; font-size: 11pt; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 8pt; line-height: 1.3; }
code { font-size: 0.87em; }
.references { font-size: 9.5pt; }
.csl-entry { margin-bottom: 2mm; }
@media screen { body { padding: 25px; } }
"""


def executable(explicit: str | None, name: str, bundled: list[Path]) -> Path:
    """Resolve an installed binary without installing dependencies."""
    candidates = [Path(explicit).expanduser()] if explicit else []
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))
    candidates.extend(bundled)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError(f"No usable {name} binary; supply --{name}")


def build(draft: Path, bibliography: Path, out_dir: Path, pandoc: Path, chromium: Path) -> tuple[Path, Path]:
    if not draft.is_file() or not bibliography.is_file():
        raise FileNotFoundError("The Markdown manuscript and bibliography must exist")
    out_dir.mkdir(parents=True, exist_ok=True)
    html = out_dir / "jev-phm-preview.html"
    pdf = out_dir / "jev-phm-preview.pdf"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    # Embedding resources makes the preview portable and avoids external assets.
    with tempfile.TemporaryDirectory(prefix="jev-preview-") as temporary:
        css = Path(temporary) / "preview.css"
        css.write_text(CSS, encoding="utf-8")
        command = [str(pandoc), str(draft), "--citeproc", "--bibliography", str(bibliography),
                   "--standalone", "--mathml", "--embed-resources", "--css", str(css),
                   "--resource-path", os.pathsep.join((str(draft.parent), str(ROOT))),
                   "--metadata", "pagetitle=Jev-PHM manuscript preview", "-o", str(html)]
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=True)
        if result.stderr.strip():
            raise RuntimeError(f"Pandoc reported unresolved build concerns:\n{result.stderr}")
    command = [str(chromium), "--headless", "--no-sandbox", "--disable-gpu",
               "--disable-dev-shm-usage", "--no-pdf-header-footer",
               f"--print-to-pdf={pdf}", html.as_uri()]
    subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=True)
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise RuntimeError("Chromium did not produce a readable PDF preview")
    return html, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, default=ROOT / "paper/draft/main.md")
    parser.add_argument("--bibliography", type=Path, default=ROOT / "paper/refs/references.bib")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "paper/preview")
    parser.add_argument("--pandoc")
    parser.add_argument("--chromium")
    args = parser.parse_args()
    pandoc = executable(args.pandoc, "pandoc", list(ROOT.glob(".venv/lib/python*/site-packages/pypandoc/files/pandoc")))
    chromium = executable(args.chromium, "chromium", sorted(Path.home().glob(".cache/ms-playwright/chromium-*/chrome-linux64/chrome")))
    html, pdf = build(args.draft.resolve(), args.bibliography.resolve(), args.out_dir.resolve(), pandoc, chromium)
    print(f"Derived HTML: {html}\nDerived PDF: {pdf}\nAuthoritative manuscript: {args.draft.resolve()}")


if __name__ == "__main__":
    main()
