# Derived manuscript preview

`jev-phm-preview.pdf` and the standalone `jev-phm-preview.html` render
[`../draft/main.md`](../draft/main.md), the sole authoritative manuscript.
These files are readable exports, not a second text source or a journal-approved
submission format. Edit the Markdown and regenerate them; do not edit exports.

From the repository root:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python src/S03_Scripts/build_jev_paper.py
```

The builder uses installed Pandoc with citeproc, the project bibliography and
embedded local figures, then local headless Chromium with GPU use disabled.
MathML renders formulas offline; no remote assets or TeX manuscript are created.
The default destination is this directory. Explicit binary locations can be
supplied using `--pandoc` and `--chromium`.
