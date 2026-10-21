# Manuscript build

Target: Physical Review E (REVTeX 4-2, `aps,pre,reprint`).

Requirements: Python 3 with numpy, scipy, matplotlib; a TeX Live with revtex4-2.

```
python paper/make_figures.py        # from the repository root; writes figures/*.pdf, numbers.tex, tables_gen.tex
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex cover_letter
```

Every number quoted in the text comes from `numbers.tex` or `tables_gen.tex`, generated from the committed records in `studies/` and `docs/_static/`.
Records and study scripts are those of commit `a3ef7a6`; the manuscript follows it on `main` (see `git log -- paper`).
SPECTRAX integration commits (branch `integration/dark-baseline`, unmerged, not a release): reservoir and coherence study `3de6b2b`, strong-drive study `4090fbe`, collisions `35231ab`; branch tip when written `729d7fe`.
The acknowledgments line is a placeholder for the author to complete.
