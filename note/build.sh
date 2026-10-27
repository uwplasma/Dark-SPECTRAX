#!/bin/sh
# Build the note: pdflatex, bibtex, pdflatex x2.
cd "$(dirname "$0")" && pdflatex -interaction=nonstopmode main >/dev/null && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main | tail -3
