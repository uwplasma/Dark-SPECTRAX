#!/bin/sh
# usage: buildB.sh CHAPNUM chfile   (minimal wrapper, compiles one chapter)
cd "$(dirname "$0")"
n=$1; f=$2
sed -e "s/CHAPFILE/$f/" -e "s/\\\\tableofcontents/\\\\setcounter{chapter}{$((n-1))}/" wrapB.tex > _wrap_$f.tex
pdflatex -interaction=nonstopmode _wrap_$f.tex >/dev/null; bibtex _wrap_$f >/dev/null
pdflatex -interaction=nonstopmode _wrap_$f.tex >/dev/null; pdflatex -interaction=nonstopmode _wrap_$f.tex | grep -E "Overfull|Undefined|undefined|Error|Output written" | head -20
