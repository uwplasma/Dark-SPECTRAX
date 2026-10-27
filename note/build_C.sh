#!/bin/bash
# Build the note-C chapters: each chapter in a minimal wrapper (cross-chapter references show ??),
# plus the combined noteC.tex (all five files, correct references). Run from note/.
set -e
cd "$(dirname "$0")"
for ch in ch8_hhs ch9_nonrel ch10_rel ch11_discussion appA_repro; do
  cat > wrap_$ch.tex <<EOF
\documentclass[11pt,a4paper]{report}
\usepackage[a4paper,margin=2.6cm]{geometry}
\usepackage[htt]{hyphenat}
\usepackage{amsmath,amssymb,bm,graphicx,booktabs,hyperref,natbib}
\setlength{\emergencystretch}{3em}
\bibliographystyle{unsrtnat}
\input{macros_C}
\begin{document}
\input{$ch}
\bibliography{refs_C}
\end{document}
EOF
  pdflatex -interaction=nonstopmode wrap_$ch.tex >/dev/null; bibtex wrap_$ch >/dev/null || true
  pdflatex -interaction=nonstopmode wrap_$ch.tex >/dev/null; pdflatex -interaction=nonstopmode wrap_$ch.tex >/dev/null
done
cat > noteC.tex <<EOF
\documentclass[11pt,a4paper]{report}
\usepackage[a4paper,margin=2.6cm]{geometry}
\usepackage[htt]{hyphenat}
\usepackage{amsmath,amssymb,bm,graphicx,booktabs,hyperref,natbib}
\setlength{\emergencystretch}{3em}
\bibliographystyle{unsrtnat}
\input{macros_C}
\title{Dark-SPECTRAX explanatory note: chapters 8--11 and Appendix A (writer C)}
\author{Rogerio Jorge}
\begin{document}
\maketitle
\part{Results}\label{part:results}
\input{ch8_hhs}\input{ch9_nonrel}\input{ch10_rel}\input{ch11_discussion}
\appendix
\input{appA_repro}
\bibliography{refs_C}
\end{document}
EOF
pdflatex -interaction=nonstopmode noteC.tex >/dev/null; bibtex noteC >/dev/null || true
pdflatex -interaction=nonstopmode noteC.tex >/dev/null; pdflatex -interaction=nonstopmode noteC.tex >/dev/null
