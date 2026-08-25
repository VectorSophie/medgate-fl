#!/usr/bin/env bash
# Regenerate every table/figure from committed experiment JSON, then
# compile paper/main.tex. Verified end-to-end on a fresh user-space TeX
# Live install during this project (scheme-basic + hyperref, booktabs,
# geometry, xcolor, natbib, enumitem, caption) — see
# docs/reproducibility_texlive.md if pdflatex isn't already on PATH.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "== regenerating tables and figures from experiments/*.json =="
PYTHONPATH=. python scripts/make_phase1_table.py
PYTHONPATH=. python scripts/make_phase1_hierarchical_table.py
PYTHONPATH=. python scripts/make_phase2_table.py
PYTHONPATH=. python scripts/make_phase3_table.py
PYTHONPATH=. python scripts/make_phase3_adapter_recovery_table.py
PYTHONPATH=. python scripts/make_phase3_integrity_table.py
PYTHONPATH=. python scripts/make_phase4_table.py
PYTHONPATH=. python scripts/make_phase5_table.py
PYTHONPATH=. python scripts/csv_to_latex_tables.py
PYTHONPATH=. python scripts/make_figures.py

echo "== compiling paper/main.tex =="
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
echo "== done: paper/main.pdf =="
