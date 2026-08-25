# Reproducing the LaTeX toolchain

`paper/main.tex` needs `pdflatex` + `bibtex` and the packages: `hyperref`,
`booktabs`, `geometry`, `xcolor`, `natbib`, `enumitem`, `caption` (plus
whatever `amsmath`/`graphicx`/base LaTeX classes scheme-basic already
ships).

This project's TeX Live is a **user-space** install (no `sudo`, since the
CI/dev sandbox doesn't have it) — not part of `requirements.lock.txt`
because it isn't a Python package. To reproduce:

```bash
curl -sL https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz -o install-tl.tar.gz
tar xzf install-tl.tar.gz && cd install-tl-*/
cat > texlive.profile << 'EOF'
selected_scheme scheme-basic
TEXDIR ~/texlive
TEXMFCONFIG ~/.texlive/texmf-config
TEXMFHOME ~/texmf
TEXMFLOCAL ~/texlive/texmf-local
TEXMFSYSCONFIG ~/texlive/texmf-config
TEXMFSYSVAR ~/texlive/texmf-var
TEXMFVAR ~/.texlive/texmf-var
option_doc 0
option_src 0
option_autobackup 0
portable 0
EOF
perl ./install-tl -profile texlive.profile   # ~2 minutes for scheme-basic
export PATH="$HOME/texlive/bin/x86_64-linux:$PATH"   # add to shell profile
tlmgr install hyperref booktabs geometry xcolor natbib url enumitem caption subcaption
```

Then `bash scripts/compile_paper.sh` from the repo root. Verified clean
(0 errors, no undefined references/citations after the standard
pdflatex/bibtex/pdflatex/pdflatex sequence) on 2026-08-25 on the hardware
in `docs/hardware_report.md`.

If a system-wide TeX Live is already available (e.g. via `apt install
texlive-latex-extra` with sudo), skip the above and just make sure the
package list above is present.
