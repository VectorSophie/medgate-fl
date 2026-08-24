# Downloading real Fed-ISIC2019 data (human action required)

This step is **stopped here on purpose**. FLamby's own dataset code will
not download ISIC2019/HAM10000 imagery until a human has accepted both
datasets' licenses in a browser — the agent will not and cannot click
"accept" on your behalf, and does not attempt to work around that gate.

## Exact official procedure (verified against `owkin/FLamby` on GitHub,
`main` branch, 2026-08-25 — re-check this file against the live repo if
much time has passed, sites change)

1. Go to the ISIC Challenge data page and accept the ISIC2019 license
   (CC BY-NC 4.0): https://challenge.isic-archive.com/data/
2. Go to the HAM10000 record on Harvard Dataverse and accept its terms
   (a portion of ISIC2019 is HAM10000-sourced and carries its own terms):
   the Harvard Dataverse HAM10000 record linked from the ISIC page above.
3. Clone FLamby (or install it) so its official download script is
   available:
   ```
   git clone https://github.com/owkin/FLamby.git third_party/FLamby
   cd third_party/FLamby
   pip install -e ".[isic2019]"   # or the relevant extras group; check
                                    # FLamby's README for the current name
   ```
4. Run FLamby's official downloader, pointing at wherever you want the raw
   data stored (**do not** point it inside this git repo — see
   `.gitignore`, which already excludes `data_manifests/raw/` and
   `data/raw/` for this reason):
   ```
   python flamby/datasets/fed_isic2019/dataset_creation_scripts/download_isic.py \
       --output-folder /absolute/path/outside/this/repo/fed_isic2019_raw
   ```
5. Follow FLamby's preprocessing step in the same directory (resize to
   224px short edge, color constancy) per its README — do not hand-roll a
   different preprocessing pipeline; matching FLamby's is what makes the
   official baseline hyperparameters (`docs/research_scope.md` §5)
   meaningful as a comparison point.
6. Record the exact local path in a machine-specific, gitignored config
   (not in any file tracked by this repo) and point
   `configs/phase1_real_fed_isic2019.yaml` (created when Phase 1's real-data
   tier starts) at it.
7. Once downloaded, run the Phase 0 manifest/audit step (to be added:
   `medgate/data/manifest.py`) to build the versioned manifest and leakage
   audit described in the project brief before any training run touches
   the data.

## What NOT to do

- Do not redistribute the downloaded images in this repository, in any
  artifact, or in any external service.
- Do not commit the raw data path's contents; `.gitignore` blocks the
  conventional locations but does not protect against `git add -f`.
- Do not attempt to script around the license-acceptance step (e.g.
  scraping the images from a mirror) — that is a license bypass, out of
  scope for this project regardless of convenience.

## Status

As of 2026-08-25: **not yet performed**. Phase 0/1 proceed on synthetic
fixtures until a human completes steps 1-2 and confirms. See
`docs/execution_plan.md` for what is gated on this.
