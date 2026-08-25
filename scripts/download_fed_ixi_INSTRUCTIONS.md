# Downloading Fed-IXITiny (human action required)

Verified 2026-08-25 by fetching `owkin/FLamby`'s
`flamby/datasets/fed_ixi/dataset_creation_scripts/download.py`: the script
calls FLamby's own `accept_license("https://brain-development.org/ixi-dataset/",
"fed_ixi")` helper before downloading anything. This is a **programmatic
consent prompt** (not a website click-through like Fed-ISIC2019's), but it
is still a human-consent gate — the script will interactively ask whoever
runs it to accept the CC BY-SA 3.0 IXI license before it proceeds. The
agent will not run this and answer "yes" on a human's behalf, for the same
reason it will not accept the ISIC2019/HAM10000 licenses (see
`scripts/download_fed_isic2019_INSTRUCTIONS.md`).

## Procedure

1. Clone FLamby if you have not already (see the ISIC instructions file
   for the exact command).
2. Install the `fed_ixi` extras (check FLamby's README for the current
   extras-group name; it also depends on TorchIO for preprocessing).
3. Run the official downloader and answer its license prompt yourself:
   ```
   python flamby/datasets/fed_ixi/dataset_creation_scripts/download.py \
       --output-folder /absolute/path/outside/this/repo/fed_ixi_raw
   ```
4. Do not point the output folder inside this git repo (same reasoning as
   the ISIC instructions — `.gitignore` covers the conventional locations
   but not `git add -f`).
5. Once downloaded, Phase 7 (`docs/execution_plan.md`) can start — but
   only after Phase 1's real Fed-ISIC2019 primary tables are complete, per
   the project's phase ordering (Fed-IXITiny is the mandatory *low-cost
   extension*, run after the primary 2D classification study, not
   alongside or before it).

## Status

As of 2026-08-25: **not yet performed**, and not yet due — Phase 1's
real-data tier itself is still `BLOCKED-LICENSE`
(`scripts/download_fed_isic2019_INSTRUCTIONS.md`), which comes first.
