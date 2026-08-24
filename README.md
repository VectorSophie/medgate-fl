# MedGate-FL

Capability-isolated federated adapters for tiered medical AI access — a
reproducible study of whether a federated medical model can keep a public
coarse-grained capability useful while isolating a fine-grained diagnostic
capability inside a restricted, encrypted adapter.

This is an active research project, not a finished result and not a
clinical tool. Start here:

- `docs/research_scope.md` — research questions, threat model, claims and
  **non-claims**, verified dataset facts, hardware constraints, limitations.
- `docs/execution_plan.md` — staged phases (0-8), current status, named
  blockers.
- `docs/literature_matrix.csv` / `.md` — every source this project cites or
  might cite, each marked `VERIFIED` or `UNVERIFIED` against a primary
  source. Only `VERIFIED` rows are cited in `paper/main.tex`.
- `docs/evidence_ledger.csv` — for every empirical claim drafted into the
  paper, which source and location backs it.
- `docs/hardware_report.md` — the actual machine this ran on; every
  resource decision traces back to this.

## Repository layout

```
configs/            experiment configs (one file per run)
data_manifests/      versioned dataset manifests + leakage audits (no raw data)
docs/                research scope, literature matrix, evidence ledger, plans
experiments/         per-run outputs (config+seed+commit+manifest hash+metrics)
logs/                raw logs
medgate/             the actual library code (data, models, federated, attacks, ...)
paper/               main.tex, references.bib, figures/, tables/
scripts/             setup, download instructions, hardware report, smoke test
tests/               the one-runnable-check per non-trivial module
```

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python scripts/report_hardware.py     # confirms what you're actually running on
```

Real Fed-ISIC2019 imagery requires a human to accept the ISIC2019/HAM10000
licenses first — see `scripts/download_fed_isic2019_INSTRUCTIONS.md`. Until
that happens, everything runs on synthetic fixtures that mirror its shape
(6 centers, 8 classes), and every result produced that way is labeled
synthetic wherever it appears.

## Smoke test

```
bash scripts/smoke_test.sh
```

## Status

Phase 0 (repo/data validation) is in progress. See
`docs/execution_plan.md` for exact status of every phase — nothing here is
claimed finished until its checkbox is.
