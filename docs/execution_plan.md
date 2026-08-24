# Staged execution plan

Ordered by scientific importance first, computational cost second (a cheap
but foundational check always precedes an expensive but secondary one).
Status legend: `DONE`, `IN PROGRESS`, `PENDING` (not started, blocked on
something specific and named), `BLOCKED-LICENSE` (needs a human to accept a
dataset license), `OUT OF SCOPE (hardware)` (explicitly not attempted on the
measured hardware in `docs/hardware_report.md`).

No phase after Phase 1 starts until Phase 0's smoke tests pass. No phase
after Phase 5 starts until Phases 0-5's primary tables exist for at least
the synthetic/tiny-real tier.

## Phase 0 — repository and data validation
Status: **IN PROGRESS**

- [x] Inspect repo, hardware, network, existing state (none — greenfield).
- [x] `docs/research_scope.md`, `docs/literature_matrix.{csv,md}`,
      `docs/evidence_ledger.csv` schemas + seed content.
- [x] Verify FLamby / Fed-ISIC2019 / Fed-IXITiny facts against primary
      sources (§5-6 of research_scope.md).
- [ ] Tiny synthetic fixture generator (`medgate/data/synthetic.py`):
      6-center, 8-class, image-shaped tensors, deterministic seed.
- [ ] Verify one centralized forward/backward pass on the synthetic fixture
      (`tests/test_phase0_forward_backward.py`).
- [ ] Verify one two-client FL round (`tests/test_phase0_fedavg_round.py`).
- [ ] Verify reproducibility from a clean environment (fresh venv +
      `requirements.lock.txt` + smoke test, documented in
      `scripts/smoke_test.sh`).
- [ ] Real Fed-ISIC2019 checksums/label counts/center counts/duplicates —
      **BLOCKED-LICENSE**: requires a human to accept the ISIC2019 and
      HAM10000 licenses (`scripts/download_fed_isic2019_INSTRUCTIONS.md`).
      Everything downstream of the real download is marked pending until
      that happens.

## Phase 1 — primary utility baseline (Fed-ISIC2019)
Status: **PENDING** (synthetic-tier can start once Phase 0 tests pass;
real-data tier is `BLOCKED-LICENSE`)

Centralized, local-only, FedAvg, FedProx, plain FedLoRA. Spec asks for 5
seeds where feasible, else 3 for development + 5 for final. Given the CPU-only
hardware (`docs/hardware_report.md`), the plan is:
1. Synthetic-fixture tier: 5 seeds, runs in seconds each — establishes that
   the training/eval/metrics-recording pipeline itself is correct.
2. Real-data tier, once unblocked: start with 3 seeds on a small
   deterministic subset + a tiny CNN backbone, time one run, then decide
   with recorded numbers (not a guess) whether 5 seeds / full 23,247 images
   / a larger backbone are actually feasible in this environment.

## Phase 2 — capability isolation
Status: **PENDING** (depends on Phase 1 baselines existing)

Coarse-head-only, hidden-fine-head, adapter isolation, adversarial
suppression, orthogonality loss, combined method — each followed by linear,
nonlinear, and few-shot probes. Implemented under `medgate/models/` and
`medgate/attacks/` (probes count as lightweight attacks and share code with
Phase 3).

## Phase 3 — security
Status: **PENDING**

Gradient inversion (DLG at minimum; iDLG/Geiping only if their citations
get verified and CPU cost is acceptable), membership inference, property
inference, adapter reconstruction, collusion, distillation, fine-tuning
recovery. Every attack script writes attacker knowledge/access/query
budget/compute budget/success condition into its result JSON — no attack is
reported without that record.

## Phase 4 — privacy mechanisms
Status: **PENDING**

No protection / secure aggregation (simulated, since this is single-machine
cross-silo) / DP-SGD via Opacus at multiple budgets / secure agg + DP.
Report privacy-utility Pareto curves from recorded per-run metrics, never
hand-typed numbers.

## Phase 5 — revocation and unlearning
Status: **PENDING**

Patient-, class-, and institution-level removal. Full retraining is the
gold standard. Reproducing a literature method (candidates:
`wu2022-fed-unlearning-kd`, `halimi-fed-unlearning-erase-client`,
`zhang2023-fedrecovery`, `deng2024-right-to-be-forgotten-medical`) is gated
on that citation being moved from UNVERIFIED to VERIFIED first, and on
Phase 1's real-data tier being unblocked (retraining needs real data to be
meaningful). `chen2026-lethe-unlearning` (Lethe) is already VERIFIED and is
the direct benchmark to compare methodology against.

## Phase 6 — external validation and fusion
Status: **PENDING**, gated on all Phase-6 dataset rows in
`docs/literature_matrix.csv` (`isic-archive-2021-descriptor`,
`milk10k-dataset-record`, `pad-ufes-20`) moving to VERIFIED, and on their
licenses being checked the same way Fed-ISIC2019's was (§5 of
research_scope.md). None of that verification has happened yet — do not
plan compute for this phase until it has.

## Phase 7 — IXI Tiny extension
Status: **PENDING**, starts only after Phase 1's primary tables (at
whichever tier turns out feasible) are complete, per the project brief.

## Phase 8 — optional blockchain and FeTS work
Status: **OUT OF SCOPE (hardware)** for FeTS/BraTS, ADNI, full FHE, and any
multi-node blockchain network on this workstation (`docs/hardware_report.md`).
A **minimal single-process permissioned-ledger simulation** (the spec's
fallback: "or a faithful simulation if resources permit") compared against
centralized IAM is still in scope, but only after Phases 0-5 are done, and
only if it can honestly be evaluated on the metrics in the brief (latency,
revocation latency, storage overhead, throughput, failure tolerance, audit
consistency, trust assumptions, operational complexity) rather than assumed
beneficial.

## Open, named blockers (not hidden inside phase checkboxes)

1. **License acceptance for Fed-ISIC2019** (ISIC2019 + HAM10000) — needs a
   human. Procedure: `scripts/download_fed_isic2019_INSTRUCTIONS.md`.
2. **43 of 59 literature-matrix rows are UNVERIFIED** — see
   `docs/literature_matrix.md` for which ones gate which phase.
3. **No GPU** — bounds model scale and which attacks (esp. gradient
   inversion at realistic image size, DP-SGD sweeps) are practical within a
   session; see `docs/hardware_report.md`.
4. **IXI Tiny's own license-acceptance mechanics** were not found in the
   fetched `fed_ixi/README.md` excerpt — re-check before Phase 7 in case a
   similar human-acceptance step applies.
