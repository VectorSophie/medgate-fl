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
- [x] Tiny synthetic fixture generator (`medgate/data/synthetic.py`):
      6-center, 8-class, image-shaped tensors, deterministic seed.
- [x] Verify one centralized forward/backward pass on the synthetic fixture
      (`tests/test_phase0_forward_backward.py` — passing, see
      `logs/phase0_smoke_test.log`).
- [x] Verify one two-client FL round (`tests/test_phase0_fedavg_round.py` —
      passing).
- [x] Verify reproducibility from a clean environment: this session's own
      venv was built from scratch (`python3 -m venv .venv`,
      `requirements.txt`, pinned in `requirements.lock.txt`) and
      `scripts/smoke_test.sh` passed 3/3 on that fresh install. Re-running
      on a second, independent machine is still open — flagged as a task,
      not silently assumed.
- [ ] Real Fed-ISIC2019 checksums/label counts/center counts/duplicates —
      **BLOCKED-LICENSE**: requires a human to accept the ISIC2019 and
      HAM10000 licenses (`scripts/download_fed_isic2019_INSTRUCTIONS.md`).
      Everything downstream of the real download is marked pending until
      that happens.

## Phase 1 — primary utility baseline (Fed-ISIC2019)
Status: **synthetic tier DONE (pipeline validation only)**; **real-data
tier BLOCKED-LICENSE**

Centralized, local-only, FedAvg, FedProx, plain FedLoRA. Spec asks for 5
seeds where feasible, else 3 for development + 5 for final. Given the CPU-only
hardware (`docs/hardware_report.md`), the plan is:
1. Synthetic-fixture tier — **done**: 5 seeds x 5 baselines = 25 runs,
   `scripts/run_phase1_synthetic.py` (config `configs/phase1_synthetic.yaml`),
   raw per-run JSON in `experiments/phase1_synthetic/`, aggregated table in
   `paper/tables/phase1_utility_synthetic.{csv,md}` via
   `scripts/make_phase1_table.py`. All 25 runs completed in ~66s wall-clock
   on the hardware in `docs/hardware_report.md`.
   **Reading the numbers**: the synthetic fixture draws images and labels
   independently at random, so there is no learnable signal. Every
   baseline lands near or below the constant-predictor macro-F1 for its
   label space (~0.22 for 3-way coarse, ~0.03 for 8-way fine — matches a
   model that collapses to predicting the marginal-majority class, the
   only thing there is to learn from noise). Centralized/FedAvg/FedProx
   land on *identical* numbers across all 5 seeds, which is the expected
   signature of that same degenerate collapse rather than a bug — with a
   fixed dataset split and no real gradient signal, different
   initializations converge to the same marginal-frequency shortcut.
   local-only and fedlora show more seed variance because each client
   model (local-only) or the frozen-random-backbone + LoRA-only path
   (fedlora) doesn't share the same aggregated trajectory. **This tier
   confirms the training/aggregation/eval/table pipeline works and does not
   produce impossible above-chance numbers (which would indicate a leakage
   bug) — it is not evidence about capability isolation or federated
   learning quality, and must never be quoted as such in `paper/main.tex`.**
2. Real-data tier, once unblocked: start with 3 seeds on a small
   deterministic subset + a tiny CNN backbone, time one run, then decide
   with recorded numbers (not a guess) whether 5 seeds / full 23,247 images
   / a larger backbone are actually feasible in this environment.

## Phase 2 — capability isolation
Status: **synthetic tier DONE (pipeline validation only)**; real-data tier
`BLOCKED-LICENSE`

Coarse-head-only, hidden-fine-head, adapter isolation, adversarial
suppression, orthogonality loss, combined method — all six trained on the
*identical* `MedGateModel` architecture (`medgate/models/backbone.py`),
differing only in the loss function (`medgate/federated/capability_isolation.py`),
so the ablation isolates the objective rather than model capacity. Each
followed by linear/nonlinear/kNN/few-shot probes on the frozen PUBLIC
representation (`medgate/attacks/probes.py`) plus an output-only probe used
to operationalize `U_public` (fine-label recoverability from the exposed
coarse output alone, before any representation access — documented in
`medgate/attacks/probes.py` and `medgate/capability_metrics.py`, since the
project brief's formula for `U_public` was not fully specified). Composite
metrics ARR/UCG/RFC/CRE implemented in `medgate/capability_metrics.py`,
always reported alongside the raw probe numbers, never in place of them.

- [x] `scripts/run_phase2_synthetic.py` + `configs/phase2_synthetic.yaml`:
      6 methods x 5 seeds = 30 runs, ~2m13s wall-clock, raw JSON under
      `experiments/phase2_synthetic/`, table in
      `paper/tables/phase2_capability_isolation_synthetic.{csv,md}` via
      `scripts/make_phase2_table.py`.
- [x] `tests/test_phase2_capability_isolation.py`: gradient-reversal sign
      is checked directly (not just "doesn't crash"), orthogonality loss
      checked against known orthogonal/parallel vectors, all six
      objectives checked for finite forward/backward, and the adversary
      head is checked to receive gradient under exactly the two methods
      that should use it.
- **Reading the numbers**: same caveat as Phase 1 — synthetic labels carry
  no signal, so every method's authorized fine macro-F1 sits at the
  8-class chance/collapse floor (~0.03), and RFC/UCG bounce in a small
  range (~0.03-0.14) that reflects test-set-size noise (only ~38 samples/
  fine-class in the pooled test set) rather than a real isolation effect.
  ARR is `n/a` for most methods because its denominator
  (`U_plain_adapter - U_public`) is ~0 when nothing is learnable — this is
  the metric correctly refusing to report a number it cannot support,
  not a bug. **None of these numbers support or refute any claim about
  whether adversarial/orthogonality objectives actually help capability
  isolation** — that question needs real, structured Fed-ISIC2019 data.

## Phase 3 — security
Status: **synthetic tier PARTIAL (pipeline validation only)**; real-data
tier `BLOCKED-LICENSE`

Implemented: gradient inversion (DLG, simplified — Adam not L-BFGS, labels
fixed not jointly optimized, deviation documented in
`medgate/attacks/gradient_inversion.py`), loss-threshold membership
inference (simplified — no shadow models, documented in
`medgate/attacks/membership_inference.py`), adapter reconstruction (A2,
`medgate/attacks/reconstruction.py`), black-box hard-label extraction (A3,
same file), two-attacker collusion (same file). Every attack result JSON
carries `attacker_knowledge`/`attacker_access`/`compute_budget` fields —
no attack is reported without that record, per the project brief.

**Not yet implemented**: property inference (source citation
`melis2019-unintended-leakage` is UNVERIFIED — gated per
`docs/literature_matrix.md`'s rule that unverified sources aren't used to
justify a method design until checked), client attribution, sign-flipping/
label-flipping/backdoor/free-rider/malformed-update/replay integrity
attacks (these target A4 malicious-client behavior during aggregation,
not yet built), full-fine-tuning recovery with a fixed compute budget
distinct from the few-shot probe already in Phase 2, low-rank adapter
completion, and known-adapter comparison. iDLG/Geiping-style stronger
gradient inversion are deferred pending their citations being verified.

- [x] `scripts/run_phase3_synthetic.py` + `configs/phase3_synthetic.yaml`:
      3 seeds (statistical protocol: 3 for this development pass, 5
      reserved for real-data final runs), ~47s wall-clock total, raw JSON
      under `experiments/phase3_synthetic/`, table in
      `paper/tables/phase3_attacks_synthetic.{csv,md}` via
      `scripts/make_phase3_table.py`.
- [x] `tests/test_phase3_attacks.py`: 6 tests, each checking a real
      property of its attack (e.g. the reconstructed adapter is a
      different object with different weights than the original, not just
      "code ran"), not merely absence-of-crash.
- **Reading the numbers**: DLG's PSNR varied enormously across seeds in
  this run (138 dB, 14.9 dB, 15.4 dB — see
  `paper/tables/phase3_attacks_synthetic.md`), which is a real,
  literature-consistent property of gradient-inversion attacks (sensitive
  to the dummy-image initialization), not a bug — but note the *absolute*
  numbers are almost certainly inflated by how tiny this model is
  (`feature_dim=64`, a few thousand parameters total): DLG is known to
  succeed far more easily against small models than realistic ones, so
  these PSNR values must not be quoted as representative of a real-scale
  backbone. Membership-inference AUC (~0.51, std 0.05) sat near the
  pre-registered target (<=0.55) here, but that is expected on
  unstructured noise with one training epoch — not evidence the real
  model would meet the target. Adapter-reconstruction and extraction
  utility stayed flat across budgets (~0.03), again the synthetic-data
  floor, not a finding about budget-scaling.

## Phase 4 — privacy mechanisms
Status: **synthetic tier DONE (pipeline validation only)**; real-data tier
`BLOCKED-LICENSE`

Four arms implemented: no protection (plain FedAvg), secure aggregation
(pairwise-additive-masking simulation of Bonawitz et al. 2017's
mathematical core — `medgate/privacy/secure_aggregation.py`, explicitly
NOT the full protocol: no Diffie-Hellman key agreement, no Shamir
secret-sharing dropout recovery, single-process simulation only), DP-SGD
via Opacus at 3 noise multipliers (`medgate/privacy/dp_sgd.py`, required
two fixes to the model itself — non-inplace ReLU, and excluding
`adversary_head` from the DP-tracked optimizer — both documented in that
file), and secure aggregation + DP-SGD combined.

- [x] Two real Opacus integration bugs found and fixed while building this
      (not routed around): inplace ReLU breaks Opacus's backward hooks;
      Opacus's optimizer requires every tracked parameter to have received
      a per-sample gradient every batch, which `model.adversary_head`
      (Phase 2's addition, unused by the plain coarse+fine objective)
      violated. Both fixes are permanent, in `medgate/models/backbone.py`
      and `medgate/attacks/gradient_inversion.py`'s `attack_params()`
      (reused here rather than duplicated).
- [x] `scripts/run_phase4_synthetic.py` + `configs/phase4_synthetic.yaml`:
      3 seeds x (no_protection, secure_agg, 3x dp_sgd, 3x
      secure_agg_plus_dp) = 24 runs, ~3m23s wall-clock, raw JSON under
      `experiments/phase4_synthetic/`, Pareto table in
      `paper/tables/phase4_privacy_synthetic.{csv,md}` via
      `scripts/make_phase4_table.py`.
- [x] `tests/test_phase4_privacy.py`: 5 tests. Correctness (masked updates
      still sum exactly right) is checked directly. **A confidentiality
      test was written wrong on the first pass and is worth recording**:
      it asserted low cosine similarity between an individual masked
      update and the true update, which failed at n=3-10 clients (measured
      mean |cos sim| 0.3-0.6) — not because masking is broken (the sum is
      still exact), but because that similarity is the wrong thing to
      measure: the real security argument is that the mask is a secret,
      uniformly-random one-time pad, which is a property of the
      *distribution*, not of one sampled geometric distance. Replaced with
      a test of the property that actually demonstrates hiding: the same
      true update, masked with two different unknown seeds, produces
      clearly different outputs. The wrong version and why it was wrong
      are kept in `medgate/privacy/secure_aggregation.py`'s
      `confidentiality_check` docstring so this mistake isn't repeated.
- **Reading the numbers**: `dp_sgd` and `secure_agg_plus_dp` produce
  IDENTICAL utility at every noise multiplier and seed in
  `paper/tables/phase4_privacy_synthetic.md` — this is correct and
  expected (masking cancels exactly in the aggregate; it changes what the
  server can *observe* per-client, never the final trained model), and is
  itself a useful correctness check on the implementation, not evidence
  the two arms are redundant to compare. `no_protection` and `secure_agg`
  are likewise identical for the same reason. Utility differences between
  DP and non-DP arms on this synthetic fixture (~0.03 vs ~0.04) are within
  noise given the fixture has no real signal — not a privacy-utility
  finding. Real epsilon-vs-utility tradeoffs need real data.

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
