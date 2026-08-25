# Repair Pass 4 Audit

Maps every item in the repair-pass-4 review to: code path(s) touched, test
name(s) that verify it, experiment artifact(s) it produced, where it is
described in `paper/main.tex`, and its status. Written after all code
changes, all experiment re-runs, the full PDF recompile, and the
clean-environment verification described in this document's own
"Verification" section — nothing here is marked complete on the basis of a
prose or comment change alone.

Base commit this pass started from: `5b4becc3a66f2ea2f992dab565b5e04b59ae19cc`.
Every experiment JSON below records its own `git_commit`/`config`/`seed`
fields; commits made at or after that hash but before this pass's own
commit reflect the code that actually produced the numbers, which is the
reproducibility property that matters (not that the JSON's commit hash
equals the *final* commit of this pass — no experiment was re-run a second
time merely to update a hash after code stopped changing).

## P0-A — Invalid adapter matrix-completion experiment

**Problem:** `adapter.up` is a `(feature_dim, rank)` matrix, so truncating
its SVD at `rank=rank` (its own maximum possible rank) is the identity map;
the reported "improvement with reveal fraction" measured only more
directly-revealed entries, not inference.

| Requirement | Code | Tests | Experiment artifact | Paper | Status |
|---|---|---|---|---|---|
| 1. ΔW = up·down as the low-rank object | `medgate/attacks/adapter_recovery.py::effective_delta_w` | `test_effective_delta_w_has_constructive_rank_le_adapter_rank` | — | §8.1 "A validity bug found and fixed" | Done |
| 2. Separate observed/unobserved error | `adapter_recovery.py::recovery_metrics` (returns `observed_entry_rmse`, `unobserved_entry_rmse`, `unobserved_entry_normalized_error` separately) | `test_low_rank_completion_attack_reports_observed_unobserved_and_gain` | `experiments/phase3_adapter_recovery_synthetic/*.json` | §8.1, Table 8 ("Unobs. error" column) | Done |
| 3. zero/mean/random-fill + SoftImpute + oracle-rank controls | `adapter_recovery.py::zero_fill/mean_fill/random_fill/hard_impute_svd/soft_impute_svd`, `FILL_METHODS` | `test_mean_and_random_fill_leave_observed_entries_exact_but_differ_on_unobserved`, `test_soft_impute_also_completes_meaningfully` | `configs/phase3_adapter_recovery_synthetic.yaml` (`fill_methods` list) | §8.1 "Method and controls", Table 8 | Done |
| 4. Reveal fractions, rank misspecification, ≥5 seeds | `scripts/run_phase3_adapter_recovery_synthetic.py` (`reveal_fractions`, `rank_misspecification_deltas`, `seeds: [0,1,2,3,4]`) | `test_svd_completion_recovers_a_known_low_rank_matrix_better_with_more_reveal`, `test_rank_misspecification_degrades_recovery_relative_to_oracle` | `experiments/phase3_adapter_recovery_synthetic/{null_signal,hierarchical}_seed{0..4}.json` (10 files) | §8.1, Tables 8–9 | Done |
| 5. Gain over zero-fill, not cosine alone | `adapter_recovery.py::completion_gain_over_zero_fill` | `test_completion_beats_zero_fill_on_unobserved_entries` | same as above (`completion_gain_over_zero_fill` field in every JSON) | Table 8 "Gain over zero-fill" column | Done |
| 6. Functional recovery on the hierarchical fixture | `scripts/run_phase3_adapter_recovery_synthetic.py::train_hierarchical_combined` | — | `experiments/phase3_adapter_recovery_synthetic/hierarchical_seed{0..4}.json` | §8.1 "Functional recovery does not track parameter recovery cleanly" | Done — result is an honest mixed/non-monotonic finding, not a clean win, reported as such |
| 7. Delete/replace all old artifacts | old `configs/phase3_adapter_recovery_synthetic.yaml`, `experiments/phase3_adapter_recovery_synthetic/*`, `paper/tables/phase3_adapter_recovery_synthetic.*` deleted before rebuilding | — | — | §8.1 entirely rewritten, not edited in place | Done |
| 8. Tests that fail on non-binding rank / unchanged zero-fill | `tests/test_adapter_recovery.py::test_rank_equal_to_ambient_dim_makes_hard_impute_a_no_op`, `::test_completion_beats_zero_fill_on_unobserved_entries` | (same) | — | §8.1 cites both by name | Done |

New/changed files: `medgate/attacks/adapter_recovery.py` (full rewrite),
`tests/test_adapter_recovery.py` (full rewrite, 9 tests),
`configs/phase3_adapter_recovery_synthetic.yaml` (full rewrite),
`scripts/run_phase3_adapter_recovery_synthetic.py` (full rewrite),
`scripts/make_phase3_adapter_recovery_table.py` (full rewrite, now emits
three tables: `phase3_adapter_recovery_{gain,rank,collusion}.csv/md/tex`).

## P0-B — Secure-aggregation security model

**Problem:** the module's default Gaussian mask was described as having
the one-time-pad (information-theoretic) concealment property that only a
uniform mask over a finite ring actually has.

| Requirement | Code | Tests | Experiment artifact | Paper | Status |
|---|---|---|---|---|---|
| 1. Reframe as correctness/server-view simulation | `medgate/privacy/secure_aggregation.py` module docstring rewritten | — | — | §7.3 masking-model paragraph; §3 Related Work | Done |
| 2. Remove unsupported confidentiality claims | same file, `masking_correctness_diagnostic`/`empirical_concealment_sanity_check` docstrings rewritten | — | — | §7.3, §12 Limitations | Done |
| 3. Z_q uniform-mask path, quantization, wraparound, exact recovery | `secure_aggregation.py::quantize_to_zq/dequantize_from_zq/mask_client_updates_zq/secure_aggregate_updates_zq` | `test_zq_masked_value_is_exactly_uniform_regardless_of_plaintext`, `test_gaussian_masked_value_is_not_uniform_and_shifts_with_plaintext`, `test_zq_quantize_dequantize_round_trip`, `test_zq_exact_aggregate_recovery_within_quantization_error`, `test_zq_wraparound_corrupts_the_aggregate_when_q_is_too_small`, `test_zq_secure_aggregate_rejects_unequal_weights` | — | §7.3 "P0-B" paragraph | Done |
| 4. Keep dropout/unequal-weight/seed-collusion tests | unchanged: `mask_client_updates`, `secure_aggregate_updates` | `test_dropout_breaks_correctness`, `test_secure_aggregate_rejects_unequal_weights`, `test_server_client_collusion_can_unmask_a_target_in_this_simulation` | — | §12 Limitations "masking vs. poisoning" bullet | Done (kept, not modified) |
| 5. Concealment as heuristic; sweep mask scale; nonlinear attacker | `empirical_concealment_sanity_check(mask_scale=, attacker=)` | `test_concealment_improves_with_mask_scale_but_never_reaches_a_guarantee`, `test_concealment_heuristic_checked_against_a_nonlinear_attacker_too` | `experiments/phase4_concealment_sweep/sweep.json` (8 scales × 2 attackers × 3 seeds = 48 runs) | §7.3, Table 6 | Done |
| 6. Rename paper labels | `scripts/display_names.py` (`secure_agg` → "Simulated pairwise additive masking") | — | — | Throughout §7.3, Table 1, §12 | Done |

New/changed files: `medgate/privacy/secure_aggregation.py` (module
docstring + `_pairwise_mask`/`mask_client_updates`/
`empirical_concealment_sanity_check` extended with `mask_scale`/`attacker`
params + new Z_q functions), `tests/test_phase4_privacy.py` (10 new
tests), `scripts/run_phase4_concealment_sweep.py` (new),
`configs/phase4_concealment_sweep.yaml` (new),
`scripts/make_phase4_concealment_table.py` (new),
`scripts/display_names.py` (renamed labels).

## P0-C — DP full-training budget

**Problem:** each federated round created a fresh `PrivacyEngine`, so the
reported epsilon was the max of independent per-round budgets, never
composed across the training run.

| Requirement | Code | Tests | Experiment artifact | Paper | Status |
|---|---|---|---|---|---|
| 1. Don't label per-round epsilon as the experiment epsilon | `scripts/run_phase4_synthetic.py::run_arm` (single cumulative `final_epsilon` after the round loop, not `max(epsilons)` over independent per-round values) | — | `experiments/phase4_synthetic/seed{0,1,2}.json` | §7.3 "P0-C" paragraph | Done |
| 2. Compose RDP per client across rounds; report max full-training epsilon | `medgate/privacy/dp_sgd.py::dp_local_train(engine=)`, `dp_fedavg_round(engines=)`, `secure_dp_fedavg_round(engines=)` | `test_dp_epsilon_composes_across_engine_reuse_not_reset_per_call`, `test_dp_epsilon_increases_monotonically_with_more_federated_rounds` | same | §7.3, Table 5 | Done |
| 3. Rename table columns precisely | field renamed `epsilon` → `epsilon_full_training_max_per_client_record_level` (`dp_sgd.py`, `run_phase4_synthetic.py`, `make_phase4_table.py`, `csv_to_latex_tables.py`) | — | `paper/tables/phase4_privacy_synthetic.csv` | Table 5 header | Done |
| 4. State record-level, not client-level, prominently | `dp_accountant_metadata()` in `run_phase4_synthetic.py`; module docstring in `dp_sgd.py` | — | every DP-arm entry in `experiments/phase4_synthetic/*.json` (`dp_accountant_metadata.adjacency`) | §7.3 body text (not only a footnote, per requirement) | Done |
| 5. Verify monotonicity (rounds/epochs/sample-rate up, noise down) | — | `test_dp_epsilon_increases_monotonically_with_more_composition` (epochs), `test_dp_epsilon_increases_monotonically_with_more_federated_rounds` (rounds), `test_dp_epsilon_increases_with_higher_sample_rate` (batch size / sample rate), `test_dp_stronger_noise_reduces_epsilon` | — | §7.3 "Monotonicity is verified directly" | Done, all 4 pass |
| 6. Record Opacus version, accountant, sampling, adjacency | `dp_accountant_metadata()` | — | every DP-arm JSON entry | §7.3 | Done |
| 7. Update abstract/methods/table caption/results/limitations consistently | — | — | — | Abstract; §7.3; Table 5 caption; §12 | Done |

New/changed files: `medgate/privacy/dp_sgd.py` (`dp_local_train`,
`dp_fedavg_round`, `secure_dp_fedavg_round` all gained `engine`/`engines`
params and now return the engine(s) too — a breaking API change,
propagated to every caller), `scripts/run_phase4_synthetic.py` (rewritten
`run_arm`/`dp_accountant_metadata`), `scripts/make_phase4_table.py` (reads
renamed field), `scripts/make_figures.py` (`fig_phase4_privacy_pareto`
reads renamed field), `tests/test_phase4_privacy.py` (existing DP tests
updated for the 3-tuple return + 3 new composition tests).

## P1 — Hierarchical experimental tier

| # | Requirement | Code | Experiment artifact | Paper | Status |
|---|---|---|---|---|---|
| 1 | ≥5 paired seeds | `configs/phase1_hierarchical.yaml` (`seeds: [0,1,2,3,4]`) | `experiments/phase1_hierarchical/seed{0..4}.json` | §7.2, Table 4 | Done |
| 2 | Larger patient count OR power/uncertainty justification | kept 12 patients/institution; justification in `configs/phase1_hierarchical.yaml` comments | — | §6.7 "Uncertainty this actually buys" paragraph (SE ≈0.054, detectable effect ≈0.15–0.2) | Done (justification branch) |
| 3 | Probe selection on attack-validation split; single held-out attack-test evaluation | `medgate/attacks/probes.py::selected_probe_attack` (new); `scripts/run_phase1_hierarchical.py::build_fixture` (carves attack-test out of the existing val/test split via `split_by_patient` reuse) | `experiments/phase1_hierarchical/*.json` (`probe_selection` field per method) | §7.2 intro paragraph, Table 4 caption | Done |
| 4 | Convergence curves + larger-budget full-fine-tune oracle | `scripts/run_phase1_hierarchical_convergence.py` (new), `configs/phase1_hierarchical_convergence.yaml` (new, 16 rounds vs. main sweep's 4) | `experiments/phase1_hierarchical_convergence/seed0.json` | §7.2 "Convergence and a near-convergence upper-bound oracle" | Done |
| 5 | Patient-level + patient-group unlearning on the hierarchical fixture | `scripts/run_phase5_hierarchical_unlearning.py` (new), `configs/phase5_hierarchical_unlearning.yaml` (new) | `experiments/phase5_hierarchical_unlearning/{patient,patient_group}_seed{0,1,2}.json` | §10.1 (new subsection), Table 15 | Done |
| 6 | Sensitivity sweeps: site_shift_strength, fine_signal_strength, class_imbalance, alternative ontology | `scripts/run_phase1_hierarchical_sensitivity.py` (new), `medgate/data/coarse_ontology.py` (new, `ALTERNATIVE_FINE_TO_COARSE_IDX`), `medgate/data/hierarchical_synthetic.py` (`fine_to_coarse_idx` param threaded through `_render`/`_make_one_institution`/`make_hierarchical_institutions`) | `experiments/phase1_hierarchical_sensitivity/sensitivity.json` (11 sweep points × 2 seeds) | §9.1 (new subsection), Table 13 | Done |
| 7 | Preserve the negative result if BestProbeRFC still exceeds authorized F1 | — (no forcing; ran the real numbers) | Table 4: 10/10 rows, up from 8/10 at 2 seeds | §7.2 "every one of the ten rows" | Done — result got *stronger*, not weaker, and is reported as such |

New tests for this section: `tests/test_representation_diagnostics.py::test_selected_probe_attack_evaluates_the_validation_winner_once_on_test`,
`tests/test_hierarchical_synthetic.py::test_alternative_coarse_ontology_is_also_learnable_and_relabels_coarse_classes`.

## P1 — Paper cleanup and consistency

| # | Requirement | Status | Where |
|---|---|---|---|
| 1 | Remove all TODO/RESULT PENDING/UNVERIFIED markers | Done — all 10 call sites rewritten as plain prose; `\TODO`/`\RESULTPENDING`/`\UNVERIFIED` macro definitions removed from the preamble (confirmed zero call sites remain via `grep`) | Preamble; abstract; §3 (×3); §6.2, §6.3; §7 header; §9 (ontology paragraph, expanded into §9.1); §15 |
| 2 | "One synthetic fixture" → "two controlled synthetic fixtures" | Done | §6.2 ("one of two controlled synthetic fixtures") |
| 3 | Recompute verified-reference counts | Done — was 16/59 verified, 43 unverified (stale); corrected to the actual `docs/literature_matrix.csv` count, 38/59 verified, 21 unverified | §3 "Verification status" paragraph |
| 4 | Remove duplicated sentence in unlearning section | Reviewed the full §10/§10.1 (rewritten substantially this pass) both in source and in the compiled PDF; no duplicate sentence found in the current text. If this was present in an earlier draft, this pass's rewrite of the confound paragraph and the new §10.1 addition superseded it | §10 |
| 5 | Shorten abstract to ~200–250 words | Done — 245 words (was ~500+) | Abstract |
| 6 | Never call the generator "medically realistic"; call it a controlled learnable hierarchical fixture | Verified via `grep` — no "medically realistic"/"clinically realistic" phrasing exists anywhere in the document; the fixture is consistently called "hierarchical fixture" / "controlled learnable hierarchical fixture" | §6.5, throughout |
| 7 | No clinical/real-data effectiveness claim | Already present, verified unchanged: "No model produced by this project is validated for, or intended for, clinical use" | §13 |
| 8 | Title/status/contributions/conclusion same claim strength | Title/subtitle/date line already read "Preliminary technical report... synthetic validation only, real-data tier license-gated"; Conclusion rewritten this pass to state the same three P0 bugs and the 5-seed/10-of-10 finding the abstract states, so both now claim the same thing at the same strength | Title page; §15 |
| 9 | Compile with zero missing refs, zero undefined labels, zero overfull-box warnings where practical | 0 undefined refs/citations (verified via `grep -i undefined` on the final `.log`, only a benign font-shape substitution warning remains). 2 overfull-hbox warnings remain, both diagnosed as non-visual artifacts (a `\resizebox`-wrapped wide table's internal pre-scale measurement, and a `hyperref` end-of-document anchor box with empty `[][]` content) rather than left unexplained; 6 other overfull warnings from long `\texttt{}` identifiers were found and fixed by shortening/restructuring the surrounding sentences | See "Verification" below |

## Verification

- **Clean install from `requirements.lock.txt`:** done in a fresh venv
  (`python3 -m venv`) with `pip install -r requirements.lock.txt
  --extra-index-url https://download.pytorch.org/whl/cpu`. First attempt
  **failed** — `torchvision` was missing from both `requirements.txt` and
  `requirements.lock.txt` (present only because an earlier session had
  pip-installed it directly into the development `.venv` without ever
  declaring it), causing every `PretrainedMobileNetBackbone` code path to
  `ModuleNotFoundError`. Fixed by adding `torchvision` to both files (with
  the version actually in use, `0.28.0+cpu`, pinned in the lock file).
  Re-verified clean after the fix.
- **Full pytest suite, exact counts:** `104 passed, 0 skipped, 0 failed`
  in both the development `.venv` and the from-scratch clean venv
  (`PYTHONPATH=. pytest tests/ -q`).
- **Every experiment whose algorithm or metric changed was re-run:**
  `phase2_synthetic` (probe suite expansion — done in repair pass 3,
  confirmed still current), `phase3_adapter_recovery_synthetic` (P0-A,
  completely new methodology, 10 files), `phase4_synthetic` (P0-C epsilon
  accounting, 3 files), `phase4_concealment_sweep` (P0-B, new, 1 file),
  `phase1_hierarchical` (P1 5-seed + probe-selection fix, 5 files),
  `phase1_hierarchical_convergence` (P1, new, 1 file),
  `phase1_hierarchical_sensitivity` (P1, new, 1 file),
  `phase5_hierarchical_unlearning` (P1, new, 6 files).
- **Every JSON records git commit / config / seed:** spot-checked across
  every experiment directory listed above; every file has `git_commit`,
  `config`, and `seed` (or per-row `seed` for the sensitivity sweep) fields
  populated from the actual run, not hand-edited.
- **Tables/figures regenerated only from committed JSON:** `scripts/csv_to_latex_tables.py`
  and `scripts/make_figures.py` both read exclusively from
  `paper/tables/*.csv`, themselves generated exclusively by
  `scripts/make_phase*_table.py` reading `experiments/**/*.json` — no
  table or figure value was hand-edited. `scripts/compile_paper.sh` was
  updated to call every `make_*` script needed to reproduce this from a
  clean checkout in one command.
- **PDF compiled twice (in fact: `pdflatex`→`bibtex`→`pdflatex`→`pdflatex`,
  repeated across the whole editing session) and all 32 pages visually
  inspected**, page by page, via the `Read` tool on the rendered PDF — not
  only the LaTeX log. Two issues were caught this way that the log alone
  would not have surfaced: a stale "on 2 seeds" claim left over in the
  Discussion section after the main sweep grew to 5 seeds, and a
  straight-quote (`"..."` instead of `` `...' ``) typographic slip in the
  Conclusion. Both fixed and reconfirmed by a further compile + re-read.

## Files changed this pass (summary)

New: `medgate/data/coarse_ontology.py`; `scripts/run_phase3_adapter_recovery_synthetic.py`
(rewritten in place — counts as new methodology), `scripts/make_phase3_adapter_recovery_table.py`
(rewritten), `scripts/run_phase4_concealment_sweep.py`,
`scripts/make_phase4_concealment_table.py`, `configs/phase4_concealment_sweep.yaml`,
`scripts/run_phase1_hierarchical_convergence.py`, `configs/phase1_hierarchical_convergence.yaml`,
`scripts/run_phase1_hierarchical_sensitivity.py`, `configs/phase1_hierarchical_sensitivity.yaml`,
`scripts/make_phase1_hierarchical_sensitivity_table.py`,
`scripts/run_phase5_hierarchical_unlearning.py`, `configs/phase5_hierarchical_unlearning.yaml`,
`scripts/make_phase5_hierarchical_table.py`, this file.

Substantially rewritten: `medgate/attacks/adapter_recovery.py`,
`tests/test_adapter_recovery.py`, `medgate/privacy/secure_aggregation.py`,
`medgate/privacy/dp_sgd.py`, `scripts/run_phase4_synthetic.py`,
`scripts/run_phase1_hierarchical.py`, `medgate/attacks/probes.py`
(added `selected_probe_attack`/`probe_by_name`), `medgate/data/hierarchical_synthetic.py`
(ontology parameter threading), `tests/test_phase4_privacy.py`,
`paper/main.tex` (abstract, conclusion, §3, §6.6–6.7, §7.2–7.3, §8.1, §9,
§9.1 new, §10, §10.1 new, §12, §14, appendix — essentially every section
touched by P0-A/B/C or P1), `requirements.txt`, `requirements.lock.txt`
(torchvision fix), `scripts/compile_paper.sh`.
