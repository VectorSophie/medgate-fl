"""Checks for the P1-7 / P0-A (repair pass 4) genuine parameter-space
adapter-recovery track (medgate/attacks/adapter_recovery.py) -- distinct
from the fresh-training recovery attacks in medgate/attacks/reconstruction.py.

Several tests below exist specifically to fail if the P0-A bug (completing
adapter.up directly, whose rank truncation is a no-op since up's own
maximum rank equals adapter_rank) is ever reintroduced.

Run: PYTHONPATH=. pytest tests/test_adapter_recovery.py -v
"""
import numpy as np
import torch

from medgate.attacks.adapter_recovery import (
    completion_gain_over_zero_fill,
    effective_delta_w,
    hard_impute_svd,
    low_rank_completion_attack,
    mean_fill,
    random_fill,
    recovery_metrics,
    simulate_partial_leak,
    soft_impute_svd,
    two_attacker_collusion_completion_attack,
    zero_fill,
)
from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, SyntheticFedISIC
from medgate.models.backbone import MedGateModel

MODEL_KWARGS = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), feature_dim=16, adapter_rank=2)


def _trained_model():
    torch.manual_seed(0)
    model = MedGateModel(**MODEL_KWARGS)
    with torch.no_grad():
        # zero-init adapter (medgate/models/backbone.py) would make "up" the
        # all-zeros matrix -- trivially "recoverable" and uninformative as a
        # test; give it real, structured (low-rank-by-construction) values.
        g = torch.Generator().manual_seed(1)
        a = torch.randn(model.adapter.up.weight.shape[0], 2, generator=g)
        b = torch.randn(2, model.adapter.up.weight.shape[1], generator=g)
        model.adapter.up.weight.copy_(a @ b)
    return model


# --- P0-A regression tests: guard against the exact bug that was found ---

def test_rank_equal_to_ambient_dim_makes_hard_impute_a_no_op():
    """Documents WHY the pass-2/3 attack was invalid: completing a (d x r)
    matrix at rank=r (its own maximum possible rank) does nothing beyond
    the initial zero-fill, because keeping ALL of a matrix's singular
    values via SVD reconstructs it exactly. This is the failure mode
    medgate/attacks/adapter_recovery.py's module docstring documents and
    fixes by completing delta_w = up @ down (ambient rank feature_dim,
    constructive rank adapter_rank) instead of up alone (ambient rank ==
    constructive rank, by construction of nn.Linear(rank, feature_dim))."""
    rng = np.random.RandomState(0)
    d, r = 16, 4
    up_like = rng.randn(d, r)  # shape (feature_dim, rank) -- what pass 2/3 completed
    partial, mask = simulate_partial_leak(torch.from_numpy(up_like).float(), reveal_fraction=0.3, seed=0)

    # rank == r == min(up_like.shape) -- exactly the non-binding case.
    recovered = hard_impute_svd(partial, mask, rank=r, iterations=50)
    zero = zero_fill(partial, mask)
    assert np.allclose(recovered, zero, atol=1e-6), (
        "hard_impute_svd at rank==min(matrix.shape) must be numerically "
        "indistinguishable from zero-fill (the P0-A bug) -- if this test "
        "fails, either the bug is fixed at the algorithm level (unlikely, "
        "it is mathematically forced) or the test itself needs updating, "
        "but this exact configuration must never silently look like real "
        "completion again."
    )


def test_completion_beats_zero_fill_on_unobserved_entries():
    """The check that would have caught the P0-A bug: on a matrix that is
    genuinely low-rank relative to its AMBIENT dimension (rank << d, not
    rank == min(shape)), hard-impute completion must recover unobserved
    entries meaningfully better than the zero-fill control."""
    rng = np.random.RandomState(0)
    d, r = 20, 3
    true = rng.randn(d, r) @ rng.randn(r, d)  # (20,20), rank 3 << 20
    partial, mask = simulate_partial_leak(torch.from_numpy(true).float(), reveal_fraction=0.5, seed=1)

    recovered = hard_impute_svd(partial, mask, rank=r, iterations=100)
    zero = zero_fill(partial, mask)

    m = recovery_metrics(torch.from_numpy(true).float(), recovered, mask, r)
    z = recovery_metrics(torch.from_numpy(true).float(), zero, mask, r)
    gain = completion_gain_over_zero_fill(m, z)

    assert gain["absolute_gain"] > 0, f"hard-impute must beat zero-fill on unobserved entries when rank({r}) << ambient dim({d}); gain={gain}"
    assert m["unobserved_entry_normalized_error"] < 0.3, m
    assert not np.allclose(recovered, zero), "completion result must differ from the zero-filled baseline"


def test_svd_completion_recovers_a_known_low_rank_matrix_better_with_more_reveal():
    rng = np.random.RandomState(0)
    true_rank = 2
    true_matrix = rng.randn(16, true_rank) @ rng.randn(true_rank, 16)  # (16,16), ambient dim 16 >> rank 2
    true_matrix_t = torch.from_numpy(true_matrix).float()

    errors = []
    for reveal_fraction in (0.2, 0.5, 0.9):
        partial, mask = simulate_partial_leak(true_matrix_t, reveal_fraction, seed=0)
        recovered = hard_impute_svd(partial, mask, rank=true_rank, iterations=100)
        metrics = recovery_metrics(true_matrix_t, recovered, mask, true_rank)
        errors.append(metrics["unobserved_entry_normalized_error"])
    assert errors[0] > errors[1] > errors[2], f"unobserved-entry error should improve monotonically with more revealed entries: {errors}"
    assert errors[2] < 0.1, f"90% reveal of an exactly-rank-{true_rank} (16x16) matrix should complete unobserved entries nearly exactly, got error {errors[2]}"


def test_rank_misspecification_degrades_recovery_relative_to_oracle():
    """A candidate_rank far from the true rank should recover unobserved
    entries worse than the oracle (true) rank -- the concrete,
    P0-A-required rank-misspecification check."""
    rng = np.random.RandomState(2)
    d, true_rank = 20, 3
    true = rng.randn(d, true_rank) @ rng.randn(true_rank, d)
    true_t = torch.from_numpy(true).float()
    partial, mask = simulate_partial_leak(true_t, reveal_fraction=0.4, seed=3)

    oracle = hard_impute_svd(partial, mask, rank=true_rank, iterations=100)
    too_low = hard_impute_svd(partial, mask, rank=1, iterations=100)
    too_high = hard_impute_svd(partial, mask, rank=10, iterations=100)

    e_oracle = recovery_metrics(true_t, oracle, mask, true_rank)["unobserved_entry_normalized_error"]
    e_low = recovery_metrics(true_t, too_low, mask, true_rank)["unobserved_entry_normalized_error"]
    e_high = recovery_metrics(true_t, too_high, mask, true_rank)["unobserved_entry_normalized_error"]

    assert e_oracle <= e_low, f"under-ranking (rank=1 < true={true_rank}) should not beat the oracle rank: oracle={e_oracle} low={e_low}"
    assert e_oracle <= e_high, f"over-ranking (rank=10 > true={true_rank}) should not beat the oracle rank: oracle={e_oracle} high={e_high}"


def test_soft_impute_also_completes_meaningfully():
    rng = np.random.RandomState(4)
    d, true_rank = 20, 3
    true = rng.randn(d, true_rank) @ rng.randn(true_rank, d)
    true_t = torch.from_numpy(true).float()
    partial, mask = simulate_partial_leak(true_t, reveal_fraction=0.5, seed=5)

    # A small threshold relative to this matrix's singular-value scale
    # should behave close to hard_impute at the true rank.
    soft = soft_impute_svd(partial, mask, lam=0.05, iterations=100)
    zero = zero_fill(partial, mask)
    m_soft = recovery_metrics(true_t, soft, mask, true_rank)
    m_zero = recovery_metrics(true_t, zero, mask, true_rank)
    assert m_soft["unobserved_entry_normalized_error"] < m_zero["unobserved_entry_normalized_error"]


def test_mean_and_random_fill_leave_observed_entries_exact_but_differ_on_unobserved():
    rng = np.random.RandomState(6)
    d, r = 10, 2
    true = rng.randn(d, r) @ rng.randn(r, d)
    partial, mask = simulate_partial_leak(torch.from_numpy(true).float(), reveal_fraction=0.4, seed=7)

    mf = mean_fill(partial, mask)
    rf = random_fill(partial, mask, seed=8)
    assert np.allclose(mf[mask], partial[mask])
    assert np.allclose(rf[mask], partial[mask])
    assert np.all(mf[~mask] == mf[~mask][0]), "mean_fill must fill every unobserved entry with the same constant"
    assert not np.allclose(rf, mf), "random_fill should not degenerate to a constant fill"


def test_effective_delta_w_has_constructive_rank_le_adapter_rank():
    model = _trained_model()
    dw = effective_delta_w(model).numpy()
    d = dw.shape[0]
    assert dw.shape == (d, d)
    sv = np.linalg.svd(dw, compute_uv=False)
    assert (sv[2:] < 1e-4).all(), f"delta_w must be rank<=2 by construction (adapter_rank=2), got singular values {sv}"
    assert d > 2, "ambient dimension must exceed adapter_rank for this to be a nontrivial completion target"


def test_low_rank_completion_attack_reports_observed_unobserved_and_gain():
    model = _trained_model()
    test_set = SyntheticFedISIC(num_samples=8, image_size=16, seed=9)
    result = low_rank_completion_attack(model, test_set, reveal_fraction=0.7, seed=5, completion_iterations=50)
    psr = result["parameter_space_recovery"]
    assert {"observed_entry_rmse", "unobserved_entry_rmse", "unobserved_entry_normalized_error", "cosine_similarity"} <= psr.keys()
    assert -1.0 <= psr["cosine_similarity"] <= 1.0
    assert "completion_gain_over_zero_fill" in result and "absolute_gain" in result["completion_gain_over_zero_fill"]
    assert 0.0 <= result["functional_recovery_fine_macro_f1"] <= 1.0
    assert result["true_rank"] == 2


def test_collusion_pools_strictly_more_information_than_either_solo_leak():
    """The pooled mask must cover at least as many entries as either
    individual leak -- the concrete, checkable claim collusion makes here."""
    model = _trained_model()
    test_set = SyntheticFedISIC(num_samples=8, image_size=16, seed=9)
    result = two_attacker_collusion_completion_attack(model, test_set, reveal_fraction_each=0.3, seed=7, completion_iterations=50)
    pooled_fraction = result["compute_budget"]["pooled_reveal_fraction"]
    assert pooled_fraction >= 0.3 - 1e-9  # union of two independent 30% leaks is >= 30%, strictly more in expectation
    assert pooled_fraction < 0.6 + 1e-9  # and at most the (overlap-free) sum of the two


if __name__ == "__main__":
    test_rank_equal_to_ambient_dim_makes_hard_impute_a_no_op()
    test_completion_beats_zero_fill_on_unobserved_entries()
    test_svd_completion_recovers_a_known_low_rank_matrix_better_with_more_reveal()
    test_rank_misspecification_degrades_recovery_relative_to_oracle()
    test_soft_impute_also_completes_meaningfully()
    test_mean_and_random_fill_leave_observed_entries_exact_but_differ_on_unobserved()
    test_effective_delta_w_has_constructive_rank_le_adapter_rank()
    test_low_rank_completion_attack_reports_observed_unobserved_and_gain()
    test_collusion_pools_strictly_more_information_than_either_solo_leak()
    print("OK")
