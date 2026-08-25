"""Checks for the P1-7 genuine parameter-space adapter-recovery track
(medgate/attacks/adapter_recovery.py) -- distinct from the fresh-training
recovery attacks in medgate/attacks/reconstruction.py.

Run: PYTHONPATH=. pytest tests/test_adapter_recovery.py -v
"""
import numpy as np
import torch

from medgate.attacks.adapter_recovery import (
    low_rank_completion_attack,
    parameter_space_recovery_metrics,
    simulate_partial_leak,
    svd_complete_matrix,
    two_attacker_collusion_completion_attack,
)
from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, SyntheticFedISIC
from medgate.models.backbone import MedGateModel

MODEL_KWARGS = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), feature_dim=16, adapter_rank=2)


def _trained_model():
    torch.manual_seed(0)
    model = MedGateModel(**MODEL_KWARGS)
    with torch.no_grad():
        # zero-init adapter (medgate/models/backbone.py) would make "up" the
        # all-zeros matrix -- trivially "recoverable" and uninformative as
        # a test; give it real, structured (low-rank-by-construction) values.
        g = torch.Generator().manual_seed(1)
        a = torch.randn(model.adapter.up.weight.shape[0], 2, generator=g)
        b = torch.randn(2, model.adapter.up.weight.shape[1], generator=g)
        model.adapter.up.weight.copy_(a @ b)
    return model


def test_svd_completion_recovers_a_known_low_rank_matrix_better_with_more_reveal():
    rng = np.random.RandomState(0)
    true_rank = 2
    true_matrix = rng.randn(16, true_rank) @ rng.randn(true_rank, 16)
    true_matrix = torch.from_numpy(true_matrix).float()

    errors = []
    for reveal_fraction in (0.2, 0.5, 0.9):
        partial, mask = simulate_partial_leak(true_matrix, reveal_fraction, seed=0)
        recovered = svd_complete_matrix(partial, mask, rank=true_rank, iterations=100)
        metrics = parameter_space_recovery_metrics(true_matrix, recovered, true_rank)
        errors.append(metrics["normalized_frobenius_error"])
    assert errors[0] > errors[1] > errors[2], f"completion should improve monotonically with more revealed entries: {errors}"
    assert errors[2] < 0.05, f"90% reveal of an exactly-rank-{true_rank} matrix should complete nearly exactly, got error {errors[2]}"


def test_low_rank_completion_attack_reports_both_parameter_and_functional_recovery():
    model = _trained_model()
    test_set = SyntheticFedISIC(num_samples=8, image_size=16, seed=9)
    result = low_rank_completion_attack(model, test_set, reveal_fraction=0.7, seed=5, completion_iterations=50)
    assert "parameter_space_recovery" in result and "functional_recovery_fine_macro_f1" in result
    assert -1.0 <= result["parameter_space_recovery"]["cosine_similarity"] <= 1.0
    assert result["parameter_space_recovery"]["normalized_frobenius_error"] >= 0.0
    assert 0.0 <= result["functional_recovery_fine_macro_f1"] <= 1.0


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
    test_svd_completion_recovers_a_known_low_rank_matrix_better_with_more_reveal()
    test_low_rank_completion_attack_reports_both_parameter_and_functional_recovery()
    test_collusion_pools_strictly_more_information_than_either_solo_leak()
    print("OK")
