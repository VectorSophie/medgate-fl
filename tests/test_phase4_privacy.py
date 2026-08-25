"""Phase 4 checks: secure-aggregation masking actually hides individual
updates while preserving the exact sum, and DP-SGD actually trains under
Opacus and returns a finite epsilon.

Run: PYTHONPATH=. pytest tests/test_phase4_privacy.py -v
"""
import torch

from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, SyntheticFedISIC, make_synthetic_centers
from medgate.models.backbone import MedGateModel
from medgate.privacy.dp_sgd import dp_local_train
from medgate.privacy.secure_aggregation import (
    confidentiality_check,
    mask_client_updates,
    secure_aggregate_updates,
    secure_fedavg_round,
)

MODEL_KWARGS = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), feature_dim=16, adapter_rank=2)


def _fake_updates(n_clients=3, seed=0):
    torch.manual_seed(seed)
    shape_a, shape_b = (5, 5), (3,)
    return [{"a": torch.randn(shape_a), "b": torch.randn(shape_b)} for _ in range(n_clients)]


def test_masking_preserves_the_exact_sum():
    updates = _fake_updates()
    masked = mask_client_updates(updates, seed=42)
    aggregated = secure_aggregate_updates(masked, weights=[1.0] * len(updates))
    true_mean = {k: sum(u[k] for u in updates) / len(updates) for k in updates[0]}
    for k in true_mean:
        assert torch.allclose(aggregated[k], true_mean[k], atol=1e-5)


def test_masking_correctness_error_is_essentially_zero():
    updates = _fake_updates()
    masked = mask_client_updates(updates, seed=42)
    check = confidentiality_check(updates, masked)
    assert check["aggregate_reconstruction_max_abs_error"] < 1e-5


def test_masking_output_depends_on_secret_mask_not_recoverable_without_it():
    """The property that actually demonstrates hiding (see the long
    caution in medgate/privacy/secure_aggregation.py confidentiality_check):
    the SAME true update, masked with two different (attacker-unknown)
    mask seeds, must produce clearly different outputs -- i.e. the masked
    value is not a stable, invertible function of the truth an observer
    could learn to undo."""
    updates = _fake_updates()
    masked_1 = mask_client_updates(updates, seed=1)
    masked_2 = mask_client_updates(updates, seed=2)
    diff = (masked_1[0]["a"] - masked_2[0]["a"]).abs().mean().item()
    true_scale = updates[0]["a"].abs().mean().item()
    assert diff > true_scale, "masked outputs should differ across mask seeds by more than the true update's own scale"


def test_secure_aggregate_rejects_unequal_weights():
    updates = _fake_updates()
    masked = mask_client_updates(updates, seed=1)
    try:
        secure_aggregate_updates(masked, weights=[1.0, 2.0, 1.0])
        assert False, "expected ValueError for unequal weights"
    except ValueError:
        pass


def test_secure_fedavg_round_changes_the_model_and_stays_finite():
    torch.manual_seed(0)
    centers = make_synthetic_centers(samples_per_center=8, image_size=16, seed=3)[:2]
    model = MedGateModel(**MODEL_KWARGS)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    new_state = secure_fedavg_round(model, centers, epochs=1, batch_size=4, lr=0.05, seed=7)
    for k, v in new_state.items():
        assert torch.isfinite(v).all(), f"non-finite value in {k}"
    moved = any(not torch.allclose(new_state[k], before[k]) for k in new_state)
    assert moved


def test_dp_local_train_produces_finite_params_and_epsilon():
    torch.manual_seed(0)
    model = MedGateModel(**MODEL_KWARGS)
    dataset = SyntheticFedISIC(num_samples=8, image_size=16, seed=4)
    state, epsilon = dp_local_train(
        model, dataset, epochs=1, batch_size=4, lr=0.05, noise_multiplier=1.0, max_grad_norm=1.0
    )
    assert epsilon > 0 and torch.isfinite(torch.tensor(epsilon))
    for k, v in state.items():
        assert torch.isfinite(v).all(), f"non-finite value in {k}"


if __name__ == "__main__":
    test_masking_preserves_the_exact_sum()
    test_masking_correctness_error_is_essentially_zero()
    test_masking_output_depends_on_secret_mask_not_recoverable_without_it()
    test_secure_aggregate_rejects_unequal_weights()
    test_secure_fedavg_round_changes_the_model_and_stays_finite()
    test_dp_local_train_produces_finite_params_and_epsilon()
    print("OK")
