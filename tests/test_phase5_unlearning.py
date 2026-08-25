"""Phase 5 checks: each unlearning method actually does what it claims to
do at the mechanism level (not just "runs without error").

Run: PYTHONPATH=. pytest tests/test_phase5_unlearning.py -v
"""
import torch

from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, SyntheticFedISIC, make_synthetic_centers
from medgate.models.backbone import MedGateModel
from medgate.unlearning.methods import (
    adapter_deletion_and_retrain,
    checkpoint_rollback,
    full_retrain,
    gradient_ascent_unlearning,
    key_revocation_only,
)

MODEL_KWARGS = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), feature_dim=16, adapter_rank=2)


def _authorized_model_and_data():
    centers = make_synthetic_centers(samples_per_center=8, image_size=16, seed=1)
    torch.manual_seed(0)
    model = MedGateModel(**MODEL_KWARGS)
    return model, centers


def test_key_revocation_only_returns_the_identical_object():
    """The whole point of this method is that it changes nothing."""
    model, _ = _authorized_model_and_data()
    result = key_revocation_only(model)
    assert result is model


def test_checkpoint_rollback_matches_a_fresh_same_seed_init():
    rollback = checkpoint_rollback(MODEL_KWARGS, seed=5)
    torch.manual_seed(5)
    fresh = MedGateModel(**MODEL_KWARGS)
    for k in rollback.state_dict():
        assert torch.allclose(rollback.state_dict()[k], fresh.state_dict()[k])


def test_adapter_deletion_changes_adapter_but_not_backbone():
    model, centers = _authorized_model_and_data()
    backbone_before = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    adapter_before = model.adapter.up.weight.clone()

    result = adapter_deletion_and_retrain(model, centers, epochs=2, batch_size=4, lr=0.05, seed=0)

    for k, v in result.backbone.state_dict().items():
        assert torch.allclose(v, backbone_before[k]), "backbone should be frozen/untouched by adapter deletion"
    assert not torch.allclose(result.adapter.up.weight, adapter_before), "adapter should have been reinitialized and retrained"


def test_gradient_ascent_unlearning_moves_the_model():
    model, centers = _authorized_model_and_data()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    removed = centers[0]
    retained = torch.utils.data.ConcatDataset(centers[1:])
    result = gradient_ascent_unlearning(model, removed, retained, ascent_steps=3, descent_steps=3, batch_size=4, lr=0.05, seed=0)
    moved = any(not torch.allclose(result.state_dict()[k], before[k]) for k in before)
    assert moved
    for v in result.state_dict().values():
        assert torch.isfinite(v).all()


def test_full_retrain_is_deterministic_given_the_same_seed():
    _, centers = _authorized_model_and_data()
    m1 = full_retrain(centers, MODEL_KWARGS, rounds=1, epochs=1, batch_size=4, lr=0.05, seed=3)
    m2 = full_retrain(centers, MODEL_KWARGS, rounds=1, epochs=1, batch_size=4, lr=0.05, seed=3)
    for k in m1.state_dict():
        assert torch.allclose(m1.state_dict()[k], m2.state_dict()[k]), f"{k}: same seed produced different weights"


if __name__ == "__main__":
    test_key_revocation_only_returns_the_identical_object()
    test_checkpoint_rollback_matches_a_fresh_same_seed_init()
    test_adapter_deletion_changes_adapter_but_not_backbone()
    test_gradient_ascent_unlearning_moves_the_model()
    test_full_retrain_is_deterministic_given_the_same_seed()
    print("OK")
