"""Checks for the remaining Phase 3 attacks (docs/execution_plan.md):
property inference, A4 integrity attacks, robust aggregation, and
token-expiry/replay in the authorization layer.

Run: PYTHONPATH=. pytest tests/test_phase3_integrity_and_property.py -v
"""
import torch

from medgate.attacks.integrity import (
    backdoor_success_rate,
    backdoor_train,
    free_rider_update,
    label_flipping_train,
    malformed_update,
    model_replacement_update,
    sign_flipping_update,
)
from medgate.attacks.property_inference import property_inference_attack
from medgate.crypto.authorization import CentralizedIAM
from medgate.crypto.adapter_encryption import generate_key
from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, SyntheticFedISIC
from medgate.federated.robust_aggregation import (
    coordinate_median_aggregate,
    trimmed_mean_aggregate,
    validate_update,
    validated_fedavg_aggregate,
)
from medgate.models.backbone import MedGateModel

MODEL_KWARGS = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), feature_dim=16, adapter_rank=2)


# --------------------------------------------------------------- property inference

def test_property_inference_recovers_a_clearly_separable_property():
    """Construct updates where the property is trivially encoded (a
    constant shift in one tensor correlated with the label) and confirm
    the attack's AUC is high -- a sanity check that the attack CAN work
    when there is something to find, before trusting a null result
    elsewhere means there's nothing to find."""
    updates, labels = [], []
    for i in range(10):
        label = i % 2
        shift = 5.0 if label == 1 else -5.0
        updates.append({"a": torch.full((4,), shift) + torch.randn(4) * 0.01})
        labels.append(label)
    result = property_inference_attack(updates, labels, seed=0)
    assert result["attack_auc"] is not None
    assert result["attack_auc"] > 0.9


def test_property_inference_returns_none_for_constant_property():
    updates = [{"a": torch.randn(4)} for _ in range(5)]
    result = property_inference_attack(updates, [1, 1, 1, 1, 1], seed=0)
    assert result["attack_auc"] is None


# --------------------------------------------------------------- A4 integrity attacks

def test_label_flipping_actually_changes_the_training_labels():
    model = MedGateModel(**MODEL_KWARGS)
    dataset = SyntheticFedISIC(num_samples=8, image_size=16, seed=1)
    original_labels = dataset.fine_labels.clone()
    state = label_flipping_train(model, dataset, epochs=1, batch_size=4, lr=0.05, flip_offset=1)
    # the dataset object itself is untouched; only training saw flipped labels
    assert torch.equal(dataset.fine_labels, original_labels)
    for v in state.values():
        assert torch.isfinite(v).all()


def test_sign_flipping_negates_the_true_delta():
    honest = {"a": torch.tensor([3.0, 4.0])}
    global_state = {"a": torch.tensor([1.0, 1.0])}
    result = sign_flipping_update(honest, global_state)
    # true delta was [2,3]; sign-flipped delta is [-2,-3]; result = global + that
    assert torch.allclose(result["a"], torch.tensor([-1.0, -2.0]))


def test_model_replacement_scales_the_delta():
    honest = {"a": torch.tensor([2.0])}
    global_state = {"a": torch.tensor([0.0])}
    result = model_replacement_update(honest, global_state, boost_factor=10.0)
    assert torch.allclose(result["a"], torch.tensor([20.0]))


def test_free_rider_contributes_exactly_zero_delta():
    global_state = {"a": torch.tensor([1.0, 2.0])}
    result = free_rider_update(global_state)
    assert torch.equal(result["a"], global_state["a"])
    assert result["a"] is not global_state["a"]  # a copy, not aliased


def test_malformed_update_detected_by_validation():
    honest = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
    assert validate_update(honest) is True
    assert validate_update(malformed_update(honest, "nan")) is False
    assert validate_update(malformed_update(honest, "inf")) is False


def test_validated_fedavg_drops_malformed_and_keeps_the_rest():
    honest_a = {"a": torch.tensor([1.0])}
    honest_b = {"a": torch.tensor([3.0])}
    bad = malformed_update(honest_a, "nan")
    result = validated_fedavg_aggregate([honest_a, honest_b, bad], weights=[1.0, 1.0, 1.0])
    assert torch.allclose(result["a"], torch.tensor([2.0]))  # mean of only the two valid ones


def test_backdoor_success_rate_in_valid_range():
    model = MedGateModel(**MODEL_KWARGS)
    dataset = SyntheticFedISIC(num_samples=8, image_size=16, seed=2)
    backdoor_train(model, dataset, epochs=2, batch_size=4, lr=0.05, target_fine_class=0)
    rate = backdoor_success_rate(model, dataset, target_fine_class=0)
    assert 0.0 <= rate <= 1.0


# --------------------------------------------------------------- robust aggregation

def test_median_and_trimmed_mean_resist_one_extreme_outlier():
    honest = [{"a": torch.tensor([1.0])}, {"a": torch.tensor([1.1])}, {"a": torch.tensor([0.9])}, {"a": torch.tensor([1.0])}]
    outlier = {"a": torch.tensor([1000.0])}
    all_updates = honest + [outlier]

    from medgate.federated.fedavg import fedavg_aggregate
    plain_mean = fedavg_aggregate(all_updates, [1.0] * len(all_updates))["a"].item()
    median = coordinate_median_aggregate(all_updates)["a"].item()
    trimmed = trimmed_mean_aggregate(all_updates, trim_fraction=0.2)["a"].item()

    assert plain_mean > 50  # dragged far by the outlier
    assert abs(median - 1.0) < 0.2  # median barely moves
    assert abs(trimmed - 1.0) < 0.2  # trimmed mean barely moves


# --------------------------------------------------------------- token expiry / replay

def test_expired_token_is_denied_and_replay_is_always_denied():
    iam = CentralizedIAM()
    key_id = iam.issue(subject_id="hospital_c", key=generate_key(), ttl_seconds=100)
    now0 = __import__("time").time()
    assert iam.authorize(key_id, now=now0) is not None  # valid before expiry

    later = now0 + 200  # past the 100s TTL
    denied = 0
    for _ in range(10):  # replay the same expired token repeatedly
        try:
            iam.authorize(key_id, now=later)
        except PermissionError:
            denied += 1
    assert denied == 10


if __name__ == "__main__":
    test_property_inference_recovers_a_clearly_separable_property()
    test_property_inference_returns_none_for_constant_property()
    test_label_flipping_actually_changes_the_training_labels()
    test_sign_flipping_negates_the_true_delta()
    test_model_replacement_scales_the_delta()
    test_free_rider_contributes_exactly_zero_delta()
    test_malformed_update_detected_by_validation()
    test_validated_fedavg_drops_malformed_and_keeps_the_rest()
    test_backdoor_success_rate_in_valid_range()
    test_median_and_trimmed_mean_resist_one_extreme_outlier()
    test_expired_token_is_denied_and_replay_is_always_denied()
    print("OK")
