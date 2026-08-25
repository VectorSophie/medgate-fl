"""Checks for the P0-3 unlearning-metric repair (docs/execution_plan.md
Phase 5): raw AUC near 0.0 is just as perfectly-separable as raw AUC near
1.0 (only the score direction differs), so treating raw AUC as "closer to
0.5 is better, closer to 0 is fine" was a real bug -- caught and fixed
2026-08-26. symmetric_auc / attack_advantage fold both directions into one
number where 0.5 is always "no signal."

Run: PYTHONPATH=. pytest tests/test_unlearning_metrics.py -v
"""
import numpy as np
from sklearn.metrics import roc_auc_score

from medgate.attacks.membership_inference import attack_advantage, symmetric_auc


def test_symmetric_auc_boundary_values():
    assert symmetric_auc(0.5) == 0.5   # no signal
    assert symmetric_auc(1.0) == 1.0   # perfectly separable, correct direction
    assert symmetric_auc(0.0) == 1.0   # perfectly separable, REVERSED direction -- the bug this fixes
    assert abs(symmetric_auc(0.9) - 0.9) < 1e-9
    assert abs(symmetric_auc(0.1) - 0.9) < 1e-9  # symmetric around 0.5


def test_attack_advantage_boundary_values():
    assert attack_advantage(0.5) == 0.0
    assert attack_advantage(1.0) == 1.0
    assert attack_advantage(0.0) == 1.0  # the same bug, same fix
    assert abs(attack_advantage(0.75) - 0.5) < 1e-9
    assert abs(attack_advantage(0.25) - 0.5) < 1e-9  # symmetric around 0.5


def test_raw_auc_near_zero_would_have_been_misread_as_good_forgetting_by_the_old_code():
    """Directly demonstrates the bug: a classifier whose scores are
    perfectly informative but anti-correlated with the label (AUC=0.0)
    must be read as maximally distinguishable, not as good forgetting.
    The OLD formula (2*auc-1) would have returned -1.0 here; a caller
    reading '2*auc-1 near 0 is good forgetting' would have wrongly called
    this a perfect result."""
    labels = [1, 1, 1, 0, 0, 0]
    scores = [-3, -2, -1, 1, 2, 3]  # perfectly separable, but higher score => label 0
    auc = roc_auc_score(labels, scores)
    assert auc == 0.0
    old_buggy_formula = 2 * auc - 1
    assert old_buggy_formula == -1.0, "sanity-check the bug actually reproduces as described"
    assert attack_advantage(auc) == 1.0, "the corrected formula must report this as maximal signal, not as near-zero"
    assert symmetric_auc(auc) == 1.0


def test_loss_threshold_mi_reports_symmetric_auc_and_advantage_consistently():
    from medgate.attacks.membership_inference import loss_threshold_membership_inference
    from medgate.data.synthetic import SyntheticFedISIC
    from medgate.models.backbone import MedGateModel

    model = MedGateModel(num_coarse=3, num_fine=8, feature_dim=16, adapter_rank=2)
    members = SyntheticFedISIC(num_samples=10, image_size=16, seed=1)
    nonmembers = SyntheticFedISIC(num_samples=10, image_size=16, seed=2)
    result = loss_threshold_membership_inference(model, members, nonmembers, batch_size=4)
    assert "symmetric_auc" in result and "attack_advantage" in result
    assert 0.5 <= result["symmetric_auc"] <= 1.0
    assert 0.0 <= result["attack_advantage"] <= 1.0
    assert abs(result["symmetric_auc"] - (0.5 + result["attack_advantage"] / 2)) < 1e-9


if __name__ == "__main__":
    test_symmetric_auc_boundary_values()
    test_attack_advantage_boundary_values()
    test_raw_auc_near_zero_would_have_been_misread_as_good_forgetting_by_the_old_code()
    test_loss_threshold_mi_reports_symmetric_auc_and_advantage_consistently()
    print("OK")
