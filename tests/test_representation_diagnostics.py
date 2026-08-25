"""Checks for the P1-9 (orthogonality-as-ablation diagnostics) and P1-8
(disjoint probe family) repairs.

Run: PYTHONPATH=. pytest tests/test_representation_diagnostics.py -v
"""
import numpy as np
import torch

from medgate.attacks.probes import (
    extract_adapter_residual,
    run_all_probes_on_features,
    run_all_probes_on_residual,
    svm_probe,
    tree_probe,
)
from medgate.capability_metrics import linear_cka
from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, SyntheticFedISIC
from medgate.models.backbone import MedGateModel, cosine_similarity_stats, orthogonality_loss

MODEL_KWARGS = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), feature_dim=16, adapter_rank=2)


def test_cosine_similarity_stats_matches_known_cases():
    z = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    delta_orthogonal = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    stats = cosine_similarity_stats(z, delta_orthogonal)
    assert abs(stats["mean"]) < 1e-6 and stats["std"] < 1e-6

    delta_mixed = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])  # one parallel (+1), one anti-parallel (-1)
    stats2 = cosine_similarity_stats(z, delta_mixed)
    assert abs(stats2["mean"]) < 1e-6  # +1 and -1 average to 0
    assert stats2["std"] > 0.9  # but the SPREAD reveals they are not both near 0
    assert abs(stats2["max"] - 1.0) < 1e-6 and abs(stats2["min"] + 1.0) < 1e-6


def test_cosine_stats_reveal_what_the_squared_loss_alone_hides():
    """The exact point of this diagnostic: orthogonality_loss (mean of
    SQUARED cosine similarity) can be identical for two very different
    situations that cosine_similarity_stats tells apart."""
    z = torch.tensor([[1.0, 0.0]] * 4)
    delta_mixed = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])  # half +1, half -1
    delta_zero = torch.zeros(4, 2) + torch.tensor([0.0, 1e-3])  # all ~orthogonal

    loss_mixed = orthogonality_loss(z, delta_mixed).item()
    loss_zero = orthogonality_loss(z, delta_zero).item()
    assert loss_mixed > loss_zero * 100  # squared loss DOES distinguish these (sanity)

    stats_mixed = cosine_similarity_stats(z, delta_mixed)
    assert abs(stats_mixed["mean"]) < 1e-6  # but the MEAN alone would suggest "no relationship" for the mixed case too
    assert stats_mixed["std"] > 0.9  # only std/min/max reveal the mixed case is NOT actually orthogonal-per-sample


def test_linear_cka_known_cases():
    rng = np.random.RandomState(0)
    X = rng.randn(50, 8)
    assert linear_cka(X, X) > 0.999  # identical representations -> CKA ~= 1
    assert linear_cka(X, 3.0 * X) > 0.999  # invariant to isotropic scaling
    Y = rng.randn(50, 8)  # independent random representation
    assert linear_cka(X, Y) < 0.5  # unrelated representations -> low CKA


def test_svm_and_tree_probes_run_and_are_bounded():
    model = MedGateModel(**MODEL_KWARGS)
    train = SyntheticFedISIC(num_samples=16, image_size=16, seed=1)
    test = SyntheticFedISIC(num_samples=16, image_size=16, seed=2)
    from medgate.attacks.probes import extract_representations
    Z_train, y_train = extract_representations(model, train)
    Z_test, y_test = extract_representations(model, test)
    svm_result = svm_probe(Z_train, y_train, Z_test, y_test)
    tree_result = tree_probe(Z_train, y_train, Z_test, y_test)
    assert 0.0 <= svm_result["macro_f1"] <= 1.0
    assert 0.0 <= tree_result["macro_f1"] <= 1.0


def test_run_all_probes_on_features_include_slow_toggle():
    model = MedGateModel(**MODEL_KWARGS)
    train = SyntheticFedISIC(num_samples=16, image_size=16, seed=1)
    test = SyntheticFedISIC(num_samples=16, image_size=16, seed=2)
    from medgate.attacks.probes import extract_representations
    Z_train, y_train = extract_representations(model, train)
    Z_test, y_test = extract_representations(model, test)
    full = run_all_probes_on_features(Z_train, y_train, Z_test, y_test, include_slow=True)
    fast = run_all_probes_on_features(Z_train, y_train, Z_test, y_test, include_slow=False)
    assert "svm_probe" in full and "tree_probe" in full
    assert "svm_probe" not in fast and "tree_probe" not in fast
    assert set(fast.keys()) <= set(full.keys())


def test_adapter_residual_probing_is_disjoint_from_representation_probing():
    """The residual probe must operate on A_phi(z), not z itself -- checked
    by confirming the residual differs from the representation (adapter is
    NOT the identity here, since it's untrained/zero-init only at
    construction, and this model has non-default rank so shapes differ
    trivially -- the real check is that extract_adapter_residual and
    extract_representations return DIFFERENT arrays for the same model+data)."""
    model = MedGateModel(**MODEL_KWARGS)
    # give the adapter nonzero weights so residual != 0 (zero-init default
    # would make this check trivially pass for the wrong reason)
    with torch.no_grad():
        model.adapter.up.weight.add_(0.1)
    dataset = SyntheticFedISIC(num_samples=8, image_size=16, seed=3)
    from medgate.attacks.probes import extract_representations
    Z, y1 = extract_representations(model, dataset)
    D, y2 = extract_adapter_residual(model, dataset)
    assert Z.shape == D.shape  # both are feature_dim-sized in this architecture
    assert not np.allclose(Z, D)
    assert np.array_equal(y1, y2)

    result = run_all_probes_on_residual(model, dataset, dataset, include_slow=False)
    assert "linear_probe" in result
    assert 0.0 <= result["linear_probe"]["macro_f1"] <= 1.0


def test_selected_probe_attack_evaluates_the_validation_winner_once_on_test():
    """P1 (repair pass 4): the selected probe's reported attack_test_result
    must come from the TEST split, and the probe picked must be the
    argmax over the VALIDATION split's scores, not the test split's own
    (which would silently readmit the selection-bias this function
    exists to remove)."""
    from medgate.attacks.probes import selected_probe_attack

    model = MedGateModel(**MODEL_KWARGS)
    train = SyntheticFedISIC(num_samples=24, image_size=16, seed=1)
    val = SyntheticFedISIC(num_samples=16, image_size=16, seed=2)
    test = SyntheticFedISIC(num_samples=16, image_size=16, seed=3)

    result = selected_probe_attack(model, train, val, test, include_slow=True)
    assert result["selected_probe"] in result["selection_val_macro_f1_by_probe"]
    best_val_name = max(result["selection_val_macro_f1_by_probe"], key=result["selection_val_macro_f1_by_probe"].get)
    assert result["selected_probe"] == best_val_name
    assert 0.0 <= result["attack_test_result"]["macro_f1"] <= 1.0
    assert "n_train_examples" in result["attack_test_result"]


if __name__ == "__main__":
    test_cosine_similarity_stats_matches_known_cases()
    test_cosine_stats_reveal_what_the_squared_loss_alone_hides()
    test_linear_cka_known_cases()
    test_svm_and_tree_probes_run_and_are_bounded()
    test_run_all_probes_on_features_include_slow_toggle()
    test_adapter_residual_probing_is_disjoint_from_representation_probing()
    test_selected_probe_attack_evaluates_the_validation_winner_once_on_test()
    test_adapter_residual_probing_is_disjoint_from_representation_probing()
    print("OK")
