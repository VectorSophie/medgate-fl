"""Checks that the hierarchical-signal fixture (medgate/data/
hierarchical_synthetic.py) actually has learnable structure, that its
patient-level split is leak-free, and that the null-signal fixture
(medgate/data/synthetic.py) genuinely stays at chance -- the two properties
the whole P0-2 repair depends on. See docs/execution_plan.md Phase 1
(hierarchical) for how these numbers are read in context.

Run: PYTHONPATH=. pytest tests/test_hierarchical_synthetic.py -v
"""
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from medgate.data.hierarchical_synthetic import HierarchicalConfig, make_hierarchical_institutions, split_by_patient
from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, SyntheticFedISIC, make_synthetic_centers
from medgate.federated.fedavg import local_train
from medgate.metrics import evaluate_both
from medgate.models.backbone import MedGateModel

CFG = HierarchicalConfig(num_patients_per_institution=30, observations_per_patient=3)


def _train_test_pools(seed=0):
    insts = make_hierarchical_institutions(CFG, seed=seed)
    train, _val, test = split_by_patient(insts, seed=seed)
    return torch.utils.data.ConcatDataset(train), torch.utils.data.ConcatDataset(test), train, test


def test_coarse_classifier_learns_coarse_task():
    torch.manual_seed(0)
    train_pool, test_pool, _, _ = _train_test_pools()
    model = MedGateModel(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), feature_dim=64, adapter_rank=4)
    local_train(model, train_pool, epochs=5, batch_size=16, lr=0.01)
    util = evaluate_both(model, test_pool)
    chance = 1.0 / len(COARSE_CLASSES)
    assert util["coarse_macro_f1"] > chance + 0.3, f"coarse macro-F1 {util['coarse_macro_f1']} not comfortably above chance {chance}"


def test_unrestricted_model_learns_fine_task():
    torch.manual_seed(1)
    train_pool, test_pool, _, _ = _train_test_pools(seed=1)
    model = MedGateModel(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), feature_dim=64, adapter_rank=4)
    local_train(model, train_pool, epochs=5, batch_size=16, lr=0.01)
    util = evaluate_both(model, test_pool)
    chance = 1.0 / len(FINE_CLASSES)
    assert util["fine_macro_f1"] > chance + 0.1, f"fine macro-F1 {util['fine_macro_f1']} not above chance {chance}"


def test_labels_not_recoverable_from_sample_position():
    """Generation order/index must carry no label information -- fit a
    trivial classifier on ONLY the within-dataset index and confirm it
    cannot beat chance at predicting the fine label."""
    insts = make_hierarchical_institutions(CFG, seed=2)
    ds = insts[0]
    n = len(ds)
    X = torch.arange(n).float().view(-1, 1).numpy()
    y = ds.fine_labels.numpy()
    split = n // 2
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X[:split], y[:split])
    preds = clf.predict(X[split:])
    f1 = f1_score(y[split:], preds, average="macro", zero_division=0)
    chance = 1.0 / len(FINE_CLASSES)
    assert f1 < chance + 0.15, f"index-only classifier scored {f1}, suspiciously above chance {chance} -- label leaks via sample order"


def test_patient_variants_never_cross_train_test_boundary():
    insts = make_hierarchical_institutions(CFG, seed=3)
    train, val, test = split_by_patient(insts, seed=3)
    for i in range(len(insts)):
        train_p = set(train[i].patient_ids.tolist())
        val_p = set(val[i].patient_ids.tolist())
        test_p = set(test[i].patient_ids.tolist())
        assert not (train_p & val_p), f"institution {i}: patient(s) in both train and val"
        assert not (train_p & test_p), f"institution {i}: patient(s) in both train and test"
        assert not (val_p & test_p), f"institution {i}: patient(s) in both val and test"


def test_null_signal_fixture_remains_at_chance():
    """Regression guard for medgate/data/synthetic.py: if this ever starts
    scoring well above chance, something (e.g. an accidental label leak)
    broke its null-signal property."""
    torch.manual_seed(4)
    centers = make_synthetic_centers(samples_per_center=32, image_size=32, seed=4)
    train_pool = torch.utils.data.ConcatDataset(centers[:5])
    test_pool = centers[5]
    model = MedGateModel(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), feature_dim=64, adapter_rank=4)
    local_train(model, train_pool, epochs=3, batch_size=8, lr=0.01)
    util = evaluate_both(model, test_pool)
    # Generous upper bound: well below what real learnable signal produces
    # (test_unrestricted_model_learns_fine_task clears fine chance+0.1
    # comfortably; this fixture must not).
    assert util["fine_macro_f1"] < 0.25, f"null-signal fixture scored {util['fine_macro_f1']} on the fine task -- no longer signal-free"


def test_hierarchical_config_signal_strength_zero_collapses_toward_chance():
    """coarse_signal_strength=0 must make the coarse task materially
    harder than the default config -- a direct check that the 'signal
    strength' knobs actually do what their names claim, not just that
    SOME config is learnable."""
    torch.manual_seed(5)
    weak_cfg = HierarchicalConfig(
        num_patients_per_institution=30, observations_per_patient=3,
        coarse_signal_strength=0.0, fine_signal_strength=0.0,
    )
    insts = make_hierarchical_institutions(weak_cfg, seed=5)
    train, _val, test = split_by_patient(insts, seed=5)
    train_pool, test_pool = torch.utils.data.ConcatDataset(train), torch.utils.data.ConcatDataset(test)
    model = MedGateModel(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), feature_dim=64, adapter_rank=4)
    local_train(model, train_pool, epochs=5, batch_size=16, lr=0.01)
    util = evaluate_both(model, test_pool)
    assert util["coarse_macro_f1"] < 0.6, f"coarse_signal_strength=0 still scored {util['coarse_macro_f1']} -- knob has no effect"


def test_alternative_coarse_ontology_is_also_learnable_and_relabels_coarse_classes():
    """P1 (repair pass 4) coarse-ontology-sensitivity sweep hook: the
    alternative (benign/malignant/ambiguous, AK isolated) ontology must
    (a) actually change which coarse class a fine label maps to relative
    to the primary ontology, and (b) still produce a learnable coarse
    task -- otherwise the sensitivity sweep would just be silently running
    the same experiment twice under a different name."""
    from medgate.data.coarse_ontology import ALTERNATIVE_FINE_TO_COARSE_IDX
    from medgate.data.synthetic import FINE_TO_COARSE_IDX

    disagreements = sum(1 for f in FINE_TO_COARSE_IDX if FINE_TO_COARSE_IDX[f] != ALTERNATIVE_FINE_TO_COARSE_IDX[f])
    assert disagreements >= 3, "alternative ontology should meaningfully differ from the primary one, not just relabel indices"

    torch.manual_seed(6)
    insts = make_hierarchical_institutions(CFG, seed=6, fine_to_coarse_idx=ALTERNATIVE_FINE_TO_COARSE_IDX)
    train, _val, test = split_by_patient(insts, seed=6)
    train_pool, test_pool = torch.utils.data.ConcatDataset(train), torch.utils.data.ConcatDataset(test)
    model = MedGateModel(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), feature_dim=64, adapter_rank=4)
    local_train(model, train_pool, epochs=5, batch_size=16, lr=0.01)
    util = evaluate_both(model, test_pool)
    chance = 1.0 / len(COARSE_CLASSES)
    assert util["coarse_macro_f1"] > chance + 0.3, f"alternative-ontology coarse macro-F1 {util['coarse_macro_f1']} not comfortably above chance {chance}"


if __name__ == "__main__":
    test_coarse_classifier_learns_coarse_task()
    test_unrestricted_model_learns_fine_task()
    test_labels_not_recoverable_from_sample_position()
    test_patient_variants_never_cross_train_test_boundary()
    test_null_signal_fixture_remains_at_chance()
    test_hierarchical_config_signal_strength_zero_collapses_toward_chance()
    test_alternative_coarse_ontology_is_also_learnable_and_relabels_coarse_classes()
    print("OK")
