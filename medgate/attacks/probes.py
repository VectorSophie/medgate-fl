"""Capability-recovery probes (Phase 2/3, docs/execution_plan.md): given a
frozen model's PUBLIC representation f_theta(x), how much fine-label
information can a probe recover? This directly measures Residual Fine
Capability (RFC, docs/research_scope.md §7) — the central test of whether
"hiding" the fine head actually isolates capability, or whether the public
representation leaks it regardless.

Uses sklearn's LogisticRegression/MLPClassifier/KNeighborsClassifier as the
probes rather than a hand-rolled gradient training loop — they are
standard, already-installed classifiers and a second bespoke training loop
here would not test anything these don't already cover.
"""
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


@torch.no_grad()
def extract_representations(model, dataset, batch_size: int = 32):
    """Frozen public representation f_theta(x) + the FINE label (index 1
    in the dataset tuple) — the probe's attack target."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    zs, ys = [], []
    for images, y_fine, _y_coarse in loader:
        zs.append(model.representation(images))
        ys.append(y_fine)
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()


def _fit_eval(clf, Z_train, y_train, Z_test, y_test) -> dict:
    start = time.time()
    clf.fit(Z_train, y_train)
    fit_seconds = time.time() - start
    preds = clf.predict(Z_test)
    return {
        "macro_f1": f1_score(y_test, preds, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, preds),
        "n_train_examples": int(len(y_train)),
        "fit_seconds": round(fit_seconds, 4),
    }


def linear_probe(Z_train, y_train, Z_test, y_test, seed: int = 0) -> dict:
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    return _fit_eval(clf, Z_train, y_train, Z_test, y_test)


def nonlinear_probe(Z_train, y_train, Z_test, y_test, seed: int = 0) -> dict:
    clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)
    return _fit_eval(clf, Z_train, y_train, Z_test, y_test)


def knn_probe(Z_train, y_train, Z_test, y_test, k: int = 5) -> dict:
    clf = KNeighborsClassifier(n_neighbors=k)
    return _fit_eval(clf, Z_train, y_train, Z_test, y_test)


def svm_probe(Z_train, y_train, Z_test, y_test, seed: int = 0) -> dict:
    """RBF-kernel SVM (P1-8: 'RBF-SVM if feasible' — feasible at this
    project's dataset scale; not attempted on anything larger without
    re-checking cost, SVC training scales poorly with N)."""
    clf = SVC(kernel="rbf", random_state=seed)  # only .predict() is used, no need for probability estimates
    return _fit_eval(clf, Z_train, y_train, Z_test, y_test)


def tree_probe(Z_train, y_train, Z_test, y_test, seed: int = 0) -> dict:
    """Random forest (P1-8: 'tree/boosting probe if feasible' — a forest
    rather than a boosted model specifically for training-time cost at
    this project's scale, and because sklearn's forest needs no extra
    tuning to be a reasonable probe out of the box)."""
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    return _fit_eval(clf, Z_train, y_train, Z_test, y_test)


def fewshot_probe(Z_train, y_train, Z_test, y_test, k_per_class: int = 5, seed: int = 0) -> dict:
    """Linear probe trained on only k_per_class examples per fine class
    (deterministically subsampled) — tests whether an attacker with very
    little auxiliary labeled data can still recover fine capability."""
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y_train):
        c_idx = np.where(y_train == c)[0]
        rng.shuffle(c_idx)
        idx.extend(c_idx[:k_per_class].tolist())
    idx = np.array(idx)
    return linear_probe(Z_train[idx], y_train[idx], Z_test, y_test, seed=seed)


@torch.no_grad()
def extract_public_outputs(model, dataset, batch_size: int = 32):
    """The model's PUBLIC OUTPUT (coarse logits) — not the internal
    representation. Used to operationalize U_public (docs/research_scope.md
    §7): the fine-label recoverability floor available from the exposed
    output alone, before any access to internals."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    outs, ys = [], []
    for images, y_fine, _y_coarse in loader:
        outs.append(model.forward_public(images))
        ys.append(y_fine)
    return torch.cat(outs).numpy(), torch.cat(ys).numpy()


def output_only_probe(model, train_dataset, test_dataset, batch_size: int = 32, seed: int = 0) -> dict:
    """U_public: a linear probe fit on the public output (coarse logits)
    alone, predicting the fine label. This is the floor an unauthorized
    user gets with zero representation access — everything else in this
    module (run_all_probes) measures how much MORE leaks once
    representation access is added on top of this floor."""
    O_train, y_train = extract_public_outputs(model, train_dataset, batch_size)
    O_test, y_test = extract_public_outputs(model, test_dataset, batch_size)
    return linear_probe(O_train, y_train, O_test, y_test, seed=seed)


def run_all_probes_on_features(Z_train, y_train, Z_test, y_test, seed: int = 0, include_slow: bool = True) -> dict:
    """The full probe family (P1-8: a disjoint battery from whatever
    adversary_head was used DURING training — none of these share any
    weights with the model's own gradient-reversal adversary; they are
    independently-fit sklearn classifiers on frozen features).
    `include_slow=False` drops SVM/tree (still available individually)
    for callers on a tight compute budget — RFC (residual_fine_capability)
    is a max over whichever probes were actually run, so dropping some
    only makes RFC more conservative, never wrong."""
    probes = {
        "linear_probe": linear_probe(Z_train, y_train, Z_test, y_test, seed),
        "nonlinear_probe": nonlinear_probe(Z_train, y_train, Z_test, y_test, seed),
        "knn_probe": knn_probe(Z_train, y_train, Z_test, y_test),
        "fewshot_probe_k5": fewshot_probe(Z_train, y_train, Z_test, y_test, k_per_class=5, seed=seed),
    }
    if include_slow:
        probes["svm_probe"] = svm_probe(Z_train, y_train, Z_test, y_test, seed)
        probes["tree_probe"] = tree_probe(Z_train, y_train, Z_test, y_test, seed)
    return probes


def run_all_probes(model, train_dataset, test_dataset, batch_size: int = 32, seed: int = 0, include_slow: bool = True) -> dict:
    Z_train, y_train = extract_representations(model, train_dataset, batch_size)
    Z_test, y_test = extract_representations(model, test_dataset, batch_size)
    return run_all_probes_on_features(Z_train, y_train, Z_test, y_test, seed, include_slow)


def probe_by_name(name: str, Z_train, y_train, Z_test, y_test, seed: int = 0) -> dict:
    """Dispatch to one probe by its run_all_probes_on_features key, with a
    single uniform call signature (some underlying probes, e.g. knn_probe,
    do not themselves take a seed) — used by selected_probe_attack below
    to refit exactly the probe TYPE selected on the validation split."""
    if name == "linear_probe":
        return linear_probe(Z_train, y_train, Z_test, y_test, seed=seed)
    if name == "nonlinear_probe":
        return nonlinear_probe(Z_train, y_train, Z_test, y_test, seed=seed)
    if name == "knn_probe":
        return knn_probe(Z_train, y_train, Z_test, y_test)
    if name == "fewshot_probe_k5":
        return fewshot_probe(Z_train, y_train, Z_test, y_test, k_per_class=5, seed=seed)
    if name == "svm_probe":
        return svm_probe(Z_train, y_train, Z_test, y_test, seed=seed)
    if name == "tree_probe":
        return tree_probe(Z_train, y_train, Z_test, y_test, seed=seed)
    raise ValueError(f"unknown probe name {name!r}")


def selected_probe_attack(
    model, train_dataset, val_dataset, test_dataset, batch_size: int = 32, seed: int = 0, include_slow: bool = True,
) -> dict:
    """P1 (repair pass 4) fix for a probe-selection data-leakage issue:
    earlier drafts reported 'BestProbeRFC' as the MAX across every probe
    family, evaluated on the SAME held-out split used for the final
    number. Selecting the best of several probes and reporting its score
    on the very data used to select it is itself a form of
    multiple-comparisons inflation — the reported "best" number is biased
    upward by however many probes were tried, not a fair estimate of what
    a single, fixed attacker would achieve.

    Fixed here: every probe is fit on `train_dataset` and SCORED on
    `val_dataset` (an attack-VALIDATION split, patient-disjoint from both
    train and test) to pick a winner by validation macro-F1; the winning
    probe TYPE is then refit on `train_dataset` and evaluated EXACTLY ONCE
    on `test_dataset` (the untouched attack-TEST split) — that single
    held-out number, not the validation-time max, is what should be
    reported as the attack's result."""
    Z_train, y_train = extract_representations(model, train_dataset, batch_size)
    Z_val, y_val = extract_representations(model, val_dataset, batch_size)
    Z_test, y_test = extract_representations(model, test_dataset, batch_size)

    val_scores = run_all_probes_on_features(Z_train, y_train, Z_val, y_val, seed, include_slow)
    selected = max(val_scores, key=lambda name: val_scores[name]["macro_f1"])
    test_result = probe_by_name(selected, Z_train, y_train, Z_test, y_test, seed=seed)

    return {
        "selected_probe": selected,
        "selection_val_macro_f1_by_probe": {name: r["macro_f1"] for name, r in val_scores.items()},
        "attack_test_result": test_result,
    }


@torch.no_grad()
def extract_adapter_residual(model, dataset, batch_size: int = 32):
    """A_phi(f_theta(x)) alone (medgate.models.backbone.MedGateModel.adapter_residual)
    + the FINE label — P1-9: probe the adapter's own residual contribution
    separately from the shared representation z, to see whether fine-label
    information specifically concentrates in the 'restricted' component or
    is already present in the 'public' one (or both)."""
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    ds, ys = [], []
    for images, y_fine, _y_coarse in loader:
        ds.append(model.adapter_residual(images))
        ys.append(y_fine)
    return torch.cat(ds).numpy(), torch.cat(ys).numpy()


def run_all_probes_on_residual(model, train_dataset, test_dataset, batch_size: int = 32, seed: int = 0, include_slow: bool = True) -> dict:
    D_train, y_train = extract_adapter_residual(model, train_dataset, batch_size)
    D_test, y_test = extract_adapter_residual(model, test_dataset, batch_size)
    return run_all_probes_on_features(D_train, y_train, D_test, y_test, seed, include_slow)
