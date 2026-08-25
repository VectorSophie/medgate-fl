"""Property inference (Melis et al. 2019-style framing,
docs/literature_matrix.csv id melis2019-unintended-leakage / bib key
melis2019exploiting): can an observer of client UPDATES infer a property
of a client's local training distribution that is unrelated to the main
task, purely from the shape of the update?

Simplified relative to Melis et al.'s original method: they train a neural
network meta-classifier over many shadow-model updates; this
implementation uses logistic regression over flattened update vectors
under leave-one-client-out cross-validation. Stated explicitly so this is
never described as a reproduction of their exact method -- it tests the
same question (does an update leak an unrelated property of its source's
data) with a much simpler classifier appropriate to this project's client
counts (single digits, not the hundreds/thousands a neural meta-classifier
would need to train on).

Attacker knowledge/access: one raw (unaggregated) update per client. This
attack targets A1 (an honest-but-curious server observing plaintext
per-client updates) — it does NOT apply once secure aggregation
(medgate/privacy/secure_aggregation.py) is in place, since then no single
client's update is ever observed in isolation. That distinction is exactly
the point of running this attack only under plaintext FedAvg.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut


def flatten_update(state_dict: dict) -> np.ndarray:
    return np.concatenate([v.detach().cpu().numpy().ravel() for v in state_dict.values()])


def majority_fine_class_property(dataset, target_class: int) -> int:
    """Ground-truth property label: 1 if `target_class` is this client's
    single most frequent fine label, else 0. Used only to SCORE the
    attack after the fact; never given to the attacker's classifier."""
    labels = dataset.fine_labels.numpy() if torch.is_tensor(dataset.fine_labels) else np.asarray(dataset.fine_labels)
    counts = np.bincount(labels, minlength=8)
    return int(counts.argmax() == target_class)


def property_inference_attack(client_updates: list, property_labels: list, seed: int = 0) -> dict:
    """Leave-one-client-out logistic regression over flattened updates.
    Needs at least one client on each side of the property to be
    meaningful; if the property is constant across all clients this seed,
    AUC is reported as undefined (None) rather than a misleading 0.5 or
    1.0 computed from a degenerate label set."""
    X = np.stack([flatten_update(u) for u in client_updates])
    y = np.array(property_labels)

    base = {
        "attacker_knowledge": "one raw per-client update each (no secure aggregation); "
                               "logistic-regression meta-classifier (simplified vs. Melis et al.'s neural meta-classifier)",
        "attacker_access": f"{len(client_updates)} clients' updates, leave-one-client-out evaluation",
        "n_clients": len(client_updates),
    }
    if len(set(y.tolist())) < 2:
        return {**base, "attack_auc": None, "note": "property label constant across clients this seed; AUC undefined"}

    loo = LeaveOneOut()
    scores, truth = [], []
    for train_idx, test_idx in loo.split(X):
        if len(set(y[train_idx].tolist())) < 2:
            # With this few clients, leaving one out can strip a fold down
            # to a single class -- logistic regression has nothing to fit
            # in that fold. Score it neutrally (0.5) rather than crashing
            # or silently biasing the AUC with a degenerate classifier.
            scores.append(0.5)
        else:
            clf = LogisticRegression(max_iter=1000, random_state=seed)
            clf.fit(X[train_idx], y[train_idx])
            scores.append(clf.predict_proba(X[test_idx])[0, 1])
        truth.append(y[test_idx][0])
    return {**base, "attack_auc": roc_auc_score(truth, scores)}
