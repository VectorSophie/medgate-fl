"""Utility metrics. Implemented here: macro-F1 and balanced accuracy — the
two used by Phase 1's baseline comparison and, later, as the utility
function U() inside the capability-isolation composites (ARR/UCG/RFC/CRE,
docs/research_scope.md §7). AUROC/AUPRC/ECE/Brier/calibration are deferred:
on the Phase 1 synthetic fixture (random images, random labels, no
class imbalance by construction) they add no information beyond what
macro-F1 already shows, and they earn their place once real, imbalanced
Fed-ISIC2019 data is in use (docs/execution_plan.md Phase 1 real-data
tier). Not implementing them now is a scoping decision, not an oversight —
tracked in docs/execution_plan.md.
"""
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score


@torch.no_grad()
def _predict_and_collect(model_forward, dataset, label_index: int, batch_size: int = 32):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    preds, labels = [], []
    for batch in loader:
        images = batch[0]
        y = batch[label_index]
        logits = model_forward(images)
        preds.append(logits.argmax(dim=1))
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def evaluate_coarse(model, dataset, batch_size: int = 32) -> dict:
    preds, labels = _predict_and_collect(model.forward_public, dataset, label_index=2, batch_size=batch_size)
    return {
        "coarse_macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "coarse_balanced_accuracy": balanced_accuracy_score(labels, preds),
    }


def evaluate_fine(model, dataset, batch_size: int = 32) -> dict:
    preds, labels = _predict_and_collect(model.forward_fine, dataset, label_index=1, batch_size=batch_size)
    return {
        "fine_macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "fine_balanced_accuracy": balanced_accuracy_score(labels, preds),
    }


def evaluate_both(model, dataset, batch_size: int = 32) -> dict:
    out = {}
    out.update(evaluate_coarse(model, dataset, batch_size))
    out.update(evaluate_fine(model, dataset, batch_size))
    return out
