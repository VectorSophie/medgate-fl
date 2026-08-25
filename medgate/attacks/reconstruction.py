"""Recovery-via-retraining attacks: A2 (unauthorized recipient with the
public backbone) and A3 (fully black-box query access), plus a minimal
two-attacker collusion PROXY.

Naming, precise on purpose (docs/execution_plan.md P1-6 attack-naming
audit): none of these functions "reconstruct" an existing protected
artifact (the real adapter's weights, an encrypted tensor, a missing
parameter) -- they each train a FRESH replacement from scratch using
auxiliary data or query access, then measure how much fine-task utility
that fresh replacement recovers. "Reconstruction" would overclaim that
something specific was recovered; what actually happens is fine-tuning
(A2) or distillation (A3) recovery. Likewise `auxiliary_data_ensemble_
collusion_proxy` is a PROXY for collusion (two independent auxiliary-data
attackers, logit-averaged) -- see medgate/attacks/property_inference.py
and medgate/attacks/integrity.py for the project's other, different uses
of "collusion" (A1 property inference over plaintext updates; A4 malicious-
client poisoning), none of which this module implements or claims to.
Genuine parameter-space adapter recovery (SVD/low-rank completion, a
known-partial-adapter attack, etc.) is a SEPARATE, not-yet-implemented
track -- see docs/execution_plan.md P1-7.

These are distinct from the frozen-feature probes in
medgate/attacks/probes.py: here the attacker actually trains new gradient-
based parameters (a fresh adapter, or a fresh full model), not just a
classifier on top of fixed features.
"""
import copy

import torch
import torch.nn as nn

from medgate.federated.fedavg import local_train


def auxiliary_data_adapter_finetuning_recovery(
    authorized_model, auxiliary_dataset, epochs: int, batch_size: int, lr: float, seed: int
):
    """A2: attacker has the public backbone + coarse head (frozen, exactly
    what they'd legitimately receive) and some auxiliary LABELED data —
    but NOT the real adapter/fine_head. They build fresh, untrained
    replacements and train only those on their auxiliary data, under a
    fixed epoch/data-size budget."""
    torch.manual_seed(seed)
    attacker_model = copy.deepcopy(authorized_model)
    for p in attacker_model.backbone.parameters():
        p.requires_grad_(False)
    for p in attacker_model.coarse_head.parameters():
        p.requires_grad_(False)

    rank = attacker_model.adapter.down.out_features
    num_fine = attacker_model.fine_head.fc.out_features
    feature_dim = attacker_model.backbone.feature_dim
    attacker_model.adapter = type(attacker_model.adapter)(feature_dim, rank=rank)
    attacker_model.fine_head = type(attacker_model.fine_head)(feature_dim, num_fine)

    trainable = list(attacker_model.adapter.parameters()) + list(attacker_model.fine_head.parameters())
    local_train(attacker_model, auxiliary_dataset, epochs, batch_size, lr, trainable_params=trainable)

    return attacker_model, {
        "attacker_knowledge": "public backbone + public coarse head (frozen), FRESH untrained adapter/fine_head",
        "attacker_access": f"{len(auxiliary_dataset)} auxiliary labeled examples, {epochs} local epochs",
        "compute_budget": {"auxiliary_examples": len(auxiliary_dataset), "epochs": epochs},
    }


@torch.no_grad()
def _query_hard_labels(authorized_model, images) -> torch.Tensor:
    authorized_model.eval()
    return authorized_model.forward_fine(images).argmax(dim=1)


def fixed_budget_hard_label_distillation(
    authorized_model, query_images: torch.Tensor, model_kwargs: dict, epochs: int, batch_size: int, lr: float, seed: int
):
    """A3: fully black-box. Attacker never sees the backbone or any
    weights — only (image -> hard predicted fine label) from a fixed query
    budget against the authorized inference endpoint. Trains a completely
    fresh model (own backbone, own adapter, own fine head) to imitate
    those labels (Tramèr et al. 2016-style extraction via hard-label
    distillation, docs/literature_matrix.csv id tramer2016-modelextraction)."""
    torch.manual_seed(seed)
    pseudo_labels = _query_hard_labels(authorized_model, query_images)
    student = _model_from_kwargs(model_kwargs)

    dataset = torch.utils.data.TensorDataset(
        query_images, pseudo_labels, torch.zeros(len(pseudo_labels), dtype=torch.long)
    )

    def fine_only_loss(model, images, y_coarse, y_fine):
        # pure hard-label distillation on the fine task; the dummy y_coarse
        # (always 0, see TensorDataset above) is never used in this loss so
        # it never pollutes the student's backbone with a fake coarse signal.
        loss_fine = nn.functional.cross_entropy(model.forward_fine(images), y_fine)
        return loss_fine, 0.0, loss_fine.item()

    local_train(student, dataset, epochs, batch_size, lr, batch_loss_fn=fine_only_loss)

    return student, {
        "attacker_knowledge": "none of the target's weights; only its output predictions",
        "attacker_access": f"{len(query_images)} black-box queries (fixed query budget), {epochs} local epochs on the student",
        "compute_budget": {"query_budget": len(query_images), "epochs": epochs},
    }


def _model_from_kwargs(model_kwargs: dict):
    from medgate.models.backbone import MedGateModel
    return MedGateModel(**model_kwargs)


def auxiliary_data_ensemble_collusion_proxy(
    authorized_model, auxiliary_dataset_a, auxiliary_dataset_b, epochs: int, batch_size: int, lr: float, seed: int
):
    """Two-attacker collusion (A4-adjacent): two independent A2 attackers,
    each with a DISJOINT slice of auxiliary data, each reconstruct their
    own adapter (auxiliary_data_adapter_finetuning_recovery) — then pool their fine
    predictions by averaging logits. Tests whether collusion recovers more
    capability than either attacker alone, using the SAME total auxiliary
    data and compute as the two solo attacks combined (no extra budget)."""
    model_a, meta_a = auxiliary_data_adapter_finetuning_recovery(authorized_model, auxiliary_dataset_a, epochs, batch_size, lr, seed)
    model_b, meta_b = auxiliary_data_adapter_finetuning_recovery(authorized_model, auxiliary_dataset_b, epochs, batch_size, lr, seed + 1)

    class ColludedEnsemble(nn.Module):
        def __init__(self, m1, m2):
            super().__init__()
            self.m1, self.m2 = m1, m2

        def forward_fine(self, x, use_adapter=True):
            return (self.m1.forward_fine(x) + self.m2.forward_fine(x)) / 2

        def forward_public(self, x):
            return self.m1.forward_public(x)

    ensemble = ColludedEnsemble(model_a, model_b)
    return ensemble, {
        "attacker_knowledge": "two colluding A2 attackers, disjoint auxiliary data, logit-averaged ensemble",
        "attacker_access": f"{len(auxiliary_dataset_a)}+{len(auxiliary_dataset_b)} auxiliary examples total (same total budget as either solo attack x2)",
        "compute_budget": {
            "auxiliary_examples_total": len(auxiliary_dataset_a) + len(auxiliary_dataset_b),
            "epochs": epochs,
        },
        "solo_a": meta_a,
        "solo_b": meta_b,
    }
