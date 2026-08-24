"""Minimal FedAvg (McMahan et al. 2017, docs/literature_matrix.csv id
mcmahan2017-fedavg): local SGD at each client, weighted average of the
resulting weights at the server.

Hand-rolled rather than built on a framework (Flower etc.) — this is a
single-workstation simulated cross-silo study (see the "simulated
cross-silo evaluation" non-claim in docs/research_scope.md), and a plain
training loop keeps every step inspectable for the attack/privacy phases
that come later.
"""
import copy

import torch
import torch.nn as nn

from medgate.models.backbone import MedGateModel


def joint_loss(model: MedGateModel, images, y_coarse, y_fine, lambda_fine: float = 1.0):
    """L_coarse + lambda_f * L_fine. Adversarial/orthogonality terms
    (lambda_a, lambda_o) are added in Phase 2 — this is the Phase 0/1
    baseline objective only."""
    ce = nn.functional.cross_entropy
    coarse_logits = model.forward_public(images)
    fine_logits = model.forward_fine(images)
    loss_coarse = ce(coarse_logits, y_coarse)
    loss_fine = ce(fine_logits, y_fine)
    return loss_coarse + lambda_fine * loss_fine, loss_coarse.item(), loss_fine.item()


def local_train(model: MedGateModel, dataset, epochs: int, batch_size: int, lr: float) -> dict:
    """Train `model` in place on one client's data; return a CPU state_dict
    (deep copy, safe to aggregate after the caller discards `model`)."""
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for images, y_fine, y_coarse in loader:
            optimizer.zero_grad()
            loss, _, _ = joint_loss(model, images, y_coarse, y_fine)
            loss.backward()
            optimizer.step()
    return copy.deepcopy(model.state_dict())


def fedavg_aggregate(state_dicts: list[dict], weights: list[float]) -> dict:
    """Weighted average of client state_dicts (McMahan et al. 2017 §2)."""
    assert len(state_dicts) == len(weights) and len(state_dicts) > 0
    total = sum(weights)
    norm_weights = [w / total for w in weights]
    keys = state_dicts[0].keys()
    averaged = {}
    for key in keys:
        stacked = torch.stack(
            [sd[key].float() * w for sd, w in zip(state_dicts, norm_weights)], dim=0
        )
        averaged[key] = stacked.sum(dim=0).to(state_dicts[0][key].dtype)
    return averaged


def run_fedavg_round(
    global_model: MedGateModel,
    client_datasets: list,
    epochs: int = 1,
    batch_size: int = 16,
    lr: float = 1e-3,
) -> dict:
    """One FedAvg round: broadcast -> local train on each client -> aggregate.
    Returns the aggregated state_dict; caller loads it back into
    global_model. Client weight = its dataset size (standard FedAvg)."""
    client_states, client_sizes = [], []
    for ds in client_datasets:
        local_model = copy.deepcopy(global_model)
        client_states.append(local_train(local_model, ds, epochs, batch_size, lr))
        client_sizes.append(len(ds))
    return fedavg_aggregate(client_states, [float(n) for n in client_sizes])
