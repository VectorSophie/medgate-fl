"""DP-SGD (Abadi et al. 2016, docs/literature_matrix.csv id abadi2016-dpsgd)
via Opacus — per-example gradient clipping + Gaussian noise, applied to one
client's LOCAL training (client-level composition across FedAvg rounds is
NOT separately accounted here; see the docstring on dp_fedavg_round below
for exactly what epsilon does and does not cover).

Two fixes required to make our architecture Opacus-compatible, both
already applied and documented in medgate/models/backbone.py:
  1. no inplace ReLU (Opacus's backward hooks break on inplace ops).
  2. the optimizer must be built over medgate.attacks.gradient_inversion.
     attack_params(model) — the params actually used by the coarse+fine
     loss — not model.parameters(), because Opacus's optimizer requires
     EVERY tracked param to have received a per-sample gradient this batch,
     and model.adversary_head (present for Phase 2's adversarial/combined
     methods but unused by the plain joint_loss objective) would violate
     that whenever it isn't part of the loss being trained.
"""
import copy

import torch.nn as nn
from opacus import PrivacyEngine

from medgate.attacks.gradient_inversion import attack_params
from medgate.models.backbone import MedGateModel


def dp_local_train(
    model: MedGateModel,
    dataset,
    epochs: int,
    batch_size: int,
    lr: float,
    noise_multiplier: float,
    max_grad_norm: float = 1.0,
    delta: float = 1e-5,
) -> tuple[dict, float]:
    """Train `model` in place under DP-SGD on one client's data. Returns
    (state_dict, achieved epsilon at the given delta). Only the coarse+
    fine joint objective is used (matches medgate.federated.fedavg.joint_loss;
    Opacus's per-sample-gradient machinery is not wired up for the Phase 2
    adversarial/orthogonality loss variants — DP-SGD here is evaluated
    against the plain adapter-isolation architecture only)."""
    import torch

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(attack_params(model), lr=lr)

    engine = PrivacyEngine()
    priv_model, priv_optimizer, priv_loader = engine.make_private(
        module=model, optimizer=optimizer, data_loader=loader,
        noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm, poisson_sampling=False,
    )
    inner = priv_model._module  # medgate/models/backbone.py's forward_public/forward_fine live here

    inner.train()
    for _ in range(epochs):
        for images, y_fine, y_coarse in priv_loader:
            priv_optimizer.zero_grad()
            loss = nn.functional.cross_entropy(inner.forward_public(images), y_coarse) + \
                nn.functional.cross_entropy(inner.forward_fine(images), y_fine)
            loss.backward()
            priv_optimizer.step()

    epsilon = engine.get_epsilon(delta=delta)
    return copy.deepcopy(inner.state_dict()), epsilon


def dp_fedavg_round(
    global_model: MedGateModel,
    client_datasets: list,
    epochs: int,
    batch_size: int,
    lr: float,
    noise_multiplier: float,
    max_grad_norm: float = 1.0,
    delta: float = 1e-5,
) -> tuple[dict, float]:
    """One DP-FedAvg round: each client trains under record-level DP-SGD
    (dp_local_train), server aggregates normally (FedAvg weighted average
    — the aggregation step itself gets no additional protection here; that
    is exactly the 'secure aggregation + DP' arm in
    scripts/run_phase4_synthetic.py, kept separate on purpose).

    Reported epsilon = the MAX across clients this round (each client's
    epsilon is independent per-client record-level DP; this does NOT
    account for cross-round composition — that would require tracking each
    client's cumulative epsilon across all rounds it participated in,
    which this simplified implementation does not do. Documented here so
    the reported epsilon is never over-claimed as a full-training-run
    guarantee.)"""
    from medgate.federated.fedavg import fedavg_aggregate

    client_states, client_sizes, epsilons = [], [], []
    for ds in client_datasets:
        local_model = copy.deepcopy(global_model)
        state, eps = dp_local_train(local_model, ds, epochs, batch_size, lr, noise_multiplier, max_grad_norm, delta)
        client_states.append(state)
        client_sizes.append(len(ds))
        epsilons.append(eps)
    aggregated = fedavg_aggregate(client_states, [float(n) for n in client_sizes])
    return aggregated, max(epsilons)


def secure_dp_fedavg_round(
    global_model: MedGateModel,
    client_datasets: list,
    epochs: int,
    batch_size: int,
    lr: float,
    noise_multiplier: float,
    seed: int,
    max_grad_norm: float = 1.0,
    delta: float = 1e-5,
) -> tuple[dict, float]:
    """Combined arm: each client trains under DP-SGD (dp_local_train) AND
    the resulting updates are pairwise-masked before the server sums them
    (medgate.privacy.secure_aggregation) — the server never sees a
    client's plaintext update, DP-noised or not. Requires uniform client
    weighting, same as secure_fedavg_round."""
    from medgate.privacy.secure_aggregation import mask_client_updates, secure_aggregate_updates

    global_state = {k: v.clone() for k, v in global_model.state_dict().items()}
    updates, epsilons = [], []
    for ds in client_datasets:
        local_model = copy.deepcopy(global_model)
        state, eps = dp_local_train(local_model, ds, epochs, batch_size, lr, noise_multiplier, max_grad_norm, delta)
        updates.append({k: state[k] - global_state[k] for k in global_state})
        epsilons.append(eps)

    masked = mask_client_updates(updates, seed)
    aggregated_update = secure_aggregate_updates(masked, weights=[1.0] * len(updates))
    new_state = {k: global_state[k] + aggregated_update[k] for k in global_state}
    return new_state, max(epsilons)
