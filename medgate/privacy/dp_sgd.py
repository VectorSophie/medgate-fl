"""DP-SGD (Abadi et al. 2016, docs/literature_matrix.csv id abadi2016-dpsgd)
via Opacus — per-EXAMPLE gradient clipping + Gaussian noise, applied to one
client's LOCAL training.

EXACT PRIVACY UNIT AND ADJACENCY (P1-10 review requirement — stated
precisely so this is never mis-called "client-level DP", which is a
different, stronger, NOT-implemented guarantee):
  - Adjacency: EXAMPLE-level (a.k.a. record-level). Opacus's PrivacyEngine
    clips and noises the PER-SAMPLE gradient inside dp_local_train's
    DataLoader batches — the unit two "adjacent" datasets differ by is ONE
    TRAINING EXAMPLE within one client's local data, not one client's
    entire dataset. A client-level guarantee (protecting a client's WHOLE
    contribution, the more relevant unit for cross-silo FL where each
    "client" is a hospital) is NOT what this module provides and is NOT
    claimed anywhere in this project.
  - Sampling mechanism: `poisson_sampling=False` (medgate/privacy/dp_sgd.py
    dp_local_train) — i.e. plain shuffled mini-batches via a standard
    DataLoader, NOT Poisson/random subsampling. This is a real, deliberate
    deviation from Opacus's recommended default (which assumes Poisson
    sampling for its tightest RDP accounting): the accountant still
    computes a numeric epsilon under `poisson_sampling=False`, but that
    epsilon is accordingly LESS TIGHT than the Poisson-sampled analysis
    the DP-SGD literature usually reports — stated here so the reported
    epsilon is never read as the tightest possible bound.
  - Composition: WITHIN one dp_local_train call, Opacus's accountant
    (RDP-based, the Opacus default) composes across every local
    mini-batch step across `epochs` — that part IS accounted. ACROSS
    FedAvg rounds and across which clients participate in which rounds,
    composition is NOT accounted (dp_fedavg_round reports the MAX of each
    round's independent per-client epsilon, not a cumulative sum/
    composition across rounds — see that function's own docstring). A
    client that participates in every round of a multi-round experiment
    has a TRUE cumulative epsilon higher than any single round's reported
    number; this project does not compute that cumulative figure.
  - delta: caller-supplied (dp_local_train's `delta` parameter, default
    1e-5 throughout this project's configs) — not derived from dataset
    size via any rule of thumb; stated as a fixed choice, not tuned.
  - Parameters COVERED by DP: exactly medgate.attacks.gradient_inversion.
    attack_params(model) — backbone + coarse_head + adapter + fine_head.
    Every one of those receives a per-sample gradient every batch (Opacus
    requires this; see test_all_dp_tracked_params_receive_per_sample_gradients
    in tests/test_phase4_privacy.py).
  - Parameters EXCLUDED from DP: model.adversary_head. It is excluded
    because it is architecturally UNUSED by the plain joint_loss objective
    this module trains (medgate.federated.fedavg.joint_loss never calls
    model.adversary_logits) — it receives NO gradient at all, private or
    otherwise, in this training path, and its weights are never updated
    nor released as a function of any private example here. This DP-SGD
    integration is NOT wired up for Phase 2's adversarial/orthogonality
    objectives (which DO use adversary_head) — if adversary_head is ever
    trained under one of those objectives while DP-SGD is also active, the
    epsilon this module reports would NOT cover that adversary head's
    parameters, and that configuration is out of scope for any DP claim in
    this project (docs/execution_plan.md Phase 4 evaluates DP-SGD only
    against the plain adapter-isolation architecture, never combined with
    Phase 2's adversarial suppression).

Two fixes required to make our architecture Opacus-compatible, both
already applied and documented in medgate/models/backbone.py:
  1. no inplace ReLU (Opacus's backward hooks break on inplace ops).
  2. the optimizer must be built over attack_params(model) — see the
     'parameters excluded from DP' point above for exactly why.
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
