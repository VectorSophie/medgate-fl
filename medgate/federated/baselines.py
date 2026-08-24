"""Phase 1 primary utility baselines (docs/execution_plan.md Phase 1):
centralized (upper bound), local-only, FedAvg, FedProx, plain FedLoRA.

Matched hyperparameter budget across baselines (project brief: "do not tune
the proposed method more extensively than the baselines") — every baseline
here takes the same `rounds`/`epochs`/`batch_size`/`lr` from one config, no
per-baseline tuning.
"""
import copy

import torch

from medgate.federated.fedavg import local_train, run_round
from medgate.models.backbone import MedGateModel


def set_seed(seed: int):
    torch.manual_seed(seed)


def train_centralized(pooled_train, model_kwargs: dict, epochs: int, batch_size: int, lr: float, seed: int):
    """Upper bound: one model trained on all clients' data pooled together
    (never done for real in the federated setting; the point of a
    federated method is to approach this without pooling)."""
    set_seed(seed)
    model = MedGateModel(**model_kwargs)
    local_train(model, pooled_train, epochs, batch_size, lr)
    return model


def train_local_only(client_train_datasets: list, model_kwargs: dict, epochs: int, batch_size: int, lr: float, seed: int):
    """Lower reference: each institution trains only on its own data, no
    communication at all. Returns one model per client, all started from
    the same seeded initialization for a fair per-client comparison."""
    set_seed(seed)
    init_model = MedGateModel(**model_kwargs)
    models = []
    for ds in client_train_datasets:
        m = copy.deepcopy(init_model)
        local_train(m, ds, epochs, batch_size, lr)
        models.append(m)
    return models


def train_fedavg(client_train_datasets: list, model_kwargs: dict, rounds: int, epochs: int, batch_size: int, lr: float, seed: int):
    set_seed(seed)
    global_model = MedGateModel(**model_kwargs)
    for _ in range(rounds):
        agg = run_round(global_model, client_train_datasets, epochs, batch_size, lr)
        global_model.load_state_dict(agg)
    return global_model


def train_fedprox(client_train_datasets: list, model_kwargs: dict, rounds: int, epochs: int, batch_size: int, lr: float, seed: int, mu: float = 0.01):
    """FedProx (Li et al. 2020, docs/literature_matrix.csv id li2020-fedprox):
    FedAvg plus a proximal term mu/2 * ||w_local - w_global||^2 added to
    each client's local loss, discouraging local drift."""
    set_seed(seed)
    global_model = MedGateModel(**model_kwargs)

    def extra_loss_fn_factory(global_model_snapshot):
        global_params = [p.detach().clone() for p in global_model_snapshot.parameters()]

        def extra_loss_fn(local_model):
            prox = torch.zeros(())
            for p_local, p_global in zip(local_model.parameters(), global_params):
                prox = prox + (p_local - p_global).pow(2).sum()
            return 0.5 * mu * prox

        return extra_loss_fn

    for _ in range(rounds):
        agg = run_round(
            global_model, client_train_datasets, epochs, batch_size, lr,
            extra_loss_fn_factory=extra_loss_fn_factory,
        )
        global_model.load_state_dict(agg)
    return global_model


def train_fedlora(client_train_datasets: list, model_kwargs: dict, rounds: int, epochs: int, batch_size: int, lr: float, seed: int):
    """Plain federated LoRA: backbone and coarse head are frozen at random
    initialization (no separate pretraining stage implemented yet — see
    docs/execution_plan.md); only the adapter and fine head are federated.
    This is the parameter-efficiency baseline the capability-isolation
    method must be compared against, WITHOUT any isolation objective."""
    set_seed(seed)
    global_model = MedGateModel(**model_kwargs)
    for p in global_model.backbone.parameters():
        p.requires_grad_(False)
    for p in global_model.coarse_head.parameters():
        p.requires_grad_(False)

    def trainable_params_fn(local_model):
        return list(local_model.adapter.parameters()) + list(local_model.fine_head.parameters())

    for _ in range(rounds):
        agg = run_round(
            global_model, client_train_datasets, epochs, batch_size, lr,
            trainable_params_fn=trainable_params_fn,
        )
        global_model.load_state_dict(agg)
    return global_model
