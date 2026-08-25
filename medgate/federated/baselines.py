"""Phase 1 primary utility baselines (docs/execution_plan.md Phase 1):
centralized (upper bound), local-only, FedAvg, FedProx, and the FedLoRA
family.

Matched hyperparameter budget across baselines (project brief: "do not tune
the proposed method more extensively than the baselines") — every baseline
here takes the same `rounds`/`epochs`/`batch_size`/`lr` from one config, no
per-baseline tuning.

FedLoRA family, repaired 2026-08-26 after a review found the ONLY prior
FedLoRA baseline compared a pretrained proposed method against a
randomly-initialized frozen backbone (an unfair comparison — see git
history and docs/execution_plan.md for the full account):

  - `train_random_frozen_lora`: the ORIGINAL implementation, kept as an
    explicitly-named NEGATIVE SANITY CONTROL only (does a frozen-but-USELESS
    backbone still let the adapter learn anything at all?), never as the
    primary FedLoRA comparison point.
  - `train_coarse_pretrained_fedlora` / `train_imagenet_pretrained_fedlora`:
    the FAIR baselines — backbone frozen at a checkpoint that actually
    encodes something (medgate/federated/pretrain.py), only then is the
    adapter+fine_head federated.
  - `train_full_finetune`: same checkpoint, everything trainable — the
    upper bound a gated method is judged against, not a deployable method.

Every function below returns (model, param_summary) where param_summary
records total/trainable/frozen parameter counts and which named modules
were in the optimizer's parameter group — the project brief's own
'no hand-typed number' rule applies to this bookkeeping too.
"""
import copy

import torch

from medgate.federated.fedavg import local_train, run_round
from medgate.models.backbone import MedGateModel


def set_seed(seed: int):
    torch.manual_seed(seed)


def param_group_summary(model, trainable_module_names: list[str]) -> dict:
    """total/trainable/frozen parameter counts, keyed by whichever named
    submodules (e.g. ["adapter", "fine_head"]) are meant to be trainable —
    cross-checked against each parameter's actual requires_grad flag
    rather than just trusting the name list, so this can never silently
    drift out of sync with what a bug elsewhere actually froze."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "trainable_module_names": trainable_module_names,
    }


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


def _freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)


def _unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad_(True)


def _lora_trainable_params_fn(local_model):
    return list(local_model.adapter.parameters()) + list(local_model.fine_head.parameters())


def _restore_frozen(model, frozen_state: dict, module_names: list[str]):
    """FedAvg's weighted average (medgate/federated/fedavg.py
    fedavg_aggregate: sum(w_i * x_i)) does not reproduce a value that is
    bit-identical across every client bit-exactly -- float multiply-then-
    sum rounding perturbs the last few mantissa bits even when every
    client's copy was untouched. That is harmless to any TRAINED
    parameter, but it means 'frozen' would silently stop being bit-exact
    without this: after each round's aggregate is loaded, the frozen
    submodules are restored directly from their known-good state rather
    than trusted to survive the round-trip through aggregation math.
    (Caught by tests/test_phase1_pretrained_baselines.py: a torch.equal
    check on the 'frozen' backbone failed before this fix was added.)"""
    for name in module_names:
        module = getattr(model, name)
        module.load_state_dict({k[len(name) + 1:]: v for k, v in frozen_state.items() if k.startswith(name + ".")})


def train_random_frozen_lora(client_train_datasets: list, model_kwargs: dict, rounds: int, epochs: int, batch_size: int, lr: float, seed: int):
    """NEGATIVE SANITY CONTROL, not the primary FedLoRA baseline. Backbone
    and coarse head are frozen at RANDOM initialization — never
    pretrained, never seeing the coarse task at all. Comparing a
    pretrained proposed method against this is an unfair experiment; this
    function exists ONLY to answer 'can the adapter learn anything above
    chance even off a useless frozen backbone', which is itself an
    informative floor, not a baseline to beat."""
    set_seed(seed)
    global_model = MedGateModel(**model_kwargs)
    _freeze(global_model.backbone)
    _freeze(global_model.coarse_head)
    frozen_state = {k: v.clone() for k, v in global_model.state_dict().items()
                     if k.startswith("backbone.") or k.startswith("coarse_head.")}
    for _ in range(rounds):
        agg = run_round(global_model, client_train_datasets, epochs, batch_size, lr, trainable_params_fn=_lora_trainable_params_fn)
        global_model.load_state_dict(agg)
        _restore_frozen(global_model, frozen_state, ["backbone", "coarse_head"])
    summary = param_group_summary(global_model, ["adapter", "fine_head"])
    return global_model, summary


def train_pretrained_fedlora(client_train_datasets: list, init_state_dict: dict, model_kwargs: dict, rounds: int, epochs: int, batch_size: int, lr: float, seed: int, backbone=None):
    """The FAIR FedLoRA baseline: load a pretrained checkpoint
    (medgate/federated/pretrain.py — coarse-pretrained or
    imagenet-pretrained, caller's choice), freeze backbone+coarse_head at
    THAT checkpoint (not random init), then federate only adapter+fine_head.

    `backbone`: pass a freshly-constructed PretrainedMobileNetBackbone
    when init_state_dict came from build_imagenet_pretrained_checkpoint —
    its architecture differs from the default SmallBackbone, so the
    state_dict's keys/shapes will not load into a default-constructed
    model. None (default) keeps the original SmallBackbone behavior."""
    set_seed(seed)
    global_model = MedGateModel(**model_kwargs, backbone=backbone)
    global_model.load_state_dict(init_state_dict)
    _freeze(global_model.backbone)
    _freeze(global_model.coarse_head)
    frozen_state = {k: v.clone() for k, v in init_state_dict.items()
                     if k.startswith("backbone.") or k.startswith("coarse_head.")}
    for _ in range(rounds):
        agg = run_round(global_model, client_train_datasets, epochs, batch_size, lr, trainable_params_fn=_lora_trainable_params_fn)
        global_model.load_state_dict(agg)
        _restore_frozen(global_model, frozen_state, ["backbone", "coarse_head"])
    summary = param_group_summary(global_model, ["adapter", "fine_head"])
    return global_model, summary


def train_full_finetune(client_train_datasets: list, init_state_dict: dict, model_kwargs: dict, rounds: int, epochs: int, batch_size: int, lr: float, seed: int, backbone=None):
    """Upper bound: same pretrained checkpoint as train_pretrained_fedlora,
    but EVERY parameter (backbone included) is federally trainable on the
    joint coarse+fine objective. Not a directly deployable gated method —
    a full-finetune model has no restricted/public split at all — used
    only as a ceiling the gated methods are compared against.

    `backbone`: see train_pretrained_fedlora's docstring — same reason.
    Unconditionally unfrozen below regardless of the backbone's own
    construction-time freeze setting (e.g. PretrainedMobileNetBackbone
    defaults to freeze=True) -- 'full_finetune' means full, and this
    function does not trust every caller to remember to pass freeze=False
    when building the backbone it hands in."""
    set_seed(seed)
    global_model = MedGateModel(**model_kwargs, backbone=backbone)
    global_model.load_state_dict(init_state_dict)
    _unfreeze_all(global_model)
    for _ in range(rounds):
        agg = run_round(global_model, client_train_datasets, epochs, batch_size, lr)
        global_model.load_state_dict(agg)
    summary = param_group_summary(global_model, ["backbone", "coarse_head", "adapter", "fine_head"])
    return global_model, summary
