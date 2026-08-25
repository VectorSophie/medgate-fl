"""Pretrained-checkpoint construction for the fair-FedLoRA-baseline repair
(docs/execution_plan.md Phase 1). Three starting points, all producing a
checkpoint that coarse_pretrained_fedlora / imagenet_pretrained_fedlora /
full_finetune / proposed_isolation can all load identically
(medgate/federated/checkpoints.py):

  1. coarse_pretrained: SmallBackbone + coarse_head trained (centrally,
     pooled) on the coarse task only, from a random init.
  2. imagenet_pretrained: PretrainedMobileNetBackbone (ImageNet weights,
     conv trunk frozen) + its proj layer and coarse_head trained on the
     coarse task -- so the checkpoint's coarse_head is meaningful before
     any client ever sees it, the same standard applied to (1).

Both stages use coarse_only_loss from capability_isolation.py (the exact
same loss function Phase 2's `coarse_only` ablation arm already uses) --
not a new, uninspected loss.
"""
import torch

from medgate.federated.baselines import set_seed
from medgate.federated.capability_isolation import coarse_only_loss
from medgate.federated.fedavg import local_train
from medgate.models.backbone import MedGateModel, PretrainedMobileNetBackbone


def pretrain_coarse_only(model, pooled_train, epochs: int, batch_size: int, lr: float, trainable_params=None) -> dict:
    """Trains whatever `trainable_params` names (default: every
    requires_grad=True param already on the model) on the coarse task
    alone via local_train + coarse_only_loss, in place. Returns nothing;
    caller reads the mutated `model`. Kept as a thin wrapper so both
    pretraining paths below share one code path, not two near-duplicates."""
    params = trainable_params if trainable_params is not None else [p for p in model.parameters() if p.requires_grad]
    local_train(model, pooled_train, epochs, batch_size, lr, trainable_params=params, batch_loss_fn=coarse_only_loss)


def build_coarse_pretrained_checkpoint(pooled_train, model_kwargs: dict, epochs: int, batch_size: int, lr: float, seed: int) -> MedGateModel:
    """SmallBackbone + coarse_head trained from random init on the coarse
    task; adapter/fine_head/adversary_head are left at their own random
    init (irrelevant -- every consumer of this checkpoint reinitializes
    or ignores them per its own method, see medgate/federated/baselines.py)."""
    set_seed(seed)
    model = MedGateModel(**model_kwargs)
    trainable = list(model.backbone.parameters()) + list(model.coarse_head.parameters())
    pretrain_coarse_only(model, pooled_train, epochs, batch_size, lr, trainable_params=trainable)
    return model


def build_imagenet_pretrained_checkpoint(pooled_train, model_kwargs: dict, epochs: int, batch_size: int, lr: float, seed: int) -> MedGateModel:
    """PretrainedMobileNetBackbone (ImageNet conv trunk frozen) + its proj
    layer and coarse_head trained on the coarse task. The conv trunk
    itself never sees this project's data at all -- only `proj` (576 ->
    feature_dim) and `coarse_head` are optimized, which is the minimum
    needed to make the frozen ImageNet features usable for a 3-way coarse
    task with this project's feature_dim, stated explicitly rather than
    silently training more of the backbone than the checkpoint's name implies."""
    set_seed(seed)
    feature_dim = model_kwargs.get("feature_dim", 64)
    backbone = PretrainedMobileNetBackbone(feature_dim=feature_dim, freeze=True)
    model = MedGateModel(**model_kwargs, backbone=backbone)
    trainable = list(model.backbone.proj.parameters()) + list(model.coarse_head.parameters())
    pretrain_coarse_only(model, pooled_train, epochs, batch_size, lr, trainable_params=trainable)
    return model
