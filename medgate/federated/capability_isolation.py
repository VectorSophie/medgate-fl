"""Phase 2: capability-isolation training objectives (docs/execution_plan.md
Phase 2). Every method below trains the exact same MedGateModel
architecture (medgate/models/backbone.py) via the exact same FedAvg loop
(medgate/federated/fedavg.py run_round) — only the loss function passed to
local_train differs. Fixing the architecture and varying only the
objective isolates what the objective itself contributes, which is the
point of this ablation; the project brief explicitly says not to assume
the combined objective works and to compare it against simpler
alternatives, so all six live side by side here with no method treated as
the "real" one during training.
"""
import torch.nn as nn

from medgate.federated.fedavg import joint_loss

_CE = nn.functional.cross_entropy

# Matched across all methods that use them — no method gets a tuning
# advantage over another (project brief: matched hyperparameter budgets).
LAMBDA_FINE = 1.0
LAMBDA_ADV = 1.0
LAMBDA_ORTH = 1.0


def coarse_only_loss(model, images, y_coarse, y_fine):
    """Never trains on the fine task at all — the ceiling case for 'what
    residual fine information exists if the backbone is never asked to
    represent it.'"""
    loss_coarse = _CE(model.forward_public(images), y_coarse)
    return loss_coarse, loss_coarse.item(), 0.0


def hidden_fine_head_loss(model, images, y_coarse, y_fine):
    """Fine head exists and is trained, but reads the SAME representation
    z as the public path (use_adapter=False) — the naive baseline the
    project brief warns about: hiding the fine head's access is not
    capability isolation if the representation underneath is unchanged."""
    loss_coarse = _CE(model.forward_public(images), y_coarse)
    loss_fine = _CE(model.forward_fine(images, use_adapter=False), y_fine)
    loss = loss_coarse + LAMBDA_FINE * loss_fine
    return loss, loss_coarse.item(), loss_fine.item()


def adapter_isolation_loss(model, images, y_coarse, y_fine):
    """Fine head reads z + A_phi(z) (the LoRA adapter path) but with no
    adversarial or orthogonality pressure — the architecture alone, no
    extra objective. This is the reference 'plain adapter' point that
    ARR (docs/research_scope.md §7) is measured against."""
    return joint_loss(model, images, y_coarse, y_fine, lambda_fine=LAMBDA_FINE)


def adversarial_loss(model, images, y_coarse, y_fine):
    """Adapter isolation + gradient-reversal adversary trying to predict
    the fine label from the PUBLIC representation z."""
    loss, lc, lf = joint_loss(model, images, y_coarse, y_fine, lambda_fine=LAMBDA_FINE)
    loss_adv = _CE(model.adversary_logits(images), y_fine)
    return loss + LAMBDA_ADV * loss_adv, lc, lf


def orthogonal_loss(model, images, y_coarse, y_fine):
    """Adapter isolation + penalty on cosine similarity between z and the
    adapter's residual contribution (medgate/models/backbone.py orth_term)."""
    loss, lc, lf = joint_loss(model, images, y_coarse, y_fine, lambda_fine=LAMBDA_FINE)
    loss_orth = model.orth_term(images)
    return loss + LAMBDA_ORTH * loss_orth, lc, lf


def combined_loss(model, images, y_coarse, y_fine):
    """Adapter isolation + adversarial suppression + orthogonality — the
    full proposed objective. Not assumed superior; compared against every
    method above under the identical training budget."""
    loss, lc, lf = joint_loss(model, images, y_coarse, y_fine, lambda_fine=LAMBDA_FINE)
    loss_adv = _CE(model.adversary_logits(images), y_fine)
    loss_orth = model.orth_term(images)
    return loss + LAMBDA_ADV * loss_adv + LAMBDA_ORTH * loss_orth, lc, lf


METHODS = {
    "coarse_only": coarse_only_loss,
    "hidden_fine_head": hidden_fine_head_loss,
    "adapter_isolation": adapter_isolation_loss,
    "adversarial": adversarial_loss,
    "orthogonal": orthogonal_loss,
    "combined": combined_loss,
}


def train_capability_isolation(
    method_name: str,
    client_train_datasets: list,
    model_kwargs: dict,
    rounds: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
):
    # local imports to avoid a circular import (baselines imports fedavg,
    # this module is imported by scripts, not by baselines.py)
    from medgate.federated.baselines import set_seed
    from medgate.federated.fedavg import run_round
    from medgate.models.backbone import MedGateModel

    set_seed(seed)
    model = MedGateModel(**model_kwargs)
    loss_fn = METHODS[method_name]
    for _ in range(rounds):
        agg = run_round(model, client_train_datasets, epochs, batch_size, lr, batch_loss_fn=loss_fn)
        model.load_state_dict(agg)
    return model
