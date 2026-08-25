"""Membership inference — simplified loss-threshold variant.

Named after Shokri et al. 2017 (docs/literature_matrix.csv id
shokri2017-mia), but stated precisely: the original paper trains shadow
models to learn an attack classifier; this is the cheaper loss-threshold
variant (attacker scores "membership" by -loss, no shadow models trained)
often used as a fast lower-bound MI baseline. Do not describe results from
this module as "the Shokri et al. attack" — describe them as "a
loss-threshold membership-inference attack."

Attacker knowledge/access:
  - white-box access to the trained model (can compute per-example loss)
  - a pool of candidate examples to score (does NOT know in advance which
    are members — ground-truth member/non-member labels are used only to
    COMPUTE the attack's AUC after the fact, never by the attack itself)
Success condition: AUC of the membership score against ground truth.
Pre-registered target (docs/research_scope.md): AUC <= 0.55.
"""
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def per_example_fine_loss(model, dataset, batch_size: int = 32):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    losses = []
    for images, y_fine, _y_coarse in loader:
        logits = model.forward_fine(images)
        losses.append(nn.functional.cross_entropy(logits, y_fine, reduction="none"))
    return torch.cat(losses).numpy()


def loss_threshold_membership_inference(model, member_dataset, nonmember_dataset, batch_size: int = 32) -> dict:
    member_losses = per_example_fine_loss(model, member_dataset, batch_size)
    nonmember_losses = per_example_fine_loss(model, nonmember_dataset, batch_size)
    scores = (-member_losses).tolist() + (-nonmember_losses).tolist()
    labels = [1] * len(member_losses) + [0] * len(nonmember_losses)
    auc = roc_auc_score(labels, scores)
    return {
        "attacker_knowledge": "white-box loss access only, no shadow models trained (simplified variant)",
        "attacker_access": "per-example fine-task loss on a candidate pool",
        "attack_auc": auc,
        "attack_advantage": 2 * auc - 1,
        "n_members": int(len(member_losses)),
        "n_nonmembers": int(len(nonmember_losses)),
        "mean_member_loss": float(member_losses.mean()),
        "mean_nonmember_loss": float(nonmember_losses.mean()),
    }
