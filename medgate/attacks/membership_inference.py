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
Success condition: SymmetricAUC / AttackAdvantage of the membership score
against ground truth (NOT raw AUC — see the note on those two functions
below; this was a real bug in an earlier version of this project, fixed
2026-08-26, see docs/execution_plan.md Phase 5).
Pre-registered target (docs/research_scope.md): SymmetricAUC <= 0.55.
"""
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def symmetric_auc(auc: float) -> float:
    """max(AUC, 1-AUC). Raw AUC near 0.5 means good forgetting/no
    membership signal -- but raw AUC near 0.0 is ALSO perfect separability,
    just with the score direction reversed (e.g. an unlearned model that
    fits removed data slightly WORSE than retained data, rather than
    better, still fully distinguishes the two). Treating raw AUC as the
    'closer to 0.5 is better, closer to 0 is fine' score was a real bug in
    an earlier version of this project (docs/execution_plan.md Phase 5) --
    it would have called AUC=0.0 an excellent result. SymmetricAUC folds
    both directions of separability into one number where 0.5 is always
    the 'no signal' point and 1.0 is always 'perfectly distinguishable',
    regardless of direction."""
    return max(auc, 1 - auc)


def attack_advantage(auc: float) -> float:
    """2*|AUC-0.5|, in [0,1]. 0 = no membership signal, 1 = perfect
    separability in either direction. This is the primary forgetting/
    leakage score this project reports; raw AUC is a diagnostic field
    only, per the same fix as symmetric_auc above."""
    return 2 * abs(auc - 0.5)


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
        "attack_auc": auc,  # DIAGNOSTIC ONLY -- direction-sensitive, do not read as "closer to 0 is safe"
        "symmetric_auc": symmetric_auc(auc),  # PRIMARY forgetting/leakage score: 0.5=no signal, 1.0=fully distinguishable
        "attack_advantage": attack_advantage(auc),  # equivalent primary score, in [0,1]
        "n_members": int(len(member_losses)),
        "n_nonmembers": int(len(nonmember_losses)),
        "mean_member_loss": float(member_losses.mean()),
        "mean_nonmember_loss": float(nonmember_losses.mean()),
    }
