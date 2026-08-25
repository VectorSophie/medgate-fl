"""Deep Leakage from Gradients (Zhu, Liu, Han 2019, docs/literature_matrix.csv
id zhu2019-dlg) — A1 threat model (docs/research_scope.md): an
honest-but-curious party that observes ONE client's raw, per-example
gradient (explicitly NOT a secure-aggregated sum — that distinction is the
whole point of Phase 4) tries to reconstruct the input image that produced
it, by optimizing a dummy image so its gradient matches the observed one.

Deviation from the original paper, stated up front: the original DLG paper
uses L-BFGS and jointly optimizes a dummy label; this implementation uses
Adam and fixes the labels to their true value (a common simplified DLG
variant) for predictable CPU cost on this hardware
(docs/hardware_report.md). Numbers here are this project's own
implementation and are not directly comparable to the original paper's
reported numbers.

Attacker knowledge/access (record on every call's result):
  - full model architecture and current weights
  - ONE raw single-example gradient
  - the example's true coarse/fine labels (simplified-DLG assumption above)
  - no other auxiliary data
Success condition: reconstruction similarity (MSE, a PSNR proxy) at a
fixed, reported step budget.
"""
import torch
import torch.nn as nn


def attack_params(model):
    """Params actually used by the coarse+fine loss this attack targets —
    deliberately excludes model.adversary_head, which some Phase 2 methods
    carry but which the standard forward_public/forward_fine loss never
    touches (Phase 0 test_phase0_forward_backward.py has the same
    restriction, for the same reason: a param unused in a given loss graph
    has no gradient to leak in the first place, by construction)."""
    modules = [model.backbone, model.coarse_head, model.adapter, model.fine_head]
    return [p for m in modules for p in m.parameters() if p.requires_grad]


def compute_true_gradient(model, image, y_coarse, y_fine):
    model.zero_grad()
    loss = nn.functional.cross_entropy(model.forward_public(image), y_coarse) + \
        nn.functional.cross_entropy(model.forward_fine(image), y_fine)
    trainable = attack_params(model)
    grads = torch.autograd.grad(loss, trainable)
    return [g.detach() for g in grads]


def simplified_known_label_gradient_inversion(model, true_image: torch.Tensor, y_coarse, y_fine, steps: int = 300, lr: float = 0.1, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    trainable = attack_params(model)
    true_grads = compute_true_gradient(model, true_image, y_coarse, y_fine)

    dummy_image = torch.randn_like(true_image, requires_grad=True)
    optimizer = torch.optim.Adam([dummy_image], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model.forward_public(dummy_image), y_coarse) + \
            nn.functional.cross_entropy(model.forward_fine(dummy_image), y_fine)
        dummy_grads = torch.autograd.grad(loss, trainable, create_graph=True)
        grad_diff = sum(((dg - tg) ** 2).sum() for dg, tg in zip(dummy_grads, true_grads))
        grad_diff.backward()
        optimizer.step()

    with torch.no_grad():
        mse = nn.functional.mse_loss(dummy_image, true_image).item()
        data_range = (true_image.max() - true_image.min()).clamp(min=1e-6).item()
        psnr_db = 10 * torch.log10(torch.tensor((data_range ** 2) / max(mse, 1e-12))).item()

    return {
        "attacker_knowledge": "full model weights + ONE raw single-example gradient + true labels (simplified DLG)",
        "attacker_access": "white-box, single client's unaggregated gradient (A1, no secure aggregation)",
        "compute_budget_steps": steps,
        "mse": mse,
        "psnr_db": psnr_db,
        "final_grad_diff": grad_diff.item(),
    }
