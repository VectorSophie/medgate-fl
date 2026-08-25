"""Phase 0 mandatory check: one centralized forward/backward pass works.

Run: PYTHONPATH=. pytest tests/test_phase0_forward_backward.py -v
"""
import torch

from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, SyntheticFedISIC
from medgate.federated.fedavg import joint_loss
from medgate.models.backbone import MedGateModel


def test_forward_backward_pass_produces_finite_gradients():
    torch.manual_seed(0)
    dataset = SyntheticFedISIC(num_samples=8, image_size=32, seed=1)
    images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    y_fine = torch.stack([dataset[i][1] for i in range(len(dataset))])
    y_coarse = torch.stack([dataset[i][2] for i in range(len(dataset))])

    model = MedGateModel(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES))

    coarse_logits = model.forward_public(images)
    fine_logits = model.forward_fine(images)
    assert coarse_logits.shape == (8, len(COARSE_CLASSES))
    assert fine_logits.shape == (8, len(FINE_CLASSES))
    assert torch.isfinite(coarse_logits).all()
    assert torch.isfinite(fine_logits).all()

    loss, loss_coarse, loss_fine = joint_loss(model, images, y_coarse, y_fine)
    assert torch.isfinite(loss)
    loss.backward()

    # joint_loss is the Phase 0/1 baseline objective: it only exercises
    # backbone/coarse_head/adapter/fine_head. model.adversary_head exists
    # for Phase 2's adversarial/combined objectives (medgate/federated/
    # capability_isolation.py) and is correctly untouched here — it is not
    # part of this check.
    touched_modules = [model.backbone, model.coarse_head, model.adapter, model.fine_head]
    grads = [p.grad for m in touched_modules for p in m.parameters() if p.requires_grad]
    assert grads, "model has no trainable parameters"
    for g in grads:
        assert g is not None, "a parameter received no gradient"
        assert torch.isfinite(g).all(), "a gradient is NaN/Inf"

    print(f"loss={loss.item():.4f} loss_coarse={loss_coarse:.4f} loss_fine={loss_fine:.4f}")


def test_zero_init_adapter_starts_as_identity_residual():
    """LoRAAdapter is zero-initialized (docs, medgate/models/backbone.py) —
    confirm that holds at construction, since Phase 2's isolation claims
    depend on this being true before any training happens."""
    model = MedGateModel(num_coarse=3, num_fine=8)
    z = torch.randn(4, model.backbone.feature_dim)
    assert torch.allclose(model.adapter(z), torch.zeros_like(z))


if __name__ == "__main__":
    test_forward_backward_pass_produces_finite_gradients()
    test_zero_init_adapter_starts_as_identity_residual()
    print("OK")
