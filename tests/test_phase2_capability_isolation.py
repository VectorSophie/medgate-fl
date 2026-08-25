"""Phase 2 checks: the six capability-isolation objectives all run, and the
two new non-trivial mechanisms (gradient reversal, orthogonality loss) do
what their docstrings claim rather than just "not crashing."

Run: PYTHONPATH=. pytest tests/test_phase2_capability_isolation.py -v
"""
import torch

from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, SyntheticFedISIC
from medgate.federated.capability_isolation import METHODS
from medgate.models.backbone import MedGateModel, grad_reverse, orthogonality_loss


def _batch():
    dataset = SyntheticFedISIC(num_samples=8, image_size=32, seed=7)
    images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    y_fine = torch.stack([dataset[i][1] for i in range(len(dataset))])
    y_coarse = torch.stack([dataset[i][2] for i in range(len(dataset))])
    return images, y_coarse, y_fine


def test_grad_reverse_negates_and_scales_the_gradient():
    """The whole point of grad_reverse is: forward = identity,
    backward = -lambd * grad. Confirm both halves directly, not just that
    training doesn't crash."""
    x = torch.randn(5, requires_grad=True)
    y = grad_reverse(x, lambd=2.0)
    assert torch.allclose(y, x)  # forward is identity

    y.sum().backward()
    # d(sum(y))/dx would be all-ones without reversal; with lambd=2 it must
    # be all -2.
    assert torch.allclose(x.grad, torch.full_like(x, -2.0))


def test_orthogonality_loss_bounded_and_zero_for_orthogonal_vectors():
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    delta_orthogonal = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    loss = orthogonality_loss(z, delta_orthogonal)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    delta_parallel = z.clone()
    loss_parallel = orthogonality_loss(z, delta_parallel)
    assert torch.allclose(loss_parallel, torch.tensor(1.0), atol=1e-6)


def test_all_six_methods_forward_backward_without_error():
    images, y_coarse, y_fine = _batch()
    for name, loss_fn in METHODS.items():
        model = MedGateModel(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES))
        loss, loss_coarse, loss_fine = loss_fn(model, images, y_coarse, y_fine)
        assert torch.isfinite(loss), f"{name}: non-finite loss"
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"{name}: non-finite gradient"


def test_adversary_head_only_gets_gradient_under_adversarial_and_combined():
    images, y_coarse, y_fine = _batch()
    for name, loss_fn in METHODS.items():
        model = MedGateModel(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES))
        loss, _, _ = loss_fn(model, images, y_coarse, y_fine)
        loss.backward()
        adv_touched = any(p.grad is not None for p in model.adversary_head.parameters())
        expected = name in ("adversarial", "combined")
        assert adv_touched == expected, f"{name}: adversary_head gradient presence was {adv_touched}, expected {expected}"


if __name__ == "__main__":
    test_grad_reverse_negates_and_scales_the_gradient()
    test_orthogonality_loss_bounded_and_zero_for_orthogonal_vectors()
    test_all_six_methods_forward_backward_without_error()
    test_adversary_head_only_gets_gradient_under_adversarial_and_combined()
    print("OK")
