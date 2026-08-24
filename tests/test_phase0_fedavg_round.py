"""Phase 0 mandatory check: one two-client FL round works.

Run: PYTHONPATH=. pytest tests/test_phase0_fedavg_round.py -v
"""
import torch

from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, make_synthetic_centers
from medgate.federated.fedavg import run_fedavg_round
from medgate.models.backbone import MedGateModel


def test_two_client_fedavg_round_changes_and_preserves_the_model():
    torch.manual_seed(0)
    centers = make_synthetic_centers(samples_per_center=32, image_size=32, seed=42)
    two_clients = centers[:2]  # mandatory check is specifically "two-client"

    global_model = MedGateModel(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES))
    initial_state = {k: v.clone() for k, v in global_model.state_dict().items()}

    aggregated_state = run_fedavg_round(
        global_model, two_clients, epochs=1, batch_size=8, lr=1e-2
    )

    # Same keys/shapes as the model it came from (aggregation didn't drop
    # or corrupt any parameter).
    assert set(aggregated_state.keys()) == set(initial_state.keys())
    for key in aggregated_state:
        assert aggregated_state[key].shape == initial_state[key].shape
        assert torch.isfinite(aggregated_state[key]).all(), f"non-finite value in {key}"

    global_model.load_state_dict(aggregated_state)

    # At least one trainable parameter actually moved — the round did work,
    # not a silent no-op.
    moved = any(
        not torch.allclose(aggregated_state[k], initial_state[k])
        for k, p in global_model.named_parameters()
        if p.requires_grad
    )
    assert moved, "no trainable parameter changed after a FedAvg round"


if __name__ == "__main__":
    test_two_client_fedavg_round_changes_and_preserves_the_model()
    print("OK")
