"""Phase 4 checks: secure-aggregation masking actually hides individual
updates while preserving the exact sum, and DP-SGD actually trains under
Opacus and returns a finite epsilon.

Run: PYTHONPATH=. pytest tests/test_phase4_privacy.py -v
"""
import torch

from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, SyntheticFedISIC, make_synthetic_centers
from medgate.models.backbone import MedGateModel
from medgate.privacy.dp_sgd import dp_local_train
from medgate.privacy.secure_aggregation import (
    empirical_concealment_sanity_check,
    mask_client_updates,
    masking_correctness_diagnostic,
    secure_aggregate_updates,
    secure_fedavg_round,
)

MODEL_KWARGS = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), feature_dim=16, adapter_rank=2)


def _fake_updates(n_clients=3, seed=0):
    torch.manual_seed(seed)
    shape_a, shape_b = (5, 5), (3,)
    return [{"a": torch.randn(shape_a), "b": torch.randn(shape_b)} for _ in range(n_clients)]


def test_masking_preserves_the_exact_sum():
    updates = _fake_updates()
    masked = mask_client_updates(updates, seed=42)
    aggregated = secure_aggregate_updates(masked, weights=[1.0] * len(updates))
    true_mean = {k: sum(u[k] for u in updates) / len(updates) for k in updates[0]}
    for k in true_mean:
        assert torch.allclose(aggregated[k], true_mean[k], atol=1e-5)


def test_masking_correctness_error_is_essentially_zero():
    updates = _fake_updates()
    masked = mask_client_updates(updates, seed=42)
    check = masking_correctness_diagnostic(updates, masked)
    assert check["aggregate_reconstruction_max_abs_error"] < 1e-5


def test_masking_correctness_multiple_client_counts_and_shapes():
    """The correctness property (masks cancel exactly) must hold beyond
    the one (n=3, two small shapes) case above — P0-4 review requirement:
    'multiple client counts and tensor shapes.'"""
    for n_clients in (2, 4, 7):
        for shape in ((1,), (3, 3), (2, 4, 4)):
            g = torch.Generator().manual_seed(n_clients * 100 + hash(shape) % 97)
            updates = [{"w": torch.randn(shape, generator=g)} for _ in range(n_clients)]
            masked = mask_client_updates(updates, seed=123)
            aggregated = secure_aggregate_updates(masked, weights=[1.0] * n_clients)
            true_mean = sum(u["w"] for u in updates) / n_clients
            assert torch.allclose(aggregated["w"], true_mean, atol=1e-5), f"n_clients={n_clients} shape={shape}"


def test_masking_is_deterministic_given_the_same_seed():
    """Same updates + same seed => bit-identical masked output (P0-4:
    'deterministic reproduction from test seeds') -- this determinism is
    what makes secure_fedavg_round's results reproducible given a fixed
    config+seed, the project's own reproducibility requirement."""
    updates = _fake_updates(seed=5)
    masked_1 = mask_client_updates(updates, seed=99)
    masked_2 = mask_client_updates(updates, seed=99)
    for k in updates[0]:
        assert torch.equal(masked_1[0][k], masked_2[0][k])


def test_masking_output_depends_on_secret_mask_not_recoverable_without_it():
    """Kept from an earlier repair (docs/execution_plan.md Phase 4): the
    property that actually demonstrates SOME concealment (the SAME true
    update, masked with two different attacker-unknown mask seeds,
    produces clearly different outputs) -- i.e. the masked value is not a
    stable, invertible function of the truth an observer could learn to
    undo without the mask. See masking_correctness_diagnostic's docstring
    for the earlier, WRONG version of this test (comparing masked vs.
    true directly) and why it was wrong."""
    updates = _fake_updates()
    masked_1 = mask_client_updates(updates, seed=1)
    masked_2 = mask_client_updates(updates, seed=2)
    diff = (masked_1[0]["a"] - masked_2[0]["a"]).abs().mean().item()
    true_scale = updates[0]["a"].abs().mean().item()
    assert diff > true_scale, "masked outputs should differ across mask seeds by more than the true update's own scale"


def test_dropout_breaks_correctness():
    """Documents, by direct demonstration rather than assertion alone, the
    module docstring's stated limitation: this simplified simulation does
    NOT implement Shamir-secret-sharing dropout recovery. If a masked
    update is produced but then dropped before aggregation (simulating a
    client disconnecting after masking but before the server receives its
    share), the surviving masks no longer cancel and the aggregate is
    WRONG -- by exactly the magnitude of the dropped client's own mask
    terms, not some small numerical error."""
    updates = _fake_updates(n_clients=4, seed=6)
    masked = mask_client_updates(updates, seed=77)
    survivors = masked[:3]  # client 3 "dropped out" after masking
    true_mean_of_survivors = sum(u["a"] for u in updates[:3]) / 3
    aggregated_survivors = secure_aggregate_updates(survivors, weights=[1.0] * 3)
    assert not torch.allclose(aggregated_survivors["a"], true_mean_of_survivors, atol=1e-3), (
        "dropout should break exact cancellation in this simplified simulation -- if this now passes, "
        "dropout recovery may have been added and this test (and the module's documented limitation) needs updating"
    )


def test_server_client_collusion_can_unmask_a_target_in_this_simulation():
    """Documents the module docstring's Diffie-Hellman caveat by direct
    demonstration: in this single-process simulation, anyone holding
    `seed` (which a real deployment's server never would) can recompute
    any pairwise mask directly and unmask a target client's contribution
    without needing every other client's cooperation -- a structural
    limitation of simulating a multi-party protocol in one process, not a
    subtle attack. This is exactly why the module claims only a
    single-process SIMULATION, never confidentiality against a
    server/client that can read `seed`."""
    from medgate.privacy.secure_aggregation import _pairwise_mask

    updates = _fake_updates(n_clients=3, seed=8)
    seed = 55
    masked = mask_client_updates(updates, seed=seed)
    # "Server" (or a colluding client) reconstructs client 0's true update
    # using only `seed` and the public protocol structure -- no secret
    # channel with client 0 was needed, unlike the real DH-based protocol.
    reconstructed = masked[0]["a"].clone()
    for j in range(1, 3):
        reconstructed = reconstructed - _pairwise_mask(updates[0]["a"].shape, seed, 0, j)
    assert torch.allclose(reconstructed, updates[0]["a"], atol=1e-6)


def test_empirical_concealment_sanity_check_runs_and_reports_both_arms():
    """Confirms the diagnostic itself works and that its UNMASKED control
    is actually a meaningful control (near-perfect separability) -- if the
    control were near-chance too, the whole check would be uninformative
    regardless of what the masked-case number says."""
    result = empirical_concealment_sanity_check(shape=(6,), seed=0, n_samples=60, mean_shift=3.0, n_bootstrap=50)
    assert result["unmasked_control_attack_auc"] > 0.9, "control classifier should trivially separate the two unmasked distributions"
    assert 0.0 <= result["masked_case_attack_auc"] <= 1.0
    lo, hi = result["masked_case_attack_auc_95ci"]
    assert lo <= result["masked_case_attack_auc"] <= hi or (lo != lo)  # allow nan CI on the rare degenerate bootstrap


def test_secure_aggregate_rejects_unequal_weights():
    updates = _fake_updates()
    masked = mask_client_updates(updates, seed=1)
    try:
        secure_aggregate_updates(masked, weights=[1.0, 2.0, 1.0])
        assert False, "expected ValueError for unequal weights"
    except ValueError:
        pass


def test_secure_fedavg_round_changes_the_model_and_stays_finite():
    torch.manual_seed(0)
    centers = make_synthetic_centers(samples_per_center=8, image_size=16, seed=3)[:2]
    model = MedGateModel(**MODEL_KWARGS)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    new_state = secure_fedavg_round(model, centers, epochs=1, batch_size=4, lr=0.05, seed=7)
    for k, v in new_state.items():
        assert torch.isfinite(v).all(), f"non-finite value in {k}"
    moved = any(not torch.allclose(new_state[k], before[k]) for k in new_state)
    assert moved


def test_dp_local_train_produces_finite_params_and_epsilon():
    torch.manual_seed(0)
    model = MedGateModel(**MODEL_KWARGS)
    dataset = SyntheticFedISIC(num_samples=8, image_size=16, seed=4)
    state, epsilon = dp_local_train(
        model, dataset, epochs=1, batch_size=4, lr=0.05, noise_multiplier=1.0, max_grad_norm=1.0
    )
    assert epsilon > 0 and torch.isfinite(torch.tensor(epsilon))
    for k, v in state.items():
        assert torch.isfinite(v).all(), f"non-finite value in {k}"


# --------------------------------------------------------------- P1-10 additions

def test_dp_excluded_adversary_head_receives_no_gradient_and_is_unchanged():
    """P1-10: the excluded parameter (adversary_head) must be genuinely
    inactive, not just 'not passed to the optimizer' -- its VALUES must be
    bit-identical before and after DP training, since joint_loss never
    calls it at all in this training path."""
    torch.manual_seed(0)
    model = MedGateModel(**MODEL_KWARGS)
    adversary_before = {k: v.clone() for k, v in model.adversary_head.state_dict().items()}
    dataset = SyntheticFedISIC(num_samples=8, image_size=16, seed=4)
    state, _epsilon = dp_local_train(model, dataset, epochs=2, batch_size=4, lr=0.05, noise_multiplier=1.0, max_grad_norm=1.0)
    for k, v in adversary_before.items():
        assert torch.equal(state[f"adversary_head.{k}"], v), f"adversary_head.{k} changed despite being excluded/inactive"


def test_dp_epsilon_increases_monotonically_with_more_composition():
    """More local epochs at the same noise_multiplier => strictly more
    composition => strictly higher epsilon (P1-10 requirement)."""
    torch.manual_seed(0)
    dataset = SyntheticFedISIC(num_samples=16, image_size=16, seed=4)
    epsilons = []
    for epochs in (1, 2, 4):
        torch.manual_seed(1)
        model = MedGateModel(**MODEL_KWARGS)
        _state, eps = dp_local_train(model, dataset, epochs=epochs, batch_size=4, lr=0.05, noise_multiplier=1.0, max_grad_norm=1.0)
        epsilons.append(eps)
    assert epsilons[0] < epsilons[1] < epsilons[2], f"epsilon did not increase monotonically with composition: {epsilons}"


def test_dp_stronger_noise_reduces_epsilon():
    """Higher noise_multiplier, identical everything else => lower
    (better/tighter) epsilon (P1-10 requirement)."""
    torch.manual_seed(0)
    dataset = SyntheticFedISIC(num_samples=16, image_size=16, seed=4)
    epsilons = []
    for nm in (0.5, 1.0, 2.0):
        torch.manual_seed(1)
        model = MedGateModel(**MODEL_KWARGS)
        _state, eps = dp_local_train(model, dataset, epochs=2, batch_size=4, lr=0.05, noise_multiplier=nm, max_grad_norm=1.0)
        epsilons.append(eps)
    assert epsilons[0] > epsilons[1] > epsilons[2], f"epsilon did not decrease with stronger noise: {epsilons}"


if __name__ == "__main__":
    test_masking_preserves_the_exact_sum()
    test_masking_correctness_error_is_essentially_zero()
    test_masking_correctness_multiple_client_counts_and_shapes()
    test_masking_is_deterministic_given_the_same_seed()
    test_masking_output_depends_on_secret_mask_not_recoverable_without_it()
    test_dropout_breaks_correctness()
    test_server_client_collusion_can_unmask_a_target_in_this_simulation()
    test_empirical_concealment_sanity_check_runs_and_reports_both_arms()
    test_secure_aggregate_rejects_unequal_weights()
    test_secure_fedavg_round_changes_the_model_and_stays_finite()
    test_dp_local_train_produces_finite_params_and_epsilon()
    test_dp_excluded_adversary_head_receives_no_gradient_and_is_unchanged()
    test_dp_epsilon_increases_monotonically_with_more_composition()
    test_dp_stronger_noise_reduces_epsilon()
    print("OK")
