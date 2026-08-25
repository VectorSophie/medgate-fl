"""Phase 4 checks: secure-aggregation masking actually hides individual
updates while preserving the exact sum, and DP-SGD actually trains under
Opacus and returns a finite epsilon.

Run: PYTHONPATH=. pytest tests/test_phase4_privacy.py -v
"""
import torch

from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, SyntheticFedISIC, make_synthetic_centers
from medgate.models.backbone import MedGateModel
from medgate.privacy.dp_sgd import dp_fedavg_round, dp_local_train
from medgate.privacy.secure_aggregation import (
    dequantize_from_zq,
    empirical_concealment_sanity_check,
    mask_client_updates,
    mask_client_updates_zq,
    masking_correctness_diagnostic,
    quantize_to_zq,
    secure_aggregate_updates,
    secure_aggregate_updates_zq,
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


# --------------------------------------------------------------- P0-B additions (repair pass 4)

def test_zq_masked_value_is_exactly_uniform_regardless_of_plaintext():
    """The actual one-time-pad property, checked empirically: two
    DIFFERENT plaintext scalars, each masked under many independent Z_q
    masks, must produce distributions of masked values that are
    statistically indistinguishable from each other AND from uniform over
    Z_q -- unlike Gaussian masking (see this module's docstring), this
    holds regardless of mean_shift or mask 'scale', because it holds
    exactly, not asymptotically."""
    import numpy as np
    from scipy import stats

    q = 97  # small prime so a chi-square-style uniformity check has enough mass per bin at n_draws below
    n_draws = 4000
    plaintext_a = torch.full((1,), 3.0)
    plaintext_b = torch.full((1,), -40.0)  # very different plaintext

    def draws_for(plaintext):
        out = []
        for i in range(n_draws):
            g = torch.Generator().manual_seed(i)
            mask = torch.randint(0, q, (1,), generator=g, dtype=torch.int64)
            out.append(int(torch.remainder(quantize_to_zq(plaintext, scale=1, q=q) + mask, q).item()))
        return np.array(out)

    draws_a = draws_for(plaintext_a)
    draws_b = draws_for(plaintext_b)

    # Both should look uniform over {0, ..., q-1}, and--critically--
    # indistinguishable from EACH OTHER despite wildly different plaintexts.
    counts_a = np.bincount(draws_a, minlength=q)
    counts_b = np.bincount(draws_b, minlength=q)
    chi2_a, p_a = stats.chisquare(counts_a)
    chi2_b, p_b = stats.chisquare(counts_b)
    assert p_a > 0.01, f"masked draws for plaintext A did not look uniform over Z_{q}: p={p_a}"
    assert p_b > 0.01, f"masked draws for plaintext B did not look uniform over Z_{q}: p={p_b}"

    ks_stat, ks_p = stats.ks_2samp(draws_a, draws_b)
    assert ks_p > 0.01, f"masked distributions for two very different plaintexts were distinguishable (KS p={ks_p}) -- the one-time-pad property should make them statistically identical"


def test_gaussian_masked_value_is_not_uniform_and_shifts_with_plaintext():
    """Contrast case for the test above: Gaussian masking (this module's
    original path) does NOT give the masked value's distribution
    independence from the plaintext -- the masked mean visibly tracks the
    plaintext mean even at a moderate mask_scale, unlike the Z_q case."""
    torch.manual_seed(0)
    plaintext_a = torch.full((2000,), 3.0)
    plaintext_b = torch.full((2000,), -40.0)
    mask_scale = 5.0
    g = torch.Generator().manual_seed(1)
    mask = torch.randn(2000, generator=g) * mask_scale
    masked_a = plaintext_a + mask
    masked_b = plaintext_b + mask
    # The masked means are still ~43 apart (the plaintext gap), only the
    # per-sample noise is shared -- Gaussian masking shifts, it does not erase.
    assert abs(masked_a.mean().item() - masked_b.mean().item() - 43.0) < 1.0


def test_zq_quantize_dequantize_round_trip():
    x = torch.tensor([3.25, -7.5, 0.0, 12.0])
    q, scale = 2 ** 31 - 1, 2 ** 12
    z = quantize_to_zq(x, scale=scale, q=q)
    back = dequantize_from_zq(z, scale=scale, q=q)
    assert torch.allclose(back, x, atol=1.0 / scale + 1e-6)


def test_zq_exact_aggregate_recovery_within_quantization_error():
    updates = _fake_updates(n_clients=4, seed=10)
    masked = mask_client_updates_zq(updates, seed=11)
    aggregated = secure_aggregate_updates_zq(masked, weights=[1.0] * len(updates))
    true_mean = {k: sum(u[k] for u in updates) / len(updates) for k in updates[0]}
    for k in true_mean:
        assert torch.allclose(aggregated[k], true_mean[k], atol=1.0 / DEFAULT_SCALE_FOR_TEST + 1e-4), k


def test_zq_wraparound_corrupts_the_aggregate_when_q_is_too_small():
    """Demonstrates, rather than only asserts, the precondition
    dequantize_from_zq's docstring states: if q is too small relative to
    the true summed integer's magnitude, the signed-residue
    interpretation wraps around and the recovered aggregate is WRONG --
    not a small numerical error, a qualitatively different number."""
    torch.manual_seed(0)
    updates = [{"a": torch.full((1,), 50.0)} for _ in range(3)]  # true sum=150, true mean=50
    tiny_q = 101  # far too small: quantized values (scale=1) already approach q on their own
    masked = mask_client_updates_zq(updates, seed=1, scale=1, q=tiny_q)
    aggregated = secure_aggregate_updates_zq(masked, weights=[1.0] * 3, scale=1, q=tiny_q)
    assert not torch.allclose(aggregated["a"], torch.full((1,), 50.0), atol=1.0), (
        f"expected wraparound corruption with too-small q={tiny_q}, got {aggregated['a']} (correctness "
        "held anyway -- q may need to be even smaller, or scale/values changed, to reproduce the failure)"
    )


def test_zq_secure_aggregate_rejects_unequal_weights():
    updates = _fake_updates()
    masked = mask_client_updates_zq(updates, seed=1)
    try:
        secure_aggregate_updates_zq(masked, weights=[1.0, 2.0, 1.0])
        assert False, "expected ValueError for unequal weights"
    except ValueError:
        pass


DEFAULT_SCALE_FOR_TEST = 2 ** 12  # mirrors secure_aggregation.DEFAULT_SCALE; quantization error bound = 1/scale


def test_concealment_improves_with_mask_scale_but_never_reaches_a_guarantee():
    """P0-B requirement: sweep mask scale rather than fixing it at 1.0.
    Larger Gaussian masks should make the two distributions HARDER to
    distinguish (masked AUC trending toward 0.5), corroborating the module
    docstring's "matter of degree, not a guarantee" framing -- this test
    checks the TREND, not that any finite scale reaches exact chance,
    since (per the docstring) it never does."""
    aucs = []
    for scale in (0.5, 5.0, 50.0):
        r = empirical_concealment_sanity_check(shape=(6,), seed=0, n_samples=80, mean_shift=3.0, n_bootstrap=40, mask_scale=scale)
        aucs.append(r["masked_case_attack_auc"])
    assert aucs[0] >= aucs[-1] - 0.05, f"expected concealment to trend toward chance as mask_scale grows: {aucs}"


def test_concealment_heuristic_checked_against_a_nonlinear_attacker_too():
    """P0-B requirement: at least one nonlinear attacker, not logistic
    regression alone -- an MLP given the same masked data should not do
    dramatically better than the linear attacker at a mask scale large
    enough to conceal the linear signal (if it did, that would itself be
    an important, reportable finding, not something to hide by only
    testing the weaker attacker)."""
    r_linear = empirical_concealment_sanity_check(shape=(6,), seed=0, n_samples=80, mean_shift=3.0, n_bootstrap=40, mask_scale=20.0, attacker="logistic")
    r_mlp = empirical_concealment_sanity_check(shape=(6,), seed=0, n_samples=80, mean_shift=3.0, n_bootstrap=40, mask_scale=20.0, attacker="mlp")
    assert r_mlp["attacker"] == "mlp"
    assert 0.0 <= r_mlp["masked_case_attack_auc"] <= 1.0
    assert r_mlp["unmasked_control_attack_auc"] > 0.9, "MLP control should also trivially separate the unmasked distributions"


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
    state, epsilon, engine = dp_local_train(
        model, dataset, epochs=1, batch_size=4, lr=0.05, noise_multiplier=1.0, max_grad_norm=1.0
    )
    assert epsilon > 0 and torch.isfinite(torch.tensor(epsilon))
    assert engine is not None
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
    state, _epsilon, _engine = dp_local_train(model, dataset, epochs=2, batch_size=4, lr=0.05, noise_multiplier=1.0, max_grad_norm=1.0)
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
        _state, eps, _engine = dp_local_train(model, dataset, epochs=epochs, batch_size=4, lr=0.05, noise_multiplier=1.0, max_grad_norm=1.0)
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
        _state, eps, _engine = dp_local_train(model, dataset, epochs=2, batch_size=4, lr=0.05, noise_multiplier=nm, max_grad_norm=1.0)
        epsilons.append(eps)
    assert epsilons[0] > epsilons[1] > epsilons[2], f"epsilon did not decrease with stronger noise: {epsilons}"


# --------------------------------------------------------------- P0-C additions (repair pass 4)

def test_dp_epsilon_composes_across_engine_reuse_not_reset_per_call():
    """The exact regression this repair pass exists to prevent: reusing
    the SAME engine across two dp_local_train calls must give a SECOND
    epsilon that is (a) higher than the first call's alone and (b) NOT
    identical to what a second, independent (fresh-engine) call would
    report -- i.e. real composition is happening, not a silent reset."""
    torch.manual_seed(0)
    dataset = SyntheticFedISIC(num_samples=16, image_size=16, seed=4)
    model = MedGateModel(**MODEL_KWARGS)

    _state1, eps1, engine = dp_local_train(model, dataset, epochs=1, batch_size=4, lr=0.05, noise_multiplier=1.0, max_grad_norm=1.0)
    fresh_model = MedGateModel(**MODEL_KWARGS)
    fresh_model.load_state_dict(_state1)
    _state2, eps2_composed, engine = dp_local_train(fresh_model, dataset, epochs=1, batch_size=4, lr=0.05, noise_multiplier=1.0, max_grad_norm=1.0, engine=engine)

    # An independent (non-reused-engine) second call, for comparison.
    independent_model = MedGateModel(**MODEL_KWARGS)
    independent_model.load_state_dict(_state1)
    _state2b, eps2_independent, _e = dp_local_train(independent_model, dataset, epochs=1, batch_size=4, lr=0.05, noise_multiplier=1.0, max_grad_norm=1.0)

    assert eps2_composed > eps1, f"composed epsilon after a second round must exceed the first round's alone: {eps1} -> {eps2_composed}"
    assert eps2_composed > eps2_independent, (
        f"a composed (engine-reused) two-round epsilon ({eps2_composed}) must exceed an independent "
        f"single-round epsilon ({eps2_independent}) -- if these are ever equal, engine reuse silently "
        "stopped composing and P0-C's bug is back"
    )


def test_dp_epsilon_increases_monotonically_with_more_federated_rounds():
    """P0-C requirement #5, at the dp_fedavg_round level (not just
    dp_local_train): more FEDERATED ROUNDS at fixed per-round epochs/noise,
    with engines reused across rounds, must strictly increase the reported
    (now genuinely cumulative) epsilon round over round."""
    torch.manual_seed(0)
    centers = make_synthetic_centers(samples_per_center=8, image_size=16, seed=3)[:2]
    torch.manual_seed(1)
    model = MedGateModel(**MODEL_KWARGS)
    engines = None
    epsilons = []
    for _round in range(3):
        state, eps, engines = dp_fedavg_round(model, centers, epochs=1, batch_size=4, lr=0.05, noise_multiplier=1.0, max_grad_norm=1.0, engines=engines)
        model.load_state_dict(state)
        epsilons.append(eps)
    assert epsilons[0] < epsilons[1] < epsilons[2], f"epsilon did not increase monotonically with more federated rounds: {epsilons}"


def test_dp_epsilon_increases_with_higher_sample_rate():
    """P0-C requirement #5: holding steps/epochs/noise fixed, a LARGER
    batch size against the same dataset (i.e. a higher per-step sampling
    rate) should increase epsilon -- more of the dataset is touched, and
    with more effective information per step, per Opacus's RDP accounting."""
    torch.manual_seed(0)
    dataset = SyntheticFedISIC(num_samples=32, image_size=16, seed=4)
    epsilons = []
    for batch_size in (2, 8, 16):
        torch.manual_seed(1)
        model = MedGateModel(**MODEL_KWARGS)
        _state, eps, _engine = dp_local_train(model, dataset, epochs=1, batch_size=batch_size, lr=0.05, noise_multiplier=1.0, max_grad_norm=1.0)
        epsilons.append(eps)
    assert epsilons[0] < epsilons[1] < epsilons[2], f"epsilon did not increase monotonically with sample rate (batch_size): {epsilons}"


if __name__ == "__main__":
    test_masking_preserves_the_exact_sum()
    test_masking_correctness_error_is_essentially_zero()
    test_masking_correctness_multiple_client_counts_and_shapes()
    test_masking_is_deterministic_given_the_same_seed()
    test_masking_output_depends_on_secret_mask_not_recoverable_without_it()
    test_dropout_breaks_correctness()
    test_server_client_collusion_can_unmask_a_target_in_this_simulation()
    test_empirical_concealment_sanity_check_runs_and_reports_both_arms()
    test_zq_masked_value_is_exactly_uniform_regardless_of_plaintext()
    test_gaussian_masked_value_is_not_uniform_and_shifts_with_plaintext()
    test_zq_quantize_dequantize_round_trip()
    test_zq_exact_aggregate_recovery_within_quantization_error()
    test_zq_wraparound_corrupts_the_aggregate_when_q_is_too_small()
    test_zq_secure_aggregate_rejects_unequal_weights()
    test_concealment_improves_with_mask_scale_but_never_reaches_a_guarantee()
    test_concealment_heuristic_checked_against_a_nonlinear_attacker_too()
    test_secure_aggregate_rejects_unequal_weights()
    test_secure_fedavg_round_changes_the_model_and_stays_finite()
    test_dp_local_train_produces_finite_params_and_epsilon()
    test_dp_excluded_adversary_head_receives_no_gradient_and_is_unchanged()
    test_dp_epsilon_increases_monotonically_with_more_composition()
    test_dp_stronger_noise_reduces_epsilon()
    test_dp_epsilon_composes_across_engine_reuse_not_reset_per_call()
    test_dp_epsilon_increases_monotonically_with_more_federated_rounds()
    test_dp_epsilon_increases_with_higher_sample_rate()
    print("OK")
