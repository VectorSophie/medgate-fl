"""Single-process SIMULATED pairwise-additive masking, in the mathematical
style of Bonawitz et al. 2017 (docs/literature_matrix.csv id
bonawitz2017-secagg): every ordered client pair (i, j) shares a random
mask; client i adds it to its update, client j subtracts it, so masks
cancel exactly in the sum.

REPAIR PASS 4 / P0-B — security-model correction, kept here rather than
only in a commit message so nobody re-introduces the overclaim:

  Earlier drafts of this module's docstrings argued that "the mask is a
  secret, uniformly-random one-time pad, i.e. a statement about the masked
  value's distribution" and cited that as the reason a near-chance
  empirical-classifier result should be read as evidence of concealment.
  That argument is only TRUE for masks drawn uniformly from a FINITE ring
  or field (Z_q) — for such masks, (plaintext + mask) mod q is EXACTLY
  uniform over Z_q regardless of the plaintext, a provable one-time-pad
  property. `mask_client_updates` below draws masks from
  `torch.randn` — a GAUSSIAN distribution over the reals — and Gaussian
  masking does NOT have that property: if X ~ N(mu_a, s^2) and
  Y ~ N(mu_b, s^2) are two possible plaintexts and M ~ N(0, sigma^2) is the
  mask, then X+M ~ N(mu_a, s^2+sigma^2) and Y+M ~ N(mu_b, s^2+sigma^2) —
  these are DIFFERENT distributions for any finite sigma (they merely
  overlap more as sigma grows), never IDENTICAL. So a Gaussian mask makes
  distinguishing plaintexts HARDER, asymptotically, but never gives the
  information-theoretic guarantee real SecAgg deployments rely on. This
  module's docstrings and `empirical_concealment_sanity_check` below were
  quietly overclaiming that Gaussian masking has that guarantee. Fixed by:
    (1) this module is now documented as a CORRECTNESS and SERVER-VIEW
        simulation ONLY — what the aggregating server can observe under
        each scheme, and that pairwise masks cancel exactly — not a
        cryptographic security implementation, and no distribution-level
        confidentiality claim is made for the Gaussian-mask functions;
    (2) `mask_client_updates_zq` / `secure_aggregate_updates_zq` below ARE
        a uniform-over-Z_q masking simulation, with quantization and
        wraparound handled explicitly, for which the one-time-pad argument
        genuinely applies — see
        test_zq_masked_value_is_exactly_uniform_regardless_of_plaintext;
    (3) `empirical_concealment_sanity_check` is now explicitly a HEURISTIC
        only, swept over mask scale (showing concealment is a matter of
        DEGREE for Gaussian masks, not a guarantee) and checked against
        both a linear and a nonlinear attacker.

Explicitly, precisely NOT implemented or claimed (repeated here because an
earlier version of this module's docstrings blurred parts of this too —
see docs/execution_plan.md Phase 4 for the full account of that repair):
  - Diffie-Hellman key agreement. `_pairwise_mask*` derive every pair's
    mask from one shared Python-level `seed` argument. In a real
    deployment each client would derive its shared secrets via DH with
    every other client and the server would never see the seed or the
    per-pair secrets; in THIS single-process simulation, whatever caller
    holds `seed` can call `_pairwise_mask*` directly and reconstruct any
    client's true update by unmasking it — there is no process boundary
    enforcing that the "server" role cannot do this. That is a structural
    limitation of simulating a multi-party protocol in one process, stated
    explicitly rather than glossed over.
  - Shamir secret-sharing dropout recovery. A dropped client here breaks
    exact cancellation (see test_dropout_breaks_correctness below, which
    demonstrates this rather than merely asserting it) — the real
    protocol's recovery mechanism is simply not implemented.
  - Authenticated/encrypted transport, server-client authentication,
    collusion-threshold guarantees. None of this exists here — masking
    values in memory is the only mechanism simulated.
  - A formal cryptographic security proof or reduction for the Gaussian
    masking path. `empirical_concealment_sanity_check` is exactly that: an
    empirical heuristic against two classifier families at a few mask
    scales, not a proof — see its docstring for why that distinction
    matters, and why only the Z_q path below has an actual
    information-theoretic argument behind it.

What this module DOES support, and what Phase 4 actually uses it for: a
faithful simulation of what the aggregating SERVER can observe under
plaintext FedAvg (every client's raw update) versus under simulated
pairwise additive masking (only masked per-client values and their exact
sum) — sufficient to compare, e.g., a property-inference attack's success
under each (medgate/attacks/property_inference.py, Phase 3). Paper/table
labels refer to this as "simulated pairwise additive masking," never as
"secure aggregation" unqualified (repair pass 4 / P0-B requirement #6).
"""
import torch

# --- Gaussian-mask correctness/server-view simulation (original path) ---
# NOT an information-theoretic concealment mechanism -- see module
# docstring. Kept because it is simpler to read for the CORRECTNESS
# property (exact cancellation) and because the empirical heuristic check
# needs a mask whose "how much does more noise help" behavior is worth
# demonstrating (a finite-field mask has no such dial: it's exact at any
# scale, which is a less pedagogically useful contrast on its own).


def _pairwise_mask(shape, seed: int, i: int, j: int, mask_scale: float = 1.0) -> torch.Tensor:
    """Deterministic Gaussian mask shared by clients i and j (i < j),
    derived from a single seed -- stands in for the pairwise-agreed secret
    the real protocol derives via Diffie-Hellman. See the module
    docstring: in this single-process simulation, anyone holding `seed`
    can recompute this, which a real deployment's process/network boundary
    would prevent. `mask_scale` is the mask's standard deviation -- higher
    values make plaintexts harder (never information-theoretically
    impossible) to distinguish from a masked value; see
    empirical_concealment_sanity_check."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + i * 9973 + j)
    return torch.randn(shape, generator=g) * mask_scale


def mask_client_updates(updates: list[dict], seed: int, mask_scale: float = 1.0) -> list[dict]:
    """updates[i] = client i's update (local_state - global_state), one
    dict of tensors per client. Returns masked copies: for every pair
    i < j, updates[i] += mask_ij, updates[j] -= mask_ij, for every
    parameter key. The masks cancel out exactly when ALL masked updates
    are summed (see test_masking_correctness_multiple_client_counts_and_shapes).
    Correctness (exact cancellation) holds at ANY mask_scale, including 0
    -- it is a property of the pairwise construction, not of the mask's
    distribution. Concealment is a different, scale-dependent property;
    see the module docstring."""
    n = len(updates)
    masked = [{k: v.clone() for k, v in u.items()} for u in updates]
    keys = updates[0].keys()
    for i in range(n):
        for j in range(i + 1, n):
            for k in keys:
                m = _pairwise_mask(updates[i][k].shape, seed, i, j, mask_scale)
                masked[i][k] = masked[i][k] + m
                masked[j][k] = masked[j][k] - m
    return masked


def secure_aggregate_updates(masked_updates: list[dict], weights: list[float]) -> dict:
    """Server-side: sum the masked updates it received. Correct ONLY under
    uniform weighting (equal client weights) -- with unequal weights the
    masks would not cancel after per-client scaling, which is why this
    function asserts near-uniform weights rather than silently producing a
    subtly-wrong aggregate. The real Bonawitz protocol handles weighted
    sums by having clients pre-scale before masking; not implemented here.

    Also correct ONLY when every masked update that was produced IS
    summed here (i.e. no client dropped out between masking and
    aggregation) — see test_dropout_breaks_correctness for a direct
    demonstration that this is NOT handled, matching the module
    docstring's stated limitation."""
    if len(set(round(w, 6) for w in weights)) > 1:
        raise ValueError(
            "secure_aggregate_updates: masks only cancel under uniform client "
            "weighting in this simplified simulation; got unequal weights "
            f"{weights}. Pre-scale updates identically before masking if you "
            "need weighted secure aggregation, or use plaintext FedAvg."
        )
    keys = masked_updates[0].keys()
    return {k: sum(u[k] for u in masked_updates) / len(masked_updates) for k in keys}


def secure_fedavg_round(global_model, client_datasets: list, epochs: int, batch_size: int, lr: float, seed: int) -> dict:
    """One federated round where the server only ever sees MASKED
    per-client updates (mask_client_updates) and their sum
    (secure_aggregate_updates) — never a plaintext individual update.
    Requires uniform client weighting (see secure_aggregate_updates)."""
    import copy

    from medgate.federated.fedavg import local_train

    global_state = {k: v.clone() for k, v in global_model.state_dict().items()}
    updates = []
    for ds in client_datasets:
        local_model = copy.deepcopy(global_model)
        local_state = local_train(local_model, ds, epochs, batch_size, lr)
        updates.append({k: local_state[k] - global_state[k] for k in global_state})

    masked = mask_client_updates(updates, seed)
    aggregated_update = secure_aggregate_updates(masked, weights=[1.0] * len(updates))
    return {k: global_state[k] + aggregated_update[k] for k in global_state}


def masking_correctness_diagnostic(updates: list[dict], masked: list[dict]) -> dict:
    """The LOAD-BEARING check (renamed from an earlier 'confidentiality_check'
    — that name overclaimed what this function actually verifies; see
    docs/execution_plan.md Phase 4). Two fields:

    - aggregate_reconstruction_max_abs_error: masked updates still sum to
      the true sum (should be ~0). This is the property this whole module
      is FOR — if this is not ~0, the masking is simply broken.
    - individual_update_cosine_similarity_to_true: how similar one
      client's masked update looks to its true update, for THIS ONE random
      mask draw. NOT a concealment measure either way — do not treat a
      low value as "proof" of hiding or a moderate value as a failure. A
      Gaussian mask does not make the masked distribution independent of
      the plaintext (see module docstring's P0-B correction); the closest
      thing to a concealment argument this module has is
      empirical_concealment_sanity_check, itself only a heuristic.
      (Measured empirically in this project: mean |cos sim| for
      this field alone is often NOT small at small client counts, ~0.4-0.6
      for n=3-6, even at mask_scale=1 -- a wrong test built around this
      exact field was written and caught during this project's own
      development, see docs/execution_plan.md Phase 4.)
    """
    key0 = next(iter(updates[0]))
    true_flat = updates[0][key0].flatten()
    masked_flat = masked[0][key0].flatten()
    cos_sim = torch.nn.functional.cosine_similarity(true_flat, masked_flat, dim=0, eps=1e-8).item()

    true_sum = {k: sum(u[k] for u in updates) for k in updates[0]}
    masked_sum = {k: sum(u[k] for u in masked) for k in masked[0]}
    max_err = max((true_sum[k] - masked_sum[k]).abs().max().item() for k in true_sum)

    return {
        "individual_update_cosine_similarity_to_true": cos_sim,  # see caution above -- not a concealment measure either way
        "aggregate_reconstruction_max_abs_error": max_err,  # should be ~0 -- THE load-bearing correctness check
    }


def empirical_concealment_sanity_check(
    shape: tuple, seed: int, n_samples: int = 100, mean_shift: float = 2.0, n_bootstrap: int = 200,
    mask_scale: float = 1.0, attacker: str = "logistic",
) -> dict:
    """An EMPIRICAL HEURISTIC, explicitly NOT a security proof — read the
    "why this is not a proof" section below before using this result for
    anything. P0-B (repair pass 4): this function now takes `mask_scale`
    (swept by test_concealment_improves_with_mask_scale_but_never_reaches_a_guarantee)
    and `attacker` (`"logistic"` or `"mlp"`, swept by
    test_concealment_heuristic_checked_against_a_nonlinear_attacker_too) —
    a single fixed scale and a single linear classifier were exactly the
    kind of narrow heuristic the module docstring warns is not a proof,
    demonstrated concretely rather than only asserted.

    Method: construct two distinguishable update "distributions" (i.i.d.
    Gaussian tensors of the given shape, means shifted by `mean_shift`,
    unit variance), draw n_samples updates from each, mask every one under
    a FRESH random mask of standard deviation `mask_scale` (a distinct
    draw per sample — unlike mask_client_updates' deterministic pairwise
    masks used for reproducible aggregation elsewhere in this module, this
    mirrors a real deployment using a fresh mask per round/client, so the
    attacker cannot exploit mask reuse). Fit a held-out classifier
    (`attacker`) to distinguish which distribution produced a given MASKED
    update; report its AUC (with a bootstrap 95% CI) against ground truth.
    As a control, fit the identical classifier on the UNMASKED updates,
    which should trivially separate the two distributions (near-1.0 AUC)
    — confirming the task is not trivially unsolvable for a reason
    unrelated to masking.

    WHY A NEAR-CHANCE MASKED-CASE AUC IS NOT A SECURITY PROOF, EVEN AT A
    LARGE mask_scale:
    (1) it is an empirical result against a FEW classifier families (here:
        logistic regression, a one-hidden-layer MLP) on ONE finite sample
        size — not a claim that holds against every possible distinguisher,
        including ones with more samples, different architectures, or a
        computationally unbounded adversary;
    (2) a REAL secure-aggregation deployment's security argument is
        INFORMATION-THEORETIC: a fresh, secret, uniformly-random mask over
        a finite ring/field makes the masked value's distribution IDENTICAL
        regardless of the plaintext (a one-time-pad argument, provable in
        closed form) — see mask_client_updates_zq /
        test_zq_masked_value_is_exactly_uniform_regardless_of_plaintext for
        the version of this module where that argument actually applies.
        THIS function's masks are Gaussian over the reals, which do NOT
        have that property at any finite mask_scale (see this module's
        docstring) — increasing mask_scale only makes the two Gaussians
        overlap more, it never makes them identical, so a near-chance AUC
        here is "a big enough mask that this attacker and this sample size
        couldn't tell," not "no attacker with any sample size ever could";
    (3) this simulation's masks are not exchanged over any real
        cryptographic channel (see the module docstring's Diffie-Hellman
        caveat) — a near-chance result here says nothing about the key
        exchange this simulation never implements.
    This function can corroborate that Gaussian masking at a given scale
    does not leak an OBVIOUS statistical signal to a standard classifier at
    this sample size. Nothing more.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier

    g = torch.Generator().manual_seed(seed)
    dist_a = torch.randn((n_samples, *shape), generator=g)
    dist_b = torch.randn((n_samples, *shape), generator=g) + mean_shift
    labels = np.array([0] * n_samples + [1] * n_samples)

    def masked_view(x, mask_seed_offset):
        mg = torch.Generator().manual_seed(seed + 500_000 + mask_seed_offset)
        mask = torch.randn(shape, generator=mg) * mask_scale
        return (x + mask).flatten().numpy()

    unmasked_X = torch.cat([dist_a, dist_b], dim=0).flatten(1).numpy()
    masked_X = np.stack(
        [masked_view(dist_a[i], i) for i in range(n_samples)]
        + [masked_view(dist_b[i], n_samples + i) for i in range(n_samples)]
    )

    def make_classifier(seed_):
        if attacker == "logistic":
            return LogisticRegression(max_iter=1000, random_state=seed_)
        if attacker == "mlp":
            return MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=seed_)
        raise ValueError(f"unknown attacker {attacker!r}; expected 'logistic' or 'mlp'")

    def fit_eval_auc(X, seed_):
        rng = np.random.RandomState(seed_)
        idx = rng.permutation(len(X))
        split = len(X) // 2
        train_idx, test_idx = idx[:split], idx[split:]
        clf = make_classifier(seed_)
        clf.fit(X[train_idx], labels[train_idx])
        scores = clf.predict_proba(X[test_idx])[:, 1]
        return roc_auc_score(labels[test_idx], scores), test_idx, scores

    masked_auc, test_idx, test_scores = fit_eval_auc(masked_X, seed)
    unmasked_auc, _, _ = fit_eval_auc(unmasked_X, seed)

    rng = np.random.RandomState(seed + 1)
    boot_aucs = []
    test_labels = labels[test_idx]
    n_test = len(test_idx)
    for _ in range(n_bootstrap):
        resample = rng.randint(0, n_test, n_test)
        if len(set(test_labels[resample].tolist())) < 2:
            continue
        boot_aucs.append(roc_auc_score(test_labels[resample], test_scores[resample]))
    ci_lo, ci_hi = (float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))) if boot_aucs else (float("nan"), float("nan"))

    return {
        "attacker": attacker,
        "mask_scale": mask_scale,
        "masked_case_attack_auc": masked_auc,
        "masked_case_attack_auc_95ci": [ci_lo, ci_hi],
        "unmasked_control_attack_auc": unmasked_auc,
        "n_samples_per_distribution": n_samples,
        "mean_shift": mean_shift,
        "not_a_security_proof": "heuristic only -- see empirical_concealment_sanity_check docstring",
    }


# --- Z_q uniform-mask simulation (P0-B addition): the path with an actual
# information-theoretic argument behind it -----------------------------

DEFAULT_Q = 2 ** 31 - 1  # a Mersenne prime; large enough that realistic update sums (see below) do not overflow it at the default SCALE/client counts used in this project's tests
DEFAULT_SCALE = 2 ** 12  # fixed-point scale: quantized_int = round(float_value * SCALE)


def quantize_to_zq(x: torch.Tensor, scale: int = DEFAULT_SCALE, q: int = DEFAULT_Q) -> torch.Tensor:
    """Fixed-point-quantize a float tensor into Z_q (int64 tensor, values
    in [0, q)). Real deployments of Bonawitz-style secure aggregation
    always quantize floats into a finite ring/field first — this is not
    optional plumbing, it is what makes the one-time-pad security argument
    apply at all (see module docstring for why the Gaussian-over-R masking
    used elsewhere in this module does NOT have that property regardless
    of scale). Negative values wrap into the upper half of [0, q) via
    modular reduction, exactly as a real fixed-point-over-Z_q scheme would;
    see dequantize_from_zq for the inverse (signed) interpretation."""
    ints = torch.round(x * scale).long()
    return torch.remainder(ints, q)


def dequantize_from_zq(y: torch.Tensor, scale: int = DEFAULT_SCALE, q: int = DEFAULT_Q) -> torch.Tensor:
    """Inverse of quantize_to_zq: interprets values above q//2 as negative
    (the standard signed-residue convention), then rescales to float. Only
    correct when the TRUE (unquantized) integer value's magnitude never
    exceeded q//2 in the first place — see
    test_zq_wraparound_corrupts_the_aggregate_when_q_is_too_small for a
    direct demonstration of what happens when that precondition is
    violated, rather than only asserting the precondition matters."""
    signed = torch.where(y > q // 2, y - q, y)
    return signed.float() / scale


def _pairwise_mask_zq(shape, seed: int, i: int, j: int, q: int = DEFAULT_Q) -> torch.Tensor:
    """Uniform-over-Z_q analogue of _pairwise_mask. Deterministic given
    seed/i/j for the same Diffie-Hellman-stand-in reason described in the
    module docstring."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + i * 9973 + j + 777)
    return torch.randint(0, q, shape, generator=g, dtype=torch.int64)


def mask_client_updates_zq(updates: list[dict], seed: int, scale: int = DEFAULT_SCALE, q: int = DEFAULT_Q) -> list[dict]:
    """Z_q analogue of mask_client_updates: quantizes each client's update
    into Z_q, then applies pairwise UNIFORM masks that cancel exactly
    (mod q) when summed. Unlike the Gaussian version, a mask drawn
    uniformly from Z_q gives the masked value a distribution that is
    EXACTLY uniform over Z_q regardless of the plaintext — the real
    one-time-pad property; see
    test_zq_masked_value_is_exactly_uniform_regardless_of_plaintext."""
    n = len(updates)
    quantized = [{k: quantize_to_zq(v, scale, q) for k, v in u.items()} for u in updates]
    masked = [{k: v.clone() for k, v in u.items()} for u in quantized]
    keys = updates[0].keys()
    for i in range(n):
        for j in range(i + 1, n):
            for k in keys:
                m = _pairwise_mask_zq(updates[i][k].shape, seed, i, j, q)
                masked[i][k] = torch.remainder(masked[i][k] + m, q)
                masked[j][k] = torch.remainder(masked[j][k] - m, q)
    return masked


def secure_aggregate_updates_zq(masked_updates: list[dict], weights: list[float], scale: int = DEFAULT_SCALE, q: int = DEFAULT_Q) -> dict:
    """Server-side sum-then-dequantize for the Z_q path. Correctness holds
    exactly (mod q) regardless of when the modular reduction happens,
    because pairwise mask terms cancel in modular arithmetic the same way
    they do over the reals — but dequantize_from_zq's SIGNED interpretation
    is only correct if the true summed integer's magnitude stayed below
    q//2; see test_zq_exact_aggregate_recovery_within_quantization_error
    for the passing case and
    test_zq_wraparound_corrupts_the_aggregate_when_q_is_too_small for the
    failure this precondition guards against, matching the pattern already
    used for dropout/unequal-weight failures elsewhere in this module."""
    if len(set(round(w, 6) for w in weights)) > 1:
        raise ValueError(
            "secure_aggregate_updates_zq: masks only cancel under uniform client "
            f"weighting in this simplified simulation; got unequal weights {weights}."
        )
    n = len(masked_updates)
    keys = masked_updates[0].keys()
    result = {}
    for k in keys:
        summed = masked_updates[0][k].clone()
        for u in masked_updates[1:]:
            summed = torch.remainder(summed + u[k], q)
        result[k] = dequantize_from_zq(summed, scale, q) / n
    return result
