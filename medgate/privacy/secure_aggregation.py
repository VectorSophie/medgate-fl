"""Single-process secure-aggregation SIMULATION — pairwise-additive masking
of the mathematical core of Bonawitz et al. 2017
(docs/literature_matrix.csv id bonawitz2017-secagg): every ordered client
pair (i, j) shares a random mask; client i adds it to its update, client j
subtracts it, so masks cancel exactly in the sum.

Explicitly, precisely NOT implemented or claimed (repeated here because an
earlier version of this module's docstrings blurred this — see
docs/execution_plan.md Phase 4 for the full account of that repair):
  - Diffie-Hellman key agreement. `_pairwise_mask` derives every pair's
    mask from one shared Python-level `seed` argument. In a real
    deployment each client would derive its shared secrets via DH with
    every other client and the server would never see the seed or the
    per-pair secrets; in THIS single-process simulation, whatever caller
    holds `seed` can call `_pairwise_mask` directly and reconstruct any
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
  - A formal cryptographic security proof or reduction. This module's
    "empirical_concealment_sanity_check" below is exactly that: an
    empirical sanity check against one classifier family, not a proof —
    see its docstring for why that distinction matters.

What this module DOES support, and what Phase 4 actually uses it for: a
faithful simulation of what the aggregating SERVER can observe under
plaintext FedAvg (every client's raw update) versus under this masking
scheme (only masked per-client values and their exact sum) — sufficient to
compare, e.g., a property-inference attack's success under each
(medgate/attacks/property_inference.py, Phase 3).
"""
import torch


def _pairwise_mask(shape, seed: int, i: int, j: int) -> torch.Tensor:
    """Deterministic mask shared by clients i and j (i < j), derived from a
    single seed -- stands in for the pairwise-agreed secret the real
    protocol derives via Diffie-Hellman. See the module docstring: in this
    single-process simulation, anyone holding `seed` can recompute this,
    which a real deployment's process/network boundary would prevent."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + i * 9973 + j)
    return torch.randn(shape, generator=g)


def mask_client_updates(updates: list[dict], seed: int) -> list[dict]:
    """updates[i] = client i's update (local_state - global_state), one
    dict of tensors per client. Returns masked copies: for every pair
    i < j, updates[i] += mask_ij, updates[j] -= mask_ij, for every
    parameter key. The masks cancel out exactly when ALL masked updates
    are summed (see test_masking_correctness_multiple_client_counts_and_shapes)."""
    n = len(updates)
    masked = [{k: v.clone() for k, v in u.items()} for u in updates]
    keys = updates[0].keys()
    for i in range(n):
        for j in range(i + 1, n):
            for k in keys:
                m = _pairwise_mask(updates[i][k].shape, seed, i, j)
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
      mask draw. NOT a confidentiality measure either way — do not treat a
      low value as "proof" of hiding or a moderate value as a failure. The
      real security argument is that the mask is a secret, uniformly-random
      one-time pad, i.e. a statement about the masked value's
      DISTRIBUTION, not about any one sampled geometric distance — see
      empirical_concealment_sanity_check below for the check that actually
      probes concealment, and its own caveats about what even that does
      not prove. (Measured empirically in this project: mean |cos sim| for
      this field alone is often NOT small at small client counts, ~0.4-0.6
      for n=3-6, even though the mask is a valid one-time pad — a wrong
      test built around this exact field was written and caught during
      this project's own development, see docs/execution_plan.md Phase 4.)
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
) -> dict:
    """An EMPIRICAL SANITY CHECK, explicitly NOT a security proof — read
    the "why this is not a proof" section below before using this result
    for anything.

    Method: construct two distinguishable update "distributions" (i.i.d.
    Gaussian tensors of the given shape, means shifted by `mean_shift`,
    unit variance), draw n_samples updates from each, mask every one under
    a FRESH random mask (a distinct draw per sample — unlike
    mask_client_updates' deterministic pairwise masks used for reproducible
    aggregation elsewhere in this module, this mirrors a real deployment
    using a fresh mask per round/client, so the attacker cannot exploit
    mask reuse). Fit a held-out logistic-regression classifier to
    distinguish which distribution produced a given MASKED update; report
    its AUC (with a bootstrap 95% CI) against ground truth. As a control,
    fit the identical classifier on the UNMASKED updates, which should
    trivially separate the two distributions (near-1.0 AUC) — confirming
    the task is not trivially unsolvable for a reason unrelated to masking.

    WHY A NEAR-CHANCE MASKED-CASE AUC IS NOT A SECURITY PROOF:
    (1) it is an empirical result against ONE classifier family (logistic
        regression) on ONE finite sample size — not a claim that holds
        against every possible distinguisher, including nonlinear ones,
        ones with more samples, or a computationally unbounded adversary;
    (2) real secure aggregation's security argument is
        INFORMATION-THEORETIC: a fresh, secret, uniformly-random mask
        makes the masked value's distribution IDENTICAL regardless of the
        plaintext (a one-time-pad argument, provable in closed form), not
        an empirical claim about whether one classifier happened to find
        the signal;
    (3) this simulation's masks are not exchanged over any real
        cryptographic channel (see the module docstring's Diffie-Hellman
        caveat) — a near-chance result here says nothing about the key
        exchange this simulation never implements.
    This function can corroborate that the masking does not leak an
    OBVIOUS statistical signal to a standard classifier at this sample
    size. Nothing more.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    g = torch.Generator().manual_seed(seed)
    dist_a = torch.randn((n_samples, *shape), generator=g)
    dist_b = torch.randn((n_samples, *shape), generator=g) + mean_shift
    labels = np.array([0] * n_samples + [1] * n_samples)

    def masked_view(x, mask_seed_offset):
        mg = torch.Generator().manual_seed(seed + 500_000 + mask_seed_offset)
        mask = torch.randn(shape, generator=mg)
        return (x + mask).flatten().numpy()

    unmasked_X = torch.cat([dist_a, dist_b], dim=0).flatten(1).numpy()
    masked_X = np.stack(
        [masked_view(dist_a[i], i) for i in range(n_samples)]
        + [masked_view(dist_b[i], n_samples + i) for i in range(n_samples)]
    )

    def fit_eval_auc(X, seed_):
        rng = np.random.RandomState(seed_)
        idx = rng.permutation(len(X))
        split = len(X) // 2
        train_idx, test_idx = idx[:split], idx[split:]
        clf = LogisticRegression(max_iter=1000, random_state=seed_)
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
        "masked_case_attack_auc": masked_auc,
        "masked_case_attack_auc_95ci": [ci_lo, ci_hi],
        "unmasked_control_attack_auc": unmasked_auc,
        "n_samples_per_distribution": n_samples,
        "mean_shift": mean_shift,
        "not_a_security_proof": "see empirical_concealment_sanity_check docstring",
    }
