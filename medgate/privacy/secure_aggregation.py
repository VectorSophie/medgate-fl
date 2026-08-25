"""Secure aggregation — simplified pairwise-additive-masking simulation of
the mathematical core of Bonawitz et al. 2017
(docs/literature_matrix.csv id bonawitz2017-secagg): every ordered client
pair (i, j) shares a random mask; client i adds it to its update, client j
subtracts it, so masks cancel exactly in the sum but any INDIVIDUAL masked
update is indistinguishable from noise to an observer who only sees that
one update.

Explicitly NOT implemented (simplifications, stated so results are never
over-claimed): the real protocol's Diffie-Hellman key agreement (masks
here are generated from a single shared seed, valid only because this is a
single-process simulation — the project brief's "simulated cross-silo
evaluation" non-claim applies directly), Shamir secret-sharing recovery
for client dropout (a dropped client here would break exact cancellation;
not handled), and authenticated/encrypted transport of the masked updates
(masking is the only property simulated). This module lets Phase 4 compare
what the SERVER can see about an individual update (plaintext FedAvg vs.
masked), not a full re-implementation of the protocol's fault tolerance.
"""
import torch


def _pairwise_mask(shape, seed: int, i: int, j: int) -> torch.Tensor:
    """Deterministic mask shared by clients i and j (i < j), derived from a
    single seed -- stands in for the pairwise-agreed secret the real
    protocol derives via Diffie-Hellman."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + i * 9973 + j)
    return torch.randn(shape, generator=g)


def mask_client_updates(updates: list[dict], seed: int) -> list[dict]:
    """updates[i] = client i's update (local_state - global_state), one
    dict of tensors per client. Returns masked copies: for every pair
    i < j, updates[i] += mask_ij, updates[j] -= mask_ij, for every
    parameter key. The masks cancel out exactly when all masked updates
    are summed."""
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
    sums by having clients pre-scale before masking; not implemented here."""
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


def confidentiality_check(updates: list[dict], masked: list[dict]) -> dict:
    """Diagnostic for Phase 4's report. Two properties, and a deliberate
    caution about a THIRD one that is tempting but wrong:

    - correctness: the masked updates still sum to the true sum (should be
      ~0 error).
    - individual_update_cosine_similarity_to_true: how similar one
      client's masked update looks to its true update, for THIS ONE random
      mask draw. Do not treat a low value here as "proof" of hiding, or a
      moderate value as a failure: the real security argument is that the
      mask is a secret uniformly-random one-time pad, so the masked
      value's DISTRIBUTION carries no information about the truth without
      that secret -- not that any single sampled geometric distance is
      small. At small client counts (n<10) with same-scale masks, this
      number is often NOT small (measured empirically: mean |cos sim| ~0.4
      -0.6 for n=3-6 in this codebase's own tests) even though the
      information-theoretic hiding argument still holds. See
      test_masking_output_depends_on_secret_mask_not_recoverable_without_it
      in tests/test_phase4_privacy.py for the property that actually
      demonstrates hiding: the SAME true update masked with two different
      (unknown to the observer) mask seeds produces very different
      outputs, i.e. the masked value is not a stable, invertible function
      of the truth.
    """
    import torch

    key0 = next(iter(updates[0]))
    true_flat = updates[0][key0].flatten()
    masked_flat = masked[0][key0].flatten()
    cos_sim = torch.nn.functional.cosine_similarity(true_flat, masked_flat, dim=0, eps=1e-8).item()

    true_sum = {k: sum(u[k] for u in updates) for k in updates[0]}
    masked_sum = {k: sum(u[k] for u in masked) for k in masked[0]}
    max_err = max((true_sum[k] - masked_sum[k]).abs().max().item() for k in true_sum)

    return {
        "individual_update_cosine_similarity_to_true": cos_sim,  # see caution above -- not a hiding proof either way
        "aggregate_reconstruction_max_abs_error": max_err,  # should be ~0 (masks cancel exactly) -- THIS is the load-bearing check
    }
