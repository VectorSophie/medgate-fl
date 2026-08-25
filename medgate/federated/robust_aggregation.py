"""Robust aggregation alternatives to plain FedAvg (medgate/federated/
fedavg.py's fedavg_aggregate), plus a validation gate for malformed
updates. Exists specifically to probe the tension the project brief names:
secure aggregation (medgate/privacy/secure_aggregation.py) HIDES individual
updates from the server, which is exactly what a robust/validating
aggregator needs to SEE in order to detect and down-weight a poisoned one.
Phase 3's integrity-attack results (scripts/run_phase3_integrity_synthetic.py)
compare plaintext FedAvg, coordinate-wise median, and trimmed mean against
the attacks in medgate/attacks/integrity.py -- this module does not claim
robust aggregation is compatible with secure aggregation; it demonstrates
why it structurally is not (a median/trimmed-mean needs each client's raw
value per coordinate, which secure aggregation never reveals).
"""
import math

import torch


def validate_update(state_dict: dict) -> bool:
    """Reject a malformed update before it ever reaches aggregation: NaN,
    Inf, or empty tensors. Does NOT detect a well-formed-but-poisoned
    update (label flipping, sign flipping, model replacement all produce
    perfectly well-formed tensors) -- that is exactly the gap robust
    aggregation below is for."""
    for v in state_dict.values():
        if v.numel() == 0:
            return False
        if torch.isnan(v).any() or torch.isinf(v).any():
            return False
    return True


def validated_fedavg_aggregate(state_dicts: list, weights: list) -> dict:
    """Plain FedAvg, but malformed updates are dropped (not averaged in)
    before weighting the rest. Raises if every update was malformed."""
    from medgate.federated.fedavg import fedavg_aggregate

    kept = [(sd, w) for sd, w in zip(state_dicts, weights) if validate_update(sd)]
    if not kept:
        raise ValueError("validated_fedavg_aggregate: every update failed validation")
    kept_states, kept_weights = zip(*kept)
    return fedavg_aggregate(list(kept_states), list(kept_weights))


def coordinate_median_aggregate(state_dicts: list) -> dict:
    """Coordinate-wise median across clients (ignores client weighting by
    design -- the median's robustness comes from ignoring exactly the kind
    of large-magnitude single-client dominance that weighting would allow;
    combining it with FedAvg-style weighting would reopen the same
    vulnerability it's meant to close)."""
    keys = state_dicts[0].keys()
    stacked = {k: torch.stack([sd[k].float() for sd in state_dicts], dim=0) for k in keys}
    return {k: v.median(dim=0).values.to(state_dicts[0][k].dtype) for k, v in stacked.items()}


def trimmed_mean_aggregate(state_dicts: list, trim_fraction: float = 0.2) -> dict:
    """Coordinate-wise trimmed mean: drop the top and bottom
    trim_fraction of values per coordinate, average the rest. Needs at
    least ceil(1/trim_fraction) clients to trim anything; falls back to
    the plain mean if too few clients are given for the requested
    trim_fraction (documented here rather than silently trimming zero)."""
    n = len(state_dicts)
    trim_count = math.floor(n * trim_fraction)
    keys = state_dicts[0].keys()
    stacked = {k: torch.stack([sd[k].float() for sd in state_dicts], dim=0) for k in keys}
    out = {}
    for k, v in stacked.items():
        sorted_v, _ = v.sort(dim=0)
        if trim_count > 0 and n - 2 * trim_count > 0:
            sorted_v = sorted_v[trim_count: n - trim_count]
        out[k] = sorted_v.mean(dim=0).to(state_dicts[0][k].dtype)
    return out
