"""Genuine PARAMETER-SPACE adapter recovery (P1-7, then substantially
REWRITTEN in repair pass 4 / P0-A after a validity bug was found in the
original version) — distinct from medgate/attacks/reconstruction.py's
auxiliary_data_adapter_finetuning_recovery and fixed_budget_hard_label_distillation,
which each train a completely FRESH adapter/model from scratch and never
touch the real adapter's actual weights at all. This module's attacks
instead start from a PARTIAL, genuinely-leaked copy of the true adapter's
own effective transformation and try to complete/recover the missing
entries — the only attacks in this project that operate directly on the
protected artifact's parameter space.

REPAIR PASS 4 / P0-A — validity bug and fix, kept here rather than only in
a commit message so nobody re-introduces it:

  The pass-2/3 version of this module completed `adapter.up.weight`
  directly. That matrix has shape (feature_dim, rank) — e.g. (64, 4) — so
  its OWN maximum possible rank is min(feature_dim, rank) == rank. Running
  truncated-SVD at rank=rank on a matrix that already has at most `rank`
  singular values keeps every one of them: the "completion" step was
  mathematically the identity map on whatever was already there. Since
  unobserved entries were zero-filled before that no-op projection, the
  reported "completion" was doing nothing but re-asserting the revealed
  entries every iteration — the improvement with reveal_fraction measured
  in pass 2/3 reflected nothing but more entries being directly known, not
  any inference about the unobserved ones. See
  tests/test_adapter_recovery.py::test_rank_equal_to_ambient_dim_makes_hard_impute_a_no_op
  for a regression test that reproduces and locks in an explanation of
  this exact failure mode, and
  test_completion_beats_zero_fill_on_unobserved_entries for the check that
  would have caught it.

  FIX: the object completed here is now the adapter's EFFECTIVE update
  matrix

      delta_w = adapter.up.weight @ adapter.down.weight

  which is (feature_dim, feature_dim) — e.g. (64, 64) — i.e. full AMBIENT
  rank up to feature_dim, but is rank <= adapter_rank BY CONSTRUCTION,
  since it is the product of a (feature_dim x rank) and a (rank x
  feature_dim) factor. Truncating delta_w's SVD at rank=adapter_rank now
  genuinely discards information (there are up to feature_dim singular
  values and only the top adapter_rank survive), so hard/soft-impute on
  delta_w is real matrix completion, and reveal_fraction / candidate rank
  / algorithm choice can now actually be compared.

Threat scenario (unchanged from pass 2): the attacker has obtained SOME
entries of the true delta_w (e.g. reconstructed by probing input/output
pairs through the adapter, a partial memory leak, or a corrupted backup)
but not the whole matrix, and exploits delta_w's low-rank structure to
complete the rest. This is NOT an attack on AES-GCM
(medgate/crypto/adapter_encryption.py) — a correctly-implemented AEAD
ciphertext reveals nothing about the plaintext bit-by-bit, so "partial leak
of ciphertext" is not equivalent to "partial leak of the adapter's
effective transformation"; this module's attacker model assumes the leak
is of the PLAINTEXT effective transformation, stated explicitly so this is
never read as "AES-GCM can be partially broken by low-rank completion" —
it cannot, and this module makes no such claim.

Reports, for every method, BOTH parameter-space and functional recovery,
and separates observed-entry error (trivially ~0 for any method that
re-asserts known entries, uninformative) from unobserved/held-out-entry
error (the actual test of whether completion inferred anything), plus
completion gain over a zero-fill control -- never cosine-to-ground-truth
alone.
"""
import copy

import numpy as np
import torch

from medgate.metrics import evaluate_fine


def effective_delta_w(model) -> torch.Tensor:
    """The adapter's effective (feature_dim x feature_dim) update matrix:
    adapter(z) = z @ down.weight.T @ up.weight.T = z @ delta_w.T, so
    delta_w = up.weight @ down.weight fully determines the adapter's
    linear effect on the representation. Full ambient rank feature_dim,
    but rank <= adapter_rank by construction (product of a (feature_dim x
    rank) and a (rank x feature_dim) factor) -- this is the object
    completed by every function below, per the P0-A fix in this module's
    docstring."""
    up = model.adapter.up.weight.detach()
    down = model.adapter.down.weight.detach()
    return up @ down


def simulate_partial_leak(matrix: torch.Tensor, reveal_fraction: float, seed: int) -> tuple:
    """Returns (partial_matrix, mask) where mask[i,j]=True means entry
    (i,j) was leaked (kept at its true value in partial_matrix) and
    mask[i,j]=False means it's unknown (zeroed in partial_matrix — every
    completion method's starting point)."""
    rng = np.random.RandomState(seed)
    m = matrix.detach().numpy() if isinstance(matrix, torch.Tensor) else matrix
    mask = rng.rand(*m.shape) < reveal_fraction
    partial = np.where(mask, m, 0.0)
    return partial, mask


# --- Completion / fill methods -----------------------------------------
# All take (partial, mask, ...) and return a completed matrix of the same
# shape. zero_fill/mean_fill/random_fill are the required "did the
# low-rank structure help at all, or would filling in a constant/guess do
# just as well" controls; hard_impute and soft_impute are the two low-rank
# completion algorithms compared; oracle_rank_hard_impute is hard_impute
# called with the TRUE rank (the best case a real attacker, who must guess
# the rank, could not assume) for the rank-misspecification sweep.

def zero_fill(partial: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The no-completion control: unobserved entries stay at 0. Any real
    completion method must beat this to be worth calling 'completion'."""
    return partial.copy()


def mean_fill(partial: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Unobserved entries filled with the mean of the REVEALED entries —
    a completion method that ignores the low-rank structure entirely."""
    out = partial.copy()
    fill_value = partial[mask].mean() if mask.any() else 0.0
    out[~mask] = fill_value
    return out


def random_fill(partial: np.ndarray, mask: np.ndarray, seed: int) -> np.ndarray:
    """Unobserved entries filled with i.i.d. draws matching the revealed
    entries' empirical mean/std -- a completion method that uses the
    revealed entries' marginal distribution but not their low-rank
    structure or positions."""
    rng = np.random.RandomState(seed)
    out = partial.copy()
    if mask.any() and mask.sum() > 1:
        mu, sigma = partial[mask].mean(), partial[mask].std()
    else:
        mu, sigma = 0.0, 1.0
    out[~mask] = rng.normal(mu, max(sigma, 1e-8), size=(~mask).sum())
    return out


def hard_impute_svd(partial: np.ndarray, mask: np.ndarray, rank: int, iterations: int = 100) -> np.ndarray:
    """Iterative hard-impute matrix completion (truncated-SVD projection
    alternated with re-asserting the known entries to their true leaked
    values; the standard "HardImpute" heuristic, not novel to this
    project). Meaningfully constrains the estimate ONLY when
    rank < min(partial.shape) -- see this module's docstring for the P0-A
    bug this fixes, and
    test_rank_equal_to_ambient_dim_makes_hard_impute_a_no_op for a
    regression test of the failure mode when that condition is violated."""
    estimate = partial.copy()
    for _ in range(iterations):
        U, S, Vt = np.linalg.svd(estimate, full_matrices=False)
        S_truncated = np.zeros_like(S)
        S_truncated[:rank] = S[:rank]
        estimate = U @ np.diag(S_truncated) @ Vt
        estimate = np.where(mask, partial, estimate)
    return estimate


def soft_impute_svd(partial: np.ndarray, mask: np.ndarray, lam: float, iterations: int = 100) -> np.ndarray:
    """SoftImpute (Mazumder, Hastie & Tibshirani 2010): same
    alternating-projection scheme as hard_impute_svd, but SOFT-thresholds
    singular values (max(sigma - lam, 0)) instead of hard-truncating to
    the top `rank`. Does not require knowing the true rank in advance —
    the rank of the result emerges from which singular values exceed
    `lam` — which is the point of including it as a second completion
    algorithm: hard_impute_svd needs the (possibly misspecified) rank as
    an input, soft_impute_svd needs a threshold instead, and the two can
    disagree under rank misspecification in informative ways."""
    estimate = partial.copy()
    for _ in range(iterations):
        U, S, Vt = np.linalg.svd(estimate, full_matrices=False)
        S_soft = np.maximum(S - lam, 0.0)
        estimate = U @ np.diag(S_soft) @ Vt
        estimate = np.where(mask, partial, estimate)
    return estimate


FILL_METHODS = ("zero_fill", "mean_fill", "random_fill", "hard_impute", "soft_impute")


def run_fill_method(method: str, partial: np.ndarray, mask: np.ndarray, *, rank: int, lam: float, iterations: int, seed: int) -> np.ndarray:
    if method == "zero_fill":
        return zero_fill(partial, mask)
    if method == "mean_fill":
        return mean_fill(partial, mask)
    if method == "random_fill":
        return random_fill(partial, mask, seed)
    if method == "hard_impute":
        return hard_impute_svd(partial, mask, rank, iterations)
    if method == "soft_impute":
        return soft_impute_svd(partial, mask, lam, iterations)
    raise ValueError(f"unknown fill method {method!r}; expected one of {FILL_METHODS}")


def recovery_metrics(true_matrix, recovered_matrix: np.ndarray, mask: np.ndarray, rank: int) -> dict:
    """Reports observed- and unobserved-entry error SEPARATELY (P0-A
    requirement #2) -- observed error is trivially ~0 for every method
    above except zero/mean/random_fill's own definition on the observed
    set (which is also exact, since all fill methods copy `partial` on
    observed positions), so the unobserved-entry error is the only number
    here that actually measures completion quality. cosine_similarity and
    normalized_frobenius_error are reported over the WHOLE matrix for
    continuity with earlier drafts, but should be read alongside, not
    instead of, the unobserved-only numbers."""
    true_np = true_matrix.detach().numpy() if isinstance(true_matrix, torch.Tensor) else true_matrix
    true_flat, rec_flat = true_np.flatten(), recovered_matrix.flatten()
    cos_sim = float(np.dot(true_flat, rec_flat) / (np.linalg.norm(true_flat) * np.linalg.norm(rec_flat) + 1e-12))
    frob_error = float(np.linalg.norm(true_np - recovered_matrix, ord="fro") / (np.linalg.norm(true_np, ord="fro") + 1e-12))

    obs_diff = (true_np - recovered_matrix)[mask]
    unobs_mask = ~mask
    unobs_diff = (true_np - recovered_matrix)[unobs_mask]
    unobs_scale = np.linalg.norm(true_np[unobs_mask]) + 1e-12

    observed_entry_rmse = float(np.sqrt((obs_diff ** 2).mean())) if obs_diff.size else 0.0
    unobserved_entry_rmse = float(np.sqrt((unobs_diff ** 2).mean())) if unobs_diff.size else 0.0
    unobserved_normalized_error = float(np.linalg.norm(unobs_diff) / unobs_scale) if unobs_diff.size else 0.0

    true_sv = np.linalg.svd(true_np, compute_uv=False)
    rec_sv = np.linalg.svd(recovered_matrix, compute_uv=False)
    return {
        "cosine_similarity": cos_sim,
        "normalized_frobenius_error": frob_error,
        "observed_entry_rmse": observed_entry_rmse,
        "unobserved_entry_rmse": unobserved_entry_rmse,
        "unobserved_entry_normalized_error": unobserved_normalized_error,
        "true_top_singular_values": true_sv[:rank].tolist(),
        "recovered_top_singular_values": rec_sv[:rank].tolist(),
    }


def completion_gain_over_zero_fill(method_metrics: dict, zero_fill_metrics: dict) -> dict:
    """P0-A requirement #5: report the gain a completion method has over
    the zero-fill control on the metric that actually matters
    (unobserved-entry error), not cosine-to-ground-truth alone. Positive
    == method beats zero-fill (lower unobserved error); non-positive ==
    the low-rank structure bought nothing over "assume unknown entries are
    0" at this reveal fraction/rank."""
    zf = zero_fill_metrics["unobserved_entry_normalized_error"]
    m = method_metrics["unobserved_entry_normalized_error"]
    return {
        "zero_fill_unobserved_normalized_error": zf,
        "method_unobserved_normalized_error": m,
        "absolute_gain": zf - m,
        "relative_gain": (zf - m) / zf if zf > 1e-12 else float("nan"),
    }


def low_rank_completion_attack(
    model, test_dataset, reveal_fraction: float, seed: int, *,
    method: str = "hard_impute", candidate_rank: int | None = None,
    soft_lam: float | None = None, completion_iterations: int = 100,
) -> dict:
    """Single-attacker completion of the adapter's EFFECTIVE update matrix
    delta_w = up @ down (P0-A fix — see module docstring; NOT `up` alone).
    `candidate_rank` lets the caller test rank misspecification: None (the
    default) uses the TRUE rank (the "oracle rank" control); an explicit
    value below or above the true rank tests what happens when the
    attacker guesses wrong. `soft_lam` is required when method=
    'soft_impute' (a threshold, not a rank)."""
    torch.manual_seed(seed)
    true_delta_w = effective_delta_w(model)
    true_rank = model.adapter.down.out_features  # adapter_rank -- the actual constructive rank
    rank_for_method = candidate_rank if candidate_rank is not None else true_rank

    partial, mask = simulate_partial_leak(true_delta_w, reveal_fraction, seed)
    lam = soft_lam if soft_lam is not None else 0.0
    recovered = run_fill_method(method, partial, mask, rank=rank_for_method, lam=lam, iterations=completion_iterations, seed=seed)
    zero_recovered = zero_fill(partial, mask)

    metrics = recovery_metrics(true_delta_w, recovered, mask, true_rank)
    zero_metrics = recovery_metrics(true_delta_w, zero_recovered, mask, true_rank)
    gain = completion_gain_over_zero_fill(metrics, zero_metrics)

    attacker_model = copy.deepcopy(model)
    with torch.no_grad():
        # Plug the recovered delta_w back in via a rank-`true_rank` SVD
        # factorization into up/down so the model's forward pass (which
        # calls up(down(z)), not delta_w directly) uses the attacker's
        # estimate, not the true weights.
        U, S, Vt = np.linalg.svd(recovered, full_matrices=False)
        r = attacker_model.adapter.down.out_features
        up_est = U[:, :r] * np.sqrt(np.maximum(S[:r], 0.0))
        down_est = (Vt[:r, :].T * np.sqrt(np.maximum(S[:r], 0.0))).T
        attacker_model.adapter.up.weight.copy_(torch.from_numpy(up_est).float())
        attacker_model.adapter.down.weight.copy_(torch.from_numpy(down_est).float())
    functional = evaluate_fine(attacker_model, test_dataset)

    return {
        "method": method,
        "attacker_knowledge": f"{reveal_fraction:.0%} of the true adapter's effective delta_w=up@down matrix's entries "
                               f"(plaintext leak, NOT an AES-GCM break), candidate_rank={rank_for_method} "
                               f"(true_rank={true_rank}, {'oracle' if rank_for_method == true_rank else 'misspecified'})",
        "attacker_access": "no auxiliary data, no queries -- pure parameter-space completion",
        "compute_budget": {"reveal_fraction": reveal_fraction, "completion_iterations": completion_iterations, "candidate_rank": rank_for_method},
        "true_rank": true_rank,
        "parameter_space_recovery": metrics,
        "completion_gain_over_zero_fill": gain,
        "functional_recovery_fine_macro_f1": functional["fine_macro_f1"],
    }


def two_attacker_collusion_completion_attack(
    model, test_dataset, reveal_fraction_each: float, seed: int, *,
    method: str = "hard_impute", completion_iterations: int = 100,
) -> dict:
    """Two attackers each independently leak a DIFFERENT random
    reveal_fraction_each of the true delta_w's entries and pool (union of)
    their revealed entries before running ONE completion — tests whether
    collusion helps PARAMETER-SPACE recovery specifically (distinct from
    medgate.attacks.reconstruction.auxiliary_data_ensemble_collusion_proxy,
    which pools functional predictions from two independently-trained
    fresh adapters, never touching the true parameters at all)."""
    torch.manual_seed(seed)
    true_delta_w = effective_delta_w(model)
    true_rank = model.adapter.down.out_features

    partial_a, mask_a = simulate_partial_leak(true_delta_w, reveal_fraction_each, seed)
    partial_b, mask_b = simulate_partial_leak(true_delta_w, reveal_fraction_each, seed + 1)
    pooled_mask = mask_a | mask_b
    pooled_partial = np.where(pooled_mask, np.where(mask_a, partial_a, partial_b), 0.0)

    recovered_solo = run_fill_method(method, partial_a, mask_a, rank=true_rank, lam=0.0, iterations=completion_iterations, seed=seed)
    recovered_colluded = run_fill_method(method, pooled_partial, pooled_mask, rank=true_rank, lam=0.0, iterations=completion_iterations, seed=seed)

    solo_metrics = recovery_metrics(true_delta_w, recovered_solo, mask_a, true_rank)
    colluded_metrics = recovery_metrics(true_delta_w, recovered_colluded, pooled_mask, true_rank)

    def functional_f1(recovered):
        m = copy.deepcopy(model)
        with torch.no_grad():
            U, S, Vt = np.linalg.svd(recovered, full_matrices=False)
            r = m.adapter.down.out_features
            up_est = U[:, :r] * np.sqrt(np.maximum(S[:r], 0.0))
            down_est = (Vt[:r, :].T * np.sqrt(np.maximum(S[:r], 0.0))).T
            m.adapter.up.weight.copy_(torch.from_numpy(up_est).float())
            m.adapter.down.weight.copy_(torch.from_numpy(down_est).float())
        return evaluate_fine(m, test_dataset)["fine_macro_f1"]

    return {
        "method": method,
        "attacker_knowledge": f"two colluding attackers, each leaked {reveal_fraction_each:.0%} of delta_w "
                               f"(pooled union: ~{pooled_mask.mean():.0%})",
        "attacker_access": "no auxiliary data, no queries -- pure parameter-space completion",
        "compute_budget": {"reveal_fraction_each": reveal_fraction_each, "pooled_reveal_fraction": float(pooled_mask.mean())},
        "solo_parameter_space_recovery": solo_metrics,
        "colluded_parameter_space_recovery": colluded_metrics,
        "solo_functional_fine_macro_f1": functional_f1(recovered_solo),
        "colluded_functional_fine_macro_f1": functional_f1(recovered_colluded),
    }
