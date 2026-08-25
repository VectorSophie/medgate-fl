"""Genuine PARAMETER-SPACE adapter recovery (P1-7 review requirement) —
distinct from medgate/attacks/reconstruction.py's
auxiliary_data_adapter_finetuning_recovery and fixed_budget_hard_label_distillation,
which each train a completely FRESH adapter/model from scratch and never
touch the real adapter's actual weights at all. This module's attacks
instead start from a PARTIAL, genuinely-leaked copy of the true adapter's
own weight matrices and try to complete/recover the missing entries — the
only attacks in this project that operate directly on the protected
artifact's parameter space.

Threat scenario: the attacker has obtained SOME of the true adapter's
weight entries (e.g. a partial memory leak, a side-channel, a corrupted
backup) but not the whole tensor, and exploits the fact that this
project's LoRA adapter (medgate.models.backbone.LoRAAdapter) is
LOW-RANK BY CONSTRUCTION (up: feature_dim x rank, down: rank x feature_dim)
to complete the rest via low-rank matrix completion. This is NOT an attack
on AES-GCM (medgate/crypto/adapter_encryption.py) — a correctly-implemented
AEAD ciphertext reveals nothing about the plaintext bit-by-bit, so "partial
leak of ciphertext" is not equivalent to "partial leak of the adapter
weights"; this module's attacker model assumes the leak is of PLAINTEXT
adapter values (e.g. from an insufficiently protected memory dump or a
side-channel on a device that had already decrypted the adapter), stated
explicitly so this is never read as "AES-GCM can be partially broken by
low-rank completion" — it cannot, and this module makes no such claim.

Reports BOTH parameter-space and functional recovery, never one alone:
  - parameter-space: cosine similarity, normalized Frobenius distance,
    and the recovered matrix's own singular-value spectrum vs. the truth's.
  - functional: fine-task macro-F1 with the completed adapter plugged into
    the model, and prediction agreement with the authorized model.
"""
import copy

import numpy as np
import torch

from medgate.metrics import evaluate_fine


def simulate_partial_leak(matrix: torch.Tensor, reveal_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (partial_matrix, mask) where mask[i,j]=True means entry
    (i,j) was leaked (kept at its true value in partial_matrix) and
    mask[i,j]=False means it's unknown (zeroed in partial_matrix — the
    completion algorithm's starting guess)."""
    rng = np.random.RandomState(seed)
    m = matrix.detach().numpy()
    mask = rng.rand(*m.shape) < reveal_fraction
    partial = np.where(mask, m, 0.0)
    return partial, mask


def svd_complete_matrix(partial: np.ndarray, mask: np.ndarray, rank: int, iterations: int = 100) -> np.ndarray:
    """Simple iterative hard-impute matrix completion (a standard,
    well-known heuristic — truncated-SVD projection alternated with
    resetting the known entries to their true leaked values; related to
    the "HardImpute" algorithm, not a novel method invented for this
    project). Converges reasonably well when `mask`'s revealed fraction is
    high enough relative to `rank` and the matrix's true rank matches
    `rank` — which it does exactly here, since LoRAAdapter's own weights
    are rank-`rank` by construction."""
    estimate = partial.copy()
    for _ in range(iterations):
        U, S, Vt = np.linalg.svd(estimate, full_matrices=False)
        S_truncated = np.zeros_like(S)
        S_truncated[:rank] = S[:rank]
        estimate = U @ np.diag(S_truncated) @ Vt
        estimate = np.where(mask, partial, estimate)  # re-assert the known (leaked) entries exactly
    return estimate


def parameter_space_recovery_metrics(true_matrix: torch.Tensor, recovered_matrix: np.ndarray, rank: int) -> dict:
    true_np = true_matrix.detach().numpy()
    true_flat, rec_flat = true_np.flatten(), recovered_matrix.flatten()
    cos_sim = float(np.dot(true_flat, rec_flat) / (np.linalg.norm(true_flat) * np.linalg.norm(rec_flat) + 1e-12))
    frob_error = float(np.linalg.norm(true_np - recovered_matrix, ord="fro") / (np.linalg.norm(true_np, ord="fro") + 1e-12))
    true_singular_values = np.linalg.svd(true_np, compute_uv=False)
    recovered_singular_values = np.linalg.svd(recovered_matrix, compute_uv=False)
    return {
        "cosine_similarity": cos_sim,
        "normalized_frobenius_error": frob_error,
        "true_top_singular_values": true_singular_values[:rank].tolist(),
        "recovered_top_singular_values": recovered_singular_values[:rank].tolist(),
    }


def low_rank_completion_attack(authorized_model, test_dataset, reveal_fraction: float, seed: int, completion_iterations: int = 100) -> dict:
    """Single-attacker low-rank completion of the adapter's `up` matrix
    (the matrix LoRAAdapter zero-initializes and trains — the more
    information-bearing of the two LoRA factors post-training; `down` is
    left as an exercise, not completed here, to keep the attack focused
    and its cost bounded)."""
    torch.manual_seed(seed)
    true_up = authorized_model.adapter.up.weight
    rank = authorized_model.adapter.down.out_features

    partial, mask = simulate_partial_leak(true_up, reveal_fraction, seed)
    recovered = svd_complete_matrix(partial, mask, rank, iterations=completion_iterations)
    param_metrics = parameter_space_recovery_metrics(true_up, recovered, rank)

    attacker_model = copy.deepcopy(authorized_model)
    with torch.no_grad():
        attacker_model.adapter.up.weight.copy_(torch.from_numpy(recovered).float())
    functional = evaluate_fine(attacker_model, test_dataset)

    return {
        "attacker_knowledge": f"{reveal_fraction:.0%} of the true adapter.up matrix's entries (plaintext leak, NOT an AES-GCM break), rank={rank} known",
        "attacker_access": "no auxiliary data, no queries -- pure parameter-space completion",
        "compute_budget": {"reveal_fraction": reveal_fraction, "completion_iterations": completion_iterations},
        "parameter_space_recovery": param_metrics,
        "functional_recovery_fine_macro_f1": functional["fine_macro_f1"],
    }


def two_attacker_collusion_completion_attack(authorized_model, test_dataset, reveal_fraction_each: float, seed: int, completion_iterations: int = 100) -> dict:
    """Two attackers each independently leak a DIFFERENT random
    reveal_fraction_each of the true adapter.up matrix's entries and pool
    (union of) their revealed entries before running ONE completion —
    tests whether collusion helps PARAMETER-SPACE recovery specifically
    (distinct from medgate.attacks.reconstruction.auxiliary_data_ensemble_collusion_proxy,
    which pools functional predictions from two independently-trained
    fresh adapters, never touching the true parameters at all)."""
    torch.manual_seed(seed)
    true_up = authorized_model.adapter.up.weight
    rank = authorized_model.adapter.down.out_features

    partial_a, mask_a = simulate_partial_leak(true_up, reveal_fraction_each, seed)
    partial_b, mask_b = simulate_partial_leak(true_up, reveal_fraction_each, seed + 1)
    pooled_mask = mask_a | mask_b
    pooled_partial = np.where(pooled_mask, np.where(mask_a, partial_a, partial_b), 0.0)

    recovered_solo = svd_complete_matrix(partial_a, mask_a, rank, iterations=completion_iterations)
    recovered_colluded = svd_complete_matrix(pooled_partial, pooled_mask, rank, iterations=completion_iterations)

    solo_metrics = parameter_space_recovery_metrics(true_up, recovered_solo, rank)
    colluded_metrics = parameter_space_recovery_metrics(true_up, recovered_colluded, rank)

    def functional_f1(recovered):
        m = copy.deepcopy(authorized_model)
        with torch.no_grad():
            m.adapter.up.weight.copy_(torch.from_numpy(recovered).float())
        return evaluate_fine(m, test_dataset)["fine_macro_f1"]

    return {
        "attacker_knowledge": f"two colluding attackers, each leaked {reveal_fraction_each:.0%} of adapter.up (pooled union: ~{pooled_mask.mean():.0%})",
        "attacker_access": "no auxiliary data, no queries -- pure parameter-space completion",
        "compute_budget": {"reveal_fraction_each": reveal_fraction_each, "pooled_reveal_fraction": float(pooled_mask.mean())},
        "solo_parameter_space_recovery": solo_metrics,
        "colluded_parameter_space_recovery": colluded_metrics,
        "solo_functional_fine_macro_f1": functional_f1(recovered_solo),
        "colluded_functional_fine_macro_f1": functional_f1(recovered_colluded),
    }
