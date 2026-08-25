"""Capability-isolation composite metrics (docs/research_scope.md §7).
Raw utility/attack numbers are always reported alongside these — never in
place of them (project brief: "Never hide failure behind a composite
score."). U(.) here is fine-task macro-F1 (medgate/metrics.py), the same
utility function used throughout Phase 1/2.

PRECISE NAMING (P1-8 review requirement: "Never define Upublic
ambiguously"). This project's metric names, and exactly which function/
field computes each:
  - PublicCoarseUtility: the public path's own coarse-task score --
    medgate.metrics.evaluate_both(...)["coarse_macro_f1"]. Not a leakage
    metric at all; the intended, sanctioned public capability.
  - AuthorizedFineUtility: the authorized path's fine-task score --
    medgate.metrics.evaluate_both(...)["fine_macro_f1"]. Referred to as
    U_authorized in the formulas below.
  - OutputLeak (a.k.a. U_public in this codebase's variable names, e.g.
    medgate/attacks/probes.py output_only_probe / scripts/run_phase2_synthetic.py
    "u_public"): fine-label recovery from the PUBLIC OUTPUT alone (a probe
    fit on coarse logits, zero representation access) -- the floor an
    unauthorized user gets with only the exposed prediction. This is the
    U_public used in unauthorized_capability_gain and
    authorized_recovery_ratio below; kept as the variable name `u_public`
    across the codebase rather than renamed everywhere, but this docstring
    is the authoritative definition to resolve any ambiguity about what it
    means -- exactly the review's concern.
  - RepLeak / BestProbeRFC (used interchangeably in this project; both
    mean residual_fine_capability below): fine-label recovery from the
    public backbone REPRESENTATION z, taking the MAXIMUM across every
    probe actually run (medgate/attacks/probes.py run_all_probes) --
    linear, nonlinear (MLP), k-NN, few-shot, and (P1-8) SVM/tree if
    feasible. "Best" = strongest attacker among those probed, a
    defender-unfavorable (conservative) choice on purpose.
  - UCG, ARR: see unauthorized_capability_gain / authorized_recovery_ratio
    below.
"""
import math


def authorized_recovery_ratio(u_authorized: float, u_public: float, u_plain_adapter: float) -> float | None:
    """ARR = (U_authorized - U_public) / (U_plain_adapter - U_public).
    1.0 = authorized users recover as much fine utility as the plain
    (unprotected) adapter baseline; less than 1.0 = isolation cost utility.
    Undefined (returns None) when the plain-adapter denominator is ~0 —
    e.g. on the synthetic fixture where nothing is learnable and
    U_plain_adapter ~= U_public; reported as null rather than a divide-by-
    near-zero number that would look meaningful but isn't."""
    denom = u_plain_adapter - u_public
    if abs(denom) < 1e-6:
        return None
    return (u_authorized - u_public) / denom


def unauthorized_capability_gain(u_unauthorized: float, u_public: float) -> float:
    """UCG = U_unauthorized - U_public. u_unauthorized is typically the
    best attack/probe result (docs/research_scope.md §7)."""
    return u_unauthorized - u_public


def residual_fine_capability(probe_results: dict) -> float:
    """RFC = U(best probe trained on the frozen public representation).
    `probe_results` maps probe name -> {"macro_f1": ..., ...} (see
    medgate/attacks/probes.py run_all_probes); RFC takes the max, i.e. the
    strongest attacker among the probes actually run — a defender-favorable
    metric would average or take the weakest, which is not what "best
    probe" means here."""
    return max(r["macro_f1"] for r in probe_results.values())


def linear_cka(X, Y) -> float:
    """Linear Centered Kernel Alignment between two representation
    matrices (N samples x D features each, same N) -- a representation-
    similarity diagnostic independent of orthogonality_loss's cosine-based
    measure (P1-9 review requirement: "CKA or another representation-
    similarity diagnostic if feasible"). 0 = maximally dissimilar linear
    structure, 1 = identical up to rotation/isotropic scaling. Unlike
    per-sample cosine similarity (medgate.models.backbone.cosine_similarity_stats),
    CKA compares the two representations' overall SIMILARITY STRUCTURE
    across the whole batch (are pairs of samples that are close in X also
    close in Y), not each sample's individual pairing -- a different, also
    only-geometric (not information-theoretic) notion of alignment,
    subject to the same "orthogonal != independent" caveat as
    orthogonality_loss's docstring."""
    import numpy as np

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2
    denom = np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro")
    if denom < 1e-12:
        return float("nan")
    return float(numerator / denom)


def capability_recovery_efficiency(u_attack: float, u_public: float, compute_cost: float) -> float:
    """CRE = (U_attack - U_public) / log(1 + compute_cost). compute_cost
    must be a positive, comparable-across-attacks cost proxy (e.g. probe
    fit_seconds, or query count for a black-box attack) — comparing CRE
    across attacks that used different cost units is meaningless and this
    function does not attempt to normalize that; the caller must."""
    return (u_attack - u_public) / math.log(1.0 + compute_cost)
