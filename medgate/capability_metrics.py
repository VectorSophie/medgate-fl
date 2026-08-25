"""Capability-isolation composite metrics (docs/research_scope.md §7).
Raw utility/attack numbers are always reported alongside these — never in
place of them (project brief: "Never hide failure behind a composite
score."). U(.) here is fine-task macro-F1 (medgate/metrics.py), the same
utility function used throughout Phase 1/2.
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


def capability_recovery_efficiency(u_attack: float, u_public: float, compute_cost: float) -> float:
    """CRE = (U_attack - U_public) / log(1 + compute_cost). compute_cost
    must be a positive, comparable-across-attacks cost proxy (e.g. probe
    fit_seconds, or query count for a black-box attack) — comparing CRE
    across attacks that used different cost units is meaningless and this
    function does not attempt to normalize that; the caller must."""
    return (u_attack - u_public) / math.log(1.0 + compute_cost)
