"""Code identifier -> human-readable publication label, shared by
scripts/csv_to_latex_tables.py (tables) and scripts/make_figures.py
(figure axis/legend labels) so the two never drift apart (P2-13 repair,
docs/execution_plan.md). Extend here, not by hand-editing a generated
.tex file or figure, whenever a new method/attack/arm is added.
"""

DISPLAY_NAMES = {
    "centralized": "Centralized",
    "local_only": "Local only",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "random_frozen_lora": "Random-frozen LoRA (negative control)",
    "coarse_pretrained_fedlora": "Coarse-pretrained FedLoRA",
    "imagenet_pretrained_fedlora": "ImageNet-pretrained FedLoRA",
    "full_finetune": "Full fine-tune (upper bound)",
    "coarse_only": "Coarse-only",
    "hidden_fine_head": "Hidden fine head",
    "adapter_isolation": "Adapter isolation",
    "adversarial": "Adversarial suppression",
    "orthogonal": "Orthogonality penalty",
    "combined": "Combined (adversarial + orthogonal)",
    "baseline": "Baseline",
    "capability_isolation": "Capability isolation",
    "simplified_known_label_gradient_inversion (single unaggregated gradient)": "Simplified known-label gradient inversion",
    "membership_inference (loss-threshold)": "Loss-threshold membership inference",
    "auxiliary_data_adapter_finetuning_recovery (A2)": "Auxiliary-data adapter fine-tuning recovery (A2)",
    "fixed_budget_hard_label_distillation (A3)": "Fixed-budget hard-label distillation (A3)",
    "auxiliary_data_ensemble_collusion_proxy (2 attackers vs solo, same total budget)": "Auxiliary-data ensemble collusion proxy",
    "none": "No attack (honest)",
    "label_flip": "Label flipping",
    "backdoor": "Backdoor insertion",
    "sign_flip": "Sign flipping",
    "model_replacement": "Model replacement",
    "free_rider": "Free rider",
    "malformed": "Malformed update",
    "coordinate_median": "Coordinate median",
    "trimmed_mean": "Trimmed mean",
    "validated_fedavg": "Validated FedAvg",
    "no_protection": "No protection",
    # P0-B (repair pass 4): never "secure aggregation" unqualified -- this
    # project's masking is a single-process, Gaussian-mask simulation with
    # no information-theoretic concealment guarantee at its default scale
    # (see medgate/privacy/secure_aggregation.py's module docstring and
    # paper/tables/phase4_concealment_sweep.tex).
    "secure_agg": "Simulated pairwise additive masking",
    "dp_sgd": "DP-SGD",
    "secure_agg_plus_dp": "Simulated masking + DP-SGD",
    "institution": "Institution-level",
    "class": "Class-level",
    "full_retrain": "Full retrain (gold standard)",
    "checkpoint_rollback": "Checkpoint rollback",
    "adapter_deletion_and_retrain": "Adapter deletion + retrain",
    "gradient_ascent_unlearning": "Gradient-ascent unlearning",
    "key_revocation_only": "Key revocation only",
    "solo": "Single attacker",
    "pooled": "Colluders (pooled)",
    "collusion_solo_half": "Colluder (pre-pooling)",
    "collusion_pooled": "Colluders (pooled)",
    "null_signal": "Null-signal fixture",
    "hierarchical": "Hierarchical fixture",
    "zero_fill": "Zero-fill (no completion)",
    "mean_fill": "Mean-fill",
    "random_fill": "Random-fill",
    "hard_impute": "Hard-impute (oracle/candidate rank)",
    "soft_impute": "Soft-impute",
    "logistic": "Logistic regression",
    "mlp": "MLP (nonlinear)",
    "primary": "Primary ontology",
    "alternative": "Alternative ontology",
    "patient": "Patient-level",
    "patient_group": "Patient-group-level",
}


def disp(code: str) -> str:
    return DISPLAY_NAMES.get(code, code.replace("_", " ").strip().title())
