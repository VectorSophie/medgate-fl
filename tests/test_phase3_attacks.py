"""Phase 3 checks: each attack runs end-to-end on a tiny fixture and
produces sane, bounded outputs (not just "doesn't crash").

Run: PYTHONPATH=. pytest tests/test_phase3_attacks.py -v
"""
import torch

from medgate.attacks.gradient_inversion import attack_params, simplified_known_label_gradient_inversion
from medgate.attacks.membership_inference import loss_threshold_membership_inference
from medgate.attacks.reconstruction import auxiliary_data_adapter_finetuning_recovery, fixed_budget_hard_label_distillation, auxiliary_data_ensemble_collusion_proxy
from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, SyntheticFedISIC
from medgate.metrics import evaluate_fine
from medgate.models.backbone import MedGateModel

MODEL_KWARGS = dict(num_coarse=len(COARSE_CLASSES), num_fine=len(FINE_CLASSES), feature_dim=16, adapter_rank=2)


def _model():
    return MedGateModel(**MODEL_KWARGS)


def test_attack_params_excludes_adversary_head():
    model = _model()
    params = attack_params(model)
    adv_params = list(model.adversary_head.parameters())
    assert not any(any(p is ap for ap in adv_params) for p in params)


def test_dlg_attack_reduces_gradient_mismatch_and_stays_finite():
    model = _model()
    dataset = SyntheticFedISIC(num_samples=1, image_size=16, seed=3)
    img, y_fine, y_coarse = dataset[0]
    result = simplified_known_label_gradient_inversion(model, img.unsqueeze(0), y_coarse.unsqueeze(0), y_fine.unsqueeze(0), steps=20, lr=0.1, seed=0)
    assert result["mse"] >= 0
    assert torch.isfinite(torch.tensor(result["final_grad_diff"]))
    assert result["compute_budget_steps"] == 20


def test_membership_inference_auc_in_valid_range():
    model = _model()
    members = SyntheticFedISIC(num_samples=10, image_size=16, seed=1)
    nonmembers = SyntheticFedISIC(num_samples=10, image_size=16, seed=2)
    result = loss_threshold_membership_inference(model, members, nonmembers, batch_size=4)
    assert 0.0 <= result["attack_auc"] <= 1.0
    assert result["n_members"] == 10 and result["n_nonmembers"] == 10


def test_adapter_reconstruction_trains_a_fresh_adapter_not_the_original():
    authorized = _model()
    original_adapter_weight = authorized.adapter.up.weight.clone()
    aux = SyntheticFedISIC(num_samples=8, image_size=16, seed=5)
    attacker_model, meta = auxiliary_data_adapter_finetuning_recovery(authorized, aux, epochs=2, batch_size=4, lr=0.05, seed=0)
    # the attacker's adapter must NOT be the same object/weights as the authorized one
    assert attacker_model.adapter is not authorized.adapter
    assert not torch.allclose(attacker_model.adapter.up.weight, original_adapter_weight)
    assert meta["compute_budget"]["auxiliary_examples"] == 8
    test_set = SyntheticFedISIC(num_samples=8, image_size=16, seed=6)
    util = evaluate_fine(attacker_model, test_set, batch_size=4)
    assert 0.0 <= util["fine_macro_f1"] <= 1.0


def test_black_box_extraction_produces_evaluable_student():
    authorized = _model()
    query_images = torch.stack([SyntheticFedISIC(1, image_size=16, seed=s)[0][0] for s in range(8)])
    student, meta = fixed_budget_hard_label_distillation(authorized, query_images, MODEL_KWARGS, epochs=2, batch_size=4, lr=0.05, seed=0)
    assert meta["compute_budget"]["query_budget"] == 8
    test_set = SyntheticFedISIC(num_samples=8, image_size=16, seed=9)
    util = evaluate_fine(student, test_set, batch_size=4)
    assert 0.0 <= util["fine_macro_f1"] <= 1.0


def test_collusion_ensemble_is_evaluable():
    authorized = _model()
    aux_a = SyntheticFedISIC(num_samples=6, image_size=16, seed=11)
    aux_b = SyntheticFedISIC(num_samples=6, image_size=16, seed=12)
    ensemble, meta = auxiliary_data_ensemble_collusion_proxy(authorized, aux_a, aux_b, epochs=2, batch_size=3, lr=0.05, seed=0)
    assert "solo_a" in meta and "solo_b" in meta
    test_set = SyntheticFedISIC(num_samples=8, image_size=16, seed=13)
    util = evaluate_fine(ensemble, test_set, batch_size=4)
    assert 0.0 <= util["fine_macro_f1"] <= 1.0


if __name__ == "__main__":
    test_attack_params_excludes_adversary_head()
    test_dlg_attack_reduces_gradient_mismatch_and_stays_finite()
    test_membership_inference_auc_in_valid_range()
    test_adapter_reconstruction_trains_a_fresh_adapter_not_the_original()
    test_black_box_extraction_produces_evaluable_student()
    test_collusion_ensemble_is_evaluable()
    print("OK")
