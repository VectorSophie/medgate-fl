"""Checks for the two P0 fixes from the repair-pass-4 handoff review:

P0-1: git_commit() must refuse to return a commit SHA when the working
tree is dirty (the failure mode was code edited+run, committed later, so
the recorded SHA didn't contain the code that actually ran).

P0-2: scripts/run_phase1_hierarchical.py's build_fixture() must use the
config's fixed data_seed for dataset generation/splitting, not the
per-run model/training seed -- the whole point of data_seed is that the
dataset stays identical while only training stochasticity varies across
the seed sweep.

Run: PYTHONPATH=. pytest tests/test_provenance.py -v
"""
from unittest.mock import patch

import pytest
import torch

from scripts.run_phase1_hierarchical import build_fixture
from scripts.run_phase1_synthetic import git_commit

CFG = {
    "data": {
        "image_size": 16,
        "num_patients_per_institution": 4,
        "observations_per_patient": 2,
        "class_imbalance_strength": 0.5,
        "sensitive_property_correlation": 0.6,
        "data_seed": 0,
        "train_frac": 0.6,
        "val_frac": 0.2,
    }
}


def _fake_run(dirty_output):
    def run(args, **kwargs):
        class R:
            stdout = "deadbeef" * 5 + "\n" if args[1] == "rev-parse" else dirty_output
        return R()
    return run


def test_git_commit_raises_on_dirty_tree_by_default():
    with patch("scripts.run_phase1_synthetic.subprocess.run", _fake_run(" M scripts/foo.py\n")):
        with pytest.raises(RuntimeError, match="uncommitted changes"):
            git_commit()


def test_git_commit_allows_dirty_tree_with_env_override(monkeypatch):
    monkeypatch.setenv("ALLOW_DIRTY_RUN", "1")
    with patch("scripts.run_phase1_synthetic.subprocess.run", _fake_run(" M scripts/foo.py\n")):
        assert git_commit().endswith("-dirty")


def test_git_commit_clean_tree_returns_plain_sha():
    with patch("scripts.run_phase1_synthetic.subprocess.run", _fake_run("")):
        commit = git_commit()
    assert commit == "deadbeef" * 5
    assert not commit.endswith("-dirty")


def test_build_fixture_dataset_is_independent_of_run_seed():
    """Same data_seed, different run seed -> byte-identical dataset/split."""
    fixture_a = build_fixture(CFG, seed=0)
    fixture_b = build_fixture(CFG, seed=1)
    train_pool_a, val_pool_a, util_a, attack_a = fixture_a[4:8]
    train_pool_b, val_pool_b, util_b, attack_b = fixture_b[4:8]
    for pool_a, pool_b in [(train_pool_a, train_pool_b), (val_pool_a, val_pool_b),
                            (util_a, util_b), (attack_a, attack_b)]:
        assert len(pool_a) == len(pool_b)
        for i in range(len(pool_a)):
            img_a, fine_a, coarse_a = pool_a[i]
            img_b, fine_b, coarse_b = pool_b[i]
            assert torch.equal(img_a, img_b)
            assert fine_a == fine_b and coarse_a == coarse_b


def test_build_fixture_dataset_changes_with_data_seed():
    fixture_a = build_fixture(CFG, seed=0)
    cfg_b = {"data": {**CFG["data"], "data_seed": 1}}
    fixture_b = build_fixture(cfg_b, seed=0)
    img_a = fixture_a[4][0][0]
    img_b = fixture_b[4][0][0]
    assert not torch.equal(img_a, img_b)
