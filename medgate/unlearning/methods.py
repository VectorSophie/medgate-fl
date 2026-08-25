"""Federated unlearning / revocation methods, compared against full
retraining as the gold standard (project brief: "Use full retraining
without removed data as the gold standard").

Reproducing a specific cited unlearning paper (Wu et al., Halimi et al.,
Zhang et al., Deng et al.) is deliberately NOT attempted here — every one
of those citations is still UNVERIFIED in docs/literature_matrix.csv, and
the project rule is not to build a method's design around, or describe it
as reproducing, an unverified source. `gradient_ascent_unlearning` below is
a generic, well-known simple technique (ascend on the loss for the data
being removed, then a short recovery descent on retained data), described
as exactly that and nothing more specific.
"""
import copy

import torch
import torch.nn as nn

from medgate.federated.fedavg import local_train, run_fedavg_round
from medgate.models.backbone import MedGateModel


def full_retrain(client_datasets_after_removal: list, model_kwargs: dict, rounds: int, epochs: int, batch_size: int, lr: float, seed: int):
    """Gold standard: train from scratch on data with the target subset
    already removed. Everything else in this module is judged against
    this, not the other way around."""
    torch.manual_seed(seed)
    model = MedGateModel(**model_kwargs)
    for _ in range(rounds):
        state = run_fedavg_round(model, client_datasets_after_removal, epochs, batch_size, lr)
        model.load_state_dict(state)
    return model


def checkpoint_rollback(model_kwargs: dict, seed: int):
    """Naive maximal rollback: the round-0 (pre-training) initialization,
    reconstructed by reinstantiating with the SAME seed the authorized
    model used — i.e. 'go back to before any of the training data,
    removed or not, was ever incorporated.' Cheap, but throws away every
    round's progress; the project brief frames this as a real,
    literature-noted limitation of rollback-based 'unlearning', which
    Phase 5's results table should confirm empirically, not assume."""
    torch.manual_seed(seed)
    return MedGateModel(**model_kwargs)


def adapter_deletion_and_retrain(authorized_model, client_datasets_after_removal: list, epochs: int, batch_size: int, lr: float, seed: int):
    """Delete (reinitialize) ONLY the adapter+fine_head — the 'restricted'
    component — and retrain just those on the remaining data; backbone and
    coarse_head are left exactly as they were (frozen, not retrained).
    Tests a question distinct from the project's key-revocation non-claim:
    not 'does revoking a key remove influence' (it plainly doesn't — see
    key_revocation_only below) but 'does deleting the adapter remove it,
    when the shared backbone is left untouched?' — the backbone may still
    carry the removed data's influence even after this."""
    torch.manual_seed(seed)
    model = copy.deepcopy(authorized_model)
    feature_dim = model.backbone.feature_dim
    rank = model.adapter.down.out_features
    num_fine = model.fine_head.fc.out_features
    model.adapter = type(model.adapter)(feature_dim, rank=rank)
    model.fine_head = type(model.fine_head)(feature_dim, num_fine)
    for p in model.backbone.parameters():
        p.requires_grad_(False)
    for p in model.coarse_head.parameters():
        p.requires_grad_(False)

    trainable = list(model.adapter.parameters()) + list(model.fine_head.parameters())
    pooled = torch.utils.data.ConcatDataset(client_datasets_after_removal)
    local_train(model, pooled, epochs, batch_size, lr, trainable_params=trainable)
    return model


def gradient_ascent_unlearning(
    authorized_model, removed_data, retained_data, ascent_steps: int, descent_steps: int, batch_size: int, lr: float, seed: int
):
    """A few steps of GRADIENT ASCENT on the fine-task loss for the data
    being removed (pushes the model away from having fit it), then a short
    gradient-DESCENT 'recovery' pass on retained data (repair whatever
    utility damage the ascent caused). See module docstring: a generic
    simple technique, not a reproduction of a specific cited paper."""
    torch.manual_seed(seed)
    model = copy.deepcopy(authorized_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    removed_loader = torch.utils.data.DataLoader(removed_data, batch_size=batch_size, shuffle=True)
    done = 0
    while done < ascent_steps:
        for images, y_fine, _y_coarse in removed_loader:
            if done >= ascent_steps:
                break
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model.forward_fine(images), y_fine)
            (-loss).backward()  # ASCEND: maximize loss on the data being forgotten
            optimizer.step()
            done += 1

    retained_loader = torch.utils.data.DataLoader(retained_data, batch_size=batch_size, shuffle=True)
    done = 0
    while done < descent_steps:
        for images, y_fine, y_coarse in retained_loader:
            if done >= descent_steps:
                break
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model.forward_public(images), y_coarse) + \
                nn.functional.cross_entropy(model.forward_fine(images), y_fine)
            loss.backward()
            optimizer.step()
            done += 1

    return model


def key_revocation_only(authorized_model):
    """No model change AT ALL — returns the authorized model UNMODIFIED.
    The literal operationalization of the project brief's non-claim:
    revoking a key (medgate/crypto/authorization.py) prevents future
    AUTHORIZED ACCESS but does nothing to the weights themselves. Kept as
    a Phase 5 arm specifically so the results table shows, numerically,
    that this method achieves zero forgetting on every metric — making
    that failure visible is the point, not claiming it as competitive."""
    return authorized_model
