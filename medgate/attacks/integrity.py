"""A4 integrity attacks: a malicious client's contribution is corrupted
before aggregation. Each function returns a state_dict with the same keys
as an honest client's, so it slots directly into
medgate.federated.fedavg.fedavg_aggregate (or the robust alternatives in
medgate.federated.robust_aggregation) alongside honest clients' states.

Attacker model throughout: a single malicious client (A4, "normal client
role in one or more rounds", docs/research_scope.md). No collusion between
multiple malicious clients is modeled here (see
medgate/attacks/reconstruction.py's collusion_attack for a DIFFERENT kind
of collusion, between two A2 attackers reconstructing an adapter, not
between A4 poisoners).
"""
import torch

from medgate.data.synthetic import NUM_FINE_CLASSES
from medgate.federated.fedavg import local_train


def label_flipping_train(model, dataset, epochs, batch_size, lr, flip_offset: int = 1) -> dict:
    """Train normally but on relabeled data: fine label y -> (y + flip_offset) % NUM_FINE_CLASSES.
    Classic data poisoning -- no protocol-level change, just corrupted
    local labels, so the resulting state_dict looks like any other
    client's to the aggregator."""
    images = dataset.images if hasattr(dataset, "images") else torch.stack([dataset[i][0] for i in range(len(dataset))])
    fine = dataset.fine_labels if hasattr(dataset, "fine_labels") else torch.stack([dataset[i][1] for i in range(len(dataset))])
    coarse = dataset.coarse_labels if hasattr(dataset, "coarse_labels") else torch.stack([dataset[i][2] for i in range(len(dataset))])
    flipped_fine = (fine + flip_offset) % NUM_FINE_CLASSES
    poisoned = torch.utils.data.TensorDataset(images, flipped_fine, coarse)
    return local_train(model, poisoned, epochs, batch_size, lr)


def sign_flipping_update(honest_state: dict, global_state: dict) -> dict:
    """Negate the client's true update before it would be sent:
    result = global - (honest - global) = 2*global - honest."""
    return {k: global_state[k] - (honest_state[k] - global_state[k]) for k in global_state}


def model_replacement_update(honest_state: dict, global_state: dict, boost_factor: float) -> dict:
    """Scale the update by boost_factor before sending. With enough
    clients and boost, a single malicious update can dominate the FedAvg
    average and approximate overwriting the aggregated model with the
    attacker's chosen state (the classic 'model replacement' attack idea:
    boost the update by ~1/(client weight fraction) to fully cancel the
    other clients' contributions in expectation)."""
    return {k: global_state[k] + boost_factor * (honest_state[k] - global_state[k]) for k in global_state}


def free_rider_update(global_state: dict) -> dict:
    """Contribute a copy of the CURRENT global state (a zero update) --
    benefits from the aggregate without doing any local training or
    contributing any signal. Harmless to model correctness (adds zero to
    the sum) but a resource-theft/incentive-compatibility problem, not a
    poisoning one."""
    return {k: v.clone() for k, v in global_state.items()}


def malformed_update(honest_state: dict, kind: str = "nan") -> dict:
    """A corrupted update a robust server should reject before
    aggregating. kind='nan'|'inf'|'wrong_shape'."""
    corrupted = {k: v.clone() for k, v in honest_state.items()}
    first_key = next(iter(corrupted))
    if kind == "nan":
        corrupted[first_key] = torch.full_like(corrupted[first_key], float("nan"))
    elif kind == "inf":
        corrupted[first_key] = torch.full_like(corrupted[first_key], float("inf"))
    elif kind == "wrong_shape":
        corrupted[first_key] = corrupted[first_key].flatten()[:1]  # deliberately wrong shape
    else:
        raise ValueError(f"unknown malformed-update kind: {kind}")
    return corrupted


def backdoor_dataset(dataset, target_fine_class: int, trigger_value: float = 5.0, patch_size: int = 4):
    """Stamp a fixed high-value patch into the top-left corner of every
    image and force the fine label to `target_fine_class` -- the standard
    backdoor-insertion recipe (a trigger pattern paired with a target
    label). Returns a NEW dataset; does not mutate the original."""
    images = dataset.images.clone() if hasattr(dataset, "images") else torch.stack([dataset[i][0] for i in range(len(dataset))]).clone()
    coarse = dataset.coarse_labels if hasattr(dataset, "coarse_labels") else torch.stack([dataset[i][2] for i in range(len(dataset))])
    images[:, :, :patch_size, :patch_size] = trigger_value
    target_fine = torch.full((len(images),), target_fine_class, dtype=torch.long)
    return torch.utils.data.TensorDataset(images, target_fine, coarse)


def _slice_to_tensor_dataset(dataset, idx) -> torch.utils.data.TensorDataset:
    return torch.utils.data.TensorDataset(dataset.images[idx], dataset.fine_labels[idx], dataset.coarse_labels[idx])


def backdoor_train(model, dataset, epochs, batch_size, lr, target_fine_class: int, poison_fraction: float = 1.0) -> dict:
    """Train on a mix of clean and triggered-and-relabeled examples
    (poison_fraction of the client's own data is backdoored; the rest
    trains normally, matching how a real malicious client would blend in).
    `dataset` must be a medgate.data.synthetic.SyntheticFedISIC (or expose
    the same .images/.fine_labels/.coarse_labels tensors)."""
    n_poison = int(len(dataset) * poison_fraction)
    parts = []
    if n_poison > 0:
        parts.append(backdoor_dataset(_slice_to_tensor_dataset(dataset, slice(0, n_poison)), target_fine_class))
    if n_poison < len(dataset):
        parts.append(_slice_to_tensor_dataset(dataset, slice(n_poison, len(dataset))))
    combined = torch.utils.data.ConcatDataset(parts)
    return local_train(model, combined, epochs, batch_size, lr)


@torch.no_grad()
def backdoor_success_rate(model, clean_test_dataset, target_fine_class: int, trigger_value: float = 5.0, patch_size: int = 4, batch_size: int = 32) -> float:
    """Fraction of triggered test images classified as target_fine_class
    (excluding examples that are already that class) -- the standard
    backdoor attack-success-rate metric."""
    triggered = backdoor_dataset(clean_test_dataset, target_fine_class, trigger_value, patch_size)
    non_target_idx = [i for i in range(len(triggered)) if clean_test_dataset.fine_labels[i].item() != target_fine_class]
    if not non_target_idx:
        return float("nan")
    subset = torch.utils.data.Subset(triggered, non_target_idx)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size)
    model.eval()
    correct, total = 0, 0
    for images, _y_fine, _y_coarse in loader:
        preds = model.forward_fine(images).argmax(dim=1)
        correct += (preds == target_fine_class).sum().item()
        total += len(preds)
    return correct / total
