"""THE HIERARCHICAL-SIGNAL FIXTURE. Unlike medgate/data/synthetic.py's
null-signal fixture (images and labels independent by construction), every
image here is generated FROM its labels and metadata by an explicit,
inspectable analytic function (medgate/data/hierarchical_synthetic.py's
`_render`) — so there is real, controllable, learnable structure a model
can actually pick up, and real structure attacks can actually exploit.

Design (all analytic, pure torch, no image library needed):

  - COARSE (3 classes, reused from medgate.data.synthetic.COARSE_CLASSES):
    determines the image's large-scale spatial pattern family — concentric
    rings, diagonal stripes, or a checkerboard grid.
  - FINE (8 classes nested in coarse, reused from
    medgate.data.synthetic.FINE_TO_COARSE_IDX): each fine class adds a
    higher-frequency oriented texture overlay at a fine-class-specific
    angle, on top of its coarse pattern.
  - SITE (institution, 0..5): a fixed per-site color tint, blur kernel
    size, brightness/contrast shift, and noise scale — an acquisition-
    shift analogue, not a label-relevant signal.
  - PATIENT (latent subject): a per-patient random rotation/translation
    "pose" applied consistently to every observation of that patient, so
    observations of one patient are more similar to each other than to
    other patients' images — and so a subject-level train/test split is
    meaningful to test in the first place.
  - SENSITIVE ATTRIBUTE: a binary label correlated with site (not baked
    into pixels — it is metadata, matching how this project's actual
    property-inference attack, medgate/attacks/property_inference.py,
    consumes client-level label statistics rather than pixel content).
  - BACKDOOR: deliberately NOT reimplemented here. medgate/attacks/
    integrity.py's backdoor_dataset()/backdoor_train() already operate on
    any dataset exposing .images/.fine_labels/.coarse_labels, which this
    module provides — reused as-is rather than duplicated. `backdoor_prevalence`
    is still accepted as a config field for interface completeness but a
    value of 0 (the default) means "apply backdoors post-hoc via
    medgate.attacks.integrity instead", which is what every script in
    this project actually does.

Every one of the "signal strength" parameters in HierarchicalConfig is a
literal scalar multiplier/probability read directly by `_render` /
`_sample_fine_labels` — inspect those two functions if a parameter's exact
effect is unclear; nothing here is hidden behind indirection.
"""
import math
from dataclasses import dataclass, field

import torch

from medgate.data.synthetic import COARSE_CLASSES, FINE_CLASSES, FINE_TO_COARSE_IDX, NUM_FINE_CLASSES

NUM_SITES = 6

# Real Fed-ISIC2019 / ISIC-2019-challenge per-class training counts, as
# verified against independent ISIC-2019 classification literature in
# docs/literature_matrix.csv (WebSearch, 2026-08-25; see
# docs/research_scope.md §5) -- used only to shape a REALISTIC class-
# imbalance target for this synthetic fixture, not asserted as this
# project's own measurement.
REAL_FED_ISIC_CLASS_COUNTS = {
    "MEL": 4522, "NV": 12875, "BCC": 3323, "AK": 867,
    "BKL": 2624, "DF": 239, "VASC": 253, "SCC": 628,
}


@dataclass
class HierarchicalConfig:
    image_size: int = 32
    num_patients_per_institution: int = 20
    observations_per_patient: int = 3
    coarse_signal_strength: float = 1.0     # 0 = no coarse pattern at all
    fine_signal_strength: float = 0.6       # 0 = no fine overlay (coarse-only becomes the whole signal)
    site_shift_strength: float = 1.0        # 0 = no site-specific tint/blur/noise/contrast
    class_imbalance_strength: float = 0.5   # 0 = uniform fine-class rate, 1 = REAL_FED_ISIC_CLASS_COUNTS skew
    sensitive_property_correlation: float = 0.6  # 0 = sensitive attribute independent of site
    backdoor_prevalence: float = 0.0        # see module docstring -- prefer medgate.attacks.integrity instead
    observation_noise: float = 0.05         # per-observation independent Gaussian noise std, on top of site noise
    patient_pose_jitter: float = 0.35       # radians/units of per-patient rotation+translation spread


def _fine_class_probabilities(imbalance_strength: float) -> torch.Tensor:
    uniform = torch.full((NUM_FINE_CLASSES,), 1.0 / NUM_FINE_CLASSES)
    real_counts = torch.tensor([REAL_FED_ISIC_CLASS_COUNTS[c] for c in FINE_CLASSES], dtype=torch.float32)
    real_dist = real_counts / real_counts.sum()
    s = min(max(imbalance_strength, 0.0), 1.0)
    return (1 - s) * uniform + s * real_dist


def _sample_fine_labels(n: int, imbalance_strength: float, generator: torch.Generator) -> torch.Tensor:
    probs = _fine_class_probabilities(imbalance_strength)
    return torch.multinomial(probs, n, replacement=True, generator=generator)


def _patient_latent_pose(patient_id: int, jitter: float, seed: int):
    """Deterministic per-patient (rotation, tx, ty) pose, stable across
    every observation of that patient and across repeated calls (a fresh
    torch.Generator keyed only by patient_id+seed, not by call order)."""
    g = torch.Generator().manual_seed(seed * 1_000_003 + patient_id)
    rot = (torch.rand((), generator=g).item() * 2 - 1) * jitter
    tx = (torch.rand((), generator=g).item() * 2 - 1) * jitter * 0.5
    ty = (torch.rand((), generator=g).item() * 2 - 1) * jitter * 0.5
    return rot, tx, ty


# Fixed per-site acquisition-shift parameters: tint (RGB multiplier),
# blur kernel size, brightness offset, contrast multiplier, noise scale.
# Deliberately hardcoded (not randomized) so "site N always looks like
# this" is a stable, inspectable property of the fixture, the way a real
# scanner/protocol difference would be.
_SITE_PARAMS = [
    {"tint": (1.10, 0.95, 0.95), "blur": 1, "brightness": 0.00, "contrast": 1.00, "noise": 0.02},
    {"tint": (0.95, 1.05, 1.00), "blur": 3, "brightness": 0.05, "contrast": 0.90, "noise": 0.05},
    {"tint": (1.00, 1.00, 1.10), "blur": 1, "brightness": -0.05, "contrast": 1.10, "noise": 0.03},
    {"tint": (1.05, 1.00, 0.90), "blur": 5, "brightness": 0.02, "contrast": 0.85, "noise": 0.08},
    {"tint": (0.90, 0.95, 1.05), "blur": 1, "brightness": -0.02, "contrast": 1.05, "noise": 0.02},
    {"tint": (1.00, 1.05, 0.95), "blur": 3, "brightness": 0.03, "contrast": 0.95, "noise": 0.06},
]


def _coarse_pattern(xx: torch.Tensor, yy: torch.Tensor, coarse_idx: torch.Tensor, freq: float = 4.0) -> torch.Tensor:
    """xx, yy: (N,H,W) per-sample (already pose-rotated/translated) grids.
    coarse_idx: (N,) long. Returns (N,H,W)."""
    r = torch.sqrt(xx ** 2 + yy ** 2 + 1e-8)
    rings = torch.sin(2 * math.pi * freq * r)
    stripes = torch.sin(2 * math.pi * freq * 0.5 * (xx + yy))
    grid = torch.sign(torch.sin(2 * math.pi * freq * 0.5 * xx) * torch.sin(2 * math.pi * freq * 0.5 * yy))
    stacked = torch.stack([rings, stripes, grid], dim=0)  # (3,N,H,W)
    return stacked.gather(0, coarse_idx.view(1, -1, 1, 1).expand(1, *stacked.shape[1:])).squeeze(0)


def _fine_overlay(xx: torch.Tensor, yy: torch.Tensor, fine_idx: torch.Tensor, freq: float = 10.0) -> torch.Tensor:
    """A higher-frequency oriented texture, one fixed orientation per fine
    class (0..7), independent of which coarse bucket it nests in -- the
    'internal pattern' that distinguishes fine classes sharing a coarse
    parent."""
    theta = fine_idx.float() * (2 * math.pi / NUM_FINE_CLASSES)  # (N,)
    theta = theta.view(-1, 1, 1)
    rotated = xx * torch.cos(theta) + yy * torch.sin(theta)
    return torch.sin(2 * math.pi * freq * rotated)


def _render(
    fine_labels: torch.Tensor, site_ids: torch.Tensor, patient_ids: torch.Tensor,
    cfg: HierarchicalConfig, seed: int,
) -> torch.Tensor:
    """Returns (N,3,H,W) float images. Every step below is analytic and
    keyed only by (fine_labels, site_ids, patient_ids, cfg, seed) -- no
    hidden state, no dependence on sample order."""
    n, hw = len(fine_labels), cfg.image_size
    coarse_idx = torch.tensor([FINE_TO_COARSE_IDX[FINE_CLASSES[f]] for f in fine_labels.tolist()])

    lin = torch.linspace(-1, 1, hw)
    base_yy, base_xx = torch.meshgrid(lin, lin, indexing="ij")
    base_xx = base_xx.unsqueeze(0).expand(n, hw, hw)
    base_yy = base_yy.unsqueeze(0).expand(n, hw, hw)

    rot = torch.zeros(n)
    tx = torch.zeros(n)
    ty = torch.zeros(n)
    for i, pid in enumerate(patient_ids.tolist()):
        r, x, y = _patient_latent_pose(pid, cfg.patient_pose_jitter, seed)
        rot[i], tx[i], ty[i] = r, x, y
    rot, tx, ty = rot.view(-1, 1, 1), tx.view(-1, 1, 1), ty.view(-1, 1, 1)
    xx = base_xx * torch.cos(rot) - base_yy * torch.sin(rot) + tx
    yy = base_xx * torch.sin(rot) + base_yy * torch.cos(rot) + ty

    pattern = cfg.coarse_signal_strength * _coarse_pattern(xx, yy, coarse_idx) + \
        cfg.fine_signal_strength * _fine_overlay(xx, yy, fine_labels)
    pattern = pattern.unsqueeze(1).expand(n, 3, hw, hw).clone()  # (N,3,H,W), replicate to RGB before tinting

    g = torch.Generator().manual_seed(seed + 777)
    for site in range(NUM_SITES):
        mask = site_ids == site
        if not mask.any():
            continue
        p = _SITE_PARAMS[site]
        tint = torch.tensor(p["tint"]).view(1, 3, 1, 1)
        pattern[mask] = pattern[mask] * tint
        pattern[mask] = pattern[mask] * (1 + cfg.site_shift_strength * (p["contrast"] - 1)) + \
            cfg.site_shift_strength * p["brightness"]
        k = p["blur"]
        if k > 1 and cfg.site_shift_strength > 0:
            blurred = torch.nn.functional.avg_pool2d(pattern[mask], kernel_size=k, stride=1, padding=k // 2)
            pattern[mask] = (1 - cfg.site_shift_strength) * pattern[mask] + cfg.site_shift_strength * blurred
        noise_scale = cfg.site_shift_strength * p["noise"] + cfg.observation_noise
        pattern[mask] = pattern[mask] + torch.randn(pattern[mask].shape, generator=g) * noise_scale

    return pattern


class HierarchicalSyntheticDataset(torch.utils.data.Dataset):
    """One institution's worth of hierarchically-generated data. Exposes
    .images/.fine_labels/.coarse_labels (same names as
    medgate.data.synthetic.SyntheticFedISIC, so every existing function
    that consumes that interface -- subset_by_indices, backdoor_dataset,
    local_train's DataLoader tuple order, etc. -- works unmodified) plus
    .patient_ids/.site_ids/.sensitive_labels for the metadata this fixture
    adds."""

    def __init__(self, images, fine_labels, coarse_labels, patient_ids, site_ids, sensitive_labels):
        self.images = images
        self.fine_labels = fine_labels
        self.coarse_labels = coarse_labels
        self.patient_ids = patient_ids
        self.site_ids = site_ids
        self.sensitive_labels = sensitive_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.fine_labels[idx], self.coarse_labels[idx]


def _make_one_institution(site: int, cfg: HierarchicalConfig, seed: int, patient_id_offset: int) -> HierarchicalSyntheticDataset:
    g = torch.Generator().manual_seed(seed + site)
    n_patients = cfg.num_patients_per_institution
    k = cfg.observations_per_patient

    patient_fine = _sample_fine_labels(n_patients, cfg.class_imbalance_strength, g)
    fine_labels = patient_fine.repeat_interleave(k)
    patient_ids = (torch.arange(n_patients) + patient_id_offset).repeat_interleave(k)
    site_ids = torch.full((n_patients * k,), site, dtype=torch.long)

    base_rate = 0.5
    site_bias = 0.35 if site < NUM_SITES // 2 else -0.35  # first half of sites skew positive, second half negative
    sensitive_prob = base_rate + cfg.sensitive_property_correlation * site_bias
    sensitive_prob = min(max(sensitive_prob, 0.0), 1.0)
    sensitive_labels = (torch.rand(n_patients * k, generator=g) < sensitive_prob).long()

    coarse_labels = torch.tensor([FINE_TO_COARSE_IDX[FINE_CLASSES[f]] for f in fine_labels.tolist()])
    images = _render(fine_labels, site_ids, patient_ids, cfg, seed)

    if cfg.backdoor_prevalence > 0:
        n_poison = int(len(images) * cfg.backdoor_prevalence)
        poison_idx = torch.randperm(len(images), generator=g)[:n_poison]
        images[poison_idx, :, :4, :4] = 5.0
        fine_labels = fine_labels.clone()
        fine_labels[poison_idx] = 0
        coarse_labels = coarse_labels.clone()
        coarse_labels[poison_idx] = FINE_TO_COARSE_IDX[FINE_CLASSES[0]]

    return HierarchicalSyntheticDataset(images, fine_labels, coarse_labels, patient_ids, site_ids, sensitive_labels)


def make_hierarchical_institutions(cfg: HierarchicalConfig = None, seed: int = 0) -> list[HierarchicalSyntheticDataset]:
    """One HierarchicalSyntheticDataset per institution (NUM_SITES=6, to
    match Fed-ISIC2019's verified center count, docs/research_scope.md §5)."""
    cfg = cfg or HierarchicalConfig()
    datasets = []
    offset = 0
    for site in range(NUM_SITES):
        ds = _make_one_institution(site, cfg, seed, patient_id_offset=offset)
        datasets.append(ds)
        offset += cfg.num_patients_per_institution
    return datasets


def split_by_patient(
    datasets: list[HierarchicalSyntheticDataset], train_frac: float = 0.6, val_frac: float = 0.2, seed: int = 0,
) -> tuple[list[HierarchicalSyntheticDataset], list[HierarchicalSyntheticDataset], list[HierarchicalSyntheticDataset]]:
    """Per-institution train/val/test split at the PATIENT level: every
    observation of a given patient_id lands in exactly one split. Returns
    three parallel lists (one entry per institution) so downstream code
    keeps its per-institution structure (test_frac = 1 - train_frac - val_frac)."""
    train_out, val_out, test_out = [], [], []
    for i, ds in enumerate(datasets):
        g = torch.Generator().manual_seed(seed + 55_000 + i)
        unique_patients = torch.unique(ds.patient_ids)
        perm = unique_patients[torch.randperm(len(unique_patients), generator=g)]
        n_train = int(len(perm) * train_frac)
        n_val = int(len(perm) * val_frac)
        train_p = set(perm[:n_train].tolist())
        val_p = set(perm[n_train:n_train + n_val].tolist())
        test_p = set(perm[n_train + n_val:].tolist())

        def subset(patient_set):
            idx = torch.tensor([j for j, pid in enumerate(ds.patient_ids.tolist()) if pid in patient_set], dtype=torch.long)
            return HierarchicalSyntheticDataset(
                ds.images[idx], ds.fine_labels[idx], ds.coarse_labels[idx],
                ds.patient_ids[idx], ds.site_ids[idx], ds.sensitive_labels[idx],
            )

        train_out.append(subset(train_p))
        val_out.append(subset(val_p))
        test_out.append(subset(test_p))
    return train_out, val_out, test_out
