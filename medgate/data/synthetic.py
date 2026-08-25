"""THE NULL-SIGNAL FIXTURE. Images and labels are drawn independently at
random (see SyntheticFedISIC.__init__ below) -- by construction there is
NO learnable relationship between an image and its label. Its only valid
use is a negative control: catching impossible above-chance behavior
(a leakage bug) or a pipeline crash. It CANNOT show whether capability
isolation, a pretrained baseline, or an attack actually works, because
there is nothing to learn, isolate, or attack in the first place. Every
result produced from this module must be reported as null-signal /
pipeline-validation, never as evidence for or against a method.

For anything that requires real learnable structure (fair pretrained
baselines, the expected authorized>public/full-finetune>=FedLoRA ranking,
attacks that need something to actually attack), use
medgate/data/hierarchical_synthetic.py instead.

Mirrors Fed-ISIC2019's shape (docs/research_scope.md §5) for pipeline
compatibility only: 6 centers, 8 fine-grained classes, a 3-way coarse
ontology, 32x32 images (small on purpose, for CPU speed). Used until the
real dataset is downloaded (license-gated, see
scripts/download_fed_isic2019_INSTRUCTIONS.md).
"""
import torch

FINE_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
NUM_FINE_CLASSES = len(FINE_CLASSES)
NUM_CENTERS = 6

# Primary coarse ontology (docs/research_scope.md §7): an experimental
# taxonomy, not a clinical access policy. AK is grouped with keratinocytic
# lesions here because it is the precursor keratinocytic lesion, per the
# project brief's explicit instruction not to fold it into a naive
# benign/malignant split. A sensitivity-analysis alternative mapping
# belongs in medgate/data/coarse_ontology.py once Phase 2 starts.
COARSE_MAP = {
    "melanocytic": ["MEL", "NV"],
    "keratinocytic": ["BCC", "SCC", "AK", "BKL"],
    "other": ["DF", "VASC"],
}
COARSE_CLASSES = list(COARSE_MAP.keys())
FINE_TO_COARSE_IDX = {
    fine: coarse_idx
    for coarse_idx, (_, fines) in enumerate(COARSE_MAP.items())
    for fine in fines
}
assert set(FINE_TO_COARSE_IDX) == set(FINE_CLASSES), "coarse map must cover every fine class"


class SyntheticFedISIC(torch.utils.data.Dataset):
    """One synthetic center's worth of (image, fine_label, coarse_label)."""

    def __init__(self, num_samples: int, image_size: int = 32, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(num_samples, 3, image_size, image_size, generator=g)
        self.fine_labels = torch.randint(0, NUM_FINE_CLASSES, (num_samples,), generator=g)
        self.coarse_labels = torch.tensor(
            [FINE_TO_COARSE_IDX[FINE_CLASSES[i]] for i in self.fine_labels.tolist()]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.fine_labels[idx], self.coarse_labels[idx]


def make_synthetic_centers(
    samples_per_center: int = 64, image_size: int = 32, seed: int = 0
) -> list[SyntheticFedISIC]:
    """Deterministic per-center synthetic datasets, one per Fed-ISIC2019 center."""
    return [
        SyntheticFedISIC(samples_per_center, image_size=image_size, seed=seed + c)
        for c in range(NUM_CENTERS)
    ]


def subset_by_indices(dataset: SyntheticFedISIC, indices) -> torch.utils.data.Dataset:
    """A plain TensorDataset over the given rows of one center's fixture —
    used by Phase 5 (medgate/unlearning) to build 'data with the target
    subset removed' scenarios without a real patient/lesion manifest
    (docs/execution_plan.md notes this as a known limitation of the
    synthetic tier: no real patient IDs exist to remove by)."""
    idx = torch.as_tensor(list(indices), dtype=torch.long)
    return torch.utils.data.TensorDataset(dataset.images[idx], dataset.fine_labels[idx], dataset.coarse_labels[idx])


def remove_fine_class(dataset: SyntheticFedISIC, fine_class_idx: int) -> torch.utils.data.Dataset:
    """Class-level removal: every example of one fine class, dropped."""
    keep = (dataset.fine_labels != fine_class_idx).nonzero(as_tuple=True)[0]
    return subset_by_indices(dataset, keep)


def select_fine_class(dataset: SyntheticFedISIC, fine_class_idx: int) -> torch.utils.data.Dataset:
    """The complement of remove_fine_class — just the removed examples,
    needed as the 'removed_data' pool for gradient_ascent_unlearning and
    for measuring residual membership signal after unlearning."""
    keep = (dataset.fine_labels == fine_class_idx).nonzero(as_tuple=True)[0]
    return subset_by_indices(dataset, keep)


def make_never_trained_class_pool(fine_class_idx: int, num_samples: int = 32, image_size: int = 32, seed: int = 0) -> torch.utils.data.Dataset:
    """A pool of examples of ONE fine class that is never included in any
    train or test center this project generates elsewhere (distinct seed
    offset, +90_000, chosen to not collide with make_synthetic_centers'
    seed+c range or make_synthetic_train_test_centers' seed+10_000 test
    offset). Used only for the WITHIN-CLASS member-vs-non-member
    unlearning comparison (medgate/unlearning -- docs/execution_plan.md
    Phase 5's class-level-removal confound fix): comparing removed
    training examples of a class against OTHER examples of the SAME class
    that no model (not even the gold-standard retrained one) ever saw,
    rather than against retained-test data of DIFFERENT classes (which
    confounds 'was this trained on' with 'is this the same class the
    model was never asked to represent at all')."""
    g = torch.Generator().manual_seed(seed + 90_000 + fine_class_idx)
    images = torch.randn(num_samples, 3, image_size, image_size, generator=g)
    fine_labels = torch.full((num_samples,), fine_class_idx, dtype=torch.long)
    coarse_labels = torch.full((num_samples,), FINE_TO_COARSE_IDX[FINE_CLASSES[fine_class_idx]], dtype=torch.long)
    return torch.utils.data.TensorDataset(images, fine_labels, coarse_labels)


def make_synthetic_train_test_centers(
    samples_per_center: int = 64, image_size: int = 32, seed: int = 0
) -> tuple[list[SyntheticFedISIC], list[SyntheticFedISIC]]:
    """Disjoint train/test synthetic datasets per center (offset seeds so
    train and test never draw the same generator state)."""
    train = make_synthetic_centers(samples_per_center, image_size, seed)
    test = make_synthetic_centers(samples_per_center, image_size, seed + 10_000)
    return train, test
