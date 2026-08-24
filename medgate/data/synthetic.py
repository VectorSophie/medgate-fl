"""Synthetic fixture mimicking Fed-ISIC2019's shape, NOT its content.

Used only until the real dataset is downloaded (license-gated, see
scripts/download_fed_isic2019_INSTRUCTIONS.md). Mirrors the verified facts
in docs/research_scope.md §5: 6 centers, 8 fine-grained classes, a 3-way
coarse ontology. Images are small (32x32) on purpose — this fixture exists
to exercise the training/aggregation code paths on CPU in seconds, not to
stand in for real-data results.
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


def make_synthetic_train_test_centers(
    samples_per_center: int = 64, image_size: int = 32, seed: int = 0
) -> tuple[list[SyntheticFedISIC], list[SyntheticFedISIC]]:
    """Disjoint train/test synthetic datasets per center (offset seeds so
    train and test never draw the same generator state)."""
    train = make_synthetic_centers(samples_per_center, image_size, seed)
    test = make_synthetic_centers(samples_per_center, image_size, seed + 10_000)
    return train, test
