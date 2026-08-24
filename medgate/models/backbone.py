"""Public/restricted capability-isolated model (docs/research_scope.md §"Experimental architecture").

    public path:    y_c_hat = h_c( f_theta(x) )
    authorized path: y_f_hat = h_f( f_theta(x) + A_phi(f_theta(x)) )

f_theta is a small CNN backbone (sized for CPU smoke tests, see
docs/hardware_report.md — not a claim about what a production backbone
should be). A_phi is a LoRA-style low-rank residual applied in
representation space, zero-initialized so an untrained adapter is a no-op
(standard LoRA init: the "up" projection starts at zero).
"""
import torch
import torch.nn as nn


class SmallBackbone(nn.Module):
    """f_theta(x): image -> feature_dim representation."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Linear(32 * 4 * 4, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        z = torch.flatten(z, 1)
        return self.proj(z)


class LinearHead(nn.Module):
    """h_c or h_f: representation -> class logits."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


class LoRAAdapter(nn.Module):
    """A_phi: low-rank residual in representation space, zero-initialized.

    Zero init means an adapter with no training is the identity residual
    (adds nothing), matching standard LoRA practice (Hu et al. 2022,
    docs/literature_matrix.csv id hu2022-lora) and giving a clean baseline:
    before any fine-task training, forward_fine == forward_public passed
    through a differently-initialized head.
    """

    def __init__(self, feature_dim: int, rank: int = 4):
        super().__init__()
        self.down = nn.Linear(feature_dim, rank, bias=False)
        self.up = nn.Linear(rank, feature_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(z))


class MedGateModel(nn.Module):
    """Public backbone + coarse head + restricted LoRA adapter + fine head."""

    def __init__(
        self,
        num_coarse: int,
        num_fine: int,
        feature_dim: int = 64,
        adapter_rank: int = 4,
        in_channels: int = 3,
    ):
        super().__init__()
        self.backbone = SmallBackbone(in_channels=in_channels, feature_dim=feature_dim)
        self.coarse_head = LinearHead(feature_dim, num_coarse)
        self.adapter = LoRAAdapter(feature_dim, rank=adapter_rank)
        self.fine_head = LinearHead(feature_dim, num_fine)

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        """f_theta(x) — the public representation. Used directly by probing
        attacks in Phase 2/3 to test whether fine-label information leaks
        through the public path even when the fine head is hidden."""
        return self.backbone(x)

    def forward_public(self, x: torch.Tensor) -> torch.Tensor:
        return self.coarse_head(self.representation(x))

    def forward_fine(self, x: torch.Tensor) -> torch.Tensor:
        z = self.representation(x)
        return self.fine_head(z + self.adapter(z))
