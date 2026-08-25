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
            nn.ReLU(),  # not inplace: Opacus's per-sample-gradient hooks (medgate/privacy/dp_sgd.py) break on inplace ops
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),  # not inplace: Opacus's per-sample-gradient hooks (medgate/privacy/dp_sgd.py) break on inplace ops
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


class GradReverse(torch.autograd.Function):
    """Gradient reversal (standard DANN-style trick): identity on the
    forward pass, negated+scaled gradient on the backward pass. Lets one
    optimizer jointly (a) train an adversary head to predict the fine
    label from the public representation as well as it can, while (b)
    pushing the backbone to make that representation *less* predictive —
    the two effects share the same loss term because of the sign flip
    here, no separate min/max loop needed."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


def orthogonality_loss(z: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """One candidate operationalization of 'separation between public and
    restricted feature subspaces' (project brief, L_orth) — squared cosine
    similarity between the public representation z and the adapter's
    residual contribution delta=A_phi(z), averaged over the batch. Not
    claimed to be the only or the best operationalization; Phase 2 compares
    it against the simpler alternatives (no orth term at all, adapter-only,
    adversarial-only) rather than assuming it helps."""
    z_n = nn.functional.normalize(z, dim=-1, eps=1e-8)
    d_n = nn.functional.normalize(delta, dim=-1, eps=1e-8)
    cos_sim = (z_n * d_n).sum(dim=-1)
    return (cos_sim ** 2).mean()


class MedGateModel(nn.Module):
    """Public backbone + coarse head + restricted LoRA adapter + fine head
    + an always-present adversary head (z -> fine-class logits, meant to be
    used only behind grad_reverse). Every Phase 2 method (coarse-only,
    hidden-fine-head, adapter-isolation, adversarial, orthogonal, combined)
    uses this exact same architecture and differs only in which loss terms
    are active — an intentional design choice so the ablation isolates the
    objective, not the model capacity."""

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
        self.adversary_head = LinearHead(feature_dim, num_fine)

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        """f_theta(x) — the public representation. Used directly by probing
        attacks in Phase 2/3 to test whether fine-label information leaks
        through the public path even when the fine head is hidden."""
        return self.backbone(x)

    def forward_public(self, x: torch.Tensor) -> torch.Tensor:
        return self.coarse_head(self.representation(x))

    def forward_fine(self, x: torch.Tensor, use_adapter: bool = True) -> torch.Tensor:
        z = self.representation(x)
        if use_adapter:
            return self.fine_head(z + self.adapter(z))
        return self.fine_head(z)  # "hidden fine head": same representation, no adapter isolation

    def adversary_logits(self, x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        z = self.representation(x)
        return self.adversary_head(grad_reverse(z, lambd))

    def orth_term(self, x: torch.Tensor) -> torch.Tensor:
        z = self.representation(x)
        return orthogonality_loss(z, self.adapter(z))
