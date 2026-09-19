"""
Loss functions used by RelFuse-Net (Sections 3.2.2, 3.3.2, 4.2.2):

  - WeightedBCELoss:      per-label class-weighted binary cross-entropy (Eq. 9's L_BCE),
                          used instead of plain/unweighted BCE to counter the severe
                          multi-label class imbalance described in Section 4.1.2.
  - VCLUBLoss:            the actual vCLUB variational upper bound on mutual information
                          (Cheng et al., 2020), replacing the earlier cosine-orthogonality
                          proxy. Used to MINIMIZE I(z_shared; z_specific) for each of the
                          three specific spaces (image/text/tabular), per Eq. 7.
  - mltm_reconstruction_loss: the masked reconstruction loss (Eq. 4/5) for MLTM
                          pretraining, computed ONLY on artificially-masked positions
                          that were originally observed (never on naturally-missing
                          entries, since there is no ground truth to reconstruct
                          against there -- Referee 2, point #8).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Per-label class-weighted BCE-with-logits. `pos_weight[c]` should be set to
    (num_negatives_c / num_positives_c) computed on the TRAINING split only,
    so that rare pathologies (e.g. "Pleural Other", 0.9% prevalence) are not
    dominated by frequent ones (e.g. "No Finding", 33.0%).
    """

    def __init__(self, pos_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight.to(logits.device)
        )

    @staticmethod
    def compute_pos_weight(labels: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
        """labels: [N, C] binary tensor over the TRAINING split."""
        pos = labels.sum(dim=0)
        neg = labels.shape[0] - pos
        return (neg + eps) / (pos + eps)


class _QNet(nn.Module):
    """Variational approximation q_theta(z_specific | z_shared) as a diagonal Gaussian."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, out_dim)
        self.logvar = nn.Linear(hidden, out_dim)

    def forward(self, z_shared: torch.Tensor):
        h = self.net(z_shared)
        return self.mu(h), self.logvar(h).clamp(-10, 10)


class VCLUBLoss(nn.Module):
    """
    vCLUB upper bound on I(z_shared; z_specific) (Cheng et al., ICML 2020).
    Call `.forward(z_shared, z_specific)` to get the MI upper-bound estimate to
    MINIMIZE (this is L_Disentangle's per-pair term in Eq. 7), and call
    `.learning_loss(z_shared, z_specific)` separately to fit the variational
    network q_theta by maximum likelihood -- this second loss must be optimized
    on its own (e.g. with its own optimizer step or a stop-gradient on z_shared/
    z_specific) and must NOT be summed into L_total, otherwise the encoder would
    be rewarded for making q's job easier rather than for disentangling.
    """

    def __init__(self, z_dim: int, hidden: int = 256):
        super().__init__()
        self.qnet = _QNet(z_dim, z_dim, hidden)

    def forward(self, z_shared: torch.Tensor, z_specific: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.qnet(z_shared)
        # Positive (matched) log-likelihood
        pos = -((mu - z_specific) ** 2) / (2 * logvar.exp())
        # Negative (shuffled) log-likelihood, sampled once per call
        perm = torch.randperm(z_specific.shape[0], device=z_specific.device)
        neg = -((mu - z_specific[perm]) ** 2) / (2 * logvar.exp())
        return (pos.sum(dim=-1) - neg.sum(dim=-1)).mean()

    def learning_loss(self, z_shared: torch.Tensor, z_specific: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.qnet(z_shared.detach())
        nll = 0.5 * (((mu - z_specific.detach()) ** 2) / logvar.exp() + logvar)
        return nll.sum(dim=-1).mean()


def mltm_reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    observed_mask: torch.Tensor,
    artificial_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Implements Eq. 4/5. `observed_mask` (m_i in the paper) marks entries that are
    genuinely present in the real EHR record (1 = observed, 0 = naturally missing).
    `artificial_mask` marks entries additionally hidden from the encoder for the
    self-supervised reconstruction objective (1 = kept visible, 0 = hidden), drawn
    ONLY from positions where observed_mask == 1 (you cannot artificially mask, or
    reconstruct against, something that was never recorded).

    Loss is computed only where observed_mask == 1 AND artificial_mask == 0, i.e.
    exactly the "artificially masked positions" in the paper's wording.
    """
    target_positions = observed_mask * (1 - artificial_mask)
    diff = target_positions * (x - x_hat)
    denom = target_positions.sum().clamp(min=1.0)
    return (diff ** 2).sum() / denom
