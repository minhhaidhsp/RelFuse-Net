"""
Loss functions used by RelFuse-Net (Sections 3.2.2, 3.3.2, 4.2.2):

  - WeightedBCELoss:      per-label class-weighted binary cross-entropy, implementing
                          Eq. 13 EXACTLY: a SYMMETRIC weight w_c = 1/(2*p_c) applied to
                          both the positive and negative terms of class c, then averaged
                          over C. This is a class-balanced weighting scheme (Cui et al.
                          style), and is NOT the same formula as PyTorch's native
                          `pos_weight=` convention in F.binary_cross_entropy_with_logits
                          (which only up-weights the positive term and does not divide
                          by C) -- do not swap this back for `pos_weight=` without also
                          changing Eq. 13 in the manuscript, or the two will disagree.
  - VCLUBLoss:            the actual vCLUB variational upper bound on mutual information
                          (Cheng et al., 2020), replacing the earlier cosine-orthogonality
                          proxy. Used to MINIMIZE I(z_shared; z_specific) for each of the
                          three specific spaces (image/text/tabular), per Eq. 7.
  - mltm_reconstruction_loss: the masked reconstruction loss (Eq. 5) for MLTM
                          pretraining, computed ONLY where the artificial-masking
                          indicator a_i == 1, i.e. positions that were genuinely
                          observed (o_i == 1) AND additionally held out for this
                          training step. Never computed against naturally-missing
                          entries, since there is no ground truth to reconstruct
                          against there (Referee 2, point #8).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Per-label class-weighted BCE, implementing Eq. 13:

        L_BCE(y_v, y_hat_v) = -(1/C) * sum_c w_c * [ y_c*log(yhat_c) + (1-y_c)*log(1-yhat_c) ]
        w_c = 1 / (2 * p_c)

    where p_c is the positive rate of label c measured on the TRAINING split only
    (Table 5), so rare pathologies (e.g. "Pleural Other") are not dominated by
    frequent ones (e.g. "No Finding"). `class_weight` must be `w_c`, shape [C],
    computed by `compute_class_weight` below -- NOT a pos/neg ratio.
    """

    def __init__(self, class_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("class_weight", class_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-element BCE from logits (numerically stable), not yet reduced: this is
        # exactly the bracketed term of Eq. 13, [y_c*log(yhat_c) + (1-y_c)*log(1-yhat_c)],
        # with the sign already folded in.
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # [N, C]
        weighted = bce * self.class_weight.to(logits.device)  # broadcast w_c over the batch
        return weighted.mean(dim=1).mean()  # mean over C (Eq. 13), then mean over the batch

    @staticmethod
    def compute_class_weight(labels: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """labels: [N, C] binary tensor over the TRAINING split. Returns w_c = 1/(2*p_c)."""
        p_c = labels.mean(dim=0).clamp(min=eps)
        return 1.0 / (2.0 * p_c)


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
    a: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Implements Eq. 5 exactly:

        L_MLTM = sum_i || a_i . (x_i - x_hat_i) ||_2^2  /  ( sum_i ||a_i||_1 + eps )

    `a` is the artificial-masking indicator a_i (1 = held out at this training
    step and used as a reconstruction target, 0 = not held out). By construction
    (see MLTMEncoder.forward in model.py), a_i is only ever sampled from positions
    where the observation indicator o_i == 1, so a_i <= o_i elementwise and this
    loss is never computed against a naturally-missing entry (Referee 2, point #8).
    """
    diff = a * (x - x_hat)
    denom = a.sum() + eps
    return (diff ** 2).sum() / denom
