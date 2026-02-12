import torch
import torch.nn.functional as F
from config import Config


def construct_clinical_graph(tabular_features, threshold=0.5):
    """
    Constructs the Inductive Patient Relation Graph (Section 3.3.1).
    Edges are established if the Cosine Similarity of clinical profiles
    (lab tests + ICD codes proxy) exceeds the threshold tau.

    Args:
        tabular_features (Tensor): [Batch_Size, Tabular_Dim]
        threshold (float): Similarity threshold tau.

    Returns:
        edge_index (Tensor): Graph connectivity in COO format.
    """
    # L2 Normalization for Cosine Similarity
    norm_feats = F.normalize(tabular_features, p=2, dim=1)

    # Compute Pairwise Similarity Matrix: S = X . X^T
    sim_matrix = torch.mm(norm_feats, norm_feats.t())

    # Remove self-loops (diagonal elements)
    sim_matrix.fill_diagonal_(0)

    # Create edges where similarity > tau
    edge_index = (sim_matrix >= threshold).nonzero().t()

    return edge_index.to(Config.DEVICE)


class DisentanglementLoss(torch.nn.Module):
    """
    Implements the Disentanglement Constraint (Section 3.3.2).

    NOTE ON IMPLEMENTATION:
    The paper proposes minimizing Mutual Information (MI) via vCLUB.
    In this official implementation, we utilize an **Orthogonality Constraint** (minimizing squared Cosine Similarity) as a computationally efficient
    proxy for minimizing MI between shared and specific representations.
    This ensures feature distinctiveness with lower memory overhead.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z_shared, z_specific):
        # Minimize the cosine similarity between shared and specific vectors
        # Target: 0 (Orthogonal)
        return torch.mean(F.cosine_similarity(z_shared, z_specific, dim=1) ** 2)


class FocalLoss(torch.nn.Module):
    """
    Focal Loss to address class imbalance (Section 4.1).
    """

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()