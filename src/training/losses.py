"""Binary cross-entropy loss with logits for link prediction."""
import torch
from torch import nn


def link_prediction_loss(
    pos_logits: torch.Tensor, neg_logits: torch.Tensor
) -> torch.Tensor:
    """Compute BCE-with-logits over concatenated positive and negative logits.

    Args:
        pos_logits: [P] logits for positive edges.
        neg_logits: [Q] logits for negative edges.
    Returns:
        scalar loss.
    """
    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat(
        [torch.ones_like(pos_logits), torch.zeros_like(neg_logits)]
    )
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)
