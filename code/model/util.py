import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor, num_logits: int, reduction='mean'):
    assert num_logits > 0

    if num_logits == 1:  # (batch, seq_len)
        # No ignore_index in BCEWithLogitsLoss, implementing manually:
        flat_logits = logits.view(-1)
        flat_labels = labels.view(-1)
        valid_mask = (flat_labels != -100)
        assert flat_logits[valid_mask].shape == flat_labels[valid_mask].shape
        loss_fct = BCEWithLogitsLoss(reduction=reduction)
        loss = loss_fct(flat_logits[valid_mask], flat_labels[valid_mask].float())
    else:  # (batch, seq_len, num_logits)
        loss_fct = CrossEntropyLoss(reduction=reduction)
        loss = loss_fct(logits.view(-1, num_logits), labels.view(-1))

    return loss
