"""Per-sample learnable curation mask used in the base session."""

from typing import Optional

import torch
import torch.nn as nn


class MaskBank(nn.Module):
    """One learnable logit per training sample.

    ``soft(idx) = sigmoid(scale * w[idx])`` is the continuous keep-probability
    used in the weighted cross-entropy; ``hard_ste`` binarises it with a
    straight-through estimator for the budget term, so the number of kept
    samples is counted discretely while gradients still flow.
    """

    def __init__(self, num_samples: int, scale: float = 65.0, init_val: float = 0.01,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.num_samples = num_samples
        self.scale = scale
        dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.w = nn.Parameter(torch.full((num_samples,), init_val, device=dev))

    def soft(self, global_idx: torch.LongTensor) -> torch.Tensor:
        """Keep-probability in ``(0, 1)`` for the given global sample indices."""
        return torch.sigmoid(self.scale * self.w[global_idx])

    @staticmethod
    def hard_ste(x: torch.Tensor) -> torch.Tensor:
        """Binarise ``x`` at 0.5 while passing gradients straight through."""
        return (x > 0.5).float().detach() - x.detach() + x

    def forward(self, global_idx: torch.LongTensor, hard: bool = False) -> torch.Tensor:
        x = self.soft(global_idx)
        return self.hard_ste(x) if hard else x
