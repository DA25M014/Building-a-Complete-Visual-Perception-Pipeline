'''Custom layers - CustomDropout with inverted scaling.

Bernoulli masking with 1/(1-p) scaling during training.
Identity during eval.
'''

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    '''Inverted dropout via Bernoulli masking.

    Unlike 'torch.nn.Dropout' (which is NOT used here), this
    implementation directly samples a binary retention mask from a
    Bernoulli distribution and applies inverted scaling in a single step.
    '''

    def __init__(self, p: float = 0.5):
        '''
        Args:
            p: probability of an element being **zeroed** (drop rate).
        '''
        super().__init__()
        if p < 0.0 or p >= 1.0:
            raise ValueError(f"drop rate p must satisfy 0 ≤ p < 1, received {p}")
        self.p = p

    def extra_repr(self) -> str:
        return f"p={self.p}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: Tensor of any shape.

        Returns:
            Same-shaped tensor with inverted-dropout applied during training.
        '''
        if not self.training or self.p == 0.0:
            return x

        retention_rate = 1.0 - self.p
        # Bernoulli mask: 1 -> keep, 0 -> drop
        keep_mask = torch.bernoulli(
            torch.full_like(x, fill_value=retention_rate, dtype=torch.float32)
        ).to(x.dtype)

        # Scale surviving activations so E[output] ~ E[input]
        return x * keep_mask * (1.0 / retention_rate)
