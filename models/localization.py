'''Bounding box localizer for Oxford-IIIT Pet dataset.

Output: (cx, cy, w, h) in pixel coordinates of the 224x224 input.
Uses clamped output instead of sigmoid to preserve gradient at edges.
'''

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    '''VGG-11 backbone -> adaptive-pool -> FC regression head -> pixel-space box.'''

    IMG_SIZE = 224

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Keep dropout moderate for regression stability
        head_drop = min(dropout_p, 0.3)
        fc_dim = 512 * 7 * 7

        self.regressor = nn.Sequential(
            nn.Linear(fc_dim, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=head_drop),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=head_drop),
            nn.Linear(4096, 4),
        )

        # Sensible initialisation for the output layer
        self._init_output_bias()

    def _init_output_bias(self):
        '''Set the final FC bias so predictions start near a centred box.'''
        output_layer = self.regressor[-1]      # last nn.Linear
        with torch.no_grad():
            nn.init.xavier_uniform_(output_layer.weight, gain=0.01)
            output_layer.bias.copy_(torch.tensor([
                self.IMG_SIZE * 0.5,   # cx  - centre of image
                self.IMG_SIZE * 0.5,   # cy
                self.IMG_SIZE * 0.40,  # w   - typical head ROI width
                self.IMG_SIZE * 0.40,  # h
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: '[B, C, 224, 224]' normalised input.
        Returns:
            '[B, 4]' as '(cx, cy, w, h)' in pixel space.
        '''
        feat = self.encoder(x)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        raw = self.regressor(feat)
        return raw.clamp(min=0.0, max=float(self.IMG_SIZE))
