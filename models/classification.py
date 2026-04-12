'''Classification head for the 37-breed Oxford-IIIT Pet task.

VGG11 encoder + adaptive pool + 3-layer FC classifier with dropout.
'''

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    '''VGG-11 backbone -> adaptive-pool -> 3-layer FC classifier.'''

    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 3,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        fc_input_dim = 512 * 7 * 7
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

        # FC head weights
        for layer in self.classifier:
            if hasattr(layer, 'weight') and layer.weight.dim() > 1:
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.zero_()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: '[B, C, H, W]' input images.
        Returns:
            '[B, num_classes]' raw logits (no softmax).
        '''
        feat = self.encoder(x)
        feat = self.avgpool(feat)
        feat = feat.view(feat.size(0), -1)
        return self.classifier(feat)
