'''U-Net segmentation with VGG-11 encoder.

Symmetric decoder with ConvTranspose2d upsampling and skip connections.
Loss: CE + soft Dice to handle class imbalance in trimaps.
'''

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class _UpsampleFuseConv(nn.Module):
    '''One decoder level: transpose-upsample -> concat skip -> double conv.'''

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        merged_ch = in_ch + skip_ch
        self.refine = nn.Sequential(
            nn.Conv2d(merged_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.upsample(x)
        fused = torch.cat([up, skip], dim=1)
        return self.refine(fused)


class VGG11UNet(nn.Module):
    '''U-Net segmentation network with VGG-11 encoder.

    Decoder channel plan (symmetric to the encoder):
        Level 5:  512 up -> cat 512 -> 512
        Level 4:  512 up -> cat 512 -> 256
        Level 3:  256 up -> cat 256 -> 128
        Level 2:  128 up -> cat 128 ->  64
        Level 1:   64 up -> cat  64 ->  64  ->  1x1 conv -> num_classes
    '''

    def __init__(
        self,
        num_classes: int = 3,
        in_channels: int = 3,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        enc_ch = self.encoder.stage_channels          # [64, 128, 256, 512, 512]
        dec_out = [512, 256, 128, 64, 64]              # decoder output channels

        # Build decoder levels 5->1 using a ModuleList for clean iteration
        self.decoder_levels = nn.ModuleList()
        prev_ch = enc_ch[-1]                           # bottleneck = 512
        for i in range(self.encoder.NUM_STAGES):
            skip_ch = enc_ch[self.encoder.NUM_STAGES - 1 - i]
            self.decoder_levels.append(
                _UpsampleFuseConv(prev_ch, skip_ch, dec_out[i])
            )
            prev_ch = dec_out[i]

        self.bottleneck_drop = CustomDropout(p=dropout_p)
        self.head = nn.Conv2d(dec_out[-1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: '[B, C, H, W]' input images.
        Returns:
            '[B, num_classes, H, W]' dense logits (no softmax).
        '''
        bottleneck, skip_maps = self.encoder(x, return_features=True)
        h = bottleneck

        # Decode from deepest (block5) to shallowest (block1)
        for lvl_idx, dec_block in enumerate(self.decoder_levels):
            skip_key = f"block{self.encoder.NUM_STAGES - lvl_idx}"
            h = dec_block(h, skip_maps[skip_key])

        h = self.bottleneck_drop(h)
        return self.head(h)
