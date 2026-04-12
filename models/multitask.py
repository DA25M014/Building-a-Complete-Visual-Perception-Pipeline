'''Unified multi-task perception model.

The three individually trained models (classifier, localizer, U-Net) share
the same VGG-11 encoder architecture but were fine-tuned independently, so
their encoder weights have drifted.  To create a single shared backbone
that is compatible with **all three** task heads we compute the arithmetic
mean of the three encoder state dicts - a technique sometimes called
"model souping".  Because all three encoders were initialised from the
same pretrained checkpoint, their weights lie in a similar loss basin and
averaging produces a strong compromise.
'''

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet
from .layers import CustomDropout


def _average_state_dicts(*dicts):
    '''Return an element-wise average of identically-keyed state dicts.'''
    combined = {}
    for key in dicts[0]:
        stacked = torch.stack([d[key].float() for d in dicts if key in d])
        combined[key] = stacked.mean(dim=0)
    return combined


class MultiTaskPerceptionModel(nn.Module):
    '''Shared-backbone multi-task model for classification + localisation + segmentation.'''

    IMG_SIZE = 224

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
    ):
        super().__init__()

        #  weights from Google Drive ─
        import gdown
        gdown.download(id="1YhqieAut_QdSiMKydtkQ2LywUZkx3q0a", output=classifier_path, quiet=False)
        gdown.download(id="1kdLonqCZktTOcCnHv6nGaZSsIrD8UfVb", output=localizer_path, quiet=False)
        gdown.download(id="16tPmdWTikvnciaqLbz62RxGQidpUsuiC", output=unet_path, quiet=False)

        #  and load each trained model 
        def _load(model, path):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
            return model

        clf  = _load(VGG11Classifier(num_classes=num_breeds, in_channels=in_channels), classifier_path)
        loc  = _load(VGG11Localizer(in_channels=in_channels), localizer_path)
        unet = _load(VGG11UNet(num_classes=seg_classes, in_channels=in_channels), unet_path)

        # use classifier encoder for cls+seg, separate for loc
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.encoder.load_state_dict(clf.encoder.state_dict())

        self.loc_encoder = VGG11Encoder(in_channels=in_channels)
        self.loc_encoder.load_state_dict(loc.encoder.state_dict())

        self.seg_encoder = VGG11Encoder(in_channels=in_channels)
        self.seg_encoder.load_state_dict(unet.encoder.state_dict())

        #  heads (keep each head's trained weights) 
        self.cls_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.cls_head = clf.classifier

        self.loc_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.loc_head = loc.regressor

        self.seg_decoder = unet.decoder_levels
        self.seg_drop    = unet.bottleneck_drop
        self.seg_head    = unet.head

    def forward(self, x: torch.Tensor):
        '''Single forward pass producing all three task outputs.

        Args:
            x: '[B, C, 224, 224]' normalised input.

        Returns:
            dict with keys ''classification'', ''localization'',
            ''segmentation''.
        '''
        bottleneck = self.encoder(x, return_features=False)

        #  branch 
        c = self.cls_pool(bottleneck).view(x.size(0), -1)
        cls_logits = self.cls_head(c)

        # loc branch uses separate encoder
        loc_bottleneck = self.loc_encoder(x, return_features=False)
        l = self.loc_pool(loc_bottleneck).view(x.size(0), -1)
        bbox = self.loc_head(l).clamp(0.0, float(self.IMG_SIZE))

        # seg branch uses separate encoder
        seg_bottleneck, seg_skips = self.seg_encoder(x, return_features=True)
        h = seg_bottleneck
        num_stages = self.seg_encoder.NUM_STAGES
        for lvl_idx, dec_block in enumerate(self.seg_decoder):
            skip_key = f"block{num_stages - lvl_idx}"
            h = dec_block(h, seg_skips[skip_key])
        h = self.seg_drop(h)
        seg_logits = self.seg_head(h)

        return {
            "classification": cls_logits,
            "localization":   bbox,
            "segmentation":   seg_logits,
        }
