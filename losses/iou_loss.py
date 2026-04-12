'''Custom IoU (Intersection over Union) loss for bounding-box regression.

'IoU_loss = 1 - IoU' so that the loss is **zero** for a perfect match
and **one** when there is no overlap at all.

Boxes are represented as '(cx, cy, w, h)' - centre-x, centre-y, width,
height.  The intersection is computed directly from this representation
without a separate conversion to corner format: the overlap along each
axis equals 'max(0, min_right_edge - max_left_edge)' where the edges
are derived on-the-fly from the centre and half-extents.
'''

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    '''Differentiable IoU loss in [0, 1] for '(cx, cy, w, h)' boxes.'''

    _VALID_REDUCTIONS = {"mean", "sum", "none"}

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        '''
        Args:
            eps:       numerical stabiliser added to the denominator.
            reduction: ''mean'' (default) | ''sum'' | ''none''.
        '''
        super().__init__()
        if reduction not in self._VALID_REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {self._VALID_REDUCTIONS}, got '{reduction}'"
            )
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> torch.Tensor:
        '''Compute '1 - IoU' for each pair of predicted / target boxes.

        Args:
            pred_boxes:   '[B, 4]' predicted  '(cx, cy, w, h)'.
            target_boxes: '[B, 4]' ground-truth '(cx, cy, w, h)'.

        Returns:
            Scalar (mean / sum) or '[B]' tensor (none).
        '''
        #  centres and half-extents 
        pcx, pcy, pw, ph = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        gcx, gcy, gw, gh = target_boxes[:, 0], target_boxes[:, 1], target_boxes[:, 2], target_boxes[:, 3]

        p_half_w, p_half_h = pw / 2, ph / 2
        g_half_w, g_half_h = gw / 2, gh / 2

        #  spans along each axis ─
        overlap_w = (
            torch.min(pcx + p_half_w, gcx + g_half_w)
            - torch.max(pcx - p_half_w, gcx - g_half_w)
        ).clamp(min=0)

        overlap_h = (
            torch.min(pcy + p_half_h, gcy + g_half_h)
            - torch.max(pcy - p_half_h, gcy - g_half_h)
        ).clamp(min=0)

        intersection = overlap_w * overlap_h

        #  = A_pred + A_gt - intersection ─
        area_pred = (pw * ph).clamp(min=0)
        area_gt = (gw * gh).clamp(min=0)
        union = area_pred + area_gt - intersection

        iou = intersection / (union + self.eps)
        per_sample_loss = 1.0 - iou          # in [0, 1]

        #  
        if self.reduction == "mean":
            return per_sample_loss.mean()
        if self.reduction == "sum":
            return per_sample_loss.sum()
        return per_sample_loss
