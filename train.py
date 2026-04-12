'''Training script for classification, localization, and segmentation.

Usage:
    python train.py --task classification --epochs 20
    python train.py --task localization   --epochs 60 --lr 1e-4
    python train.py --task segmentation   --epochs 60 --lr 1e-4
'''

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import f1_score
import wandb

from data.pets_dataset import (
    OxfordIIITPetDataset,
    get_train_transforms,
    get_val_transforms,
    IMG_SIZE,
)
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.layers import CustomDropout
from losses.iou_loss import IoULoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device():
    '''Pick the best available accelerator: CUDA -> MPS -> CPU.'''
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True


def dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3, eps: float = 1e-6):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    dice = 0.0
    for c in range(num_classes):
        p = (pred_flat == c).float()
        t = (target_flat == c).float()
        inter = (p * t).sum()
        dice += (2 * inter + eps) / (p.sum() + t.sum() + eps)
    return dice / num_classes


def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor):
    return (pred == target).float().mean().item()


def compute_iou(pred_box, gt_box, eps=1e-6):
    px1 = pred_box[0] - pred_box[2] / 2
    py1 = pred_box[1] - pred_box[3] / 2
    px2 = pred_box[0] + pred_box[2] / 2
    py2 = pred_box[1] + pred_box[3] / 2
    gx1 = gt_box[0] - gt_box[2] / 2
    gy1 = gt_box[1] - gt_box[3] / 2
    gx2 = gt_box[0] + gt_box[2] / 2
    gy2 = gt_box[1] + gt_box[3] / 2
    ix1 = max(px1, gx1); iy1 = max(py1, gy1)
    ix2 = min(px2, gx2); iy2 = min(py2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_p = max(0, px2 - px1) * max(0, py2 - py1)
    area_g = max(0, gx2 - gx1) * max(0, gy2 - gy1)
    return inter / (area_p + area_g - inter + eps)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

PERIODIC_INTERVAL = 10   # save every N epochs
KEEP_LAST_N = 2          # rolling window of periodic checkpoints

def save_checkpoint(model, optimizer, scheduler, epoch, metric, best_metric,
                    task_name: str, is_best: bool):
    '''Save best checkpoint + periodic checkpoints (rolling window).'''
    os.makedirs("checkpoints", exist_ok=True)

    payload = {
        "state_dict": model.state_dict(),
        "best_metric": best_metric,
    }
    # full checkpoint for resume
    full_payload = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "metric": metric,
        "best_metric": best_metric,
    }

    # Always save best
    if is_best:
        best_path = f"checkpoints/{task_name}.pth"
        torch.save(payload, best_path)  # slim
        print(f"  [SAVED] Best {task_name} saved (metric={best_metric:.4f})")

    # Periodic save every PERIODIC_INTERVAL epochs
    if epoch % PERIODIC_INTERVAL == 0:
        periodic_path = f"checkpoints/{task_name}_epoch{epoch}.pth"
        torch.save(full_payload, periodic_path)  # full for resume
        print(f"  [CKPT] Periodic checkpoint: {periodic_path}")

        # Clean old periodic checkpoints (keep only last KEEP_LAST_N)
        import glob
        periodics = sorted(glob.glob(f"checkpoints/{task_name}_epoch*.pth"))
        while len(periodics) > KEEP_LAST_N:
            old = periodics.pop(0)
            os.remove(old)
            print(f"  [DEL] Removed old: {old}")


def load_checkpoint(model, optimizer, scheduler, task_name: str, device):
    '''Try to resume from the latest periodic or best checkpoint.
    Returns start_epoch and best_metric.'''
    import glob

    # Prefer latest periodic, fall back to best
    periodics = sorted(glob.glob(f"checkpoints/{task_name}_epoch*.pth"))
    if periodics:
        ckpt_path = periodics[-1]
    elif os.path.exists(f"checkpoints/{task_name}.pth"):
        ckpt_path = f"checkpoints/{task_name}.pth"
    else:
        return 1, 0.0   # no checkpoint found, start fresh

    print(f"  [RESUME] Resuming from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt and scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_metric = ckpt.get("best_metric", 0.0)
    print(f"  [RESUME] Resuming at epoch {start_epoch}, best_metric={best_metric:.4f}")
    return start_epoch, best_metric


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def build_datasets(data_root: str, val_frac: float = 0.15):
    '''Build train / val splits from trainval set with proper isolation.'''
    full_ds = OxfordIIITPetDataset(data_root, split="trainval", transform=None)
    n = len(full_ds)
    n_val = int(n * val_frac)
    n_train = n - n_val

    generator = torch.Generator().manual_seed(42)
    indices = list(range(n))
    perm = torch.randperm(n, generator=generator).tolist()
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    train_ds = OxfordIIITPetDataset(data_root, split="trainval", transform=get_train_transforms())
    val_ds = OxfordIIITPetDataset(data_root, split="trainval", transform=get_val_transforms())

    train_ds.samples = [full_ds.samples[i] for i in train_indices]
    val_ds.samples = [full_ds.samples[i] for i in val_indices]
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Encoder helpers
# ---------------------------------------------------------------------------

def load_pretrained_encoder(model, ckpt_path: str):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"  [INFO] No pretrained encoder at {ckpt_path}, training from scratch.")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    encoder_sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = model.encoder.load_state_dict(encoder_sd, strict=False)
    print(f"  [INFO] Loaded pretrained encoder ({len(encoder_sd)} keys, "
          f"missing={len(missing)}, unexpected={len(unexpected)})")


def freeze_encoder(model, mode: str):
    if mode == "none":
        return
    if mode == "full_freeze":
        for p in model.encoder.parameters():
            p.requires_grad = False
    elif mode == "partial":
        for i, block in enumerate(model.encoder.conv_stages):
            if i < 3:
                for p in block.parameters():
                    p.requires_grad = False


def get_param_groups(model, encoder_lr: float, head_lr: float, weight_decay: float = 1e-4):
    '''Differential LR: lower for pretrained encoder, higher for new head.'''
    encoder_params = list(model.encoder.parameters())
    encoder_ids = set(id(p) for p in encoder_params)
    head_params = [p for p in model.parameters() if id(p) not in encoder_ids and p.requires_grad]
    encoder_trainable = [p for p in encoder_params if p.requires_grad]

    groups = []
    if encoder_trainable:
        groups.append({"params": encoder_trainable, "lr": encoder_lr, "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    return groups


# ---------------------------------------------------------------------------
# Task 1: Classification
# ---------------------------------------------------------------------------

def train_classifier(args):
    device = get_device()
    train_ds, val_ds = build_datasets(args.data_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda"))

    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout).to(device)

    if args.no_batchnorm:
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                parent = model
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
                if parts[-1].isdigit():
                    parent[int(parts[-1])] = nn.Identity()
                else:
                    setattr(parent, parts[-1], nn.Identity())
        print("  [ABLATION] Removed all BatchNorm layers.")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_f1 = 0.0
    start_epoch = 1

    if args.resume:
        start_epoch, best_f1 = load_checkpoint(model, optimizer, scheduler, "classifier", device)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item() * images.size(0)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_ds)
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        scheduler.step()

        wandb.log({"cls/train_loss": train_loss, "cls/val_loss": val_loss,
                    "cls/macro_f1": macro_f1, "epoch": epoch})
        print(f"[CLS] Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  F1={macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
        save_checkpoint(model, optimizer, scheduler, epoch, macro_f1, best_f1,
                        "classifier", is_best=(macro_f1 >= best_f1))

    print(f"\nBest classification F1: {best_f1:.4f}")
    return model


# ---------------------------------------------------------------------------
# Task 2: Localization
# ---------------------------------------------------------------------------

def train_localizer(args):
    device = get_device()
    train_ds, val_ds = build_datasets(args.data_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda"))

    model = VGG11Localizer(dropout_p=args.dropout).to(device)

    if args.pretrained_encoder:
        load_pretrained_encoder(model, args.pretrained_encoder)
    freeze_encoder(model, args.freeze_mode)

    # --- Losses (MSE + IoU as required by assignment) ---
    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")

    # Differential LR: encoder 10x lower than head
    param_groups = get_param_groups(model, encoder_lr=args.lr * 0.1,
                                    head_lr=args.lr, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_iou50 = 0.0
    start_epoch = 1

    if args.resume:
        start_epoch, best_iou50 = load_checkpoint(model, optimizer, scheduler, "localizer", device)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)
            has_bbox = batch["has_bbox"]  # bool list

            optimizer.zero_grad()
            pred = model(images)

            # Only compute loss on samples with valid bbox annotations
            valid = torch.tensor(has_bbox, dtype=torch.bool, device=device)
            if valid.sum() == 0:
                continue
            pred_v = pred[valid]
            bbox_v = bboxes[valid]

            # Normalise to [0,1] for MSE stability, IoU on raw pixel coords
            loss_mse = mse_loss(pred_v / IMG_SIZE, bbox_v / IMG_SIZE)
            loss_iou = iou_loss(pred_v, bbox_v)
            loss = loss_mse + loss_iou

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        ious = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                has_bbox = batch["has_bbox"]
                pred = model(images)

                valid = torch.tensor(has_bbox, dtype=torch.bool, device=device)
                if valid.sum() == 0:
                    continue
                pred_v = pred[valid]
                bbox_v = bboxes[valid]

                loss = mse_loss(pred_v / IMG_SIZE, bbox_v / IMG_SIZE) + iou_loss(pred_v, bbox_v)
                val_loss += loss.item() * valid.sum().item()

                for i in range(pred_v.size(0)):
                    ious.append(compute_iou(pred_v[i].cpu().numpy(), bbox_v[i].cpu().numpy()))

        val_loss /= max(len(ious), 1)
        ious_arr = np.array(ious)
        mean_iou = ious_arr.mean()
        acc_50 = (ious_arr >= 0.5).mean() * 100
        acc_75 = (ious_arr >= 0.75).mean() * 100
        scheduler.step()

        wandb.log({"loc/train_loss": train_loss, "loc/val_loss": val_loss,
                    "loc/mean_iou": mean_iou, "loc/acc@0.5": acc_50,
                    "loc/acc@0.75": acc_75, "epoch": epoch})
        print(f"[LOC] Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  "
              f"mIoU={mean_iou:.4f}  Acc@0.5={acc_50:.1f}%  Acc@0.75={acc_75:.1f}%")

        if acc_50 > best_iou50:
            best_iou50 = acc_50
        save_checkpoint(model, optimizer, scheduler, epoch, acc_50, best_iou50,
                        "localizer", is_best=(acc_50 >= best_iou50))

    print(f"\nBest localization Acc@0.5: {best_iou50:.1f}%")
    return model


# ---------------------------------------------------------------------------
# Task 3: Segmentation
# ---------------------------------------------------------------------------

def train_segmentation(args):
    device = get_device()
    train_ds, val_ds = build_datasets(args.data_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda"))

    model = VGG11UNet(num_classes=3, dropout_p=args.dropout).to(device)

    if args.pretrained_encoder:
        load_pretrained_encoder(model, args.pretrained_encoder)
    freeze_encoder(model, args.freeze_mode)

    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Differential LR
    param_groups = get_param_groups(model, encoder_lr=args.lr * 0.1,
                                    head_lr=args.lr, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_dice = 0.0
    start_epoch = 1

    if args.resume:
        start_epoch, best_dice = load_checkpoint(model, optimizer, scheduler, "unet", device)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["trimap"].to(device)
            optimizer.zero_grad()
            logits = model(images)

            loss_ce = ce_loss(logits, masks)

            # Soft Dice loss
            probs = torch.softmax(logits, dim=1)
            loss_dice = 0.0
            for c in range(3):
                p = probs[:, c]
                t = (masks == c).float()
                inter = (p * t).sum()
                loss_dice += 1.0 - (2.0 * inter + 1.0) / (p.sum() + t.sum() + 1.0)
            loss_dice = loss_dice / 3.0

            loss = loss_ce + loss_dice
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_ds)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        dice_scores_list = []
        pixel_accs = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["trimap"].to(device)
                logits = model(images)
                val_loss += ce_loss(logits, masks).item() * images.size(0)
                preds = logits.argmax(dim=1)
                for i in range(preds.size(0)):
                    dice_scores_list.append(dice_score(preds[i], masks[i]).item())
                    pixel_accs.append(pixel_accuracy(preds[i], masks[i]))

        val_loss /= len(val_ds)
        mean_dice = np.mean(dice_scores_list)
        mean_pix = np.mean(pixel_accs)
        scheduler.step()

        wandb.log({"seg/train_loss": train_loss, "seg/val_loss": val_loss,
                    "seg/dice_score": mean_dice, "seg/pixel_accuracy": mean_pix,
                    "epoch": epoch})
        print(f"[SEG] Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  Dice={mean_dice:.4f}  PixAcc={mean_pix:.4f}")

        if mean_dice > best_dice:
            best_dice = mean_dice
        save_checkpoint(model, optimizer, scheduler, epoch, mean_dice, best_dice,
                        "unet", is_best=(mean_dice >= best_dice))

    print(f"\nBest segmentation Dice: {best_dice:.4f}")
    return model


# ---------------------------------------------------------------------------
# Task 4: Multi-task fine-tuning (aligns heads with shared backbone)
# ---------------------------------------------------------------------------

def train_multitask(args):
    '''Fine-tune the unified multi-task model for a few epochs.

    This is the critical step: after training individual models, their encoder
    weights have drifted apart.  Joint fine-tuning on all three tasks
    simultaneously aligns the heads with the averaged shared backbone.
    '''
    device = get_device()
    train_ds, val_ds = build_datasets(args.data_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda"))

    # Build multi-task model (without gdown - load locally)
    from models.vgg11 import VGG11Encoder
    from models.multitask import _average_state_dicts

    clf = VGG11Classifier(num_classes=37)
    clf_ckpt = torch.load("checkpoints/classifier.pth", map_location="cpu", weights_only=False)
    clf.load_state_dict(clf_ckpt.get("state_dict", clf_ckpt))

    loc = VGG11Localizer()
    loc_ckpt = torch.load("checkpoints/localizer.pth", map_location="cpu", weights_only=False)
    loc.load_state_dict(loc_ckpt.get("state_dict", loc_ckpt))

    seg = VGG11UNet(num_classes=3)
    seg_ckpt = torch.load("checkpoints/unet.pth", map_location="cpu", weights_only=False)
    seg.load_state_dict(seg_ckpt.get("state_dict", seg_ckpt))

    # Build a simple multi-task wrapper for training
    class _MTModel(nn.Module):
        def __init__(self):
            super().__init__()
            avg_sd = _average_state_dicts(
                clf.encoder.state_dict(),
                loc.encoder.state_dict(),
                seg.encoder.state_dict(),
            )
            self.encoder = VGG11Encoder()
            self.encoder.load_state_dict(avg_sd)

            self.cls_pool = nn.AdaptiveAvgPool2d((7, 7))
            self.cls_head = clf.classifier
            self.loc_pool = nn.AdaptiveAvgPool2d((7, 7))
            self.loc_head = loc.regressor

            self.seg_decoder = seg.decoder_levels
            self.seg_drop = seg.bottleneck_drop
            self.seg_head = seg.head

        def forward(self, x):
            bottleneck, skips = self.encoder(x, return_features=True)

            c = self.cls_pool(bottleneck).view(x.size(0), -1)
            cls_logits = self.cls_head(c)

            l = self.loc_pool(bottleneck).view(x.size(0), -1)
            bbox = self.loc_head(l).clamp(0, 224)

            h = bottleneck
            num_stages = self.encoder.NUM_STAGES
            for lvl_idx, dec_block in enumerate(self.seg_decoder):
                skip_key = f"block{num_stages - lvl_idx}"
                h = dec_block(h, skips[skip_key])
            h = self.seg_drop(h)
            seg_logits = self.seg_head(h)

            return cls_logits, bbox, seg_logits

    model = _MTModel().to(device)

    cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    mse_loss_fn = nn.MSELoss()
    iou_loss_fn = IoULoss(reduction="mean")
    seg_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"cls": 0, "loc": 0, "seg": 0}
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            bboxes = batch["bbox"].to(device)
            masks = batch["trimap"].to(device)
            has_bbox = torch.tensor(batch["has_bbox"], dtype=torch.bool, device=device)

            optimizer.zero_grad()
            cls_logits, pred_bbox, seg_logits = model(images)

            loss_cls = cls_criterion(cls_logits, labels)

            if has_bbox.sum() > 0:
                loss_loc = (mse_loss_fn(pred_bbox[has_bbox] / IMG_SIZE, bboxes[has_bbox] / IMG_SIZE)
                            + iou_loss_fn(pred_bbox[has_bbox], bboxes[has_bbox]))
            else:
                loss_loc = torch.tensor(0.0, device=device)

            loss_seg_ce = seg_ce(seg_logits, masks)
            probs = torch.softmax(seg_logits, dim=1)
            loss_dice = 0.0
            for c_i in range(3):
                p = probs[:, c_i]
                t = (masks == c_i).float()
                inter = (p * t).sum()
                loss_dice += 1.0 - (2.0 * inter + 1.0) / (p.sum() + t.sum() + 1.0)
            loss_seg = loss_seg_ce + loss_dice / 3.0

            # Weighted sum — emphasise localization since it's hardest
            loss = loss_cls + 2.0 * loss_loc + loss_seg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running["cls"] += loss_cls.item() * images.size(0)
            running["loc"] += loss_loc.item() * images.size(0)
            running["seg"] += loss_seg.item() * images.size(0)

        scheduler.step()

        # --- Validate ---
        model.eval()
        all_preds_cls, all_labels_cls = [], []
        ious = []
        dice_scores_list = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                bboxes = batch["bbox"].to(device)
                masks = batch["trimap"].to(device)
                has_bbox_v = batch["has_bbox"]

                cls_logits, pred_bbox, seg_logits = model(images)

                all_preds_cls.extend(cls_logits.argmax(1).cpu().numpy())
                all_labels_cls.extend(labels.cpu().numpy())

                for i in range(pred_bbox.size(0)):
                    if has_bbox_v[i]:
                        ious.append(compute_iou(pred_bbox[i].cpu().numpy(),
                                                bboxes[i].cpu().numpy()))

                seg_preds = seg_logits.argmax(dim=1)
                for i in range(seg_preds.size(0)):
                    dice_scores_list.append(dice_score(seg_preds[i], masks[i]).item())

        f1 = f1_score(all_labels_cls, all_preds_cls, average="macro")
        ious_arr = np.array(ious) if ious else np.array([0.0])
        acc50 = (ious_arr >= 0.5).mean() * 100
        acc75 = (ious_arr >= 0.75).mean() * 100
        m_dice = np.mean(dice_scores_list)

        wandb.log({
            "mt/cls_loss": running["cls"] / len(train_ds),
            "mt/loc_loss": running["loc"] / len(train_ds),
            "mt/seg_loss": running["seg"] / len(train_ds),
            "mt/f1": f1, "mt/acc@0.5": acc50, "mt/acc@0.75": acc75,
            "mt/dice": m_dice, "epoch": epoch,
        })
        print(f"[MT] Epoch {epoch}/{args.epochs}  F1={f1:.4f}  "
              f"Acc@0.5={acc50:.1f}%  Acc@0.75={acc75:.1f}%  Dice={m_dice:.4f}")

    # --- Save individual checkpoints back ---
    # Re-extract components so MultiTaskPerceptionModel can load them
    os.makedirs("checkpoints", exist_ok=True)

    # Classifier = encoder + cls head
    clf_new = VGG11Classifier(num_classes=37)
    clf_new.encoder.load_state_dict(model.encoder.state_dict())
    clf_new.classifier.load_state_dict(model.cls_head.state_dict())
    torch.save({"state_dict": clf_new.state_dict()}, "checkpoints/classifier.pth")

    # Localizer = encoder + loc head
    loc_new = VGG11Localizer()
    loc_new.encoder.load_state_dict(model.encoder.state_dict())
    loc_new.regressor.load_state_dict(model.loc_head.state_dict())
    torch.save({"state_dict": loc_new.state_dict()}, "checkpoints/localizer.pth")

    # U-Net = encoder + decoder
    seg_new = VGG11UNet(num_classes=3)
    seg_new.encoder.load_state_dict(model.encoder.state_dict())
    seg_new.decoder_levels.load_state_dict(model.seg_decoder.state_dict())
    seg_new.bottleneck_drop = model.seg_drop
    seg_new.head.load_state_dict(model.seg_head.state_dict())
    torch.save({"state_dict": seg_new.state_dict()}, "checkpoints/unet.pth")

    print("\n[DONE] Saved all three checkpoints with aligned shared backbone.")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DA6401 Assignment 2 Training")
    parser.add_argument("--task", type=str, required=True,
                        choices=["classification", "localization", "segmentation", "multitask"])
    parser.add_argument("--data-root", type=str, default="data/oxford-iiit-pet")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--project", type=str, default="da6401-assignment2")
    parser.add_argument("--pretrained-encoder", type=str, default=None)
    parser.add_argument("--freeze-mode", type=str, default="none",
                        choices=["none", "partial", "full_freeze"])
    parser.add_argument("--no-batchnorm", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint for this task.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    wandb.init(project=args.project,
               name=args.run_name or f"{args.task}-dp{args.dropout}-lr{args.lr}",
               config=vars(args))

    if args.task == "classification":
        train_classifier(args)
    elif args.task == "localization":
        train_localizer(args)
    elif args.task == "segmentation":
        train_segmentation(args)
    elif args.task == "multitask":
        train_multitask(args)

    wandb.finish()


if __name__ == "__main__":
    main()
