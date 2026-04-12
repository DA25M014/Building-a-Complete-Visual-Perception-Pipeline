'''
Inference and W&B report visualisations.

Covers all W&B report sections:
    Q2.1 - activations: BN vs no-BN activation distributions overlaid
    Q2.4 - feature_maps: first and last conv layer feature maps  
    Q2.5 - detection_table: bbox with confidence + IoU + color-coded boxes
    Q2.6 - seg_samples: segmentation with Dice vs Pixel Accuracy analysis
    Q2.7 - novel: full pipeline on 3 internet images
'''

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb
from PIL import Image

from data.pets_dataset import (
    OxfordIIITPetDataset,
    get_val_transforms,
    IMG_SIZE, MEAN, STD,
)
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet


def get_device():
    # pick best available accelerator
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def denormalize(img_tensor):
    # convert normalised CHW tensor to HWC numpy uint8 for display
    img = img_tensor.cpu().clone()
    for c in range(3):
        img[c] = img[c] * STD[c] + MEAN[c]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


# list of breed names matching class indices 0-36
BREED_NAMES = [
    "Abyssinian", "American_Bulldog", "American_Pit_Bull_Terrier", "Basset_Hound",
    "Beagle", "Bengal", "Birman", "Bombay", "Boxer", "British_Shorthair",
    "Chihuahua", "Egyptian_Mau", "English_Cocker_Spaniel", "English_Setter",
    "German_Shorthaired", "Great_Pyrenees", "Havanese", "Japanese_Chin",
    "Keeshond", "Leonberger", "Maine_Coon", "Miniature_Pinscher", "Newfoundland",
    "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian_Blue",
    "Saint_Bernard", "Samoyed", "Scottish_Terrier", "Shiba_Inu",
    "Siamese", "Sphynx", "Staffordshire_Bull_Terrier", "Wheaten_Terrier",
    "Yorkshire_Terrier",
]


# -------------------------------------------------------
# Q2.1 - Activation distributions: BN vs no-BN overlaid
# -------------------------------------------------------

def log_activation_distributions(args):
    '''
    Pass same input through model WITH batchnorm and WITHOUT batchnorm.
    Extract activations from the 3rd conv layer and overlay histograms.
    '''
    device = get_device()
    val_ds = OxfordIIITPetDataset(args.data_root, split="trainval", transform=get_val_transforms())

    # load model with batchnorm
    model_bn = VGG11Classifier(num_classes=37).to(device)
    ckpt_bn = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_bn.load_state_dict(ckpt_bn.get("state_dict", ckpt_bn))
    model_bn.eval()

    # load model without batchnorm (if checkpoint provided)
    model_nobn = None
    if args.ckpt_nobn and os.path.exists(args.ckpt_nobn):
        model_nobn = VGG11Classifier(num_classes=37).to(device)
        # replace batchnorm with identity to match saved weights
        for name, module in model_nobn.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                parent = model_nobn
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
                if parts[-1].isdigit():
                    parent[int(parts[-1])] = nn.Identity()
                else:
                    setattr(parent, parts[-1], nn.Identity())
        ckpt_nobn = torch.load(args.ckpt_nobn, map_location=device, weights_only=False)
        model_nobn.load_state_dict(ckpt_nobn.get("state_dict", ckpt_nobn))
        model_nobn.eval()

    # hook to capture activations at 3rd conv layer
    activations = {}
    def make_hook(tag):
        def hook_fn(module, inp, out):
            activations[tag] = out.detach().cpu().numpy().flatten()
        return hook_fn

    # 3rd conv is the first conv in stage 3 (index 2)
    # stage3 = Sequential(Sequential(Conv,BN,ReLU), Sequential(Conv,BN,ReLU))
    # the Conv2d is at conv_stages[2][0][0]
    model_bn.encoder.conv_stages[2][0][0].register_forward_hook(make_hook("with_bn"))
    if model_nobn is not None:
        model_nobn.encoder.conv_stages[2][0][0].register_forward_hook(make_hook("without_bn"))

    # use same input for both
    sample_img = val_ds[0]["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        model_bn(sample_img)
        if model_nobn is not None:
            model_nobn(sample_img)

    # create overlaid histogram plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(activations["with_bn"], bins=100, alpha=0.6, label="With BatchNorm", color="blue", density=True)
    if "without_bn" in activations:
        ax.hist(activations["without_bn"], bins=100, alpha=0.6, label="Without BatchNorm", color="red", density=True)
    ax.set_title("Q2.1: Activation Distribution at 3rd Conv Layer")
    ax.set_xlabel("Activation Value")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()

    # log as wandb interactive plot via histogram
    wandb.log({"Q2.1/activation_overlay": wandb.Image(fig)})

    # also log raw histograms as wandb.Histogram for interactivity
    wandb.log({
        "Q2.1/hist_with_bn": wandb.Histogram(activations["with_bn"], num_bins=100),
    })
    if "without_bn" in activations:
        wandb.log({
            "Q2.1/hist_without_bn": wandb.Histogram(activations["without_bn"], num_bins=100),
        })

    plt.close(fig)
    print("Logged Q2.1 activation distributions to W&B.")


# -------------------------------------------------------
# Q2.4 - Feature map visualisation
# -------------------------------------------------------

def log_feature_maps(args):
    '''
    Pass a single dog image through the trained classifier.
    Visualise feature maps from first conv (low-level edges)
    and last conv (high-level semantic patterns like snouts/ears).
    '''
    device = get_device()
    val_ds = OxfordIIITPetDataset(args.data_root, split="trainval", transform=get_val_transforms())

    model = VGG11Classifier(num_classes=37).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()

    # hooks to capture feature maps
    activations = {}
    def make_hook(tag):
        def hook_fn(m, i, o):
            activations[tag] = o.detach().cpu()
        return hook_fn

    # first conv: conv_stages[0][0] (the Conv2d in stage 1)
    model.encoder.conv_stages[0][0].register_forward_hook(make_hook("first_conv"))

    # last conv: conv_stages[4] is stage5 = Sequential(cbr, cbr)
    # last Conv2d is conv_stages[4][1][0] (second cbr block, Conv2d)
    last_stage = model.encoder.conv_stages[4]
    # find the last Conv2d in the last stage
    last_conv_found = False
    for i in range(len(last_stage) - 1, -1, -1):
        sub = last_stage[i]
        if isinstance(sub, nn.Sequential):
            for j in range(len(sub) - 1, -1, -1):
                if isinstance(sub[j], nn.Conv2d):
                    sub[j].register_forward_hook(make_hook("last_conv"))
                    last_conv_found = True
                    break
            if last_conv_found:
                break
        elif isinstance(sub, nn.Conv2d):
            sub.register_forward_hook(make_hook("last_conv"))
            break

    # pick an image and run forward pass
    sample = val_ds[0]
    img_tensor = sample["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        model(img_tensor)

    # log the original input image
    img_np = denormalize(sample["image"])
    wandb.log({"Q2.4/input_image": wandb.Image(img_np, caption="Input Image")})

    # visualise and log feature maps for both layers
    for layer_name in ["first_conv", "last_conv"]:
        feat = activations[layer_name][0]  # shape [C, H, W]
        n_show = min(16, feat.size(0))

        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        fig.suptitle(f"Q2.4: Feature Maps - {layer_name} ({feat.size(0)} channels, showing first {n_show})")
        for i, ax in enumerate(axes.flat):
            if i < n_show:
                ax.imshow(feat[i].numpy(), cmap="viridis")
            ax.axis("off")
        plt.tight_layout()
        wandb.log({f"Q2.4/{layer_name}": wandb.Image(fig)})
        plt.close(fig)

    print("Logged Q2.4 feature maps to W&B.")


# -------------------------------------------------------
# Q2.5 - Detection table with confidence + IoU
# -------------------------------------------------------

def compute_iou_single(pred_box, gt_box, eps=1e-6):
    # compute IoU between two boxes in cxcywh format
    px1 = pred_box[0] - pred_box[2] / 2
    py1 = pred_box[1] - pred_box[3] / 2
    px2 = pred_box[0] + pred_box[2] / 2
    py2 = pred_box[1] + pred_box[3] / 2
    gx1 = gt_box[0] - gt_box[2] / 2
    gy1 = gt_box[1] - gt_box[3] / 2
    gx2 = gt_box[0] + gt_box[2] / 2
    gy2 = gt_box[1] + gt_box[3] / 2
    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_p = max(0, px2 - px1) * max(0, py2 - py1)
    area_g = max(0, gx2 - gx1) * max(0, gy2 - gy1)
    return inter / (area_p + area_g - inter + eps)


def log_detection_table(args):
    '''
    Log wandb table with 15 val images showing:
    - image with GT box (green) and predicted box (red) overlaid
    - confidence score (max softmax of a dummy classifier or regression confidence)
    - IoU value
    - color coding: green=GT, red=prediction
    Identifies failure cases (high confidence + low IoU).
    '''
    device = get_device()
    val_ds = OxfordIIITPetDataset(args.data_root, split="trainval", transform=get_val_transforms())

    # load localizer
    model = VGG11Localizer().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()

    # also load classifier for confidence scores
    clf = None
    clf_path = args.ckpt.replace("localizer", "classifier")
    if os.path.exists(clf_path):
        clf = VGG11Classifier(num_classes=37).to(device)
        clf_ckpt = torch.load(clf_path, map_location=device, weights_only=False)
        clf.load_state_dict(clf_ckpt.get("state_dict", clf_ckpt))
        clf.eval()

    # create wandb table with all required columns
    columns = ["Image", "Confidence", "IoU", "Pred Box (cxcywh)", "GT Box (cxcywh)", "Analysis"]
    table = wandb.Table(columns=columns)

    n_samples = min(15, len(val_ds))
    for idx in range(n_samples):
        sample = val_ds[idx]
        if not sample["has_bbox"]:
            continue

        img_tensor = sample["image"].unsqueeze(0).to(device)
        gt_box = sample["bbox"].numpy()

        with torch.no_grad():
            pred_box = model(img_tensor)[0].cpu().numpy()
            # get confidence from classifier if available
            if clf is not None:
                cls_logits = clf(img_tensor)
                probs = torch.softmax(cls_logits, dim=1)
                confidence = probs.max().item()
            else:
                # use 1-IoU_loss as a proxy confidence
                iou_val = compute_iou_single(pred_box, gt_box)
                confidence = iou_val

        iou = compute_iou_single(pred_box, gt_box)

        # draw the image with both boxes
        img_np = denormalize(sample["image"])
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(img_np)

        # ground truth box in GREEN
        gx1 = gt_box[0] - gt_box[2] / 2
        gy1 = gt_box[1] - gt_box[3] / 2
        rect_gt = patches.Rectangle(
            (gx1, gy1), gt_box[2], gt_box[3],
            linewidth=2, edgecolor="green", facecolor="none", label="GT"
        )
        ax.add_patch(rect_gt)

        # predicted box in RED
        px1 = pred_box[0] - pred_box[2] / 2
        py1 = pred_box[1] - pred_box[3] / 2
        rect_pred = patches.Rectangle(
            (px1, py1), pred_box[2], pred_box[3],
            linewidth=2, edgecolor="red", facecolor="none", label="Pred"
        )
        ax.add_patch(rect_pred)

        ax.set_title(f"IoU: {iou:.3f} | Conf: {confidence:.3f}")
        ax.legend(loc="upper right")
        ax.axis("off")
        plt.tight_layout()

        # analyse the result
        if iou >= 0.75:
            analysis = "Good detection"
        elif iou >= 0.5:
            analysis = "Acceptable detection"
        elif confidence > 0.5 and iou < 0.3:
            analysis = "FAILURE: high confidence but low IoU"
        else:
            analysis = "Poor detection"

        table.add_data(
            wandb.Image(fig),
            round(confidence, 4),
            round(iou, 4),
            str(np.round(pred_box, 1)),
            str(np.round(gt_box, 1)),
            analysis,
        )
        plt.close(fig)

    wandb.log({"Q2.5/detection_results": table})
    print("Logged Q2.5 detection table to W&B.")


# -------------------------------------------------------
# Q2.6 - Segmentation: Dice vs Pixel Accuracy
# -------------------------------------------------------

def dice_score_per_image(pred, target, num_classes=3, eps=1e-6):
    # compute mean dice across classes for a single image
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    total_dice = 0.0
    for c in range(num_classes):
        p = (pred_flat == c).float()
        t = (target_flat == c).float()
        inter = (p * t).sum()
        total_dice += (2 * inter + eps) / (p.sum() + t.sum() + eps)
    return total_dice / num_classes


def pixel_accuracy_per_image(pred, target):
    return (pred == target).float().mean().item()


def log_seg_samples(args):
    '''
    Log 5 segmentation samples showing original, GT trimap, predicted trimap.
    Track both Pixel Accuracy and Dice Score.
    Include mathematical explanation of why Dice > Pixel Accuracy for imbalanced data.
    '''
    device = get_device()
    val_ds = OxfordIIITPetDataset(args.data_root, split="trainval", transform=get_val_transforms())

    model = VGG11UNet(num_classes=3).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()

    # table with all required columns
    columns = ["Original", "GT Trimap", "Predicted Trimap", "Dice Score", "Pixel Accuracy"]
    table = wandb.Table(columns=columns)

    # color map for trimaps: 0=foreground(yellow), 1=background(blue), 2=boundary(red)
    trimap_cmap = plt.cm.get_cmap("tab10", 3)

    for idx in range(min(5, len(val_ds))):
        sample = val_ds[idx]
        img_tensor = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["trimap"]

        with torch.no_grad():
            logits = model(img_tensor)
            pred_mask = logits.argmax(dim=1)[0].cpu()

        # compute metrics
        dice = dice_score_per_image(pred_mask, gt_mask).item()
        pix_acc = pixel_accuracy_per_image(pred_mask, gt_mask)

        # create side-by-side visualisation
        img_np = denormalize(sample["image"])

        # original image
        fig_orig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img_np)
        ax.set_title("Original")
        ax.axis("off")
        plt.tight_layout()

        # gt trimap
        fig_gt, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(gt_mask.numpy(), cmap=trimap_cmap, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title("GT Trimap")
        ax.axis("off")
        plt.tight_layout()

        # predicted trimap
        fig_pred, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(pred_mask.numpy(), cmap=trimap_cmap, vmin=0, vmax=2, interpolation="nearest")
        ax.set_title(f"Predicted (Dice={dice:.3f})")
        ax.axis("off")
        plt.tight_layout()

        table.add_data(
            wandb.Image(fig_orig),
            wandb.Image(fig_gt),
            wandb.Image(fig_pred),
            round(dice, 4),
            round(pix_acc, 4),
        )
        plt.close(fig_orig)
        plt.close(fig_gt)
        plt.close(fig_pred)

    wandb.log({"Q2.6/segmentation_samples": table})

    # Dice vs Pixel Accuracy explanation
    explanation = (
        "Pixel Accuracy = (correctly classified pixels) / (total pixels). "
        "In trimaps, background class dominates (often >70% of pixels). "
        "A model predicting ALL pixels as background gets ~70% pixel accuracy "
        "but 0% on foreground. "
        "Dice = 2*|P intersect G| / (|P| + |G|) computed per-class then averaged. "
        "It treats each class equally regardless of pixel count. "
        "A model that ignores the minority foreground class gets Dice ~ 0.33 "
        "(only background class correct) vs Pixel Accuracy ~ 0.7. "
        "This is why Pixel Accuracy appears artificially high in early epochs "
        "while Dice stays low - the model has not yet learned the minority classes."
    )
    wandb.log({"Q2.6/dice_vs_pixacc_explanation": wandb.Html(f"<p>{explanation}</p>")})

    print("Logged Q2.6 segmentation samples to W&B.")


# -------------------------------------------------------
# Q2.7 - Novel images pipeline
# -------------------------------------------------------

def log_novel_pipeline(args):
    '''
    Run the full pipeline (classify + localise + segment) on 3 novel
    pet images downloaded from the internet. Log results to W&B.
    '''
    device = get_device()

    clf_path = os.path.join(args.ckpt_dir, "classifier.pth")
    loc_path = os.path.join(args.ckpt_dir, "localizer.pth")
    unet_path = os.path.join(args.ckpt_dir, "unet.pth")

    # load all three models
    classifier = VGG11Classifier(num_classes=37).to(device)
    clf_ckpt = torch.load(clf_path, map_location=device, weights_only=False)
    classifier.load_state_dict(clf_ckpt.get("state_dict", clf_ckpt))
    classifier.eval()

    localizer = VGG11Localizer().to(device)
    loc_ckpt = torch.load(loc_path, map_location=device, weights_only=False)
    localizer.load_state_dict(loc_ckpt.get("state_dict", loc_ckpt))
    localizer.eval()

    segmenter = VGG11UNet(num_classes=3).to(device)
    seg_ckpt = torch.load(unet_path, map_location=device, weights_only=False)
    segmenter.load_state_dict(seg_ckpt.get("state_dict", seg_ckpt))
    segmenter.eval()

    # set up transform for novel images
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    transform = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

    novel_dir = args.novel_dir
    if not os.path.isdir(novel_dir):
        print(f"Novel image directory '{novel_dir}' not found. Skipping Q2.7.")
        return

    columns = ["Pipeline Output", "Predicted Breed", "Confidence", "Bbox (cxcywh)", "Segmentation Mask"]
    table = wandb.Table(columns=columns)

    for fname in sorted(os.listdir(novel_dir))[:3]:
        fpath = os.path.join(novel_dir, fname)
        # skip non-image files
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        image = np.array(Image.open(fpath).convert("RGB"))
        transformed = transform(image=image)
        img_tensor = transformed["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            cls_logits = classifier(img_tensor)
            bbox = localizer(img_tensor)[0].cpu().numpy()
            seg_logits = segmenter(img_tensor)
            seg_mask = seg_logits.argmax(dim=1)[0].cpu().numpy()

        # get breed prediction and confidence
        probs = torch.softmax(cls_logits, dim=1)
        conf_val, pred_cls = probs.max(dim=1)
        cls_idx = pred_cls.item()
        breed = BREED_NAMES[cls_idx] if cls_idx < len(BREED_NAMES) else str(cls_idx)
        confidence = conf_val.item()

        # create 3-panel figure: classification + bbox + segmentation
        img_show = denormalize(transformed["image"])
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # panel 1: classification result
        axes[0].imshow(img_show)
        axes[0].set_title(f"{breed} ({confidence:.1%})")
        axes[0].axis("off")

        # panel 2: bounding box overlay
        axes[1].imshow(img_show)
        px1 = bbox[0] - bbox[2] / 2
        py1 = bbox[1] - bbox[3] / 2
        rect = patches.Rectangle(
            (px1, py1), bbox[2], bbox[3],
            linewidth=2, edgecolor="red", facecolor="none"
        )
        axes[1].add_patch(rect)
        axes[1].set_title("Bounding Box")
        axes[1].axis("off")

        # panel 3: segmentation mask
        axes[2].imshow(seg_mask, cmap="tab10", vmin=0, vmax=2, interpolation="nearest")
        axes[2].set_title("Segmentation Mask")
        axes[2].axis("off")

        plt.tight_layout()

        # separate seg mask figure for the table
        fig_seg, ax_seg = plt.subplots(figsize=(4, 4))
        ax_seg.imshow(seg_mask, cmap="tab10", vmin=0, vmax=2, interpolation="nearest")
        ax_seg.axis("off")
        plt.tight_layout()

        table.add_data(
            wandb.Image(fig),
            breed,
            round(confidence, 4),
            str(np.round(bbox, 1)),
            wandb.Image(fig_seg),
        )
        plt.close(fig)
        plt.close(fig_seg)

    wandb.log({"Q2.7/novel_pipeline": table})
    print("Logged Q2.7 novel pipeline results to W&B.")


# -------------------------------------------------------
# Main entry point
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inference and W&B visualisations")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["activations", "feature_maps", "detection_table",
                                 "seg_samples", "novel"])
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to model checkpoint.")
    parser.add_argument("--ckpt-nobn", type=str, default=None,
                        help="Path to no-batchnorm classifier checkpoint (for Q2.1).")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints",
                        help="Directory with all checkpoints (for novel mode).")
    parser.add_argument("--data-root", type=str, default="data/oxford-iiit-pet")
    parser.add_argument("--novel-dir", type=str, default="novel_images")
    parser.add_argument("--project", type=str, default="da6401-assignment2")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    wandb.init(project=args.project, name=args.run_name or f"inference-{args.mode}")

    if args.mode == "activations":
        log_activation_distributions(args)
    elif args.mode == "feature_maps":
        log_feature_maps(args)
    elif args.mode == "detection_table":
        log_detection_table(args)
    elif args.mode == "seg_samples":
        log_seg_samples(args)
    elif args.mode == "novel":
        log_novel_pipeline(args)

    wandb.finish()


if __name__ == "__main__":
    main()
