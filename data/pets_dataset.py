'''Dataset for Oxford-IIIT Pet - returns image, class label, bbox, and trimap.

Annotation files used:
    annotations/list.txt      - image name, class-id (1-37), species, breed
    annotations/xmls/         - PASCAL VOC XML with head bounding box
    annotations/trimaps/      - PNG segmentation masks (1=fg, 2=bg, 3=boundary)

Bounding boxes are converted to (x_center, y_center, w, h) in the *resized*
224x224 pixel space.
'''

import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMG_SIZE = 224

# ImageNet statistics for normalisation
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-15, 15),
                     border_mode=0, p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.5),
            A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 32),
                            hole_width_range=(16, 32), p=0.3),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_labels"],
                                 min_area=100, min_visibility=0.3),
        additional_targets={"mask": "mask"},
    )


def get_val_transforms():
    return A.Compose(
        [
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_labels"],
                                 min_area=1, min_visibility=0.1),
        additional_targets={"mask": "mask"},
    )


def _parse_bbox_xml(xml_path: str):
    '''Return (xmin, ymin, xmax, ymax) from PASCAL VOC XML or None.'''
    if not os.path.exists(xml_path):
        return None
    tree = ET.parse(xml_path)
    root = tree.getroot()
    obj = root.find("object")
    if obj is None:
        return None
    bndbox = obj.find("bndbox")
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)
    return [xmin, ymin, xmax, ymax]


def _xyxy_to_cxcywh(box, img_w, img_h):
    '''Convert (xmin,ymin,xmax,ymax) in original coords to
    (cx,cy,w,h) in resized IMG_SIZE coords.'''
    xmin, ymin, xmax, ymax = box
    # Scale to resized image
    sx = IMG_SIZE / img_w
    sy = IMG_SIZE / img_h
    xmin_r = xmin * sx
    ymin_r = ymin * sy
    xmax_r = xmax * sx
    ymax_r = ymax * sy
    cx = (xmin_r + xmax_r) / 2.0
    cy = (ymin_r + ymax_r) / 2.0
    w = xmax_r - xmin_r
    h = ymax_r - ymin_r
    return [cx, cy, w, h]


class OxfordIIITPetDataset(Dataset):
    '''Oxford-IIIT Pet multi-task dataset loader.

    Args:
        root: Path to dataset root containing 'images/' and 'annotations/'.
        split: 'trainval' or 'test'.
        transform: albumentations Compose (use helpers above).
    '''

    def __init__(self, root: str, split: str = "trainval", transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.img_dir = os.path.join(root, "images")
        self.trimap_dir = os.path.join(root, "annotations", "trimaps")
        self.xml_dir = os.path.join(root, "annotations", "xmls")

        # Parse list file
        list_file = os.path.join(root, "annotations", "list.txt")
        self.samples = []  # (image_name, class_id_0indexed)
        with open(list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                img_name = parts[0]
                class_id = int(parts[1]) - 1  # 1-indexed -> 0-indexed
                self.samples.append((img_name, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, class_id = self.samples[idx]

        # --- Image ---
        img_path = os.path.join(self.img_dir, img_name + ".jpg")
        image = np.array(Image.open(img_path).convert("RGB"))
        orig_h, orig_w = image.shape[:2]

        # --- Trimap mask ---
        trimap_path = os.path.join(self.trimap_dir, img_name + ".png")
        if os.path.exists(trimap_path):
            trimap = np.array(Image.open(trimap_path))
            # Original trimap: 1=foreground, 2=background, 3=boundary
            # Map to 0-indexed: 0=foreground, 1=background, 2=boundary
            trimap = trimap.astype(np.int64) - 1
            trimap = np.clip(trimap, 0, 2)
        else:
            trimap = np.zeros((orig_h, orig_w), dtype=np.int64)

        # --- Bounding box ---
        xml_path = os.path.join(self.xml_dir, img_name + ".xml")
        bbox_raw = _parse_bbox_xml(xml_path)
        has_bbox = bbox_raw is not None

        if has_bbox:
            # Clip to image bounds
            xmin = max(0, min(bbox_raw[0], orig_w - 1))
            ymin = max(0, min(bbox_raw[1], orig_h - 1))
            xmax = max(0, min(bbox_raw[2], orig_w))
            ymax = max(0, min(bbox_raw[3], orig_h))
            bbox_voc = [xmin, ymin, xmax, ymax]
        else:
            bbox_voc = [0, 0, orig_w, orig_h]  # dummy full image

        # --- Apply transforms ---
        if self.transform is not None:
            transformed = self.transform(
                image=image,
                mask=trimap,
                bboxes=[bbox_voc],
                bbox_labels=[class_id],
            )
            image = transformed["image"]  # [C, H, W] float tensor
            trimap = torch.tensor(transformed["mask"], dtype=torch.long)

            if len(transformed["bboxes"]) > 0:
                tb = transformed["bboxes"][0]  # (xmin, ymin, xmax, ymax) in resized space
                cx = (tb[0] + tb[2]) / 2.0
                cy = (tb[1] + tb[3]) / 2.0
                w = tb[2] - tb[0]
                h = tb[3] - tb[1]
                bbox = torch.tensor([cx, cy, w, h], dtype=torch.float32)
            else:
                # bbox got removed by augmentation - mark as invalid
                has_bbox = False
                bbox = torch.tensor(
                    [IMG_SIZE / 2, IMG_SIZE / 2, IMG_SIZE, IMG_SIZE], dtype=torch.float32
                )
        else:
            # Manual resize fallback
            image = np.array(Image.fromarray(image).resize((IMG_SIZE, IMG_SIZE)))
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            trimap = torch.tensor(
                np.array(Image.fromarray(trimap.astype(np.uint8)).resize(
                    (IMG_SIZE, IMG_SIZE), Image.NEAREST
                )),
                dtype=torch.long,
            )
            bbox = torch.tensor(
                _xyxy_to_cxcywh(bbox_voc, orig_w, orig_h), dtype=torch.float32
            )

        label = torch.tensor(class_id, dtype=torch.long)

        return {
            "image": image,
            "label": label,
            "bbox": bbox,
            "trimap": trimap,
            "has_bbox": has_bbox,
            "img_name": img_name,
        }
