# DA6401 - Assignment 2

## VGG11 Visual Perception Pipeline

> **Roll No:** DA25M014 · **Course:** DA6401 Deep Learning · **IIT Madras**

---

### 🔗 Quick Links

|   |   |
|---|---|
| 📊 **W&B Report** | [View Report](https://api.wandb.ai/links/da25m014-iitm/ee73ad37) |
| 💻 **GitHub Repo** | [View Code](https://github.com/DA25M014/da6401_assignment_2) |

---

### 📌 About

A **complete multi-task visual perception pipeline built from scratch** on the Oxford-IIIT Pet Dataset (37 breeds). VGG11 architecture with custom BatchNorm and Dropout — no pretrained weights, no pre-built models.

| Task | Model | Key Metric |
|:---:|:---|:---|
| Classification | VGG11 + FC Head | Macro F1 = 0.70 |
| Localization | VGG11 + Regression Head | Acc@0.5 = 86%, Acc@0.75 = 60% |
| Segmentation | VGG11 U-Net | Mean Dice = 0.84, PixAcc = 0.90 |

---

### 🧪 W&B Experiments

| # | Section | Finding |
|:---:|:---|:---|
| 2.1 | BatchNorm Ablation | BN keeps activation std ~0.98, without BN collapses to 0.09 |
| 2.2 | Dropout Ablation | p=0.2 halves generalization gap while preserving F1 |
| 2.3 | Transfer Learning | Full fine-tune (Dice 0.83) > Partial (0.82) > Frozen (0.75) |
| 2.4 | Feature Maps | Edges at conv1 → semantic concepts at conv5 |
| 2.5 | Object Detection | Height underestimation in weakest cases |
| 2.6 | Segmentation Eval | Dice superior to PixAcc for imbalanced trimaps |
| 2.7 | Novel Images | Segmentation generalizes better than classification to OOD |
| 2.8 | Meta-Analysis | CE+Dice loss, separate encoders eliminate task interference |

---

### 📂 Project Structure

```
├── data/
│   └── pets_dataset.py          # Dataset class with transforms
├── models/
│   ├── vgg11.py                 # VGG11 encoder (from scratch)
│   ├── layers.py                # CustomDropout (inverted scaling)
│   ├── classification.py        # VGG11Classifier
│   ├── localization.py          # VGG11Localizer
│   ├── segmentation.py          # VGG11UNet
│   └── multitask.py             # MultiTaskPerceptionModel
├── losses/
│   └── iou_loss.py              # Custom IoU loss
├── train.py                     # Training script (all tasks)
├── inference.py                 # W&B visualization and inference
├── multitask.py                 # Gradescope entry point
└── requirements.txt
```

---

### 🚀 Usage

```bash
# Classification
python train.py --task classification --epochs 20 --lr 1e-3 --dropout 0.5

# Localization
python train.py --task localization --epochs 60 --lr 1e-4

# Segmentation (full fine-tune)
python train.py --task segmentation --epochs 60 --lr 1e-4 --freeze-mode none
```

---

### 📦 Dependencies

```
torch
torchvision
wandb
numpy
matplotlib
gdown
```

---

<div align="center">

**DA25M014** · M.Tech · IIT Madras

[GitHub](https://github.com/DA25M014/da6401_assignment_2) · [W&B Report](https://api.wandb.ai/links/da25m014-iitm/ee73ad37)

---

<sub>DA6401 · Deep Learning · IIT Madras · 2025–26</sub>

</div>
