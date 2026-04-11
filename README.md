**Wandb Report Link:** https://api.wandb.ai/links/agrawalritik2001-/agzg27at  
**Github Repo Link:** https://github.com/RitikAgrawal01/da6401_assignment_2  
**Wandb  Project Link:** https://wandb.ai/agrawalritik2001-/da6401-assignment2

**Name:** Ritik Agrawal  
**Roll No:** DA25M026

# 🐾 Multi-Task Pet Pipeline: Classification, Localization & Segmentation

This project implements a unified **multi-task deep learning architecture** using a shared VGG11 encoder to simultaneously perform breed classification, head localization (bounding boxes), and pixel-level segmentation on the Oxford-IIIT Pet Dataset.

---

## 📌 Project Highlights

- ✅ **Unified Multi-Task Architecture:** Single shared encoder with three specialized task heads.
- ✅ **Extensive Experimentation:** - Impact of **Batch Normalization** on convergence.
  - **Dropout Ablation** studies ($p=0.0$ to $0.5$).
  - **Transfer Learning** strategies (Frozen vs. Fine-tuned).
- ✅ **Visual Analysis:** Deep dive into feature maps, detection overlays, and segmentation masks.
- ✅ **Real-World Evaluation:** End-to-end testing on "in-the-wild" images.
- ✅ **Experiment Tracking:** Fully integrated with **Weights & Biases (W&B)**.

---

## 🏗️ Architecture Overview

The model utilizes a **VGG11 backbone** as a shared feature extractor, branching into three distinct heads:

1.  **Classification Head:** Fully connected layers + Dropout → Breed Prediction (37 classes).
2.  **Localization Head:** Regression head → Bounding Box ($x, y, w, h$).
3.  **Segmentation Head:** U-Net style decoder → Trimap Mask (3 classes)

---

## 📊 Dataset: Oxford-IIIT Pet
- **Scope:** 37 pet breeds (cats & dogs).
- **Annotations:** - Pixel-level segmentation masks (trimaps).
  - Bounding box annotations (head region).

---

## ⚙️ Training & Experiments

### 🔹 Task 2.1 — Batch Normalization Analysis
Compared models with and without BatchNorm. Findings showed that BatchNorm significantly improved training stability and gradient flow, allowing for higher learning rates and faster convergence.

### 🔹 Task 2.2 — Dropout Ablation
Tested dropout rates $p \in \{0.0, 0.2, 0.5\}$. We observed that $p=0.2$ provided the optimal balance between training accuracy and validation generalization.

### 🔹 Task 2.3 — Transfer Learning
| Strategy | Description | Result |
| :--- | :--- | :--- |
| **Frozen Encoder** | Weights locked; only heads trained. | Limited performance. |
| **Partial Fine-tuning** | Last few layers of VGG11 unfrozen. | Good trade-off. |
| **Full Fine-tuning** | All weights updated. | **Best overall performance.** |

---

## 📈 Evaluation Metrics

| Task | Metrics | Final Score |
| :--- | :--- | :--- |
| **Classification** | Accuracy / Macro F1 | **1.00 / 1.00** |
| **Localization** | Mean IoU | **0.79** |
| **Segmentation** | Mean Dice Score | **0.86** |
| **Segmentation** | Pixel Accuracy | **0.93** |

> **Note:** While Pixel Accuracy is high, the **Dice Score** proved more reliable due to the class imbalance inherent in segmentation trimaps.

---

## 🔍 Key Insights

- **Feature Maps:** Early layers capture edges/textures; deeper layers represent semantic features like ears and snouts.
- **Localization Challenges:** Model performance dips in cluttered backgrounds or unusual animal poses.
- **Generalization:** The pipeline is robust for clear subjects in "in-the-wild" images, though localization is more sensitive to environment noise than classification.

---

## 🛠️ Tech Stack

- **Framework:** PyTorch & Torchvision
- **Tracking:** Weights & Biases (W&B)
- **Environment:** Kaggle GPU / Jupyter
- **Libraries:** NumPy, Matplotlib, OpenCV

---

## 📦 Project Structure

```text
├── models/
│   ├── vgg11.py        # Backbone definitions
│   ├── multitask.py    # Multi-head logic
│   └── unet.py         # Segmentation decoder
├── datasets/
│   └── oxford_pet.py   # Custom Dataset & Transforms
├── wandb_exp.ipynb
├── checkpoints/        # Saved model weights
├── utils/
│   └── helpers.py      # IoU & Dice calculation logic
└── README.md
```
