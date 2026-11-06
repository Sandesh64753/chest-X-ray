# ChestVision AI

**ChestVision AI** is an end-to-end, explainable deep learning platform for automated chest radiograph analysis. It integrates a **DenseNet121 multi-class classifier**, a **custom U-Net lung segmenter**, and **Gradient-weighted Class Activation Mapping (Grad-CAM)** to deliver transparent, reproducible, and verifiable clinical insights.

---

## Table of Contents

- [Overview](#overview)
- [Dataset & Data Pipeline](#dataset--data-pipeline)
- [Model Architectures & Training](#model-architectures--training)
  - [1. DenseNet121 Classifier (Transfer Learning)](#1-densenet121-classifier-transfer-learning)
  - [2. U-Net Lung Segmenter (Trained from Scratch)](#2-u-net-lung-segmenter-trained-from-scratch)
  - [3. Explainability (Grad-CAM & Spatial Localization)](#3-explainability-grad-cam--spatial-localization)
  - [4. Longitudinal Comparative Analysis](#4-longitudinal-comparative-analysis)
- [Model Evaluation & Benchmarks](#model-evaluation--benchmarks)
  - [Classification Performance](#classification-performance)
  - [Segmentation Performance](#segmentation-performance)
- [How It Works (Inference Pipeline)](#how-it-works-inference-pipeline)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
  - [Docker Compose (Recommended)](#docker-compose-recommended)
  - [Local Development Setup](#local-development-setup)
- [API Reference](#api-reference)
- [Medical Disclaimer](#medical-disclaimer)

---

## Overview

Chest radiography is the most widely performed diagnostic imaging examination globally. ChestVision AI provides an explainable AI pipeline capable of:
1. **Multi-class Pathological Classification:** Detecting COVID-19, Lung Opacity, Normal, and Viral Pneumonia with calibrated class confidence probabilities.
2. **Lung Field Segmentation:** Delineating precise anatomical lung boundaries to quantify aerated lung surface area and assist radiologic assessment.
3. **Visual Attribution (Grad-CAM):** Highlighting the specific spatial regions and features that guided the neural network's classification decisions.
4. **Comparative Analysis:** Quantifying longitudinal differences between two scans of a patient (class transitions, probability shifts, lung volume changes, and attention overlap).

---

## Dataset & Data Pipeline

The models are trained, validated, and evaluated on the peer-reviewed **COVID-19 Radiography Database** (Kaggle: `tawsifurrahman/covid19-radiography-database`).

### Dataset Statistics
- **Total Images & Ground-Truth Masks:** 21,165 paired 256x256 / 224x224 radiographs
- **Class Breakdown:**
  - **Normal:** 10,192 images (48.15%)
  - **Lung Opacity:** 6,012 images (28.41%)
  - **COVID-19:** 3,616 images (17.08%)
  - **Viral Pneumonia:** 1,345 images (6.36%)

### Stratified Data Split (Seed = 42)
To guarantee strict independence and prevent data leakage:
- **Train Set (70%):** 14,815 scans
- **Validation Set (15%):** 3,175 scans
- **Test Set (15%):** 3,175 scans (held-out benchmark)

### Preprocessing & Data Augmentation
- **Resolution:** Resized to standard $224 \times 224 \times 3$ resolution.
- **Normalization:** 
  - Classifier inputs are normalized via DenseNet-specific ImageNet zero-centering (`tf.keras.applications.densenet.preprocess_input`).
  - Segmenter inputs are scaled to $[0.0, 1.0]$.
- **Data Augmentation:**
  - Anatomically sound horizontal flipping ($L \leftrightarrow R$ mirroring). Vertical flipping is explicitly disabled because chest radiographs maintain a strict upright anatomical orientation.
  - Random contrast adjustments ($0.9$ to $1.1$) and brightness fluctuations ($\pm 0.1$).
  - Shared stateless seed transform applied in lockstep to both X-ray and mask to preserve strict spatial coordinate alignment.
- **Class Balancing:** Computed inverse frequency class weights to address representation imbalance across categories.

---

## Model Architectures & Training

### 1. DenseNet121 Classifier (Transfer Learning)

DenseNet121 (Densely Connected Convolutional Networks) connects all layers directly with each other, maximizing feature reuse, mitigating the vanishing-gradient problem, and retaining low-level edge features alongside high-level semantic representations.

- **Input Dimension:** $224 \times 224 \times 3$
- **Backbone:** ImageNet pre-trained DenseNet121 (427 layers total, output shape $7 \times 7 \times 1024$)
- **Custom Classification Head:**
  $$\text{Input}(224, 224, 3) \to \text{DenseNet121} \to \text{GlobalAveragePooling2D} \to \text{BatchNorm} \to \text{Dropout}(0.4) \to \text{Dense}(256, \text{ReLU}, L_2=10^{-4}) \to \text{Dropout}(0.3) \to \text{Dense}(4, \text{Softmax})$$

#### Two-Stage Training Protocol
1. **Stage 1 (Feature Extraction):**
   - Backbone is completely frozen (`trainable = False`).
   - Only the custom classification head is trained for 15 epochs.
   - Optimizer: Adam ($\eta = 10^{-3}$).
   - Loss: Weighted Categorical Cross-Entropy.
2. **Stage 2 (Fine-Tuning):**
   - Backbone is partially unfrozen: the bottom 70% of layers remain frozen, while the top 30% (last 129 layers) are unfrozen.
   - All `BatchNormalization` layers remain strictly frozen throughout fine-tuning to protect pre-trained ImageNet mean/variance statistics from degrading on medical image distributions.
   - Trained for 20 epochs with a low learning rate: Adam ($\eta = 10^{-5}$).
   - Callbacks: `ModelCheckpoint` (saving best `val_loss`), `EarlyStopping` (patience: 6), and `ReduceLROnPlateau` (factor: 0.5, patience: 3).

---

### 2. U-Net Lung Segmenter (Trained from Scratch)

A custom, symmetrical 4-level U-Net convolutional network specifically designed to extract precise lung parenchymal masks.

- **Architecture Details:**
  - **Contracting Path (Encoder):** 4 downsampling stages. Each stage consists of two $3 \times 3$ Conv2D layers (padding: `same`), followed by Batch Normalization, ReLU activation, and a $2 \times 2$ Max Pooling layer. Filters expand progressively: $32 \to 64 \to 128 \to 256$.
  - **Bottleneck:** Dual $3 \times 3$ convolutions with 512 filters + Batch Normalization + ReLU.
  - **Expansive Path (Decoder):** 4 upsampling stages using $2 \times 2$ Transposed Convolutions (strides: 2) concatenated with skip-connection feature maps from the corresponding encoder level, followed by dual $3 \times 3$ convolutions ($256 \to 128 \to 64 \to 32$ filters).
  - **Output Layer:** $1 \times 1$ Conv2D with **Sigmoid** activation to output continuous per-pixel lung probabilities $[0.0, 1.0]$.
- **Compound Objective Function (BCE + Dice Loss):**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{Dice}}$$
  $$\text{Dice}(Y, \hat{Y}) = \frac{2 |Y \cap \hat{Y}| + \epsilon}{|Y| + |\hat{Y}| + \epsilon}, \quad \mathcal{L}_{\text{Dice}} = 1 - \text{Dice}$$
  *Why BCE + Dice?* Binary Cross-Entropy regulates local pixel-level classification, while Dice loss directly optimizes macro spatial overlap and counters the severe background-to-lung pixel imbalance.
- **Training Parameters:** 40 epochs from scratch, Adam optimizer ($\eta = 10^{-4}$), monitoring validation Dice coefficient.

---

### 3. Explainability (Grad-CAM & Spatial Localization)

To avoid black-box predictions, ChestVision AI computes **Gradient-weighted Class Activation Mapping (Grad-CAM)**:

1. **Feature Extraction:** Extracts activation feature maps $A^k$ from the final convolutional block of DenseNet121 (`conv5_block16_concat`, dimension $7 \times 7 \times 1024$).
2. **Gradient Computation:** Using `tf.GradientTape`, computes the gradient of the predicted class score $y^c$ with respect to each feature activation map:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
3. **Linear Combination & ReLU:**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
4. **Colorization & Anatomical Mapping:**
   - Normalizes the activation matrix to $[0, 1]$ and maps it to a JET colormap overlaid onto the original radiograph.
   - Calculates the centroid $(\bar{x}, \bar{y})$ of high-activation pixels ($\ge 0.6$ threshold) and translates it into an intuitive spatial description (e.g., *upper-left lung field*, *central lung field*, *lower-right lung field*).

---

### 4. Longitudinal Comparative Analysis

When tracking a patient across two sequential scans, the comparative module performs differential analysis:
- **Class State Changes:** Flags categorical transitions (e.g., `Normal` $\to$ `COVID-19`).
- **Confidence Shifts:** Computes the delta in prediction confidence ($\Delta = \text{conf}_2 - \text{conf}_1$).
- **Per-Class Probability Deltas:** Tracks changes across all four diagnostic categories.
- **Aerated Lung Area Difference:** Calculates the difference in segmented lung surface percentage ($\Delta\% = \text{mask}_2\% - \text{mask}_1\%$).
- **Attention Overlap (IoU):** Evaluates the spatial Intersection-over-Union of Grad-CAM attention regions ($\ge 0.6$ activation) to measure shift in model focus.

---

## Model Evaluation & Benchmarks

The models were evaluated on the held-out test set consisting of **3,175 radiographs** never seen during training or validation.

### Classification Performance

- **Overall Test Accuracy:** **93.54%**
- **Macro F1-Score:** **94.06%**
- **Weighted F1-Score:** **93.53%**
- **Macro Precision:** **94.73%**
- **Macro Recall:** **93.44%**
- **Macro ROC-AUC:** **0.9917**

#### Detailed Per-Class Metrics Table

| Diagnosis Class | Precision | Recall | F1-Score | ROC-AUC | Test Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **COVID-19** | **96.45%** | **95.20%** | **95.82%** | **0.9979** | 542 |
| **Lung Opacity** | **93.25%** | **88.80%** | **90.97%** | **0.9855** | 902 |
| **Normal** | **92.30%** | **95.68%** | **93.96%** | **0.9837** | 1,529 |
| **Viral Pneumonia** | **96.94%** | **94.06%** | **95.48%** | **0.9995** | 202 |
| **Macro Average** | **94.73%** | **93.44%** | **94.06%** | **0.9917** | **3,175** |
| **Weighted Average** | **93.57%** | **93.54%** | **93.53%** | **—** | **3,175** |

---

### Segmentation Performance

Evaluated against ground-truth radiologist lung annotations across 3,175 test masks:

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Mean Dice Coefficient (DSC)** | **98.25% $\pm$ 1.64%** | Overlap fidelity between predicted and ground-truth lung masks |
| **Mean Intersection over Union (IoU)** | **97.05% $\pm$ 2.78%** | Jaccard index measuring region similarity |
| **Pixel-wise Binary Accuracy** | **99.33%** | Correct classification across all pixel coordinates |
| **Test BCE + Dice Loss** | **0.0352** | Combined objective convergence on test set |

---

## How It Works (Inference Pipeline)

```
[User Radiograph / Demo Image]
            │
            ▼
┌───────────────────────────────┐
│     FastAPI Image Ingestion   │
│  - Validate MIME & Extension  │
│  - Stream buffer to PIL Image │
└──────────────┬────────────────┘
               │
               ▼
┌───────────────────────────────┐
│        Preprocessing          │
│  - Resize to 224x224 RGB      │
│  - DenseNet Zero-Centering    │
│  - [0, 1] Segmenter Scaling   │
└──────────────┬────────────────┘
               │
       ┌───────┴────────────────────────┐
       ▼                                ▼
┌───────────────────────────┐    ┌───────────────────────────┐
│    DenseNet121 Classifier │    │      U-Net Segmenter      │
│  - 4-class probabilities  │    │  - Per-pixel probability  │
│  - Predicted class & conf │    │  - Binarize (threshold 0.5)│
└──────────────┬────────────┘    │  - Lung area percentage   │
               │                 └──────────────┬────────────┘
               ▼                                │
┌───────────────────────────┐                   │
│         Grad-CAM          │                   │
│  - Compute conv gradients │                   │
│  - Thermal JET overlay    │                   │
│  - Spatial region centroid│                   │
└──────────────┬────────────┘                   │
               │                                │
               └───────────────┬────────────────┘
                               │
                               ▼
                ┌───────────────────────────────┐
                │     Response Serialization    │
                │  - Base64 encoded PNG visuals │
                │  - Formatted medical metadata │
                │  - Structured JSON response   │
                └──────────────┬────────────────┘
                               │
                               ▼
                [React 19 Interactive Dashboard]
```

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Docker Compose Stack                          │
│                                                                        │
│  ┌─────────────────────────────┐      ┌─────────────────────────────┐  │
│  │      frontend container     │      │      backend container      │  │
│  │   React 19 + Vite + Tailwind│─────▶│   FastAPI + Uvicorn + TF    │  │
│  │   Nginx (Port :3000 -> :80) │      │   TensorFlow CPU (Port 8000)│  │
│  └─────────────────────────────┘      └─────────────────────────────┘  │
│                                                      ▲                 │
│                                                      │                 │
│                                        ┌─────────────┴──────────────┐  │
│                                        │  Mounted Models & Weights  │  │
│                                        │  - DenseNet121 (53.16 MB)  │  │
│                                        │  - U-Net (89.19 MB)        │  │
│                                        └────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Docker Compose (Recommended)

Run both the frontend and backend with a single command:

```bash
# Clone the repository and enter the directory
git clone https://github.com/your-username/Chest-X-Ray.git
cd "Chest X Ray"

# Build and start services
docker compose up --build

# Open the application in your browser
open http://localhost:3000
```

### Local Development Setup

#### 1. Backend (FastAPI)
```bash
cd "Chest X Ray"
python -m venv venv
venv\Scripts\activate          # On Windows
# source venv/bin/activate     # On Linux / macOS

pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

#### 2. Frontend (React + Vite)
```bash
cd "Chest X Ray/frontend"
npm install
npm run dev
# Vite runs at http://localhost:3000
```

---

## API Reference

| HTTP Method | Route | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health check endpoint returning model load status. |
| `GET` | `/api/ready` | Readiness check verifying classifier and segmenter availability. |
| `GET` | `/api/info` | Returns model architecture metadata, class labels, and config. |
| `POST` | `/api/analyze` | Accepts an uploaded image or sample key (`sample_1`, `sample_2`). Returns classification, segmentation mask, and Grad-CAM visualizations. |
| `POST` | `/api/compare` | Accepts two images. Returns side-by-side comparative metrics, probability deltas, lung area changes, and attention overlap. |
| `GET` | `/api/export/{fmt}` | Exports the latest analysis report as formatted plaintext (`txt`) or structured JSON (`json`). |

---

## Classification Classes

- **COVID-19:** Bilateral ground-glass opacities, peripheral consolidation, and multifocal infiltrates.
- **Lung Opacity:** Generalized airspace opacities, fluid accumulation, atelectasis, or consolidation.
- **Normal:** Clear lung fields without focal infiltrates, consolidation, or effusion.
- **Viral Pneumonia:** Diffuse interstitial markings, bronchial wall thickening, and peribronchial infiltration.

---

## Medical Disclaimer

> **IMPORTANT NOTICE:**  
> **ChestVision AI is an educational and research prototype.** Its predictions, segmentation masks, confidence scores, and Grad-CAM visualizations describe neural network behavior and statistical correlations. **They do not constitute a medical diagnosis, clinical opinion, or treatment recommendation.** All diagnostic decisions, interpretations, and patient care workflows must be conducted by qualified healthcare professionals and licensed radiologists.
