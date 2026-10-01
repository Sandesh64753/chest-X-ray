# 🫁 ChestVision AI

**Explainable Chest X-ray Classification & Semantic Lung Field Segmentation Workspace**

ChestVision AI is a production-grade **FastAPI** web application designed for automated chest radiograph analysis. It integrates deep learning computer vision models for multi-class pathology classification, pixel-level semantic lung field segmentation, and Gradient-weighted Class Activation Mapping (Grad-CAM) explainability into a modern, high-performance medical AI dashboard.

---

## 🌟 Key Features

* **🎯 Multi-Class Pathology Classification:**
  * Powered by DenseNet121 transfer learning pretrained on ImageNet.
  * Classifies chest X-rays into 4 distinct categories: `COVID`, `Lung_Opacity`, `Normal`, and `Viral Pneumonia`.
  * Interactive probability distribution visualizations rendered via Chart.js.

* **🫁 Semantic Lung Field Segmentation:**
  * Custom U-Net encoder-decoder architecture with transposed convolutions and skip connections.
  * Generates pixel-level binary lung masks and calculates precise lung-mask area coverage percentages.
  * Displays original X-rays, binary masks, and semi-transparent medical overlays.

* **🧠 Explainable Grad-CAM Visualizations:**
  * Computes visual feature activation maps from DenseNet121 deep convolutional representations.
  * Calculates spatial activation centroids to generate plain descriptions (e.g., `central lung field`, `upper-right lung field`).
  * Displays visual heatmaps and overlays directly on target radiographs.

* **⚖️ Two X-ray Comparative Analysis:**
  * Side-by-side comparative analysis of two X-ray scans.
  * Quantifies deltas in predictions, softmax confidence, lung-mask coverage shift, per-class probability changes, and Grad-CAM spatial focus IoU overlap.

* **🖼️ Interactive Frontend UX:**
  * Instant client-side image upload preview with `❌ Cancel / Remove Image` options.
  * In-button animated loading spinners for smooth, non-intrusive inference feedback.

* **📄 Exportable Structured Reports:**
  * Downloadable analysis summaries in formatted `.TXT` and machine-readable `.JSON` formats.

---

## 📁 Repository Structure

```
d:/Chest X Ray/
├── app.py                      # Main FastAPI application server & routing
├── config.py                   # Configuration, model paths, class colors & disclaimers
├── requirements.txt            # Dependency specification (FastAPI, Uvicorn, TensorFlow, etc.)
├── README.md                   # Comprehensive project documentation
├── Dockerfile                  # Containerization specification for production deployment
├── docker-compose.yml          # Docker Compose orchestration configuration
├── .dockerignore               # Files and patterns ignored during Docker build
├── .gitignore                  # Git repository ignore rules
│
├── models/                     # Pretrained Keras deep learning model files
│   ├── classifier_stage2_best.keras
│   └── lung_segmentation.keras
│
├── templates/                  # Jinja2 HTML Templates
│   ├── base.html              # Master layout with sidebar & Chart.js script
│   ├── dashboard.html         # Homepage dashboard view
│   ├── analyze.html           # Single X-ray analysis view
│   ├── compare.html           # Side-by-side 2-image comparison view
│   ├── explainability.html    # Grad-CAM methodology & mathematical formulation
│   ├── model_info.html        # Model architectures & preprocessing specifications
│   └── about.html            # Project overview & technology stack
│
├── utils/                      # Modular backend processing utilities
│   ├── __init__.py
│   ├── model_loader.py         # Cached model loading (@functools.lru_cache)
│   ├── preprocessing.py        # DenseNet & U-Net image preprocessing
│   ├── classification.py       # DenseNet inference engine
│   ├── segmentation.py        # U-Net mask & area calculation engine
│   ├── gradcam.py             # Grad-CAM heatmap & spatial centroid analyzer
│   ├── comparison.py           # Quantitative 2-image comparison & IoU engine
│   └── report.py              # Exportable text & JSON report generators
│
└── assets/                     # UI static assets & styling
    ├── style.css              # Custom CSS medical AI stylesheet
    └── samples/               # Demo X-ray images for instant testing
        ├── sample_xray_1.png
        └── sample_xray_2.png
```

---

## 🛠️ Local Installation & Development

### 1. Prerequisites
Ensure Python 3.9+ and `pip` are installed on your system.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Model Files Placement
Ensure the trained Keras models are placed inside the `models/` directory:
- `models/classifier_stage2_best.keras`
- `models/lung_segmentation.keras`

### 4. Run Development Server
Start the FastAPI server using Uvicorn:
```bash
uvicorn app:app --reload
```
or launch directly via Python:
```bash
python app.py
```
Open your browser and navigate to: `http://127.0.0.1:8000`

---

## 🐳 Docker Deployment Guide

### Option 1: Deploy with Docker Compose (Recommended)

Run the containerized application using Docker Compose:

```bash
docker-compose up -d --build
```

To view running logs:
```bash
docker-compose logs -f
```

To stop the deployment:
```bash
docker-compose down
```

---

### Option 2: Deploy with Docker CLI

1. **Build the Docker Image:**
```bash
docker build -t chestvision-ai:latest .
```

2. **Run the Container:**
```bash
docker run -d \
  --name chestvision_container \
  -p 8000:8000 \
  --restart unless-stopped \
  chestvision-ai:latest
```

3. **Verify Application Health:**
Access the dashboard at `http://localhost:8000`

---

## 📊 Dataset & Model Architecture Summary

* **Dataset:** Trained and evaluated on the **COVID-19 Radiography Database**.
* **Classification Architecture:**
  * Backbone: DenseNet121 pretrained on ImageNet
  * Head: `GlobalAveragePooling2D` ➔ `BatchNormalization` ➔ `Dropout(0.4)` ➔ `Dense(256, ReLU)` ➔ `Dropout(0.3)` ➔ `Dense(4, Softmax)`
  * Resolution: 224 × 224 × 3
* **Segmentation Architecture:**
  * Custom U-Net with 4 Conv Encoder Blocks (32, 64, 128, 256 filters), Bottleneck (512 filters), 4 Transposed Conv Decoder Blocks with skip connections, and a 1×1 Conv Sigmoid output.
  * Resolution: 224 × 224 × 3
  * Mask Threshold: 0.5

---

## ⚠️ Medical Disclaimer

> **IMPORTANT:** ChestVision AI is an educational and research prototype. Its predictions, segmentation masks, confidence scores, and Grad-CAM visualizations describe model behavior and **must not** be interpreted as a medical diagnosis. Clinical decisions must always be made by qualified healthcare professionals.
