import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model File Configuration
CLASSIFIER_PATH = os.getenv(
    "CLASSIFIER_PATH",
    os.path.join(BASE_DIR, "models", "classifier_stage2_best.keras")
)
if not os.path.exists(CLASSIFIER_PATH):
    # Fallback to artifacts if models folder copy isn't found
    CLASSIFIER_PATH = os.path.join(BASE_DIR, "artifacts", "classifier_stage2_best.keras")

SEGMENTER_PATH = os.getenv(
    "SEGMENTER_PATH",
    os.path.join(BASE_DIR, "models", "lung_segmentation.keras")
)
if not os.path.exists(SEGMENTER_PATH):
    # Fallback to artifacts segmenter_best.keras if needed
    SEGMENTER_PATH = os.path.join(BASE_DIR, "artifacts", "segmenter_best.keras")

# Dataset & Class Constants
CLASS_NAMES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224
MASK_SIZE = 224
MASK_BINARIZATION_THRESHOLD = 0.5

# Color Palette for Visualizations (Medical UI Theme)
CLASS_COLORS = {
    "COVID": "#ef4444",         # Coral Red
    "Lung_Opacity": "#f59e0b",  # Amber/Orange
    "Normal": "#10b981",        # Medical Emerald
    "Viral Pneumonia": "#3b82f6"# Deep Blue
}

# Medical Disclaimer Text
MEDICAL_DISCLAIMER = (
    "ChestVision AI is an educational and research prototype. Its predictions, "
    "segmentation masks, confidence scores, and Grad-CAM visualizations describe model behavior "
    "and should not be interpreted as a medical diagnosis. Clinical decisions must be made by "
    "qualified healthcare professionals."
)

COMPARISON_DISCLAIMER = (
    "These are quantitative differences in MODEL OUTPUTS between the two images, "
    "not a medical assessment of disease progression. A change in predicted class, "
    "confidence, probabilities, segmented lung-opacity area, or Grad-CAM attention "
    "region does not by itself indicate clinical improvement or worsening. This "
    "system does not determine whether a change is 'good' or 'bad' -- that "
    "interpretation requires a qualified radiologist reviewing both images directly, "
    "ideally alongside clinical history and other diagnostic information."
)
