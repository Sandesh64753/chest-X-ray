import datetime
import json
import base64
from io import BytesIO
from PIL import Image

def generate_text_report(analysis_result):
    """
    Generates a formatted text analysis report.
    """
    cls_res = analysis_result["classification"]
    seg_res = analysis_result["segmentation"]
    gcam_res = analysis_result["gradcam"]

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        "=" * 60,
        "CHESTVISION AI - MODEL ANALYSIS REPORT",
        "=" * 60,
        f"Generated At: {now}",
        f"System Version: ChestVision AI v1.0",
        "-" * 60,
        "1. CLASSIFICATION RESULT (DenseNet121)",
        f"   Predicted Category: {cls_res['predicted_class']}",
        f"   Model Confidence:   {cls_res['confidence'] * 100:.2f}%",
        "",
        "   Class Probabilities:",
    ]

    for cls_name, prob in cls_res["class_probabilities"].items():
        report_lines.append(f"     - {cls_name:<16}: {prob * 100:>6.2f}%")

    report_lines.extend([
        "",
        "-" * 60,
        "2. LUNG SEGMENTATION RESULT (U-Net)",
        f"   Segmented Lung Area: {seg_res['mask_area_pct']:.2f}% of total image area",
        "",
        "-" * 60,
        "3. EXPLAINABLE AI (Grad-CAM)",
        f"   Primary Activation Region: {gcam_res['region_description']}",
        "   Attention Interpretation:  Highlights image regions that strongly influenced",
        "                              the classifier's softmax probability decision.",
        "",
        "=" * 60,
        "IMPORTANT MEDICAL DISCLAIMER",
        "=" * 60,
        "ChestVision AI is an educational and research prototype. Its predictions,",
        "segmentation masks, confidence scores, and Grad-CAM visualizations describe",
        "model behavior and MUST NOT be interpreted as a medical diagnosis. Clinical",
        "decisions must be made by qualified healthcare professionals.",
        "=" * 60,
    ])

    return "\n".join(report_lines)

def generate_json_report(analysis_result):
    """
    Generates a structured JSON string of the analysis results.
    """
    cls_res = analysis_result["classification"]
    seg_res = analysis_result["segmentation"]
    gcam_res = analysis_result["gradcam"]

    data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "classification": {
            "predicted_class": cls_res["predicted_class"],
            "confidence": cls_res["confidence"],
            "probabilities": cls_res["class_probabilities"]
        },
        "segmentation": {
            "lung_mask_area_pct": seg_res["mask_area_pct"]
        },
        "explainability": {
            "gradcam_activation_region": gcam_res["region_description"]
        },
        "disclaimer": (
            "ChestVision AI is an educational and research prototype. Predictions, "
            "masks, confidence scores, and heatmaps describe model behavior and are not "
            "a clinical diagnosis."
        )
    }
    return json.dumps(data, indent=2)
