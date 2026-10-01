import numpy as np
from config import CLASS_NAMES, COMPARISON_DISCLAIMER

def _gradcam_region_overlap(heatmap1, heatmap2, threshold=0.6):
    """
    IoU between the high-activation regions (>= threshold) of two Grad-CAM heatmaps.
    Lower = model attended to different regions across the two images;
    Higher = model attended to similar regions.
    Geometric similarity measure only.
    """
    h1 = (heatmap1 >= threshold)
    h2 = (heatmap2 >= threshold)
    union = np.logical_or(h1, h2).sum()
    if union == 0:
        return None
    return float(np.logical_and(h1, h2).sum() / union)

def compare_xrays(analysis1, analysis2):
    """
    Runs quantitative comparison between two model analysis results.

    Args:
        analysis1: dict from analyze_single_xray for Image 1
        analysis2: dict from analyze_single_xray for Image 2

    Returns:
        dict containing measurable differences, per-class deltas, and disclaimers.
    """
    class1 = analysis1["classification"]["predicted_class"]
    class2 = analysis2["classification"]["predicted_class"]
    
    conf1 = analysis1["classification"]["confidence"]
    conf2 = analysis2["classification"]["confidence"]

    probs1 = analysis1["classification"]["class_probabilities"]
    probs2 = analysis2["classification"]["class_probabilities"]

    area1 = analysis1["segmentation"]["mask_area_pct"]
    area2 = analysis2["segmentation"]["mask_area_pct"]

    # Calculate per-class deltas
    probability_deltas = {
        cls: float(probs2[cls] - probs1[cls])
        for cls in CLASS_NAMES
    }

    # Grad-CAM IoU overlap
    heatmap1 = analysis1["gradcam"]["heatmap"]
    heatmap2 = analysis2["gradcam"]["heatmap"]
    gradcam_iou = _gradcam_region_overlap(heatmap1, heatmap2)

    measurable_differences = {
        "predicted_class_changed": (class1 != class2),
        "class1": class1,
        "class2": class2,
        "confidence_delta": float(conf2 - conf1),
        "class_probability_deltas": probability_deltas,
        "lung_mask_area_pct_change": float(area2 - area1),
        "area1": area1,
        "area2": area2,
        "gradcam_region_overlap_iou": gradcam_iou
    }

    return {
        "image_1": {
            "predicted_class": class1,
            "confidence": conf1,
            "probabilities": probs1,
            "lung_mask_area_pct": area1
        },
        "image_2": {
            "predicted_class": class2,
            "confidence": conf2,
            "probabilities": probs2,
            "lung_mask_area_pct": area2
        },
        "measurable_differences": measurable_differences,
        "disclaimer": COMPARISON_DISCLAIMER
    }
