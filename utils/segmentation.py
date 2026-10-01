import cv2
import numpy as np
from config import MASK_BINARIZATION_THRESHOLD, IMG_SIZE
from utils.preprocessing import preprocess_for_segmenter

def predict_segmentation(segmenter, raw_img_uint8, threshold=MASK_BINARIZATION_THRESHOLD):
    """
    Runs U-Net segmentation inference on an X-ray image.

    Args:
        segmenter: tf.keras.Model
        raw_img_uint8: uint8 NumPy array (224, 224, 3)
        threshold: float threshold (default 0.5)

    Returns:
        dict containing:
            - mask_prob: raw sigmoid probability map (224, 224)
            - mask_binary: uint8 binary mask (224, 224) with values 0 or 255
            - mask_area_pct: float percentage of total image area covered by predicted mask
            - overlay: uint8 RGB image with lung mask overlay
    """
    input_tensor = preprocess_for_segmenter(raw_img_uint8)
    pred_mask = segmenter.predict(input_tensor, verbose=0)[0]  # shape (224, 224, 1)
    mask_prob = np.squeeze(pred_mask)  # shape (224, 224)
    
    mask_binary_bool = mask_prob > threshold
    mask_binary_uint8 = (mask_binary_bool * 255).astype(np.uint8)

    # Calculate Lung Mask Area percentage
    total_pixels = IMG_SIZE * IMG_SIZE
    active_pixels = np.sum(mask_binary_bool)
    mask_area_pct = float((active_pixels / total_pixels) * 100.0)

    # Create segmentation overlay
    overlay = create_segmentation_overlay(raw_img_uint8, mask_binary_bool)

    return {
        "mask_prob": mask_prob,
        "mask_binary": mask_binary_uint8,
        "mask_binary_bool": mask_binary_bool,
        "mask_area_pct": mask_area_pct,
        "overlay": overlay
    }

def create_segmentation_overlay(raw_img_uint8, mask_binary_bool, color=(14, 165, 233), alpha=0.4):
    """
    Creates a semi-transparent colored overlay of the lung mask on the original image.
    Default color: Medical Teal (RGB 14, 165, 233).
    """
    overlay = raw_img_uint8.copy()
    color_mask = np.zeros_like(raw_img_uint8)
    color_mask[mask_binary_bool] = color

    # Blend original image and colored mask where mask is True
    blended = cv2.addWeighted(raw_img_uint8, 1 - alpha, color_mask, alpha, 0)
    
    # Keep original image pixels where mask is False
    result = np.where(mask_binary_bool[..., None], blended, raw_img_uint8)
    return result
