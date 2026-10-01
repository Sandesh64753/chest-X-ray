import os
from functools import lru_cache
import tensorflow as tf
from config import CLASSIFIER_PATH, SEGMENTER_PATH

@lru_cache(maxsize=1)
def load_models():
    """
    Loads and caches TensorFlow/Keras models for classification and segmentation.
    Uses @lru_cache so models are loaded only once per FastAPI server process.
    """
    classifier = None
    segmenter = None
    errors = []

    # 1. Load Classifier Model
    if not os.path.exists(CLASSIFIER_PATH):
        errors.append(f"Classifier model file not found at: {CLASSIFIER_PATH}")
    else:
        try:
            classifier = tf.keras.models.load_model(CLASSIFIER_PATH)
        except Exception as e:
            errors.append(f"Failed to load classification model: {str(e)}")

    # 2. Load Segmentation Model
    if not os.path.exists(SEGMENTER_PATH):
        errors.append(f"Segmentation model file not found at: {SEGMENTER_PATH}")
    else:
        try:
            segmenter = tf.keras.models.load_model(SEGMENTER_PATH, compile=False)
        except Exception as e:
            errors.append(f"Failed to load segmentation model: {str(e)}")

    if errors:
        error_msg = "\n".join(errors)
        raise RuntimeError(f"Model Loading Error:\n{error_msg}")

    return classifier, segmenter
