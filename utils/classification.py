import numpy as np
from config import CLASS_NAMES
from utils.preprocessing import preprocess_for_classifier

def predict_classification(classifier, raw_img_uint8):
    """
    Runs classification inference using the DenseNet121 model.

    Args:
        classifier: tf.keras.Model
        raw_img_uint8: uint8 NumPy array of shape (224, 224, 3)

    Returns:
        dict containing:
            - predicted_class: str
            - confidence: float (0.0 to 1.0)
            - class_probabilities: dict mapping class name -> probability float
            - pred_index: int
    """
    input_tensor = preprocess_for_classifier(raw_img_uint8)
    probs = classifier.predict(input_tensor, verbose=0)[0]
    
    pred_index = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[pred_index]
    confidence = float(probs[pred_index])

    class_probabilities = {
        CLASS_NAMES[i]: float(probs[i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "pred_index": pred_index,
        "raw_probs": probs
    }
