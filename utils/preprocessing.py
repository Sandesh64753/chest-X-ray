import io
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from config import IMG_SIZE

def load_and_preprocess_pil_image(image_input):
    """
    Accepts a file path, PIL Image, bytes, or FastAPI UploadFile/BytesIO object,
    converts it to an RGB PIL image, resizes to (IMG_SIZE, IMG_SIZE),
    and returns a uint8 NumPy array of shape (224, 224, 3).
    """
    if isinstance(image_input, (str, bytes, io.BytesIO)):
        img = Image.open(image_input)
    elif hasattr(image_input, "read"):  # FastAPI UploadFile or BytesIO
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input)
    else:
        raise ValueError("Unsupported image input type.")

    # Convert to RGB (handles grayscale, RGBA, palette)
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.uint8)

def preprocess_for_classifier(raw_img_uint8):
    """
    Applies DenseNet121 preprocess_input to an RGB uint8 image.
    Returns tensor with shape (1, 224, 224, 3) ready for model.predict().
    """
    img_float = tf.cast(raw_img_uint8, tf.float32)
    preprocessed = densenet_preprocess(img_float)
    return tf.expand_dims(preprocessed, axis=0)

def preprocess_for_segmenter(raw_img_uint8):
    """
    Applies simple [0, 1] scaling to an RGB uint8 image for U-Net segmentation.
    Returns tensor with shape (1, 224, 224, 3) ready for model.predict().
    """
    img_float = tf.cast(raw_img_uint8, tf.float32) / 255.0
    return tf.expand_dims(img_float, axis=0)
