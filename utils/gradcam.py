import cv2
import numpy as np
import tensorflow as tf
from utils.preprocessing import preprocess_for_classifier

def get_densenet_base(classifier):
    """Retrieves the DenseNet121 base layer from the classifier model."""
    try:
        return classifier.get_layer("densenet121")
    except ValueError:
        # Fallback search for nested keras Model
        for layer in classifier.layers:
            if isinstance(layer, tf.keras.Model) or "densenet" in layer.name.lower():
                return layer
        raise ValueError("Could not locate DenseNet backbone in classifier model.")

def make_gradcam_heatmap(img_batch_cls, classifier, densenet_base=None, pred_index=None):
    """
    Computes Grad-CAM heatmap for a given input batch tensor using GradientTape.
    Follows notebook implementation.
    """
    if densenet_base is None:
        densenet_base = get_densenet_base(classifier)

    with tf.GradientTape() as tape:
        # Forward pass through DenseNet backbone
        conv_outputs = densenet_base(img_batch_cls, training=False)
        tape.watch(conv_outputs)

        # Forward pass through classifier head
        x = conv_outputs
        
        # Sequence of head layers in DenseNet classifier
        head_layers = ["gap", "batch_normalization", "dropout", "dense", "dropout_1", "predictions"]
        for l_name in head_layers:
            try:
                layer = classifier.get_layer(l_name)
                x = layer(x, training=False)
            except ValueError:
                pass

        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    # Calculate gradients of target class output with respect to conv_outputs
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise ValueError("Gradients are None. Classifier head is not connected to DenseNet output.")

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by pooled gradients
    conv_outputs_0 = conv_outputs[0]
    heatmap = conv_outputs_0 @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU activation on heatmap
    heatmap = tf.maximum(heatmap, 0.0)

    # Normalize heatmap to [0, 1] range
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / (max_val + 1e-8)

    return heatmap.numpy(), int(pred_index), predictions.numpy()[0]

def overlay_gradcam(raw_img_uint8, heatmap, alpha=0.4):
    """
    Resizes heatmap, applies JET colormap, and overlays onto the original image.
    """
    heatmap_resized = cv2.resize(heatmap, (raw_img_uint8.shape[1], raw_img_uint8.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(raw_img_uint8, 1 - alpha, heatmap_color, alpha, 0)
    return overlay, heatmap_color, heatmap_resized

def describe_activation_region(heatmap, threshold=0.6):
    """
    Locates the centroid of high-activation pixels in the Grad-CAM heatmap
    and translates it into a plain spatial description. (Notebook exact implementation)
    """
    ys, xs = np.where(heatmap >= threshold)

    if len(ys) == 0:
        return "diffuse regions across the lung fields"

    h, w = heatmap.shape
    cy, cx = ys.mean() / h, xs.mean() / w

    vertical = (
        "upper" if cy < 0.33
        else ("lower" if cy > 0.66 else "central")
    )

    horizontal = (
        "left" if cx < 0.33
        else ("right" if cx > 0.66 else "central")
    )

    if vertical == "central" and horizontal == "central":
        return "central lung field"

    return f"{vertical}-{horizontal} lung field"

def generate_gradcam(classifier, raw_img_uint8, target_pred_index=None):
    """
    Full Grad-CAM generation pipeline for an X-ray image.

    Returns:
        dict containing:
            - heatmap: 2D float heatmap (224, 224)
            - heatmap_color: RGB heatmap visualization
            - overlay: RGB image with heatmap overlay
            - region_description: spatial text description
            - pred_index: index of target class
    """
    input_tensor = preprocess_for_classifier(raw_img_uint8)
    heatmap_raw, pred_index, probs = make_gradcam_heatmap(input_tensor, classifier, pred_index=target_pred_index)
    overlay, heatmap_color, heatmap_resized = overlay_gradcam(raw_img_uint8, heatmap_raw)
    region_desc = describe_activation_region(heatmap_resized)

    return {
        "heatmap": heatmap_resized,
        "heatmap_raw": heatmap_raw,
        "heatmap_color": heatmap_color,
        "overlay": overlay,
        "region_description": region_desc,
        "pred_index": pred_index
    }
