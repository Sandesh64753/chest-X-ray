import os
import sys
from functools import lru_cache
import tensorflow as tf

# Import Keras safely
try:
    import keras
except ImportError:
    keras = tf.keras


def _apply_deserialization_patches():
    """
    Patches Keras deserialization functions to strip 'quantization_config'
    which causes 'Unrecognized keyword arguments passed to Dense: quantization_config'
    when models saved in newer Keras 3 versions are loaded in environments with version mismatches.
    """
    targets = []

    # Keras entrypoints
    try:
        from keras.saving import deserialize_keras_object
        targets.append(('keras.saving', 'deserialize_keras_object', deserialize_keras_object))
    except Exception:
        pass

    try:
        from keras.src.saving import serialization_lib
        if hasattr(serialization_lib, 'deserialize_keras_object'):
            targets.append(('keras.src.saving.serialization_lib', 'deserialize_keras_object', serialization_lib.deserialize_keras_object))
    except Exception:
        pass

    try:
        from keras.utils import deserialize_keras_object
        targets.append(('keras.utils', 'deserialize_keras_object', deserialize_keras_object))
    except Exception:
        pass

    # TensorFlow.Keras entrypoints
    try:
        from tensorflow.keras.utils import deserialize_keras_object as tf_deser
        targets.append(('tensorflow.keras.utils', 'deserialize_keras_object', tf_deser))
    except Exception:
        pass

    try:
        from tensorflow.keras.saving import deserialize_keras_object as tf_deser2
        targets.append(('tensorflow.keras.saving', 'deserialize_keras_object', tf_deser2))
    except Exception:
        pass

    for mod_name, fn_name, orig_fn in targets:
        if not getattr(orig_fn, '_safe_patched', False):
            def make_safe_wrapper(target_fn):
                def safe_deserialize(config, *args, **kwargs):
                    if isinstance(config, dict) and 'config' in config and isinstance(config['config'], dict):
                        config = config.copy()
                        config['config'] = config['config'].copy()
                        config['config'].pop('quantization_config', None)
                    return target_fn(config, *args, **kwargs)
                safe_deserialize._safe_patched = True
                return safe_deserialize

            wrapped_fn = make_safe_wrapper(orig_fn)

            # Apply patch to module attributes
            if mod_name == 'keras.saving':
                import keras.saving
                keras.saving.deserialize_keras_object = wrapped_fn
            elif mod_name == 'keras.src.saving.serialization_lib':
                import keras.src.saving.serialization_lib
                keras.src.saving.serialization_lib.deserialize_keras_object = wrapped_fn
            elif mod_name == 'keras.utils':
                import keras.utils
                keras.utils.deserialize_keras_object = wrapped_fn
            elif mod_name == 'tensorflow.keras.utils':
                import tensorflow.keras.utils
                tensorflow.keras.utils.deserialize_keras_object = wrapped_fn
            elif mod_name == 'tensorflow.keras.saving':
                import tensorflow.keras.saving
                tensorflow.keras.saving.deserialize_keras_object = wrapped_fn


def _create_safe_custom_objects():
    """
    Creates custom layer classes that pop 'quantization_config' during from_config.
    """
    custom_objs = {}
    layer_names = [
        'Dense', 'Conv2D', 'Conv2DTranspose', 'BatchNormalization', 'Dropout',
        'GlobalAveragePooling2D', 'Flatten', 'InputLayer', 'Concatenate', 'Add',
        'ZeroPadding2D', 'Activation', 'Rescaling', 'MaxPooling2D', 'UpSampling2D'
    ]
    keras_layers = getattr(keras, 'layers', getattr(tf.keras, 'layers', None))
    if keras_layers:
        for name in layer_names:
            if hasattr(keras_layers, name):
                base_cls = getattr(keras_layers, name)
                class SafeLayer(base_cls):
                    @classmethod
                    def from_config(cls, config):
                        if isinstance(config, dict):
                            config = config.copy()
                            config.pop('quantization_config', None)
                        return super().from_config(config)
                SafeLayer.__name__ = f'Safe{name}'
                custom_objs[name] = SafeLayer
    return custom_objs


def _load_single_model(path):
    _apply_deserialization_patches()
    custom_objects = _create_safe_custom_objects()

    loaders = []
    if hasattr(keras, 'models') and hasattr(keras.models, 'load_model'):
        loaders.append(keras.models.load_model)
    if hasattr(tf.keras, 'models') and hasattr(tf.keras.models, 'load_model'):
        loaders.append(tf.keras.models.load_model)

    last_error = None
    for loader in loaders:
        for compile_flag in [False, True]:
            try:
                return loader(path, compile=compile_flag, custom_objects=custom_objects)
            except Exception as e:
                last_error = e

    raise last_error


@lru_cache(maxsize=1)
def load_models():
    from config import CLASSIFIER_PATH, SEGMENTER_PATH

    classifier = None
    segmenter = None
    errors = []

    # 1. Load Classifier Model
    if not os.path.exists(CLASSIFIER_PATH):
        errors.append(f"Classifier model file not found at: {CLASSIFIER_PATH}")
    else:
        try:
            classifier = _load_single_model(CLASSIFIER_PATH)
        except Exception as e:
            errors.append(f"Failed to load classification model: {str(e)}")

    # 2. Load Segmentation Model
    if not os.path.exists(SEGMENTER_PATH):
        errors.append(f"Segmentation model file not found at: {SEGMENTER_PATH}")
    else:
        try:
            segmenter = _load_single_model(SEGMENTER_PATH)
        except Exception as e:
            errors.append(f"Failed to load segmentation model: {str(e)}")

    if errors:
        error_msg = "\n".join(errors)
        raise RuntimeError(f"Model Loading Error:\n{error_msg}")

    return classifier, segmenter
