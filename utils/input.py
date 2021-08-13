from typing import Dict, Text
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils


def input_layer(feature_spec):
    input_layers = {}
    for key, spec in feature_spec.items():
        if isinstance(spec, tf.io.VarLenFeature):
            input_layers[key] = tf.keras.layers.Input(
                shape=spec.shape, name=key, dtype=spec.dtype, sparse=True)
        elif isinstance(spec, tf.io.FixedLenFeature):
            input_layers[key] = tf.keras.layers.Input(
                shape=spec.shape, name=key, dtype=spec.dtype)
        else:
            raise ValueError('Spec type is not supported: ', key, spec)
    return input_layers
