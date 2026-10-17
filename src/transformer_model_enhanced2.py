import os
import numpy as np
import tensorflow as tf
import keras_hub

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==============================================================================
# 1. DeBERTa-v3 Preprocessor
# ==============================================================================
class DebertaPreprocessor:
    """Handles tokenization and sequence formatting for DeBERTa-v3."""
    def __init__(self, model_preset="deberta_v3_small_en", max_seq_length=128):
        self.max_seq_length = max_seq_length
        self.preprocessor = keras_hub.models.DebertaV3Preprocessor.from_preset(
            model_preset, sequence_length=max_seq_length
        )

    def tokenize_pairs(self, questions, titles=None):
        q_tensor = tf.constant(questions, dtype=tf.string)
        if titles is not None:
            t_tensor = tf.constant(titles, dtype=tf.string)
            processed = self.preprocessor(x=(q_tensor, t_tensor))
        else:
            processed = self.preprocessor(x=q_tensor)

        features = processed[0] if isinstance(processed, tuple) else processed
        return {
            "token_ids": np.array(features["token_ids"]),
            "padding_mask": np.array(features["padding_mask"])
        }

# ==============================================================================
# 2. DeBERTa-v3 Model Factory
# ==============================================================================
class DebertaHumorModelFactory:
    """Builds the DeBERTa-v3 classifier."""
    def __init__(self, model_preset="deberta_v3_small_en", max_seq_length=128):
        self.model_preset = model_preset
        self.max_seq_length = max_seq_length

    def build_classifier(self, use_title=True, dropout_rate=0.3):
        input_ids = tf.keras.Input(shape=(self.max_seq_length,), dtype=tf.int32, name="token_ids")
        padding_mask = tf.keras.Input(shape=(self.max_seq_length,), dtype=tf.int32, name="padding_mask")
        inputs = {"token_ids": input_ids, "padding_mask": padding_mask}

        deberta_backbone = keras_hub.models.DebertaV3Backbone.from_preset(self.model_preset)
        backbone_outputs = deberta_backbone(inputs)

        if isinstance(backbone_outputs, dict):
            sequence_output = backbone_outputs["sequence_output"]
        else:
            sequence_output = backbone_outputs

        cls_token_representation = tf.keras.layers.Lambda(
            lambda x: x[:, 0, :], name="cls_token_extraction"
        )(sequence_output)

        x = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(cls_token_representation)
        
        output_probability = tf.keras.layers.Dense(
            1, activation="sigmoid", dtype="float32", name="classification_output"
        )(x)

        return tf.keras.models.Model(inputs=inputs, outputs=output_probability, name="DeBERTa_v3_Humor_Classifier")
