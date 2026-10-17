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
    """Handles tokenization and auxiliary signal packaging for DeBERTa-v3."""
    def __init__(self, model_preset="deberta_v3_small_en", max_seq_length=128):
        self.max_seq_length = max_seq_length
        self.preprocessor = keras_hub.models.DebertaV3Preprocessor.from_preset(
            model_preset, sequence_length=max_seq_length
        )

    def tokenize_pairs(self, questions, titles=None, inc_signals=None, sub_signals=None):
        """Tokenizes text pairs and safely packages numerical signal arrays."""
        q_tensor = tf.constant(questions, dtype=tf.string)
        if titles is not None:
            t_tensor = tf.constant(titles, dtype=tf.string)
            processed = self.preprocessor(x=(q_tensor, t_tensor))
        else:
            processed = self.preprocessor(x=q_tensor)

        features = processed[0] if isinstance(processed, tuple) else processed
        
        model_inputs = {
            "token_ids": np.array(features["token_ids"]),
            "padding_mask": np.array(features["padding_mask"])
        }

        # Include auxiliary signals formatted as (batch_size, 1)
        if inc_signals is not None:
            model_inputs["inc_signal"] = np.array(inc_signals, dtype=np.float32).reshape(-1, 1)
        if sub_signals is not None:
            model_inputs["sub_signal"] = np.array(sub_signals, dtype=np.float32).reshape(-1, 1)

        return model_inputs

# ==============================================================================
# 2. DeBERTa-v3 Model Factory
# ==============================================================================
class DebertaHumorModelFactory:
    """Builds a Unified Feature-Fused DeBERTa-v3 classifier."""
    def __init__(self, model_preset="deberta_v3_small_en", max_seq_length=128):
        self.model_preset = model_preset
        self.max_seq_length = max_seq_length

    def build_classifier(self, use_title=True, use_inc=False, use_sub=False, dropout_rate=0.3):
        input_ids = tf.keras.Input(shape=(self.max_seq_length,), dtype=tf.int32, name="token_ids")
        padding_mask = tf.keras.Input(shape=(self.max_seq_length,), dtype=tf.int32, name="padding_mask")
        
        inputs = {"token_ids": input_ids, "padding_mask": padding_mask}

        deberta_backbone = keras_hub.models.DebertaV3Backbone.from_preset(self.model_preset)
        backbone_outputs = deberta_backbone(inputs)

        if isinstance(backbone_outputs, dict):
            sequence_output = backbone_outputs["sequence_output"]
        else:
            sequence_output = backbone_outputs

        # [CLS] representation (Index 0)
        cls_token_representation = tf.keras.layers.Lambda(
            lambda x: x[:, 0, :], name="cls_token_extraction"
        )(sequence_output)

        features_to_fuse = [cls_token_representation]

        # Conditionally append numerical feature pathways
        if use_inc:
            inc_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name="inc_signal")
            inputs["inc_signal"] = inc_input
            features_to_fuse.append(inc_input)

        if use_sub:
            sub_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name="sub_signal")
            inputs["sub_signal"] = sub_input
            features_to_fuse.append(sub_input)

        # Concatenate / Fuse representations
        if len(features_to_fuse) > 1:
            combined_features = tf.keras.layers.Concatenate(name="feature_fusion")(features_to_fuse)
        else:
            combined_features = cls_token_representation

        # Classification Head
        x = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(combined_features)
        output_probability = tf.keras.layers.Dense(
            1, activation="sigmoid", dtype="float32", name="classification_output"
        )(x)

        return tf.keras.models.Model(inputs=inputs, outputs=output_probability, name="Feature_Fused_DeBERTa")
