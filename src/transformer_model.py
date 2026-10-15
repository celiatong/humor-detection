import os
import numpy as np
import tensorflow as tf
import keras_hub

# Force CPU mode for execution stability on Apple Silicon M-series Macs & standard environments
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
tf.config.set_visible_devices([], 'GPU')

# ==============================================================================
# 1. DEBERTA-V3 TEXT PIPELINE PREPROCESSOR
# ==============================================================================
class DebertaPreprocessor:
    """Handles native Keras 3 tokenization and sequence formatting for DeBERTa-v3."""
    
    def __init__(self, model_preset="deberta_v3_small_en", max_seq_length=128):
        self.max_seq_length = max_seq_length
        # Loads the official pre-trained DeBERTa-v3 tokenizer natively via KerasHub
        self.preprocessor = keras_hub.models.DebertaV3Preprocessor.from_preset(
            model_preset,
            sequence_length=max_seq_length
        )
  
    def tokenize_pairs(self, questions, titles=None, inc_signals=None, sub_signals=None):
        """
        Tokenizes inputs based on the experimental setup:
        - Single input: [CLS] Question [SEP]
        - Paired input: [CLS] Question [SEP] Product Title [SEP]
        Also structures auxiliary input arrays (INC, SUB) if provided.
        """
        if titles is not None:
            # Multi-sequence packing requires explicit string tensors and keyword argument x=(q, t)
            q_tensor = tf.constant(questions, dtype=tf.string)
            t_tensor = tf.constant(titles, dtype=tf.string)
            processed = self.preprocessor(x=(q_tensor, t_tensor))
        else:
            q_tensor = tf.constant(questions, dtype=tf.string)
            processed = self.preprocessor(x=q_tensor)
            
        # Safely unwrap if the preprocessor returned an (x, y) tuple
        features = processed[0] if isinstance(processed, tuple) else processed
        
        # Convert KerasHub preprocessing tensors to standard numpy arrays
        model_inputs = {
            "token_ids": np.array(features["token_ids"]),
            "padding_mask": np.array(features["padding_mask"])
        }
        
        # Include optional pre-computed linguistic signals formatted as (batch_size, 1)
        if inc_signals is not None:
            model_inputs["inc_signal"] = np.array(inc_signals, dtype=np.float32).reshape(-1, 1)
        if sub_signals is not None:
            model_inputs["sub_signal"] = np.array(sub_signals, dtype=np.float32).reshape(-1, 1)
            
        return model_inputs


# ==============================================================================
# 2. DEBERTA-V3 MODEL FACTORY (NATIVE KERAS 3)
# ==============================================================================
class DebertaHumorModelFactory:
    """Builds a Unified Cross-Attention DeBERTa-v3 classifier natively in Keras 3."""
    
    def __init__(self, model_preset="deberta_v3_small_en", max_seq_length=128):
        self.model_preset = model_preset
        self.max_seq_length = max_seq_length
  
    def build_classifier(self, use_title=True, use_inc=False, use_sub=False, dropout_rate=0.3):
        """
        Attaches a binary classification head to the native DeBERTa-v3 backbone.
        Dynamically configures inputs based on the 5 methodology variants:
        1. Q (use_title=False, use_inc=False, use_sub=False)
        2. Q+T (use_title=True, use_inc=False, use_sub=False)
        3. INC_Q+T (use_title=True, use_inc=True, use_sub=False)
        4. SUB_Q+T (use_title=True, use_inc=False, use_sub=True)
        5. INC_SUB_Q+T (use_title=True, use_inc=True, use_sub=True)
        """
        # --- 1. Define Base Text Inputs ---
        input_ids = tf.keras.Input(
            shape=(self.max_seq_length,), 
            dtype=tf.int32, 
            name="token_ids"
        )
        padding_mask = tf.keras.Input(
            shape=(self.max_seq_length,), 
            dtype=tf.int32, 
            name="padding_mask"
        )
        
        inputs = {
            "token_ids": input_ids,
            "padding_mask": padding_mask
        }
        
        # --- 2. Load the Native Pre-trained DeBERTa-v3 Backbone ---
        deberta_backbone = keras_hub.models.DebertaV3Backbone.from_preset(
            self.model_preset
        )
  
        # --- 3. Pass Inputs through Encoder Layers ---
        backbone_outputs = deberta_backbone({
            "token_ids": input_ids,
            "padding_mask": padding_mask
        })
  
        # Handle dictionary or raw tensor output from DeBERTa-v3 Backbone
        if isinstance(backbone_outputs, dict):
            sequence_output = backbone_outputs["sequence_output"]
        else:
            sequence_output = backbone_outputs

        # Extract the [CLS] representation (Index 0 of sequence dimension)
        cls_token_representation = tf.keras.layers.Lambda(
            lambda x: x[:, 0, :], 
            name="cls_token_extraction"
        )(sequence_output)
        
        features_to_concatenate = [cls_token_representation]
        
        # --- 4. Define and Integrate Auxiliary Inputs (INC & SUB) ---
        if use_inc:
            inc_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name="inc_signal")
            inputs["inc_signal"] = inc_input
            features_to_concatenate.append(inc_input)
            
        if use_sub:
            sub_input = tf.keras.Input(shape=(1,), dtype=tf.float32, name="sub_signal")
            inputs["sub_signal"] = sub_input
            features_to_concatenate.append(sub_input)
            
        # Concatenate features if auxiliary signals are active
        if len(features_to_concatenate) > 1:
            combined_features = tf.keras.layers.Concatenate(name="feature_fusion")(features_to_concatenate)
        else:
            combined_features = cls_token_representation
  
        # --- 5. Classification Head ---
        x = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(combined_features)
        output_probability = tf.keras.layers.Dense(
            1, 
            activation="sigmoid", 
            name="classification_output"
        )(x)
  
        # Assemble Functional Keras Model
        model = tf.keras.models.Model(
            inputs=inputs, 
            outputs=output_probability, 
            name="DeBERTa_v3_Humor_Classifier"
        )
  
        return model


# ==============================================================================
# 3. PIPELINE TEST / INTEGRATION CHECK
# ==============================================================================
if __name__ == "__main__":
    print("🚀 Initializing Native Keras 3 Pipeline...")
    
    # Sample training data simulating PQA Dataset
    sample_questions = [
        "Will this banana slicer cut away my existential dread?",
        "What is the exact width of the blade in inches?"
    ]
    sample_titles = [
        "Hutzler 571 Banana Slicer",
        "Stainless Steel Chef Knife"
    ]
    
    # Dummy pre-computed auxiliary outputs (batch_size=2)
    dummy_inc_signals = [0.89, 0.12]
    dummy_sub_signals = [0.75, 0.05]
    
    max_len = 64
    preprocessor = DebertaPreprocessor(max_seq_length=max_len)
    
    # --------------------------------------------------------------------------
    # Scenario: Building Configuration 5 (INC_SUB_Q+T)
    # --------------------------------------------------------------------------
    print("\n📦 Preprocessing Inputs for configuration [INC_SUB_Q+T]...")
    tokenized_inputs = preprocessor.tokenize_pairs(
        questions=sample_questions,
        titles=sample_titles,
        inc_signals=dummy_inc_signals,
        sub_signals=dummy_sub_signals
    )
    
    print("\n[SUCCESS] Tokenization Output Keys and Shapes:")
    for key, val in tokenized_inputs.items():
        print(f" --> {key:<15} shape: {val.shape}")
        
    # Build Classifier Architecture matching Configuration 5
    factory = DebertaHumorModelFactory(max_seq_length=max_len)
    model = factory.build_classifier(use_title=True, use_inc=True, use_sub=True)
    
    print("\n" + "="*60)
    print("    DEBERTA-V3 CLASSIFIER MODEL SUMMARY [INC_SUB_Q+T]")
    print("="*60)
    model.summary()
    
    # Run dummy forward-pass inference check
    print("\n🔮 Running test forward-pass prediction...")
    test_predictions = model.predict(tokenized_inputs, verbose=0)
    
    print("\nPredictions output vector:")
    for q, p in zip(sample_questions, test_predictions):
        print(f" --> Predicted P(Humor): {p[0]*100:.2f}% | Question: \"{q}\"")
