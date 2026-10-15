import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformer_model import DebertaPreprocessor, DebertaHumorModelFactory

# Force CPU mode for Mac stability
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
tf.config.set_visible_devices([], 'GPU')

# Path Configuration
PROJECT_DIR = "/Users/sisi/Desktop/Masters_degree/5018/humor-detection"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
SAVED_DIR = os.path.join(PROJECT_DIR, "saved_models")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

os.makedirs(SAVED_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================================================================
# 1. MATHEMATICAL FOCAL LOSS (Proposed Methodology)
# ==============================================================================
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss to focus learning on hard, borderline-ironic questions."""
    def __init__(self, gamma=2.0, alpha=0.25, name="focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        cross_entropy = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        modulating_factor = tf.math.pow(1.0 - p_t, self.gamma)
        
        loss = modulating_factor * cross_entropy
        return tf.reduce_mean(loss)


# ==============================================================================
# 2. MAIN TRAINING PIPELINE
# ==============================================================================
def train_pipeline():
    print("Loading AWS e-commerce data splits...")
    humorous_df = pd.read_csv(os.path.join(DATA_DIR, "Humorous.csv"))
    unbiased_df = pd.read_csv(os.path.join(DATA_DIR, "Non-humorous-unbiased.csv"))
    
    # Clean columns
    humorous_df['question'] = humorous_df['question'].fillna('').astype(str)
    unbiased_df['question'] = unbiased_df['question'].fillna('').astype(str)
    humorous_df['title'] = humorous_df['product_description'].fillna('').astype(str)
    unbiased_df['title'] = unbiased_df['product_description'].fillna('').astype(str)
    
    humorous_df['label'] = 1
    unbiased_df['label'] = 0
    
    dataset = pd.concat([humorous_df, unbiased_df], ignore_index=True)
    
    # 70% Train, 15% Val, 15% Test splits
    train_val, test_df = train_test_split(dataset, test_size=0.15, random_state=42, stratify=dataset['label'])
    train_df, val_df = train_test_split(train_val, test_size=0.1765, random_state=42, stratify=train_val['label'])
    
    print(f"Dataset Split Sizes: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    # Preprocess text pairs into tokens
    print("\nTokenizing sequence pairs using DeBERTa preprocessor...")
    preprocessor = DebertaPreprocessor(max_seq_length=64)
    train_inputs = preprocessor.tokenize_pairs(train_df['question'].tolist(), train_df['title'].tolist())
    val_inputs = preprocessor.tokenize_pairs(val_df['question'].tolist(), val_df['title'].tolist())
    test_inputs = preprocessor.tokenize_pairs(test_df['question'].tolist(), test_df['title'].tolist())
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values

    # Build Native DeBERTa-v3 Classifier
    print("\nBuilding model architecture...")
    factory = DebertaHumorModelFactory(max_seq_length=64)
    model = factory.build_classifier(use_title=True, use_inc=False, use_sub=False)

    # --------------------------------------------------------------------------
    # WARMUP & COSINE LR SCHEDULE SETUP (Native Keras 3) [source: 2, 5]
    # --------------------------------------------------------------------------
    batch_size = 32
    epochs = 5
    steps_per_epoch = len(train_df) // batch_size # 419 steps [source: 9]
    total_steps = steps_per_epoch * epochs       # 2095 steps
    warmup_steps = int(total_steps * 0.1)        # 210 steps (10% of training) [source: 2]
    decay_steps = total_steps - warmup_steps     # 1885 steps [source: 2]

    # Create the CosineDecay schedule with integrated warmup [source: 5]
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,              # Starts at 0.0 for warmup [source: 5]
        decay_steps=decay_steps,                # Decay steps [source: 5]
        alpha=0.0,                              # Decays down to near 0.0 [source: 5]
        warmup_target=2e-5,                     # Target peak learning rate [source: 5]
        warmup_steps=warmup_steps               # Linear warmup steps [source: 5]
    )

    # Compile with Proposed AdamW Optimizer and Focal Loss
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.01),
        loss=FocalLoss(gamma=2.0),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    # Callbacks for Early Stopping
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3, 
            restore_best_weights=True,
            verbose=1
        )
    ]

    print(f"\nStarting fine-tuning training loop ({epochs} Epochs)...")
    print(f" --> Peak Learning Rate: 2e-5 | Warmup Steps: {warmup_steps} | Decay Steps: {decay_steps}")
    
    model.fit(
        train_inputs, y_train,
        validation_data=(val_inputs, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print("\nEvaluating model on the Unbiased Test Set...")
    eval_results = model.evaluate(test_inputs, y_test, verbose=0)
    
    loss = float(eval_results[0])
    acc = float(eval_results)
    precision = float(eval_results)
    recall = float(eval_results)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)

    print(f"\n[SUCCESS] Final Trained Test Accuracy: {acc * 100:.2f}%")
    print(f"          Final F1-Score: {f1:.4f}")
    
    # -------------------------------------------------------------
    # SAVE MODEL AND SPREADSHEETS
    # -------------------------------------------------------------
    model_save_path = os.path.join(SAVED_DIR, "DeBERTa_v3_FineTuned_Unbiased.keras")
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

    results_df = pd.DataFrame([{
        "Model Name": "DeBERTa-v3-Small (Proposed)",
        "Dataset": "Unbiased_PQA",
        "Test Accuracy (%)": round(acc * 100, 2),
        "F1-Score": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Test Loss": round(loss, 4),
        "Max Seq Length": 64,
        "Optimizer": "AdamW + Cosine Warmup (peak_lr=2e-5, decay=0.01)",
        "Loss Function": "Focal Loss (gamma=2.0)",
        "Trainable Params": model.count_params()
    }])

    excel_path = os.path.join(RESULTS_DIR, "transformer_finetuned_results.xlsx")
    results_df.to_excel(excel_path, index=False, sheet_name="Transformer Baselines")
    print(f"Results successfully written to: {excel_path}")

    # Free memory
    del model
    tf.keras.backend.clear_session()
    gc.collect()

if __name__ == "__main__":
    train_pipeline()
