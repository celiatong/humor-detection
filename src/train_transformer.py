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
# MAIN TRAINING PIPELINE
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

    # Preprocess text pairs into tokens using your preprocessor class
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

    # STABILIZED COMPILATION:
    # 1. We increase learning rate to 1e-4 to force CPU gradient updates [source: 9]
    # 2. We use standard BinaryCrossentropy to provide robust initial gradient flows
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(),
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
            patience=2, 
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("\nStarting fine-tuning training loop (5 Epochs)...")
    model.fit(
        train_inputs, y_train,
        validation_data=(val_inputs, y_val),
        epochs=5,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    print("\nEvaluating model on the Unbiased Test Set...")
    eval_results = model.evaluate(test_inputs, y_test, verbose=0)
    
    loss = float(eval_results[0])
    acc = float(eval_results[1])
    precision = float(eval_results[2])
    recall = float(eval_results[3])
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)

    print(f"\n[SUCCESS] Final Trained Test Accuracy: {acc * 100:.2f}%")
    print(f"          Final F1-Score: {f1:.4f}")
    
    # -------------------------------------------------------------
    # 1. SAVE MODEL INTO THE SPECIFIED FOLDER
    # -------------------------------------------------------------
    model_save_path = os.path.join(SAVED_DIR, "DeBERTa_v3_FineTuned_Unbiased.keras")
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

    # -------------------------------------------------------------
    # 2. SAVE RESULTS TO EXCEL
    # -------------------------------------------------------------
    results_df = pd.DataFrame([{
        "Model Name": "DeBERTa-v3-Small (Proposed)",
        "Dataset": "Unbiased_PQA",
        "Test Accuracy (%)": round(acc * 100, 2),
        "F1-Score": round(f1, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "Test Loss": round(loss, 4),
        "Max Seq Length": 64,
        "Optimizer": "AdamW (lr=1e-4, decay=0.01)",
        "Loss Function": "Binary Crossentropy",
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
