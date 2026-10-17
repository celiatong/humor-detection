import os
import gc
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformer_model import DebertaPreprocessor, DebertaHumorModelFactory

# ------------------------------------------------------------------------------
# 1. GPU Configuration & Mixed Precision
# ------------------------------------------------------------------------------
try:
    # Crucial for maximizing Tensor Cores on H800
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("[INFO] Mixed precision (float16) enabled for faster GPU training.")
except ValueError as e:
    print(f"[WARN] Mixed precision policy could not be set: {e}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] Detected {len(gpus)} GPU(s). Dynamic memory growth enabled.")
    except RuntimeError as e:
        print(f"[WARN] Error setting memory growth: {e}")
else:
    print("[WARN] No GPU detected. Running on CPU.")

# ------------------------------------------------------------------------------
# 2. Argument Parsing
# ------------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="DeBERTa-v3 Humor Detection Fine-Tuning")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing dataset CSVs")
    parser.add_argument("--output_dir", type=str, default="./saved_models", help="Directory to save weights")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory for metric summaries")
    # ENHANCED: Increased default batch size to 128 to leverage H800 memory
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate for scheduler")
    # ENHANCED: Increased max sequence length to 128 to prevent text clipping
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum token length")
    return parser.parse_args()

# ------------------------------------------------------------------------------
# 3. Training & Evaluation Pipeline
# ------------------------------------------------------------------------------
def train_pipeline():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    humorous_path = os.path.join(args.data_dir, "Humorous.csv")
    unbiased_path = os.path.join(args.data_dir, "Non-humorous-unbiased.csv")

    if not os.path.exists(humorous_path) or not os.path.exists(unbiased_path):
        raise FileNotFoundError(
            f"Dataset files missing in '{args.data_dir}'."
        )

    print(f"[INFO] Loading datasets from: {args.data_dir}")
    humorous_df = pd.read_csv(humorous_path)
    unbiased_df = pd.read_csv(unbiased_path)

    humorous_df['question'] = humorous_df['question'].fillna('').astype(str)
    unbiased_df['question'] = unbiased_df['question'].fillna('').astype(str)
    humorous_df['title'] = humorous_df['product_description'].fillna('').astype(str)
    unbiased_df['title'] = unbiased_df['product_description'].fillna('').astype(str)

    humorous_df['label'] = 1
    unbiased_df['label'] = 0
    dataset = pd.concat([humorous_df, unbiased_df], ignore_index=True)

    # 70% Train, 15% Val, 15% Test
    train_val, test_df = train_test_split(dataset, test_size=0.15, random_state=42, stratify=dataset['label'])
    train_df, val_df = train_test_split(train_val, test_size=0.1765, random_state=42, stratify=train_val['label'])

    print(f"[INFO] Split sizes: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

    preprocessor = DebertaPreprocessor(max_seq_length=args.max_seq_length)
    train_inputs = preprocessor.tokenize_pairs(train_df['question'].tolist(), train_df['title'].tolist())
    val_inputs = preprocessor.tokenize_pairs(val_df['question'].tolist(), val_df['title'].tolist())
    test_inputs = preprocessor.tokenize_pairs(test_df['question'].tolist(), test_df['title'].tolist())

    y_train = train_df['label'].values.astype(np.float32)
    y_val = val_df['label'].values.astype(np.float32)
    y_test = test_df['label'].values.astype(np.float32)

    print("[INFO] Constructing DeBERTa-v3 architecture...")
    factory = DebertaHumorModelFactory(max_seq_length=args.max_seq_length)
    model = factory.build_classifier(use_title=True)

    # Cosine Decay warmup configuration (10% of total steps)
    steps_per_epoch = len(train_df) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.1)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr * 0.1,
        decay_steps=total_steps,
        alpha=0.01,
        warmup_target=args.lr,
        warmup_steps=warmup_steps
    )

    # Binary Focal Cross-entropy Loss to focus on hard misclassifications
    loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=False,
        gamma=2.0
    )

    # ENHANCED: Added clipnorm to prevent gradient explosions in mixed-precision scaling
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule, 
            weight_decay=0.01,
            clipnorm=1.0
        ),
        loss=loss_fn,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print(f"[INFO] Launching training ({args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr})...")
    history = model.fit(
        train_inputs, y_train,
        validation_data=(val_inputs, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print("[INFO] Evaluating on unbiased test set...")
    eval_results = model.evaluate(test_inputs, y_test, verbose=0)
    loss = float(eval_results[0])
    acc = float(eval_results)
    precision = float(eval_results)
    recall = float(eval_results)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)

    print(f"\n[METRICS] Test Loss:     {loss:.4f}")
    print(f"[METRICS] Test Accuracy: {acc * 100:.2f}%")
    print(f"[METRICS] Precision:     {precision:.4f}")
    print(f"[METRICS] Recall:        {recall:.4f}")
    print(f"[METRICS] F1-Score:      {f1:.4f}\n")

    # Save Model Weights
    model_save_path = os.path.join(args.output_dir, "DeBERTa_v3_FineTuned_Unbiased_Enhanced.keras")
    model.save(model_save_path)
    print(f"[SAVED] Checkpoint: {model_save_path}")

    # Export Metrics Summary
    results_dict = {
        "Model Name": ["DeBERTa-v3-Small (Enhanced)"],
        "Dataset": ["Unbiased_PQA"],
        "Test Accuracy (%)": [round(acc * 100, 2)],
        "F1-Score": [round(f1, 4)],
        "Precision": [round(precision, 4)],
        "Recall": [round(recall, 4)],
        "Test Loss": [round(loss, 4)],
        "Max Seq Length": [args.max_seq_length],
        "Optimizer": [f"AdamW (lr={args.lr}, CosineDecay)"],
        "Batch Size": [args.batch_size],
        "Epochs Trained": [len(history.epoch)],
        "Trainable Params": [model.count_params()]
    }
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(os.path.join(args.results_dir, "transformer_enhanced_results.csv"), index=False)
    try:
        results_df.to_excel(os.path.join(args.results_dir, "transformer_enhanced_results.xlsx"), index=False)
    except Exception:
        pass
    print("[SAVED] Enhanced results reports.")

    del model
    tf.keras.backend.clear_session()
    gc.collect()

if __name__ == "__main__":
    train_pipeline()
