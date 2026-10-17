import os
import gc
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformer_model import DebertaPreprocessor, DebertaHumorModelFactory

# GPU Configuration
try:
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
except ValueError as e:
    print(f"[WARN] Mixed precision policy could not be set: {e}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"[WARN] Error setting memory growth: {e}")
else:
    print("[WARN] No GPU detected.")

# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="DeBERTa-v3 Humor Detection")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./saved_models")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=128)
    return parser.parse_args()

# Training Pipeline
def train_pipeline():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    humorous_path = os.path.join(args.data_dir, "Humorous.csv")
    unbiased_path = os.path.join(args.data_dir, "Non-humorous-unbiased.csv")

    humorous_df = pd.read_csv(humorous_path)
    unbiased_df = pd.read_csv(unbiased_path)

    humorous_df['question'] = humorous_df['question'].fillna('').astype(str)
    unbiased_df['question'] = unbiased_df['question'].fillna('').astype(str)
    humorous_df['title'] = humorous_df['product_description'].fillna('').astype(str)
    unbiased_df['title'] = unbiased_df['product_description'].fillna('').astype(str)
    humorous_df['label'] = 1
    unbiased_df['label'] = 0
    dataset = pd.concat([humorous_df, unbiased_df], ignore_index=True)

    train_val, test_df = train_test_split(dataset, test_size=0.15, random_state=42, stratify=dataset['label'])
    train_df, val_df = train_test_split(train_val, test_size=0.1765, random_state=42, stratify=train_val['label'])

    preprocessor = DebertaPreprocessor(max_seq_length=args.max_seq_length)
    train_inputs = preprocessor.tokenize_pairs(train_df['question'].tolist(), train_df['title'].tolist())
    val_inputs = preprocessor.tokenize_pairs(val_df['question'].tolist(), val_df['title'].tolist())
    test_inputs = preprocessor.tokenize_pairs(test_df['question'].tolist(), test_df['title'].tolist())

    y_train = train_df['label'].values.astype(np.float32)
    y_val = val_df['label'].values.astype(np.float32)
    y_test = test_df['label'].values.astype(np.float32)

    factory = DebertaHumorModelFactory(max_seq_length=args.max_seq_length)
    model = factory.build_classifier(use_title=True)

    steps_per_epoch = len(train_df) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.1)
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr * 0.1, decay_steps=total_steps,
        alpha=0.01, warmup_target=args.lr, warmup_steps=warmup_steps
    )
    
    loss_fn = tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=False, gamma=2.0)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=0.01, clipnorm=1.0),
        loss=loss_fn,
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)]

    history = model.fit(
        train_inputs, y_train, validation_data=(val_inputs, y_val),
        epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=1
    )

    print("[INFO] Evaluating on unbiased test set...")
    eval_results = model.evaluate(test_inputs, y_test, verbose=0)

    # Correct unpacking of evaluation metrics
    loss, acc, precision, recall = eval_results
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)

    print(f"\n[METRICS] Test Loss:     {loss:.4f}")
    print(f"[METRICS] Test Accuracy: {acc * 100:.2f}%")
    print(f"[METRICS] Precision:     {precision:.4f}")
    print(f"[METRICS] Recall:        {recall:.4f}")
    print(f"[METRICS] F1-Score:      {f1:.4f}\n")

    model_save_path = os.path.join(args.output_dir, "DeBERTa_v3_FineTuned_Enhanced.keras")
    model.save(model_save_path)
    print(f"[SAVED] Checkpoint: {model_save_path}")

    # ... (rest of reporting logic)

    del model
    tf.keras.backend.clear_session()
    gc.collect()

if __name__ == "__main__":
    train_pipeline()
