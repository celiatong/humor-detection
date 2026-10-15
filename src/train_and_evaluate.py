import os
import gc
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Concatenate, 
    Lambda, Conv1D, GlobalMaxPooling1D, Dropout
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. HARDWARE & ENVIRONMENT SETUP
# ==============================================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
tf.config.set_visible_devices([], 'GPU') # CPU stability mode

# ==============================================================================
# 2. CONFIGURATION CLASS
# ==============================================================================
class ExperimentConfig:
    """Central configuration for hyperparameters and paths."""
    def __init__(self, epochs=3, batch_size=32, patience=5):
        self.project_dir = "/Users/sisi/Desktop/Masters_degree/5018/humor-detection"
        self.data_dir = os.path.join(self.project_dir, "data")
        self.saved_models_dir = os.path.join(self.project_dir, "saved_models")
        self.results_dir = os.path.join(self.project_dir, "results")
        
        self.max_vocab_size = 10000
        self.max_seq_length = 50
        self.embedding_dim = 300
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = 42

        self.encoders = ["CNN", "LSTM"]
        self.feature_variants = ["Q", "Q+T", "INC_Q+T", "SUB_Q+T", "INC_SUB_Q+T"]
        
        os.makedirs(self.saved_models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Permanent live checkpoint path
        self.excel_path = os.path.join(
            self.results_dir, 
            f"humor_ablation_results_{self.epochs}epochs_live.xlsx"
        )

# ==============================================================================
# 3. DATASET MANAGER CLASS
# ==============================================================================
class DatasetManager:
    """Loads and formats the Biased and Unbiased datasets."""
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def _resolve_column(self, df, candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def load_raw_data(self):
        humorous_path = os.path.join(self.config.data_dir, "Humorous.csv")
        unbiased_path = os.path.join(self.config.data_dir, "Non-humorous-unbiased.csv")
        
        biased_candidates = ["Non-humorous-biased.csv", "Non-humours-biased.csv"]
        biased_path = None
        for b_name in biased_candidates:
            p = os.path.join(self.config.data_dir, b_name)
            if os.path.exists(p):
                biased_path = p
                break

        if not (os.path.exists(humorous_path) and os.path.exists(unbiased_path) and biased_path):
            raise FileNotFoundError(f"AWS data files missing in {self.config.data_dir}.")

        return pd.read_csv(humorous_path), pd.read_csv(unbiased_path), pd.read_csv(biased_path)

    def prepare_datasets(self):
        humorous_df, unbiased_df, biased_df = self.load_raw_data()

        def clean_df(df, default_label):
            q_col = self._resolve_column(df, ['question', 'question_text', 'text'])
            t_col = self._resolve_column(df, ['product_description', 'product_title', 'title', 'description'])
            l_col = self._resolve_column(df, ['label', 'is_humorous', 'target'])

            cleaned = pd.DataFrame()
            cleaned['question'] = df[q_col].fillna('').astype(str) if q_col else ""
            cleaned['title'] = df[t_col].fillna('').astype(str) if t_col else "Product"
            cleaned['label'] = df[l_col].astype(int) if l_col else default_label
            return cleaned

        h_clean = clean_df(humorous_df, default_label=1)
        u_clean = clean_df(unbiased_df, default_label=0)
        b_clean = clean_df(biased_df, default_label=0)

        biased_dataset = pd.concat([h_clean, b_clean], ignore_index=True)
        unbiased_dataset = pd.concat([h_clean, u_clean], ignore_index=True)

        print(f" Loaded Biased PQA: {len(biased_dataset)} samples")
        print(f" Loaded Unbiased PQA: {len(unbiased_dataset)} samples")

        return {
            "Biased_PQA": biased_dataset,
            "Unbiased_PQA": unbiased_dataset
        }

# ==============================================================================
# 4. TEXT PREPROCESSOR & FEATURE EXTRACTOR
# ==============================================================================
class TextPreprocessor:
    """Handles tokenization and sequence padding."""
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tokenizer = Tokenizer(num_words=config.max_vocab_size, oov_token="<unk>")

    def fit_and_transform(self, train_df, val_df, test_df):
        all_text = train_df['question'].tolist() + train_df['title'].tolist()
        self.tokenizer.fit_on_texts(all_text)

        def encode_seqs(df):
            q_seq = pad_sequences(
                self.tokenizer.texts_to_sequences(df['question']),
                maxlen=self.config.max_seq_length,
                padding='post'
            )
            t_seq = pad_sequences(
                self.tokenizer.texts_to_sequences(df['title']),
                maxlen=self.config.max_seq_length,
                padding='post'
            )
            y = df['label'].values
            return q_seq, t_seq, y

        X_q_train, X_t_train, y_train = encode_seqs(train_df)
        X_q_val, X_t_val, y_val = encode_seqs(val_df)
        X_q_test, X_t_test, y_test = encode_seqs(test_df)

        return (X_q_train, X_t_train, y_train), (X_q_val, X_t_val, y_val), (X_q_test, X_t_test, y_test)

# ==============================================================================
# 5. MODEL FACTORY CLASS (Keras 3 Scope Safe)
# ==============================================================================
class ModelFactory:
    """Builds functional Keras 3 neural network architectures."""
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def build_incongruity_extractor(self):
        embedding = Embedding(self.config.max_vocab_size, self.config.embedding_dim)
        lstm_branch = LSTM(64)
        q_in = Input(shape=(self.config.max_seq_length,), name="inc_q_in")
        t_in = Input(shape=(self.config.max_seq_length,), name="inc_t_in")

        q_vec = lstm_branch(embedding(q_in))
        t_vec = lstm_branch(embedding(t_in))

        dist = Lambda(
            lambda x: tf.exp(-tf.reduce_sum(tf.abs(x[0] - x[1]), axis=1, keepdims=True)),
            name="manhattan_similarity"
        )([q_vec, t_vec])

        return Model(inputs=[q_in, t_in], outputs=dist, name="Incongruity_Extractor")

    def build_subjectivity_extractor(self):
        embedding = Embedding(self.config.max_vocab_size, self.config.embedding_dim)
        q_in = Input(shape=(self.config.max_seq_length,), name="subj_q_in")
        rep = LSTM(64)(embedding(q_in))
        return Model(inputs=q_in, outputs=rep, name="Subjectivity_Extractor")

    def build_classifier(self, encoder_type: str, variant: str):
        safe_name = f"Humor_{encoder_type}_{variant.replace('+', '_')}"
        embedding = Embedding(self.config.max_vocab_size, self.config.embedding_dim)

        def encode(x_input, prefix):
            emb = embedding(x_input)
            if encoder_type == "CNN":
                conv = Conv1D(128, kernel_size=5, activation='relu', name=f"{prefix}_conv1d")(emb)
                return GlobalMaxPooling1D(name=f"{prefix}_gpool")(conv)
            else:
                return LSTM(64, name=f"{prefix}_lstm")(emb)

        inputs = []
        features = []

        # Question input
        q_in = Input(shape=(self.config.max_seq_length,), name="input_question")
        inputs.append(q_in)
        features.append(encode(q_in, "q"))

        # Product Title input
        if "T" in variant:
            t_in = Input(shape=(self.config.max_seq_length,), name="input_title")
            inputs.append(t_in)
            features.append(encode(t_in, "t"))

        # Incongruity input
        if "INC" in variant:
            inc_in = Input(shape=(1,), name="input_incongruity_score")
            inputs.append(inc_in)
            features.append(inc_in)

        # Subjectivity input
        if "SUB" in variant:
            sub_in = Input(shape=(64,), name="input_subjectivity_vector")
            inputs.append(sub_in)
            features.append(sub_in)

        merged = Concatenate(name="feature_concat")(features) if len(features) > 1 else features[0]
        dense = Dense(128, activation='relu', name="dense_hidden")(merged)
        dropout = Dropout(0.3, name="dense_dropout")(dense)
        output = Dense(1, activation='sigmoid', name="classification_output")(dropout)

        model = Model(inputs=inputs, outputs=output, name=safe_name)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        return model

# ==============================================================================
# 6. MODEL TRAINER CLASS
# ==============================================================================
class ModelTrainer:
    """Executes model training, evaluation, and saves artifacts immediately."""
    def __init__(self, config: ExperimentConfig):
        self.config = config

    def train_and_eval(self, model: Model, train_inputs, y_train, val_inputs, y_val, test_inputs, y_test, model_display_name):
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0)
        ]
        
        if self.config.epochs >= 10:
            callbacks.append(
                EarlyStopping(monitor='val_loss', patience=self.config.patience, restore_best_weights=True, verbose=0)
            )

        print(f"  --> [1/2 Training] {model_display_name} for {self.config.epochs} epochs...")
        model.fit(
            train_inputs, y_train,
            validation_data=(val_inputs, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=0
        )

        print(f"  --> [2/2 Evaluating] Evaluating {model_display_name} immediately on Test Set...")
        eval_results = model.evaluate(test_inputs, y_test, verbose=0)
        
        loss = float(eval_results[0])
        acc = float(eval_results[1])
        precision = float(eval_results[2])
        recall = float(eval_results[3])
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)

        # Save model weights to disk
        save_path = os.path.join(self.config.saved_models_dir, f"{model.name}.keras")
        model.save(save_path)

        # Flush Keras computation graph memory
        del model
        tf.keras.backend.clear_session()
        gc.collect()

        return {
            "Test Accuracy (%)": round(acc * 100, 2),
            "F1-Score": round(float(f1), 4),
            "Precision": round(float(precision), 4),
            "Recall": round(float(recall), 4),
            "Test Loss": round(float(loss), 4)
        }

# ==============================================================================
# 7. ABLATION STUDY PIPELINE (WITH LIVE INCREMENTAL EXCEL SAVE)
# ==============================================================================
class AblationStudyPipeline:
    """Orchestrates runs and writes results to Excel immediately after each model."""
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.dataset_manager = DatasetManager(config)
        self.preprocessor = TextPreprocessor(config)
        self.factory = ModelFactory(config)
        self.trainer = ModelTrainer(config)
        self.all_results = []

    def _prepare_inputs(self, variant, q, t, inc, sub):
        if variant == "Q":
            return q
        elif variant == "Q+T":
            return [q, t]
        elif variant == "INC_Q+T":
            return [q, t, inc]
        elif variant == "SUB_Q+T":
            return [q, t, sub]
        elif variant == "INC_SUB_Q+T":
            return [q, t, inc, sub]
        raise ValueError(f"Unknown variant: {variant}")

    def _save_incremental_excel(self):
        """Writes current completed results to disk after every single model run."""
        if not self.all_results:
            return

        df_all = pd.DataFrame(self.all_results)
        
        with pd.ExcelWriter(self.config.excel_path, engine='openpyxl') as writer:
            # 1. Raw Detailed Results sheet
            df_all.to_excel(writer, sheet_name="All_Completed_Runs", index=False)

            # 2. Side-by-Side Comparison Pivot (if data is present)
            if "Dataset" in df_all.columns and "Model Architecture" in df_all.columns:
                try:
                    pivot_df = df_all.pivot(
                        index="Model Architecture", 
                        columns="Dataset", 
                        values="Test Accuracy (%)"
                    ).reset_index()

                    if "Biased_PQA" in pivot_df.columns and "Unbiased_PQA" in pivot_df.columns:
                        pivot_df["Accuracy Delta (Biased - Unbiased)"] = (
                            pivot_df["Biased_PQA"] - pivot_df["Unbiased_PQA"]
                        ).round(2)

                    pivot_df.to_excel(writer, sheet_name="Comparison_Summary", index=False)
                except Exception:
                    pass

            # 3. Metadata sheet
            metadata_df = pd.DataFrame({
                "Parameter": ["Epochs", "Batch Size", "Max Vocab Size", "Seq Length", "Embedding Dim", "Last Update"],
                "Value": [self.config.epochs, self.config.batch_size, self.config.max_vocab_size, 
                          self.config.max_seq_length, self.config.embedding_dim, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            })
            metadata_df.to_excel(writer, sheet_name="Metadata", index=False)

    def run(self):
        datasets = self.dataset_manager.prepare_datasets()

        for dataset_name, df in datasets.items():
            print(f"\n=======================================================")
            print(f" BENCHMARKING DATASET: {dataset_name}")
            print(f"=======================================================")

            # 70% Train, 15% Val, 15% Test
            train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=self.config.random_state, stratify=df['label'])
            train_df, val_df = train_test_split(train_val_df, test_size=0.1765, random_state=self.config.random_state, stratify=train_val_df['label'])

            (X_q_tr, X_t_tr, y_tr), (X_q_val, X_t_val, y_val), (X_q_te, X_t_te, y_te) = \
                self.preprocessor.fit_and_transform(train_df, val_df, test_df)

            print(" Generating linguistic signals (Incongruity & Subjectivity)...")
            incong_extractor = self.factory.build_incongruity_extractor()
            inc_tr = incong_extractor.predict([X_q_tr, X_t_tr], verbose=0)
            inc_val = incong_extractor.predict([X_q_val, X_t_val], verbose=0)
            inc_te = incong_extractor.predict([X_q_te, X_t_te], verbose=0)
            del incong_extractor
            tf.keras.backend.clear_session()
            gc.collect()

            subj_extractor = self.factory.build_subjectivity_extractor()
            sub_tr = subj_extractor.predict(X_q_tr, verbose=0)
            sub_val = subj_extractor.predict(X_q_val, verbose=0)
            sub_te = subj_extractor.predict(X_q_te, verbose=0)
            del subj_extractor
            tf.keras.backend.clear_session()
            gc.collect()

            for encoder in self.config.encoders:
                for variant in self.config.feature_variants:
                    model_display_name = f"{encoder}_{variant}"
                    
                    tr_in = self._prepare_inputs(variant, X_q_tr, X_t_tr, inc_tr, sub_tr)
                    val_in = self._prepare_inputs(variant, X_q_val, X_t_val, inc_val, sub_val)
                    te_in = self._prepare_inputs(variant, X_q_te, X_t_te, inc_te, sub_te)

                    model = self.factory.build_classifier(encoder, variant)
                    
                    # 1. Train and Evaluate immediately
                    metrics = self.trainer.train_and_eval(
                        model, tr_in, y_tr, val_in, y_val, te_in, y_te, model_display_name
                    )

                    # 2. Append metrics
                    record = {
                        "Dataset": dataset_name,
                        "Model Architecture": model_display_name,
                        "Encoder": encoder,
                        "Feature Config": variant,
                        **metrics
                    }
                    self.all_results.append(record)

                    # 3. Save to Excel right away!
                    self._save_incremental_excel()
                    print(f"  [SAVED TO EXCEL] {dataset_name} | {model_display_name}: "
                          f"Acc={metrics['Test Accuracy (%)']}%, F1={metrics['F1-Score']}")

        print(f"\nAll experiments complete! Live file fully updated at:\n  --> {self.config.excel_path}\n")

# ==============================================================================
# 8. ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Humor Detection Ablation Study")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (e.g. 3, 10, 50, 100)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    print(f"Starting pipeline with {args.epochs} epochs...")
    config = ExperimentConfig(epochs=args.epochs, batch_size=args.batch_size)
    pipeline = AblationStudyPipeline(config)
    pipeline.run()
