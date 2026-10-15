import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Force CPU mode for Mac stability
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
tf.config.set_visible_devices([], 'GPU')

# ==============================================================================
# QUALITATIVE ERROR ANALYZER CLASS (OOP)
# ==============================================================================
class QualitativeAnalyzer:
    """Performs qualitative error analysis on trained humor detection models."""

    def __init__(self, dataset_name="Unbiased_PQA", model_filename="Humor_CNN_INC_SUB_Q_T.keras"):
        self.project_dir = "/Users/sisi/Desktop/Masters_degree/5018/humor-detection"
        self.data_dir = os.path.join(self.project_dir, "data")
        self.models_dir = os.path.join(self.project_dir, "saved_models")
        self.results_dir = os.path.join(self.project_dir, "results")
        
        self.dataset_name = dataset_name
        self.model_filename = model_filename
        self.max_vocab_size = 10000
        self.max_seq_length = 50
        self.random_state = 42

        os.makedirs(self.results_dir, exist_ok=True)

    def _resolve_column(self, df, candidates):
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def load_data_and_split(self):
        """Loads and prepares the exact 15% test split used in training."""
        humorous_path = os.path.join(self.data_dir, "Humorous.csv")
        unbiased_path = os.path.join(self.data_dir, "Non-humorous-unbiased.csv")
        biased_candidates = ["Non-humorous-biased.csv", "Non-humours-biased.csv"]
        biased_path = next((os.path.join(self.data_dir, b) for b in biased_candidates if os.path.exists(os.path.join(self.data_dir, b))), None)

        humorous_df = pd.read_csv(humorous_path)
        unbiased_df = pd.read_csv(unbiased_path)
        biased_df = pd.read_csv(biased_path)

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

        target_df = pd.concat([h_clean, u_clean if "Unbiased" in self.dataset_name else b_clean], ignore_index=True)

        # Recreate exact 70/15/15 test split
        train_val_df, test_df = train_test_split(target_df, test_size=0.15, random_state=self.random_state, stratify=target_df['label'])
        train_df, _ = train_test_split(train_val_df, test_size=0.1765, random_state=self.random_state, stratify=train_val_df['label'])

        return train_df, test_df

    def prepare_test_inputs(self, train_df, test_df, model):
        """Tokenizes sequences and formats model inputs based on model requirements."""
        tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token="<unk>")
        tokenizer.fit_on_texts(train_df['question'].tolist() + train_df['title'].tolist())

        X_q_test = pad_sequences(tokenizer.texts_to_sequences(test_df['question']), maxlen=self.max_seq_length, padding='post')
        X_t_test = pad_sequences(tokenizer.texts_to_sequences(test_df['title']), maxlen=self.max_seq_length, padding='post')

        # Check model expected inputs
        num_inputs = len(model.inputs)
        if num_inputs == 1:
            return X_q_test
        elif num_inputs == 2:
            return [X_q_test, X_t_test]
        elif num_inputs == 3:
            dummy_inc = np.zeros((len(test_df), 1))
            return [X_q_test, X_t_test, dummy_inc]
        elif num_inputs == 4:
            dummy_inc = np.zeros((len(test_df), 1))
            dummy_sub = np.zeros((len(test_df), 64))
            return [X_q_test, X_t_test, dummy_inc, dummy_sub]
        else:
            return X_q_test

    def analyze(self, top_k=10):
        # 1. Load Model
        model_path = os.path.join(self.models_dir, self.model_filename)
        if not os.path.exists(model_path):
            available = [f for f in os.listdir(self.models_dir) if f.endswith(".keras")]
            if available:
                print(f"[!] '{self.model_filename}' not found. Defaulting to available model: '{available[0]}'")
                model_path = os.path.join(self.models_dir, available[0])
                self.model_filename = available[0]
            else:
                raise FileNotFoundError(f"No trained .keras models found in {self.models_dir}")

        print(f"Loading trained model: {self.model_filename}...")
        model = tf.keras.models.load_model(model_path, compile=False)

        # 2. Prepare Data
        train_df, test_df = self.load_data_and_split()
        test_inputs = self.prepare_test_inputs(train_df, test_df, model)

        # 3. Predict Probabilities
        print(f"Running inference on {len(test_df)} test records...")
        probabilities = model.predict(test_inputs, verbose=0).flatten()

        results_df = test_df.copy().reset_index(drop=True)
        results_df['humor_probability'] = np.round(probabilities, 4)
        results_df['predicted_label'] = (probabilities >= 0.5).astype(int)

        # 4. Extract Key Error Groups
        # True Positives: Ground Truth = 1, Predicted = 1 (Highest probability first)
        true_positives = results_df[
            (results_df['label'] == 1) & (results_df['predicted_label'] == 1)
        ].sort_values(by='humor_probability', ascending=False)

        # False Negatives (Missed Humor): Ground Truth = 1, Predicted = 0 (Lowest probability first)
        false_negatives = results_df[
            (results_df['label'] == 1) & (results_df['predicted_label'] == 0)
        ].sort_values(by='humor_probability', ascending=True)

        # False Positives: Ground Truth = 0, Predicted = 1 (Highest probability first)
        false_positives = results_df[
            (results_df['label'] == 0) & (results_df['predicted_label'] == 1)
        ].sort_values(by='humor_probability', ascending=False)

        # 5. Display Formatted Previews in Console
        self._print_records("TRUE POSITIVES (Highly Confident Humor)", true_positives.head(top_k))
        self._print_records("FALSE NEGATIVES (Missed Humor - 'Should be funny, but model said NO')", false_negatives.head(top_k))
        self._print_records("FALSE POSITIVES (Serious Questions Falsely Flagged as Funny)", false_positives.head(top_k))

        # 6. Save to Excel
        output_excel = os.path.join(self.results_dir, f"qualitative_records_{self.model_filename.replace('.keras', '')}.xlsx")
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            true_positives.to_excel(writer, sheet_name="True_Positives", index=False)
            false_negatives.to_excel(writer, sheet_name="False_Negatives_Missed", index=False)
            false_positives.to_excel(writer, sheet_name="False_Positives", index=False)
            results_df.to_excel(writer, sheet_name="All_Test_Predictions", index=False)

        print(f"\n [SUCCESS] Full qualitative analysis saved to:\n  --> {output_excel}\n")

    def _print_records(self, title, df_subset):
        print("\n" + "=" * 80)
        print(f" {title}")
        print("=" * 80)
        if df_subset.empty:
            print("No records found for this category.")
            return

        for idx, row in df_subset.iterrows():
            prob = row['humor_probability']
            q = row['question']
            t = row['title']
            print(f"• Prob: {prob*100:5.1f}% | Product : {t[:50]}")
            print(f"               | Question: \"{q}\"")
            print("-" * 80)


# ==============================================================================
# SCRIPT ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qualitative Error Analysis for Humor Detection")
    parser.add_argument("--dataset", type=str, default="Unbiased_PQA", choices=["Unbiased_PQA", "Biased_PQA"])
    parser.add_argument("--model", type=str, default="Humor_CNN_INC_SUB_Q_T.keras", help="Model file in saved_models/")
    parser.add_argument("--top_k", type=int, default=5, help="Number of records to display in console")
    args = parser.parse_args()

    analyzer = QualitativeAnalyzer(dataset_name=args.dataset, model_filename=args.model)
    analyzer.analyze(top_k=args.top_k)
