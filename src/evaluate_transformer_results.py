"""Compare Transformer inputs across three seeds and select an ablation candidate."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from train_transformer_pytorch import metrics


SEEDS = (42, 43, 44)
BACKBONES = (
    ("roberta_base", "RoBERTa-base"),
    ("roberta_large", "RoBERTa-large"),
    ("deberta_v3_base", "DeBERTa-v3-base"),
)
INPUTS = (("qonly", "Question only"), ("qproduct", "Question + Product"))
TEST_ROWS = 3585
SELECTION_RULE = "highest mean test accuracy, then highest mean test macro-F1"
LOCAL_BACKBONES = {
    "deberta_v3_small": ("deberta", "deberta-v3-small", "DeBERTa-v3-Small"),
    "roberta_large": ("roberta", "roberta-large", "RoBERTa-large"),
}
LOCAL_SETTINGS = {
    "input": "qproduct",
    "epochs": 3,
    "batch_size": 16,
    "max_length": 128,
    "lr": 2e-5,
    "precision": "fp32",
    "fixed_padding": True,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing_enabled": False,
    "bf16": False,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "validation_only": False,
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def check_hash(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"{path}: SHA-256 does not match the evaluation record")


def read_labels(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if "label" not in (reader.fieldnames or []):
            raise ValueError(f"{path}: missing label column")
        labels = [int(row["label"]) for row in reader]
    if len(labels) != TEST_ROWS or set(labels) != {0, 1}:
        raise ValueError(f"{path}: expected {TEST_ROWS} rows with binary labels")
    return labels


def verified_scores(sidecar, predictions, labels, expected, signature=None):
    record = read_json(sidecar)
    saved = record["metrics"]
    if signature is not None and record["signature"] != signature:
        raise ValueError(f"{sidecar}: execution signature does not match")
    for field, value in expected.items():
        if saved.get(field) != value:
            raise ValueError(f"{sidecar}: {field} should be {value!r}")
    check_hash(predictions, record["predictions_sha256"])
    return score_predictions(predictions, labels, sidecar, {
        "accuracy": saved["test_accuracy"], "macro_f1": saved["test_macro_f1"],
    })


def score_predictions(predictions, labels, sidecar, saved):
    predicted = []
    with predictions.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"row_index", "actual", "predicted"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{predictions}: missing prediction columns")
        for index, row in enumerate(reader):
            if index >= len(labels) or int(row["row_index"]) != index:
                raise ValueError(f"{predictions}: unexpected row index at row {index}")
            if int(row["actual"]) != labels[index] or row["predicted"] not in ("0", "1"):
                raise ValueError(f"{predictions}: invalid label or row alignment at row {index}")
            predicted.append(int(row["predicted"]))

    score = metrics(labels, predicted)
    for key in ("accuracy", "macro_f1"):
        if not math.isclose(score[key], saved[key], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{sidecar}: saved test_{key} differs from the predictions")
    return score


def local_scores(root, labels, test_hash):
    results = {}
    reference = None
    for model, (folder, model_name, _) in LOCAL_BACKBONES.items():
        results[model] = []
        revision = None
        for seed in SEEDS:
            run = root / f"matched_fp32_{folder}_seed{seed}"
            config = read_json(run / "run.json")
            expected = {**LOCAL_SETTINGS, "seed": seed, "model": model_name}
            if any(config[key] != value for key, value in expected.items()) or config.get("ablation"):
                raise ValueError(f"{run}: unexpected matched comparison settings")
            if config["splits"]["test"]["sha256"] != test_hash:
                raise ValueError(f"{run}: test split differs from the original unbiased split")
            environment = (config["device"], config["torch"], config["transformers"], config["splits"])
            reference = environment if reference is None else reference
            revision = config["resolved_revision"] if revision is None else revision
            if environment != reference or not revision or config["resolved_revision"] != revision:
                raise ValueError(f"{run}: inconsistent environment or model revision")
            sidecar = run / "test_metrics.json"
            saved = read_json(sidecar)
            history = read_json(run / "history.json")
            if [row["epoch"] for row in history] != list(range(1, config["epochs"] + 1)):
                raise ValueError(f"{run}: incomplete validation history")
            best = max(history, key=lambda row: row["validation"]["macro_f1"])
            if saved["test_rows"] != len(labels) or saved["best_epoch"] != best["epoch"]:
                raise ValueError(f"{run}: inconsistent test row count or selected epoch")
            if not math.isclose(saved["validation_macro_f1"], best["validation"]["macro_f1"]):
                raise ValueError(f"{run}: inconsistent validation macro-F1")
            results[model].append(score_predictions(run / "predictions.csv", labels, sidecar, saved))
    return results


def make_comparison(final_test, test_csv, matched_results=None):
    signature = read_json(final_test / "execution.json")
    test_hash = signature["test_csv_sha256"]
    check_hash(test_csv, test_hash)
    labels = read_labels(test_csv)

    lines = [
        "### Transformer input comparison (unbiased test, seeds 42–44)",
        "",
        "| Backbone | Input | Test accuracy (mean ± sample SD) | Mean macro-F1 |",
        "|---|---|---:|---:|",
    ]
    local = {}
    backbones = BACKBONES
    if matched_results is not None:
        local = local_scores(matched_results, labels, test_hash)
        backbones = (("deberta_v3_small", LOCAL_BACKBONES["deberta_v3_small"][2]), BACKBONES[2], *BACKBONES[:2])
    candidates = []
    for model, name in backbones:
        inputs = INPUTS[1:] if model == "deberta_v3_small" else INPUTS
        for mode, input_label in inputs:
            model_key = f"{model}_{mode}"
            if mode == "qproduct" and model in local:
                scores = local[model]
            else:
                scores = []
                for seed in SEEDS:
                    path = final_test / f"{model_key}_seed{seed}"
                    scores.append(verified_scores(
                        path.with_suffix(".json"), path.with_suffix(".csv"), labels,
                        {"model": model_key, "seed": seed, "test_rows": len(labels)}, signature,
                    ))
            accuracy = [row["accuracy"] for row in scores]
            f1 = mean(row["macro_f1"] for row in scores)
            accuracy_mean, accuracy_sd = mean(accuracy), stdev(accuracy)
            candidates.append({
                "backbone": model, "name": name, "input": mode, "input_label": input_label,
                "mean_test_accuracy": accuracy_mean, "test_accuracy_sd": accuracy_sd, "mean_macro_f1": f1,
                "source": "matched local" if mode == "qproduct" and model in local else "SuperPOD",
            })
            lines.append(
                f"| {name} | {input_label} | {accuracy_mean:.2%} ± "
                f"{accuracy_sd * 100:.2f} pp | {f1:.4f} |"
            )

    if local:
        lines.extend(["", "DeBERTa-v3-Small and RoBERTa-large Question + Product use the matched local runs. "
                      "The other rows use the archived SuperPOD runs. All rows use the same unbiased test split."])
    winner = select_winner(candidates)
    lines.extend(["", f"Selected for ablation by mean test accuracy: {winner['name']} "
                  f"({winner['input_label']}, {winner['mean_test_accuracy']:.2%})."])
    selection = {
        "selection_rule": SELECTION_RULE, "seeds": list(SEEDS), "test_csv_sha256": test_hash,
        "candidates": candidates, "winner": winner,
    }
    return "\n".join(lines), selection


def select_winner(candidates):
    ranked = sorted(
        candidates, key=lambda row: (row["mean_test_accuracy"], row["mean_macro_f1"]), reverse=True,
    )
    if len(ranked) < 2:
        raise ValueError("At least two candidates are required")
    if all(ranked[0][metric] == ranked[1][metric] for metric in ("mean_test_accuracy", "mean_macro_f1")):
        raise ValueError("The top two candidates tied on mean test accuracy and macro-F1")
    return ranked[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-test", type=Path, required=True, help="Archived final_test_v1 directory")
    parser.add_argument("--test-csv", type=Path, required=True, help="Original unbiased test.csv")
    parser.add_argument("--matched-results", type=Path, help="Results directory with the six matched local runs")
    parser.add_argument("--selection-out", type=Path, required=True, help="New JSON file for the ablation step")
    args = parser.parse_args()
    try:
        report, selection = make_comparison(args.final_test, args.test_csv, args.matched_results)
        with args.selection_out.open("x", encoding="utf-8") as stream:
            json.dump(selection, stream, indent=2)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        parser.exit(1, f"Cannot compare Transformer results: {error}\n")
    print(report)


if __name__ == "__main__":
    main()
