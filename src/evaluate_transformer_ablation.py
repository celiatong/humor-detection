"""Rebuild the ablation table for the selected Transformer configuration."""

import argparse
import csv
from pathlib import Path

from evaluate_transformer_results import (
    SEEDS, SELECTION_RULE, check_hash, read_json, read_labels, select_winner, verified_scores,
)


ABLATIONS = (
    ("control", "Question + Product", "Control", False, False),
    ("with_inc", "with_inc", "+INC", True, False),
    ("with_sub", "with_sub", "+SUB", False, True),
    ("with_inc_sub", "with_inc_sub", "+INC+SUB", True, True),
)


def make_ablation(selection_path, cross_dataset, test_csv):
    selection = read_json(selection_path)
    if selection["seeds"] != list(SEEDS) or selection["selection_rule"] != SELECTION_RULE:
        raise ValueError("Expected a selection from the three-seed Transformer comparison")
    winner = select_winner(selection["candidates"])
    if selection["winner"] != winner:
        raise ValueError("The saved winner does not match the comparison ranking")
    if (winner["backbone"], winner["input"]) != ("roberta_large", "qproduct"):
        raise ValueError(f"No archived ablations for {winner['backbone']} ({winner['input']})")

    fusion_run = read_json(cross_dataset / "execution.json")
    test_hash = fusion_run["data_sha256"]["unbiased"]["test"]
    if selection["test_csv_sha256"] != test_hash:
        raise ValueError("The comparison and ablation use different test splits")
    check_hash(test_csv, test_hash)
    labels = read_labels(test_csv)
    checkpoints = {}
    for run in fusion_run["runs"]:
        if run["training_dataset"] == "unbiased":
            model = run["model"]
            if model in checkpoints:
                raise ValueError(f"Duplicate unbiased fusion checkpoint: {model}")
            checkpoints[model] = run

    lines = [
        "### RoBERTa-large ablation (unbiased train/test, seed 42)",
        "",
        "| RoBERTa-large | Report label | Test accuracy |",
        "|---|---|---:|",
    ]
    for model, configuration, name, use_inc, use_sub in ABLATIONS:
        checkpoint = checkpoints[model]
        if (checkpoint["use_inc"], checkpoint["use_sub"]) != (use_inc, use_sub):
            raise ValueError(f"Incorrect feature flags for {model}")
        key = f"unbiased_to_unbiased_{model}"
        score = verified_scores(
            cross_dataset / f"metrics_{key}.json",
            cross_dataset / f"predictions_{key}.csv",
            labels,
            {
                "model": model, "seed": 42, "learning_rate": 1e-5,
                "training_dataset": "unbiased", "test_dataset": "unbiased",
                "test_rows": len(labels), "selected_epoch": checkpoint["validation"]["epoch"],
            },
        )
        lines.append(f"| {configuration} | {name} | {score['accuracy']:.2%} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True, help="JSON from evaluate_transformer_results.py")
    parser.add_argument("--cross-dataset", type=Path, required=True, help="Archived 25_large_cross_dataset directory")
    parser.add_argument("--test-csv", type=Path, required=True, help="Original unbiased test.csv")
    args = parser.parse_args()
    try:
        report = make_ablation(args.selection, args.cross_dataset, args.test_csv)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        parser.exit(1, f"Cannot evaluate Transformer ablation: {error}\n")
    print(report)


if __name__ == "__main__":
    main()
