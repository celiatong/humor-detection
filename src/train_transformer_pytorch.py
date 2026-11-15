"""Fine-tune a PyTorch Transformer or run a RoBERTa-large feature ablation."""

import argparse
import csv
import hashlib
import json
import math
import unicodedata
from itertools import combinations, islice
from pathlib import Path


MODELS = {
    "roberta-base": "FacebookAI/roberta-base",
    "roberta-large": "FacebookAI/roberta-large",
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-small": "microsoft/deberta-v3-small",
}
SPLITS = ("train", "validation", "test")


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def load_splits(root, split_names=SPLITS):
    datasets, hashes, groups, titles = {}, {}, {}, {}
    for split in split_names:
        path = root / f"{split}.csv"
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"question", "product_description", "product_group", "label"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{path}: requires {', '.join(sorted(required))}")
            records = list(reader)
        if not records or any(
            not normalize(r.get(k) or "") for r in records for k in required
        ):
            raise ValueError(f"{path}: missing rows or required values")
        if {r["label"].strip() for r in records} != {"0", "1"}:
            raise ValueError(f"{path}: both binary labels 0 and 1 are required")
        groups[split] = {normalize(r["product_group"]) for r in records}
        titles[split] = {normalize(r["product_description"]) for r in records}
        datasets[split] = records
    for a, b in combinations(split_names, 2):
        if groups[a] & groups[b] or titles[a] & titles[b]:
            raise ValueError(f"Product overlap between {a} and {b}")
    return datasets, hashes


def metrics(actual, predicted):
    if not actual or len(actual) != len(predicted):
        raise ValueError("Expected a nonempty prediction for each label")
    scores = []
    for label in (0, 1):
        tp = sum(a == p == label for a, p in zip(actual, predicted))
        total = actual.count(label) + predicted.count(label)
        scores.append(2 * tp / total if total else 0.0)
    return {
        "test_rows": len(actual),
        "accuracy": sum(a == p for a, p in zip(actual, predicted)) / len(actual),
        "macro_f1": sum(scores) / 2,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--revision", help="Optional pinned Hugging Face commit")
    parser.add_argument("--model-path", type=Path, help="Local pretrained model and tokenizer")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input", choices=("qproduct", "qonly"), default="qproduct")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("auto", "fp32"), default="auto")
    parser.add_argument("--fixed-padding", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--ablation", choices=("control", "inc", "sub", "inc-sub"))
    parser.add_argument("--inc-features", type=Path, help="Directory of exported INC train.pt and validation.pt")
    parser.add_argument("--sub-features", type=Path, help="Directory of exported SUB train.pt and validation.pt")
    parser.add_argument("--test-features", type=Path, help="Combined INC/SUB cache from cross-dataset evaluation")
    args = parser.parse_args()
    if args.ablation and (args.model != "roberta-large" or args.input != "qproduct"):
        parser.error("--ablation requires --model roberta-large and --input qproduct")
    if not args.ablation and (args.inc_features or args.sub_features or args.test_features):
        parser.error("auxiliary feature files require --ablation")
    if args.model_path and not args.model_path.is_dir():
        parser.error("--model-path must be an existing local pretrained model directory")
    if args.model_path and args.revision:
        parser.error("use --model-path for a local snapshot or --revision for a public snapshot")
    try:
        splits, hashes = load_splits(args.data_dir, SPLITS[:2] if args.validation_only else SPLITS)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"Cannot load PQA splits: {error}\n")
    data_log = {s: {"rows": len(records), "sha256": hashes[s]} for s, records in splits.items()}
    print(json.dumps(data_log, indent=2), flush=True)
    if args.check_data:
        return
    if not args.model or not args.output:
        parser.error("--model and --output are required for training")
    if min(args.epochs, args.batch_size, args.max_length, args.gradient_accumulation_steps) < 1 or not (
        math.isfinite(args.lr) and args.lr > 0
    ):
        parser.error("epochs, batch size, max length, accumulation steps and learning rate must be positive")
    if args.output.exists():
        parser.error("--output must be a new directory")
    try:
        train(args, splits, data_log)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Training failed: {error}\n")


def load_auxiliary_features(args, splits, data_log):
    import torch

    from transformer_model_pytorch import ABLATIONS, FEATURES, FEATURE_WIDTH

    output = {}
    provenance = {}
    active = ABLATIONS[args.ablation]
    checkpoints = {}
    for split, records in splits.items():
        rows = len(records)
        csv_hash = data_log[split]["sha256"]
        combined = torch.zeros(rows, FEATURE_WIDTH)
        provenance[split] = {}
        if split == "test" and active:
            if not args.test_features:
                raise ValueError("--test-features is required for test evaluation with INC or SUB")
            test_cache = torch.load(args.test_features, map_location="cpu", weights_only=True)
            test_values = test_cache.get("features")
            if not isinstance(test_values, torch.Tensor) or test_values.shape != (rows, FEATURE_WIDTH):
                raise ValueError(f"{args.test_features}: expected {rows} × {FEATURE_WIDTH} combined features")
            signature = test_cache.get("signature", {})
        for name in active:
            if split == "test":
                path = args.test_features
                expected_hash = signature.get("test_csv_sha256")
                extractor_hash = signature.get(f"{name}_checkpoint_sha256")
                values = test_values[:, FEATURES[name]]
            else:
                directory = getattr(args, f"{name}_features")
                if directory is None:
                    raise ValueError(f"--{name}-features is required for {args.ablation}")
                path = directory / f"{split}.pt"
                saved = torch.load(path, map_location="cpu", weights_only=True)
                expected_hash = saved.get("source_csv_sha256")
                extractor_hash = saved.get(f"{name}_checkpoint_sha256")
                values = saved.get("features")
                if saved.get("split", split) != split:
                    raise ValueError(f"{path}: unexpected split")
                if name == "inc" and saved.get("score_meaning") != "probability_of_original_product":
                    raise ValueError(f"{path}: expected original-product matching probabilities")
            if expected_hash != csv_hash:
                raise ValueError(f"{path}: feature rows do not match the {split} CSV hash")
            if not extractor_hash or checkpoints.setdefault(name, extractor_hash) != extractor_hash:
                raise ValueError(f"{path}: missing or inconsistent {name.upper()} extractor hash")
            columns = FEATURES[name]
            shape = (rows, columns.stop - columns.start)
            if not isinstance(values, torch.Tensor) or values.shape != shape:
                raise ValueError(f"{path}: expected {name.upper()} features with shape {shape}")
            values = values.float()
            if not torch.isfinite(values).all():
                raise ValueError(f"{path}: non-finite {name.upper()} feature values")
            if name == "inc" and ((values < 0) | (values > 1)).any():
                raise ValueError(f"{path}: INC probabilities must be in [0, 1]")
            combined[:, columns] = values
            provenance[split][name] = {
                "file": str(path),
                "checkpoint_sha256": extractor_hash,
            }
        output[split] = combined
    return output, provenance


def train(args, splits, data_log):
    import torch
    import transformers
    from torch.utils.data import DataLoader
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
        set_seed,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the CUDA PyTorch installation.")
    features, feature_sources = ({}, {})
    if args.ablation:
        from transformer_model_pytorch import RobertaHumorFusion

        features, feature_sources = load_auxiliary_features(args, splits, data_log)

    set_seed(args.seed)
    model_id = str(args.model_path.resolve()) if args.model_path else MODELS[args.model]
    if args.model_path:
        expected_type = "roberta" if args.model.startswith("roberta") else "deberta-v2"
        actual_type = AutoConfig.from_pretrained(model_id).model_type
        if actual_type != expected_type:
            raise ValueError(f"{model_id}: expected a {expected_type} checkpoint for {args.model}")
    if args.ablation:
        model = RobertaHumorFusion.from_pretrained(model_id, args.revision, args.ablation)
        config = model.backbone.config
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=args.revision, num_labels=2,
        )
        config = model.config
    revision = getattr(config, "_commit_hash", None) or args.revision
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, use_fast=bool(args.ablation),
    )
    model.to(args.device)
    mixed = args.precision == "auto" and args.device == "cuda" and torch.cuda.is_bf16_supported()
    checkpointing = args.gradient_checkpointing == "on" or (
        args.gradient_checkpointing == "auto" and args.device == "cuda" and args.model == "roberta-large"
    )
    if checkpointing:
        encoder = model.backbone if args.ablation else model
        encoder.gradient_checkpointing_enable()
        del encoder
    set_seed(args.seed)

    def collate(records):
        q = [r["question"] for r, _ in records]
        p = [r["product_description"] for r, _ in records] if args.input == "qproduct" else None
        batch = tokenizer(
            q, text_pair=p, truncation=True, padding="max_length" if args.fixed_padding else True,
            max_length=args.max_length, return_tensors="pt",
        )
        batch["labels"] = torch.tensor([int(r["label"]) for r, _ in records])
        if args.ablation:
            batch["features"] = torch.stack([row for _, row in records])
        return batch

    loaders = {
        split: DataLoader(
            list(zip(records, features[split] if args.ablation else [None] * len(records))),
            batch_size=args.batch_size, shuffle=split == "train",
            generator=torch.Generator().manual_seed(args.seed), collate_fn=collate,
        )
        for split, records in splits.items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(loaders["train"]) / args.gradient_accumulation_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(steps * 0.1), steps)
    args.output.mkdir(parents=True)

    def save_json(name, data):
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    save_json("run.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "splits": data_log, "model_id": model_id, "resolved_revision": revision,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "bf16": mixed, "weight_decay": 0.01, "warmup_ratio": 0.1,
        "gradient_checkpointing_enabled": checkpointing,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "feature_sources": feature_sources,
    })

    def epoch(active_model, split):
        training = split == "train"
        active_model.train(training)
        actual, predicted = [], []
        loader = iter(loaders[split])
        group_size = args.gradient_accumulation_steps if training else 1
        loss_total = 0.0
        for update, group in enumerate(iter(lambda: list(islice(loader, group_size)), []), 1):
            group_rows = sum(len(batch["labels"]) for batch in group)
            if training:
                optimizer.zero_grad(set_to_none=True)
            for batch in group:
                labels = batch.pop("labels").to(args.device)
                inputs = {k: v.to(args.device) for k, v in batch.items()}
                with torch.set_grad_enabled(training):
                    with torch.autocast(args.device, dtype=torch.bfloat16, enabled=mixed):
                        output = active_model(**inputs)
                        logits = output if args.ablation else output.logits
                        loss = torch.nn.functional.cross_entropy(logits.float(), labels)
                    if not torch.isfinite(loss).item():
                        raise RuntimeError(f"Non-finite loss in {split}")
                    if training:
                        (loss * (len(labels) / group_rows)).backward()
                loss_total += loss.item() * len(labels)
                actual.extend(labels.tolist())
                predicted.extend(logits.detach().argmax(dim=1).tolist())
            if training:
                torch.nn.utils.clip_grad_norm_(active_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                if update % 100 == 0:
                    print(f"Train update {update} | rows {len(actual)} | loss {loss_total / len(actual):.4f}", flush=True)
        result = metrics(actual, predicted)
        if args.validation_only:
            result["rows"] = result.pop("test_rows")
            result["loss"] = loss_total / len(actual)
            result["predicted_label_counts"] = {label: predicted.count(label) for label in (0, 1)}
        return result, actual, predicted

    best, best_epoch, history = -1.0, None, []
    checkpoint = args.output / "best_model"
    for i in range(1, args.epochs + 1):
        tr, _, _ = epoch(model, "train")
        va, _, _ = epoch(model, "validation")
        result = {"epoch": i, "train": tr, "validation": va}
        history.append(result)
        print(json.dumps(result), flush=True)
        if va["macro_f1"] > best:
            best, best_epoch = va["macro_f1"], i
            if args.ablation:
                checkpoint.mkdir(exist_ok=True)
                torch.save(model.state_dict(), checkpoint / "model.pt")
                model.backbone.config.save_pretrained(checkpoint)
            else:
                model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
        save_json("history.json", history)

    if args.validation_only:
        result = {"best_epoch": best_epoch, **history[best_epoch - 1]["validation"]}
        save_json("validation_metrics.json", result)
        print(json.dumps(result, indent=2), flush=True)
        return

    del optimizer, scheduler
    if args.ablation:
        model.load_state_dict(torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True))
        selected = model
    else:
        del model
        selected = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(args.device)
    if args.device == "cuda":
        torch.cuda.empty_cache()
    result, actual, predicted = epoch(selected, "test")
    result.update(best_epoch=best_epoch, validation_macro_f1=best)
    with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_index", "actual", "predicted"])
        writer.writerows((i, a, p) for i, (a, p) in enumerate(zip(actual, predicted)))
    save_json("test_metrics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
