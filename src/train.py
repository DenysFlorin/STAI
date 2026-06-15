import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

from .data import AbsaDataset, read_jsonl
from .evaluate import span_f1
from .labels import ID_TO_LABEL, LABELS, LABEL_TO_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERT for end-to-end ABSA.")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--output-dir", default="models/bert-absa")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--metrics-file", help="Optional JSON file for training metrics.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--freeze-encoder", action="store_true", help="Train only the token-classification head.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def decode_predictions(logits: torch.Tensor, labels: torch.Tensor) -> tuple[list[list[str]], list[list[str]]]:
    predictions = logits.argmax(dim=-1).detach().cpu().tolist()
    gold_ids = labels.detach().cpu().tolist()
    pred_sequences: list[list[str]] = []
    gold_sequences: list[list[str]] = []

    for pred_row, gold_row in zip(predictions, gold_ids):
        pred_tags: list[str] = []
        gold_tags: list[str] = []
        for pred_id, gold_id in zip(pred_row, gold_row):
            if gold_id == -100:
                continue
            pred_tags.append(ID_TO_LABEL[pred_id])
            gold_tags.append(ID_TO_LABEL[gold_id])
        pred_sequences.append(pred_tags)
        gold_sequences.append(gold_tags)
    return gold_sequences, pred_sequences


def evaluate(model, dataloader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_gold: list[list[str]] = []
    all_pred: list[list[str]] = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            gold, pred = decode_predictions(outputs.logits, batch["labels"])
            all_gold.extend(gold)
            all_pred.extend(pred)

    metrics = span_f1(all_gold, all_pred)
    metrics["loss"] = total_loss / max(len(dataloader), 1)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    output_dir = Path(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)
    if args.freeze_encoder:
        for parameter in model.base_model.parameters():
            parameter.requires_grad = False
    total_parameters, trainable_parameters = count_parameters(model)
    print(f"trainable parameters: {trainable_parameters:,} / {total_parameters:,}")

    train_dataset = AbsaDataset(read_jsonl(args.train_file), tokenizer, max_length=args.max_length)
    dev_dataset = AbsaDataset(read_jsonl(args.dev_file), tokenizer, max_length=args.max_length)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    best_f1 = -1.0
    best_epoch = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", disable=args.no_progress)
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += outputs.loss.item()
            progress.set_postfix(loss=total_loss / max(progress.n, 1))

        train_loss = total_loss / max(len(train_loader), 1)
        metrics = evaluate(model, dev_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "dev_loss": metrics["loss"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "per_sentiment": metrics["per_sentiment"],
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"dev_loss={metrics['loss']:.4f} p={metrics['precision']:.4f} "
            f"r={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_epoch = epoch
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"saved best model to {output_dir}")

    metrics_path = Path(args.metrics_file) if args.metrics_file else output_dir / "train_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "model_name": args.model_name,
        "train_file": args.train_file,
        "dev_file": args.dev_file,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
        "freeze_encoder": args.freeze_encoder,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "history": history,
    }
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote training metrics to {metrics_path}")


if __name__ == "__main__":
    main()
