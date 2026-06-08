import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

from .data import AbsaDataset, read_jsonl
from .evaluate import span_f1
from .labels import ID_TO_LABEL, LABELS, LABEL_TO_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Romanian RoBERT for end-to-end ABSA.")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--output-dir", default="models/robert-absa")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)

    train_dataset = AbsaDataset(read_jsonl(args.train_file), tokenizer, max_length=args.max_length)
    dev_dataset = AbsaDataset(read_jsonl(args.dev_file), tokenizer, max_length=args.max_length)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_f1 = -1.0
    output_dir = Path(args.output_dir)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += outputs.loss.item()
            progress.set_postfix(loss=total_loss / max(progress.n, 1))

        metrics = evaluate(model, dev_loader, device)
        print(
            f"epoch={epoch} train_loss={total_loss / max(len(train_loader), 1):.4f} "
            f"dev_loss={metrics['loss']:.4f} p={metrics['precision']:.4f} "
            f"r={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"saved best model to {output_dir}")


if __name__ == "__main__":
    main()
