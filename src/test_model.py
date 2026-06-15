import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

from .data import AbsaDataset, read_jsonl
from .labels import ID_TO_LABEL, LABELS, LABEL_TO_ID
from .train import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned ABSA model on a test set.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--metrics-file", help="Optional JSON file for test metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_dir,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    ).to(device)

    dataset = AbsaDataset(read_jsonl(args.test_file), tokenizer, max_length=args.max_length)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    metrics = evaluate(model, dataloader, device)
    print(
        f"test_loss={metrics['loss']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )
    for sentiment, sentiment_metrics in metrics["per_sentiment"].items():
        print(
            f"{sentiment.lower()}_precision={sentiment_metrics['precision']:.4f} "
            f"{sentiment.lower()}_recall={sentiment_metrics['recall']:.4f} "
            f"{sentiment.lower()}_f1={sentiment_metrics['f1']:.4f}"
        )
    metrics_path = Path(args.metrics_file) if args.metrics_file else Path(args.model_dir) / "test_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "model_dir": args.model_dir,
        "test_file": args.test_file,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "loss": metrics["loss"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "per_sentiment": metrics["per_sentiment"],
    }
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote test metrics to {metrics_path}")


if __name__ == "__main__":
    main()
