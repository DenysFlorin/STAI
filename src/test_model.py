import argparse

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


if __name__ == "__main__":
    main()

