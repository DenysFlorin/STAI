import argparse

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .evaluate import tags_to_spans
from .labels import ID_TO_LABEL, LABELS, LABEL_TO_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Romanian aspect sentiments.")
    parser.add_argument("--model-dir", default="models/bert-restaurants-absa")
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def simple_tokenize(text: str) -> list[str]:
    for punctuation in [",", ".", "!", "?", ";", ":"]:
        text = text.replace(punctuation, f" {punctuation} ")
    return text.split()


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
    model.eval()

    tokens = simple_tokenize(args.text)
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**{key: value.to(device) for key, value in encoding.items()})

    predictions = outputs.logits.argmax(dim=-1).squeeze(0).cpu().tolist()
    word_ids = encoding.word_ids(batch_index=0)

    word_tags = ["O"] * len(tokens)
    seen_word_ids: set[int] = set()
    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_word_ids:
            continue
        word_tags[word_id] = ID_TO_LABEL[predictions[token_index]]
        seen_word_ids.add(word_id)

    print("Tokens and predicted tags:")
    for token, tag in zip(tokens, word_tags):
        print(f"{token}\t{tag}")

    spans = tags_to_spans(word_tags)
    if not spans:
        print("\nNo aspects found.")
        return

    print("\nAspects:")
    for start, end, sentiment in sorted(spans):
        aspect = " ".join(tokens[start : end + 1])
        print(f"{aspect}\t{sentiment}")


if __name__ == "__main__":
    main()
