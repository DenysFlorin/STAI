import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


SENTIMENT_MAP = {
    "positive": "POS",
    "negative": "NEG",
    "neutral": "NEU",
}


@dataclass(frozen=True)
class TokenOffset:
    token: str
    start: int
    end: int


@dataclass(frozen=True)
class AspectTerm:
    start: int
    end: int
    sentiment: str


def tokenize_with_offsets(text: str) -> list[TokenOffset]:
    return [
        TokenOffset(token=match.group(), start=match.start(), end=match.end())
        for match in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    ]


def tag_tokens(tokens: list[TokenOffset], aspects: list[AspectTerm]) -> list[str]:
    tags = ["O"] * len(tokens)
    for aspect in aspects:
        covered = [
            index
            for index, token in enumerate(tokens)
            if token.start >= aspect.start and token.end <= aspect.end
        ]
        if not covered:
            continue

        if len(covered) == 1:
            tags[covered[0]] = f"S-{aspect.sentiment}"
            continue

        tags[covered[0]] = f"B-{aspect.sentiment}"
        for index in covered[1:-1]:
            tags[index] = f"I-{aspect.sentiment}"
        tags[covered[-1]] = f"E-{aspect.sentiment}"
    return tags


def parse_sentence(sentence_node: ET.Element, skip_conflict: bool) -> dict[str, list[str]] | None:
    text_node = sentence_node.find("text")
    if text_node is None or not text_node.text:
        return None

    text = text_node.text
    tokens = tokenize_with_offsets(text)
    aspects: list[AspectTerm] = []

    aspect_terms_node = sentence_node.find("aspectTerms")
    if aspect_terms_node is not None:
        for aspect_node in aspect_terms_node.findall("aspectTerm"):
            polarity = aspect_node.attrib.get("polarity")
            if polarity == "conflict" and skip_conflict:
                continue
            if polarity not in SENTIMENT_MAP:
                continue

            aspects.append(
                AspectTerm(
                    start=int(aspect_node.attrib["from"]),
                    end=int(aspect_node.attrib["to"]),
                    sentiment=SENTIMENT_MAP[polarity],
                )
            )

    if not aspects:
        return None

    return {
        "tokens": [token.token for token in tokens],
        "tags": tag_tokens(tokens, aspects),
    }


def convert_xml(input_file: str | Path, skip_conflict: bool = True) -> list[dict[str, list[str]]]:
    tree = ET.parse(input_file)
    root = tree.getroot()
    records: list[dict[str, list[str]]] = []

    for sentence_node in root.findall(".//sentence"):
        record = parse_sentence(sentence_node, skip_conflict=skip_conflict)
        if record is not None:
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: list[dict[str, list[str]]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert SemEval ABSA XML into BIES JSONL.")
    parser.add_argument("--input", required=True, help="Path to a SemEval XML file.")
    parser.add_argument("--output", required=True, help="Output JSONL file, or train file when --dev-output is used.")
    parser.add_argument("--dev-output", help="Optional dev JSONL path.")
    parser.add_argument("--dev-ratio", type=float, default=0.1, help="Hold-out ratio when --dev-output is used.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--keep-conflict", action="store_true", help="Keep conflict labels if you add a mapping first.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = convert_xml(args.input, skip_conflict=not args.keep_conflict)

    if args.dev_output:
        random.Random(args.seed).shuffle(records)
        dev_size = max(1, int(len(records) * args.dev_ratio))
        dev_records = records[:dev_size]
        train_records = records[dev_size:]
        write_jsonl(args.output, train_records)
        write_jsonl(args.dev_output, dev_records)
        print(f"wrote {len(train_records)} train records to {args.output}")
        print(f"wrote {len(dev_records)} dev records to {args.dev_output}")
    else:
        write_jsonl(args.output, records)
        print(f"wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()

