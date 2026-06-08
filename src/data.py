import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None
    Dataset = object

from .labels import LABEL_TO_ID


@dataclass(frozen=True)
class AbsaExample:
    tokens: list[str]
    tags: list[str]


def read_jsonl(path: str | Path) -> list[AbsaExample]:
    examples: list[AbsaExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            tokens = record["tokens"]
            tags = record["tags"]
            if len(tokens) != len(tags):
                raise ValueError(f"{path}:{line_number} has {len(tokens)} tokens but {len(tags)} tags")
            unknown = [tag for tag in tags if tag not in LABEL_TO_ID]
            if unknown:
                raise ValueError(f"{path}:{line_number} contains unknown tags: {unknown}")
            examples.append(AbsaExample(tokens=tokens, tags=tags))
    return examples


class AbsaDataset(Dataset):
    def __init__(self, examples: Iterable[AbsaExample], tokenizer, max_length: int = 128):
        if torch is None:
            raise RuntimeError("AbsaDataset requires torch. Install dependencies with `pip install -r requirements.txt`.")
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        encoding = self.tokenizer(
            example.tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        labels: list[int] = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != previous_word_id:
                labels.append(LABEL_TO_ID[example.tags[word_id]])
            else:
                labels.append(-100)
            previous_word_id = word_id

        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(labels, dtype=torch.long)
        return item
