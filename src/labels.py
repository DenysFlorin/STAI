LABELS = [
    "O",
    "B-POS",
    "I-POS",
    "E-POS",
    "S-POS",
    "B-NEG",
    "I-NEG",
    "E-NEG",
    "S-NEG",
    "B-NEU",
    "I-NEU",
    "E-NEU",
    "S-NEU",
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

