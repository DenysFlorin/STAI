from collections import Counter


SENTIMENTS = ("POS", "NEG", "NEU")


def tags_to_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    """Extract complete BIES sentiment spans.

    Malformed fragments such as I/E without B, or B sequences without E, are
    ignored. This keeps the metric strict at the span+sentiment level.
    """
    spans: set[tuple[int, int, str]] = set()
    index = 0
    while index < len(tags):
        tag = tags[index]
        if tag == "O":
            index += 1
            continue

        prefix, sentiment = tag.split("-", maxsplit=1)
        if prefix == "S":
            spans.add((index, index, sentiment))
            index += 1
            continue

        if prefix != "B":
            index += 1
            continue

        start = index
        index += 1
        while index < len(tags) and tags[index] == f"I-{sentiment}":
            index += 1
        if index < len(tags) and tags[index] == f"E-{sentiment}":
            spans.add((start, index, sentiment))
            index += 1
        else:
            continue
    return spans


def _scores(counts: Counter) -> dict[str, float]:
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def span_f1(gold_sequences: list[list[str]], pred_sequences: list[list[str]]) -> dict:
    counts = Counter()
    by_sentiment = {sentiment: Counter() for sentiment in SENTIMENTS}
    for gold_tags, pred_tags in zip(gold_sequences, pred_sequences):
        gold_spans = tags_to_spans(gold_tags)
        pred_spans = tags_to_spans(pred_tags)
        counts["tp"] += len(gold_spans & pred_spans)
        counts["fp"] += len(pred_spans - gold_spans)
        counts["fn"] += len(gold_spans - pred_spans)

        for sentiment in SENTIMENTS:
            gold_sentiment = {span for span in gold_spans if span[2] == sentiment}
            pred_sentiment = {span for span in pred_spans if span[2] == sentiment}
            sentiment_counts = by_sentiment[sentiment]
            sentiment_counts["tp"] += len(gold_sentiment & pred_sentiment)
            sentiment_counts["fp"] += len(pred_sentiment - gold_sentiment)
            sentiment_counts["fn"] += len(gold_sentiment - pred_sentiment)

    scores = _scores(counts)
    scores["per_sentiment"] = {
        sentiment: _scores(sentiment_counts)
        for sentiment, sentiment_counts in by_sentiment.items()
    }
    return scores
