# BERT End-to-End ABSA

Minimal implementation of the paper idea from **"Exploiting BERT for End-to-End Aspect-based Sentiment Analysis"** using BERT as an encoder plus a token-classification head.

The implemented architecture is:

```text
sentence -> BERT encoder -> linear token-classification head -> ABSA tags
```

This corresponds to the paper's simple `BERT + Linear` baseline. The model predicts aspect spans and sentiment jointly as a sequence-tagging task.

## Labels

The project uses the paper-style BIES labels:

```text
O
B-POS I-POS E-POS S-POS
B-NEG I-NEG E-NEG S-NEG
B-NEU I-NEU E-NEU S-NEU
```

Example:

```text
The food was excellent , but the service was slow .
O   S-POS O   O         O O   O   S-NEG  O   O    O
```

## Setup

Use a Python version supported by PyTorch, such as Python 3.10-3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train on the sample data

```powershell
python -m src.train --train-file data/sample_train.jsonl --dev-file data/sample_dev.jsonl --output-dir models/bert-sample-absa --epochs 3
```

The sample dataset is intentionally tiny and only verifies that the pipeline runs. For a real project report, train on a larger annotated dataset.

## Use SemEval ABSA Data

For the paper-focused experiment, use SemEval 2014 Task 4 restaurant or laptop XML files. The official task page describes the XML format with aspect terms, sentiment polarity, and `from`/`to` character offsets.

Download and convert the restaurant and laptop datasets:

```powershell
python -m src.prepare_semeval2014
```

This creates:

```text
data/semeval2014/restaurants_train.jsonl
data/semeval2014/restaurants_dev.jsonl
data/semeval2014/restaurants_test.jsonl
data/semeval2014/laptops_train.jsonl
data/semeval2014/laptops_dev.jsonl
data/semeval2014/laptops_test.jsonl
```

Convert a SemEval train XML file into this project's JSONL format:

```powershell
python -m src.convert_semeval --input path\to\Restaurants_Train_v2.xml --output data\restaurants_train.jsonl --dev-output data\restaurants_dev.jsonl
```

Then fine-tune:

```powershell
python -m src.train --model-name bert-base-uncased --train-file data\semeval2014\restaurants_train.jsonl --dev-file data\semeval2014\restaurants_dev.jsonl --output-dir models\bert-restaurants-absa --epochs 3
```

Evaluate on the test set:

```powershell
python -m src.test_model --model-dir models\bert-restaurants-absa --test-file data\semeval2014\restaurants_test.jsonl
```

## Predict

```powershell
python -m src.predict --model-dir models/bert-restaurants-absa --text "The food was excellent, but the service was slow."
```

## Data Format

Each line is a JSON object with tokenized words and one tag per word:

```json
{"tokens": ["The", "food", "was", "excellent", "."], "tags": ["O", "S-POS", "O", "O", "O"]}
```

The tokenizer may split words into subword pieces. During training, only the first subword of each original token receives the word label; continuation subwords are ignored in the loss.
