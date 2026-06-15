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

### Optional GPU setup

The training scripts automatically use CUDA when a CUDA-enabled PyTorch build is installed.
On this Windows machine with an NVIDIA GPU, the CUDA 12.8 PyTorch wheel worked:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

Verify that PyTorch can see the GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
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

No-aspect sentences are retained with all-`O` tags. This gives the current split sizes:

```text
restaurants_train: 2737  restaurants_dev: 304  restaurants_test: 800
laptops_train:     2741  laptops_dev:     304  laptops_test:     800
```

Convert a SemEval train XML file into this project's JSONL format:

```powershell
python -m src.convert_semeval --input path\to\Restaurants_Train_v2.xml --output data\restaurants_train.jsonl --dev-output data\restaurants_dev.jsonl
```

Use `--drop-no-aspect` if you intentionally want to remove no-aspect sentences.

Then fine-tune:

```powershell
python -m src.train --model-name bert-base-uncased --train-file data\semeval2014\restaurants_train.jsonl --dev-file data\semeval2014\restaurants_dev.jsonl --output-dir models\bert-restaurants-absa --epochs 3
```

Training saves the best checkpoint and `train_metrics.json` in the output directory.

To train the comparison baseline with frozen BERT embeddings and only the linear head trainable:

```powershell
python -m src.train --model-name bert-base-uncased --train-file data\semeval2014\restaurants_train.jsonl --dev-file data\semeval2014\restaurants_dev.jsonl --output-dir models\frozen-bert-restaurants-absa --epochs 10 --batch-size 8 --learning-rate 1e-3 --freeze-encoder
```

Evaluate on the test set:

```powershell
python -m src.test_model --model-dir models\bert-restaurants-absa --test-file data\semeval2014\restaurants_test.jsonl
```

Evaluation saves `test_metrics.json` in the model directory.

Run a multi-seed experiment and aggregate mean/std metrics:

```powershell
python -m src.run_experiments --model-name bert-base-uncased --train-file data\semeval2014\restaurants_train.jsonl --dev-file data\semeval2014\restaurants_dev.jsonl --test-file data\semeval2014\restaurants_test.jsonl --output-dir models\multiseed-restaurants-bert-linear --seeds 13,21,42 --epochs 3 --batch-size 8 --test-batch-size 16 --no-progress
```

The summary is written to `summary.json` in the experiment output directory. Test metrics include overall span F1 and POS/NEG/NEU per-sentiment metrics.

Run a heavier single-seed GPU experiment:

```powershell
python -m src.train --model-name bert-base-uncased --train-file data\semeval2014\restaurants_train.jsonl --dev-file data\semeval2014\restaurants_dev.jsonl --output-dir models\restaurants-bert-linear-8ep-seed13 --epochs 8 --batch-size 8 --seed 13 --no-progress
python -m src.test_model --model-dir models\restaurants-bert-linear-8ep-seed13 --test-file data\semeval2014\restaurants_test.jsonl --batch-size 16
```

This reached test F1 `0.7418`; the best checkpoint was selected at epoch 7 by dev F1.

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
