import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-seed ABSA experiments and aggregate metrics.",
        allow_abbrev=False,
    )
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--dev-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="13,21,42", help="Comma-separated random seeds.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> float:
    start = time.perf_counter()
    subprocess.run(command, check=True)
    return time.perf_counter() - start


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def aggregate_seed_metrics(seed_results: list[dict]) -> dict:
    test_metrics = [result["test_metrics"] for result in seed_results]
    aggregate = {
        metric: mean_std([metrics[metric] for metrics in test_metrics])
        for metric in ["loss", "precision", "recall", "f1"]
    }
    aggregate["per_sentiment"] = {}
    for sentiment in ["POS", "NEG", "NEU"]:
        aggregate["per_sentiment"][sentiment] = {
            metric: mean_std([metrics["per_sentiment"][sentiment][metric] for metrics in test_metrics])
            for metric in ["precision", "recall", "f1"]
        }
    return aggregate


def main() -> None:
    args = parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_results: list[dict] = []

    for seed in seeds:
        seed_dir = output_dir / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        train_metrics_file = seed_dir / "train_metrics.json"
        test_metrics_file = seed_dir / "test_metrics.json"

        train_command = [
            sys.executable,
            "-m",
            "src.train",
            "--model-name",
            args.model_name,
            "--train-file",
            args.train_file,
            "--dev-file",
            args.dev_file,
            "--output-dir",
            str(seed_dir),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--max-length",
            str(args.max_length),
            "--seed",
            str(seed),
            "--metrics-file",
            str(train_metrics_file),
        ]
        if args.freeze_encoder:
            train_command.append("--freeze-encoder")
        if args.no_progress:
            train_command.append("--no-progress")

        test_command = [
            sys.executable,
            "-m",
            "src.test_model",
            "--model-dir",
            str(seed_dir),
            "--test-file",
            args.test_file,
            "--batch-size",
            str(args.test_batch_size),
            "--max-length",
            str(args.max_length),
            "--metrics-file",
            str(test_metrics_file),
        ]

        print(f"running seed={seed}")
        train_seconds = run_command(train_command)
        test_seconds = run_command(test_command)

        seed_results.append(
            {
                "seed": seed,
                "output_dir": str(seed_dir),
                "train_seconds": train_seconds,
                "test_seconds": test_seconds,
                "train_metrics": read_json(train_metrics_file),
                "test_metrics": read_json(test_metrics_file),
            }
        )

    summary = {
        "model_name": args.model_name,
        "train_file": args.train_file,
        "dev_file": args.dev_file,
        "test_file": args.test_file,
        "output_dir": args.output_dir,
        "seeds": seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_batch_size": args.test_batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "freeze_encoder": args.freeze_encoder,
        "train_seconds": mean_std([result["train_seconds"] for result in seed_results]),
        "test_seconds": mean_std([result["test_seconds"] for result in seed_results]),
        "test_metrics": aggregate_seed_metrics(seed_results),
        "runs": seed_results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote multi-seed summary to {summary_path}")


if __name__ == "__main__":
    main()
