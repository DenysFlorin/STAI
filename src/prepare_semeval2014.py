import argparse
import random
import urllib.parse
import urllib.request
from pathlib import Path

from .convert_semeval import convert_xml, write_jsonl


HF_BASE_URL = "https://huggingface.co/datasets/alexcadillon/SemEval2014Task4/resolve/main"

FILES = {
    "restaurants_train_xml": "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train_v2.xml",
    "restaurants_test_xml": "ABSA_Gold_TestData/Restaurants_Test_Gold.xml",
    "laptops_train_xml": "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Train_v2.xml",
    "laptops_test_xml": "ABSA_Gold_TestData/Laptops_Test_Gold.xml",
}


def hf_url(path: str) -> str:
    return f"{HF_BASE_URL}/{urllib.parse.quote(path)}?download=true"


def download_file(path: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"exists: {output_path}")
        return

    print(f"downloading: {path}")
    with urllib.request.urlopen(hf_url(path), timeout=60) as response:
        output_path.write_bytes(response.read())
    print(f"wrote: {output_path}")


def prepare_domain(
    domain: str,
    train_xml: Path,
    test_xml: Path,
    output_dir: Path,
    dev_ratio: float,
    seed: int,
) -> None:
    records = convert_xml(train_xml)
    random.Random(seed).shuffle(records)
    dev_size = max(1, int(len(records) * dev_ratio))
    dev_records = records[:dev_size]
    train_records = records[dev_size:]
    test_records = convert_xml(test_xml)

    write_jsonl(output_dir / f"{domain}_train.jsonl", train_records)
    write_jsonl(output_dir / f"{domain}_dev.jsonl", dev_records)
    write_jsonl(output_dir / f"{domain}_test.jsonl", test_records)

    print(
        f"{domain}: train={len(train_records)} dev={len(dev_records)} "
        f"test={len(test_records)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and convert SemEval 2014 ABSA data.")
    parser.add_argument("--output-dir", default="data/semeval2014")
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    xml_dir = output_dir / "xml"

    local_paths: dict[str, Path] = {}
    for key, remote_path in FILES.items():
        local_path = xml_dir / Path(remote_path).name
        download_file(remote_path, local_path)
        local_paths[key] = local_path

    prepare_domain(
        "restaurants",
        local_paths["restaurants_train_xml"],
        local_paths["restaurants_test_xml"],
        output_dir,
        args.dev_ratio,
        args.seed,
    )
    prepare_domain(
        "laptops",
        local_paths["laptops_train_xml"],
        local_paths["laptops_test_xml"],
        output_dir,
        args.dev_ratio,
        args.seed,
    )


if __name__ == "__main__":
    main()
