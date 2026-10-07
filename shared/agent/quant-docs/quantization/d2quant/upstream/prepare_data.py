"""Download the calibration and perplexity datasets used by D²Quant.

The datasets are saved with ``datasets.save_to_disk`` under the layout expected
by the D²Quant data loaders. Set ``D2QUANT_DATA_ROOT`` to the same directory
when it is not the default ``./data``.
"""

import argparse
import os

from datasets import load_dataset


def _save(dataset, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset.save_to_disk(path)
    print(f"[saved] {path} ({len(dataset)} rows)")


def prepare_wikitext(data_root):
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    _save(train, os.path.join(data_root, "wikitext", "traindata"))
    _save(test, os.path.join(data_root, "wikitext", "testdata"))


def prepare_ptb(data_root):
    train = load_dataset(
        "ptb_text_only", "penn_treebank", split="train", trust_remote_code=True
    )
    test = load_dataset(
        "ptb_text_only", "penn_treebank", split="test", trust_remote_code=True
    )
    _save(train, os.path.join(data_root, "ptb", "traindata"))
    _save(test, os.path.join(data_root, "ptb", "testdata"))


def prepare_c4(data_root):
    train = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    validation = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )
    _save(train, os.path.join(data_root, "c4", "traindata"))
    _save(validation, os.path.join(data_root, "c4", "valdata"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data")
    parser.add_argument(
        "--datasets",
        default="wikitext,ptb,c4",
        help="Comma-separated subset of {wikitext,ptb,c4}.",
    )
    args = parser.parse_args()

    requested = {name.strip() for name in args.datasets.split(",") if name.strip()}
    if "wikitext" in requested:
        prepare_wikitext(args.data_root)
    if "ptb" in requested:
        prepare_ptb(args.data_root)
    if "c4" in requested:
        prepare_c4(args.data_root)


if __name__ == "__main__":
    main()
