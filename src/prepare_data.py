"""Create a reproducible, source-disjoint subset of OPUS Books en-fr."""

import argparse
import hashlib
import re
from collections import Counter

from datasets import load_dataset

from .common import DATA, seed_everything, write_json, write_jsonl

DATASET = "Helsinki-NLP/opus_books"
CONFIG = "en-fr"
SEED = 42


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prepare(train_size: int, val_size: int, test_size: int, seed: int) -> dict:
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("All split sizes must be positive")
    seed_everything(seed)
    raw = load_dataset(DATASET, CONFIG, split="train")
    candidates = []
    seen_sources = set()
    counters = Counter()
    for item in raw:
        source = clean(item["translation"]["en"])
        target = clean(item["translation"]["fr"])
        if not source or not target:
            counters["empty"] += 1
            continue
        if not (15 <= len(source) <= 300 and 15 <= len(target) <= 350):
            counters["length_filtered"] += 1
            continue
        key = source.casefold()
        if key in seen_sources:
            counters["duplicate_source"] += 1
            continue
        seen_sources.add(key)
        candidates.append({"id": str(item["id"]), "source": source, "target": target})
    import random

    random.Random(seed).shuffle(candidates)
    required = train_size + val_size + test_size
    if len(candidates) < required:
        raise ValueError(f"Only {len(candidates)} usable pairs; requested {required}")
    selected = candidates[:required]
    splits = {
        "train": selected[:train_size],
        "validation": selected[train_size : train_size + val_size],
        "test": selected[train_size + val_size :],
    }
    for name, rows in splits.items():
        write_jsonl(DATA / f"{name}.jsonl", rows)
    digest = hashlib.sha256()
    for name in ("train", "validation", "test"):
        for row in splits[name]:
            digest.update((row["id"] + "\n").encode())
    manifest = {
        "dataset": DATASET,
        "config": CONFIG,
        "source_url": "https://huggingface.co/datasets/Helsinki-NLP/opus_books",
        "direction": "English to French",
        "seed": seed,
        "original_examples": len(raw),
        "usable_examples": len(candidates),
        "filtered": dict(counters),
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "selected_id_sha256": digest.hexdigest(),
        "preprocessing": "Collapse whitespace; remove empty, extreme-length, and repeated English sources; seed-shuffle; disjoint split.",
    }
    write_json(DATA / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=8000)
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    print(prepare(args.train_size, args.val_size, args.test_size, args.seed))


if __name__ == "__main__":
    main()
