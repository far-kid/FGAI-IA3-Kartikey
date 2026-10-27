"""Shared paths, reproducibility settings, and JSONL helpers."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
MODELS = ("transformer", "mt5", "llama", "gemma")
HF_MODELS = {
    "mt5": "google/mt5-small",
    "llama": "meta-llama/Llama-3.2-1B-Instruct",
    "gemma": "google/gemma-3-270m-it",
}
PROMPT = "Translate the following English text into French. Output only the French translation.\nEnglish: {source}\nFrench:"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: python -m src.prepare_data")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


def env_info() -> dict:
    return {"device": gpu_name(), "torch": torch.__version__, "python": os.sys.version.split()[0]}
