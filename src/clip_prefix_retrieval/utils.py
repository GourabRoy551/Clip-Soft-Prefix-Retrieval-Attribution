"""Small I/O and reproducibility helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Make the main random sources deterministic."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Use CUDA when available; otherwise fall back to CPU."""

    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_project_dirs(output_dir: Path, checkpoint_dir: Path) -> None:
    """Create the standard output directories used by the scripts."""

    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


def save_json(payload: Mapping, path: Path) -> None:
    """Write a JSON file with readable indentation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path) -> dict:
    """Load a JSON file."""

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_results(results: Mapping[str, float], label: str = "") -> None:
    """Print Recall@k values in a compact table."""

    if label:
        print(f"\n{label}")
    print(f"{'Metric':<12} {'Score':>8}")
    print("-" * 22)
    for key, value in results.items():
        print(f"{key:<12} {value:>7.2f}%")

