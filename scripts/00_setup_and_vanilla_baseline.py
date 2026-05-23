"""Section 0: environment setup, Flickr30k loading, and vanilla CLIP baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clip_prefix_retrieval.config import DEFAULT_CHECKPOINT_DIR, DEFAULT_OUTPUT_DIR, EVAL_BATCH_SIZE, SEED
from clip_prefix_retrieval.data import make_flickr30k_loaders
from clip_prefix_retrieval.evaluation import compute_recall_at_k
from clip_prefix_retrieval.modeling import load_frozen_clip
from clip_prefix_retrieval.plotting import plot_recall_bars
from clip_prefix_retrieval.utils import ensure_project_dirs, get_device, print_results, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    ensure_project_dirs(args.output_dir, args.checkpoint_dir)

    device = get_device()
    print(f"Using device: {device}")
    model, preprocess, tokenizer = load_frozen_clip(device)

    loaders = make_flickr30k_loaders(preprocess, eval_batch_size=args.eval_batch_size)
    test_loader = loaders["test_loader"]

    # Baseline protocol: one first caption per image, raw caption text.
    baseline = compute_recall_at_k(
        model,
        test_loader,
        tokenizer,
        device,
        prefix=None,
        desc="Vanilla CLIP on Flickr30k",
    )
    print_results(baseline, "Vanilla CLIP - raw captions")

    save_json(baseline, args.output_dir / "results" / "baseline_vanilla.json")
    plot_recall_bars(
        {"Vanilla CLIP": baseline},
        "Vanilla CLIP on Flickr30k test set",
        args.output_dir / "figures" / "02_vanilla_clip_baseline.png",
    )


if __name__ == "__main__":
    main()

