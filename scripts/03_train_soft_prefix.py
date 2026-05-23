"""Section 3: train a CoOp-style soft prefix with frozen CLIP encoders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clip_prefix_retrieval.config import DEFAULT_CHECKPOINT_DIR, DEFAULT_OUTPUT_DIR, SEED
from clip_prefix_retrieval.data import make_flickr30k_loaders
from clip_prefix_retrieval.modeling import load_frozen_clip
from clip_prefix_retrieval.plotting import plot_training_curves
from clip_prefix_retrieval.training import train_soft_prefix
from clip_prefix_retrieval.utils import ensure_project_dirs, get_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--n-ctx", type=int, default=32, help="Number of learned prefix vectors.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    ensure_project_dirs(args.output_dir, args.checkpoint_dir)

    device = get_device()
    model, preprocess, tokenizer = load_frozen_clip(device)
    loaders = make_flickr30k_loaders(preprocess)

    # Only the prefix module has trainable parameters. CLIP stays frozen.
    _, history = train_soft_prefix(
        model,
        tokenizer,
        loaders["train_loader"],
        loaders["val_loader"],
        device,
        checkpoint_dir=args.checkpoint_dir,
        n_ctx=args.n_ctx,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_name=f"prefix_k{args.n_ctx}",
    )

    save_json(history, args.output_dir / "results" / f"training_history_k{args.n_ctx}.json")
    plot_training_curves(history, args.output_dir / "figures" / f"training_curves_k{args.n_ctx}.png")


if __name__ == "__main__":
    main()

