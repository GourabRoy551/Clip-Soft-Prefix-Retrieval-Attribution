"""Section 5: compare vanilla and prefix token attribution."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clip_prefix_retrieval.attribution import aggregate_top_tokens, summarize_attribution_entropy
from clip_prefix_retrieval.config import DEFAULT_CHECKPOINT_DIR, DEFAULT_OUTPUT_DIR, SEED
from clip_prefix_retrieval.data import make_flickr30k_loaders
from clip_prefix_retrieval.modeling import load_frozen_clip, load_prefix
from clip_prefix_retrieval.plotting import plot_entropy_summary, plot_top_tokens
from clip_prefix_retrieval.utils import ensure_project_dirs, get_device, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--prefix-path", type=Path, default=None)
    parser.add_argument("--n-eval", type=int, default=500)
    parser.add_argument("--n-vocab", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    ensure_project_dirs(args.output_dir, args.checkpoint_dir)

    prefix_path = args.prefix_path or args.checkpoint_dir / "prefix_k32_best.pt"
    device = get_device()
    model, preprocess, tokenizer = load_frozen_clip(device)
    loaders = make_flickr30k_loaders(preprocess)
    test_data = loaders["test_data"]
    prefix = load_prefix(prefix_path, device)

    summary = summarize_attribution_entropy(
        model,
        tokenizer,
        test_data,
        preprocess,
        device,
        prefix,
        n_eval=args.n_eval,
    )
    save_json(summary, args.output_dir / "results" / "attribution_entropy_summary.json")

    compact_summary = {k: v for k, v in summary.items() if not k.endswith("_values")}
    with (args.output_dir / "results" / "attribution_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in compact_summary.items():
            writer.writerow([key, value])

    plot_entropy_summary(
        summary["vanilla_entropy_values"],
        summary["prefix_entropy_values"],
        args.output_dir / "figures" / "11_attribution_entropy.png",
    )

    top_tokens = aggregate_top_tokens(
        model,
        tokenizer,
        test_data,
        preprocess,
        device,
        prefix,
        n_vocab=args.n_vocab,
    )
    save_json(top_tokens, args.output_dir / "results" / "top_attributed_tokens.json")
    with (args.output_dir / "results" / "top_attributed_tokens.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "vanilla_token", "vanilla_score", "prefix_token", "prefix_score"])
        for rank, (vanilla_row, prefix_row) in enumerate(zip(top_tokens["vanilla"], top_tokens["prefix"]), start=1):
            writer.writerow([rank, vanilla_row[0], vanilla_row[1], prefix_row[0], prefix_row[1]])

    plot_top_tokens(top_tokens, args.output_dir / "figures" / "12_top_attributed_tokens.png")


if __name__ == "__main__":
    main()

