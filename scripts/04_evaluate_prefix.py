"""Section 4: evaluate the learned prefix against all Flickr30k baselines."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clip_prefix_retrieval.config import DEFAULT_CHECKPOINT_DIR, DEFAULT_OUTPUT_DIR, METRICS, SEED
from clip_prefix_retrieval.data import make_flickr30k_loaders
from clip_prefix_retrieval.evaluation import compute_recall_at_k
from clip_prefix_retrieval.modeling import load_frozen_clip, load_prefix
from clip_prefix_retrieval.plotting import plot_prefix_delta, plot_recall_bars
from clip_prefix_retrieval.utils import ensure_project_dirs, get_device, load_json, print_results, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--prefix-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    ensure_project_dirs(args.output_dir, args.checkpoint_dir)

    prefix_path = args.prefix_path or args.checkpoint_dir / "prefix_k32_best.pt"
    device = get_device()
    model, preprocess, tokenizer = load_frozen_clip(device)
    test_loader = make_flickr30k_loaders(preprocess)["test_loader"]
    prefix = load_prefix(prefix_path, device)

    prefix_results = compute_recall_at_k(
        model,
        test_loader,
        tokenizer,
        device,
        prefix=prefix,
        desc="Prefix on Flickr30k",
    )
    print_results(prefix_results, "Prefix k=32")
    save_json(prefix_results, args.output_dir / "results" / "prefix_k32_test.json")

    vanilla = load_json(args.output_dir / "results" / "baseline_vanilla.json")
    single = load_json(args.output_dir / "results" / "baseline_single_template.json")
    ensemble = load_json(args.output_dir / "results" / "baseline_ensemble.json")
    all_results = {
        "Vanilla raw": vanilla,
        "Single template": single,
        "Ensemble 5": ensemble,
        "Prefix k=32": prefix_results,
    }

    with (args.output_dir / "results" / "flickr_retrieval_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", *METRICS])
        for condition, values in all_results.items():
            writer.writerow([condition, *[values[m] for m in METRICS]])

    plot_recall_bars(
        all_results,
        "Flickr30k retrieval comparison",
        args.output_dir / "figures" / "08_flickr_retrieval_comparison.png",
    )
    plot_prefix_delta(
        prefix_results,
        ensemble,
        args.output_dir / "figures" / "09_prefix_vs_ensemble_delta.png",
    )


if __name__ == "__main__":
    main()

