"""Section 2: evaluate single-template and handcrafted-template ensemble baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clip_prefix_retrieval.config import DEFAULT_CHECKPOINT_DIR, DEFAULT_OUTPUT_DIR, ENSEMBLE_TEMPLATES, SEED, SINGLE_TEMPLATE
from clip_prefix_retrieval.data import make_flickr30k_loaders
from clip_prefix_retrieval.evaluation import compute_recall_at_k, compute_recall_ensemble, compute_recall_single_template
from clip_prefix_retrieval.modeling import load_frozen_clip
from clip_prefix_retrieval.plotting import plot_recall_bars
from clip_prefix_retrieval.utils import ensure_project_dirs, get_device, load_json, print_results, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    ensure_project_dirs(args.output_dir, args.checkpoint_dir)

    device = get_device()
    model, preprocess, tokenizer = load_frozen_clip(device)
    test_loader = make_flickr30k_loaders(preprocess)["test_loader"]

    baseline_path = args.output_dir / "results" / "baseline_vanilla.json"
    if baseline_path.exists():
        vanilla = load_json(baseline_path)
    else:
        vanilla = compute_recall_at_k(model, test_loader, tokenizer, device, desc="Vanilla CLIP")
        save_json(vanilla, baseline_path)

    single = compute_recall_single_template(
        model,
        test_loader,
        tokenizer,
        device,
        template=SINGLE_TEMPLATE,
        desc="Single template",
    )
    ensemble = compute_recall_ensemble(
        model,
        test_loader,
        tokenizer,
        device,
        templates=ENSEMBLE_TEMPLATES,
        desc="Template ensemble",
    )

    print_results(single, "Single template")
    print_results(ensemble, "Template ensemble")

    save_json(single, args.output_dir / "results" / "baseline_single_template.json")
    save_json(ensemble, args.output_dir / "results" / "baseline_ensemble.json")
    plot_recall_bars(
        {"Vanilla raw": vanilla, "Single template": single, "Ensemble 5": ensemble},
        "Handcrafted prompt baselines on Flickr30k",
        args.output_dir / "figures" / "05_handcrafted_baselines.png",
    )


if __name__ == "__main__":
    main()

