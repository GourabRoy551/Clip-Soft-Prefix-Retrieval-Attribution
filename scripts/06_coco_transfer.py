"""Section 6: test zero-shot transfer of the Flickr30k prefix on MS-COCO."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clip_prefix_retrieval.config import DEFAULT_CHECKPOINT_DIR, DEFAULT_OUTPUT_DIR, ENSEMBLE_TEMPLATES, METRICS, SEED
from clip_prefix_retrieval.data import make_coco_loader
from clip_prefix_retrieval.evaluation import compute_recall_at_k, compute_recall_ensemble
from clip_prefix_retrieval.modeling import load_frozen_clip, load_prefix
from clip_prefix_retrieval.utils import ensure_project_dirs, get_device, load_json, print_results, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--prefix-path", type=Path, default=None)
    parser.add_argument("--n-eval", type=int, default=500)
    return parser.parse_args()


def _plot_transfer(flickr_results, coco_results, out_path: Path) -> None:
    conditions = [
        ("Vanilla", "vanilla", "#4C72B0"),
        ("Ensemble", "ensemble", "#55A868"),
        ("Prefix", "prefix", "#C44E52"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(3)
    width = 0.13
    k_labels = ["R@1", "R@5", "R@10"]

    for ax, metric_keys, title in [
        (axes[0], ["i2t_R@1", "i2t_R@5", "i2t_R@10"], "Image to Text"),
        (axes[1], ["t2i_R@1", "t2i_R@5", "t2i_R@10"], "Text to Image"),
    ]:
        bar_idx = 0
        for label, key, color in conditions:
            for dataset_name, source, alpha, hatch in [
                ("Flickr", flickr_results, 0.9, None),
                ("COCO", coco_results, 0.55, "//"),
            ]:
                vals = [source[key][m] for m in metric_keys]
                offset = (bar_idx - 3 + 0.5) * width
                ax.bar(
                    x + offset,
                    vals,
                    width * 0.9,
                    label=f"{label} {dataset_name}",
                    color=color,
                    alpha=alpha,
                    hatch=hatch,
                    edgecolor="white",
                )
                bar_idx += 1
        ax.set_xticks(x)
        ax.set_xticklabels(k_labels)
        ax.set_ylabel("Recall (%)")
        ax.set_ylim(0, 105)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=7, ncol=2, loc="lower right")
    fig.suptitle("Recall@k: Flickr30k vs MS-COCO")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_transfer_gap(flickr_results, coco_results, out_path: Path) -> None:
    labels = ["I2T\nR@1", "I2T\nR@5", "I2T\nR@10", "T2I\nR@1", "T2I\nR@5", "T2I\nR@10"]
    x = np.arange(len(METRICS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (name, key, color) in enumerate([
        ("Vanilla", "vanilla", "#4C72B0"),
        ("Ensemble", "ensemble", "#55A868"),
        ("Prefix", "prefix", "#C44E52"),
    ]):
        gaps = [coco_results[key][m] - flickr_results[key][m] for m in METRICS]
        bars = ax.bar(x + (i - 1) * width, gaps, width * 0.9, label=name, color=color, alpha=0.85)
        for bar, value in zip(bars, gaps):
            if abs(value) > 0.5:
                va = "bottom" if value >= 0 else "top"
                offset = 0.4 if value >= 0 else -0.4
                ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.1f}", ha="center", va=va, fontsize=7.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Transfer gap (COCO - Flickr30k, pp)")
    ax.set_title("Transfer gap per condition")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    ensure_project_dirs(args.output_dir, args.checkpoint_dir)

    prefix_path = args.prefix_path or args.checkpoint_dir / "prefix_k32_best.pt"
    device = get_device()
    model, preprocess, tokenizer = load_frozen_clip(device)
    _, coco_loader = make_coco_loader(preprocess, n_eval=args.n_eval)
    prefix = load_prefix(prefix_path, device)

    coco_vanilla = compute_recall_at_k(model, coco_loader, tokenizer, device, desc="COCO vanilla")
    coco_ensemble = compute_recall_ensemble(
        model,
        coco_loader,
        tokenizer,
        device,
        templates=ENSEMBLE_TEMPLATES,
        desc="COCO ensemble",
    )
    coco_prefix = compute_recall_at_k(model, coco_loader, tokenizer, device, prefix=prefix, desc="COCO prefix")

    print_results(coco_vanilla, "COCO vanilla")
    print_results(coco_ensemble, "COCO ensemble")
    print_results(coco_prefix, "COCO prefix")

    save_json(coco_vanilla, args.output_dir / "results" / "coco_vanilla.json")
    save_json(coco_ensemble, args.output_dir / "results" / "coco_ensemble.json")
    save_json(coco_prefix, args.output_dir / "results" / "coco_prefix_k32.json")

    flickr_results = {
        "vanilla": load_json(args.output_dir / "results" / "baseline_vanilla.json"),
        "ensemble": load_json(args.output_dir / "results" / "baseline_ensemble.json"),
        "prefix": load_json(args.output_dir / "results" / "prefix_k32_test.json"),
    }
    coco_results = {"vanilla": coco_vanilla, "ensemble": coco_ensemble, "prefix": coco_prefix}

    with (args.output_dir / "results" / "transfer_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "metric", "flickr30k", "coco", "gap_pp"])
        for condition in ["vanilla", "ensemble", "prefix"]:
            for metric in METRICS:
                gap = coco_results[condition][metric] - flickr_results[condition][metric]
                writer.writerow([condition, metric, flickr_results[condition][metric], coco_results[condition][metric], gap])

    _plot_transfer(flickr_results, coco_results, args.output_dir / "figures" / "13_flickr_vs_coco_transfer.png")
    _plot_transfer_gap(flickr_results, coco_results, args.output_dir / "figures" / "14_transfer_gap.png")


if __name__ == "__main__":
    main()

