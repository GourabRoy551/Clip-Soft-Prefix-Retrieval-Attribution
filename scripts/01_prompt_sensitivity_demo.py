"""Section 1: short demonstration that CLIP similarity shifts with wording."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clip_prefix_retrieval.config import DEFAULT_CHECKPOINT_DIR, DEFAULT_OUTPUT_DIR, SEED
from clip_prefix_retrieval.data import make_flickr30k_loaders
from clip_prefix_retrieval.modeling import load_frozen_clip
from clip_prefix_retrieval.utils import ensure_project_dirs, get_device, save_json, set_seed


DEMO_ITEMS = [
    {
        "idx": 0,
        "label": "Man in orange hat",
        "paraphrases": [
            "A man wearing glasses and an orange hat.",
            "An orange hat is worn by a man with glasses.",
            "A bespectacled man in an orange hat.",
            "Man. Hat. Orange. Glasses.",
        ],
    },
    {
        "idx": 10,
        "label": "Children playing",
        "paraphrases": [
            "Children are playing outside in a field.",
            "Kids run and play in an open outdoor area.",
            "Young children enjoying outdoor activities.",
            "Outside. Kids. Playing.",
        ],
    },
    {
        "idx": 25,
        "label": "Dog on grass",
        "paraphrases": [
            "A dog is running through the grass.",
            "A canine sprints across a green lawn.",
            "On a grassy surface, a dog is moving quickly.",
            "Dog. Grass. Running.",
        ],
    },
    {
        "idx": 50,
        "label": "Street performer",
        "paraphrases": [
            "A street performer is entertaining a crowd.",
            "An outdoor entertainer performs for onlookers.",
            "A crowd watches a performer on the street.",
            "Performer. Street. Crowd.",
        ],
    },
    {
        "idx": 75,
        "label": "Cyclist on road",
        "paraphrases": [
            "A cyclist rides along a road.",
            "Someone is biking down a paved street.",
            "A person on a bicycle travels down the road.",
            "Bicycle. Road. Rider.",
        ],
    },
]

PARAPHRASE_LABELS = ["Full sentence", "Passive voice", "Nominal phrase", "Keywords only"]


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
    loaders = make_flickr30k_loaders(preprocess)
    test_data = loaders["test_data"]

    results = []
    model.eval()
    with torch.no_grad():
        for item in DEMO_ITEMS:
            raw_image = test_data[item["idx"]]["image"].convert("RGB")
            img_tensor = preprocess(raw_image).unsqueeze(0).to(device)
            img_feat = model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            tokens = tokenizer(item["paraphrases"]).to(device)
            txt_feat = model.encode_text(tokens)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ txt_feat.T).squeeze().cpu().numpy()
            results.append({**item, "image": raw_image, "sims": sims})

    summary_rows = []
    for result in results:
        sims = result["sims"]
        rank_order = np.argsort(sims)[::-1]
        keyword_rank = int(np.where(rank_order == 3)[0][0]) + 1
        summary_rows.append(
            {
                "label": result["label"],
                "range": float(sims.max() - sims.min()),
                "keyword_rank": keyword_rank,
            }
        )

    save_json(
        {
            "mean_range": float(np.mean([row["range"] for row in summary_rows])),
            "mean_keyword_rank": float(np.mean([row["keyword_rank"] for row in summary_rows])),
            "items": summary_rows,
        },
        args.output_dir / "results" / "phrasing_sensitivity_summary.json",
    )
    with (args.output_dir / "results" / "phrasing_sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "range", "keyword_rank"])
        writer.writeheader()
        writer.writerows(summary_rows)

    fig = plt.figure(figsize=(13, 11))
    outer = gridspec.GridSpec(len(results), 2, figure=fig, width_ratios=[1, 2.8], hspace=0.55, wspace=0.08)
    colors = ["#4C72B0", "#4C72B0", "#4C72B0", "#C44E52"]
    for row, result in enumerate(results):
        ax_img = fig.add_subplot(outer[row, 0])
        ax_img.imshow(result["image"])
        ax_img.axis("off")
        ax_img.set_title(result["label"], fontsize=8, loc="left")

        ax_bar = fig.add_subplot(outer[row, 1])
        sims = result["sims"]
        bars = ax_bar.barh(np.arange(len(sims)), sims, color=colors, alpha=0.85, height=0.55)
        for bar, sim in zip(bars, sims):
            ax_bar.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2, f"{sim:.4f}", va="center", fontsize=7.5)
        ax_bar.set_yticks(np.arange(len(sims)))
        ax_bar.set_yticklabels(PARAPHRASE_LABELS, fontsize=8)
        ax_bar.set_xlim(sims.min() - 0.02, sims.max() + 0.04)
        ax_bar.set_xlabel("Cosine similarity", fontsize=8)
        ax_bar.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Cosine similarity varies with phrasing", fontsize=10, y=1.01)
    fig.savefig(args.output_dir / "figures" / "03_phrasing_sensitivity_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    sim_matrix = np.stack([r["sims"] for r in results])
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".3f",
        cmap="Blues",
        xticklabels=PARAPHRASE_LABELS,
        yticklabels=[r["label"] for r in results],
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Cosine similarity heatmap")
    ax.set_xlabel("Phrasing condition")
    ax.set_ylabel("Image")
    fig.tight_layout()
    fig.savefig(args.output_dir / "figures" / "04_phrasing_similarity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved phrasing sensitivity figures and summary files.")


if __name__ == "__main__":
    main()

