"""Plotting helpers used by the section scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .config import METRICS


def plot_recall_bars(
    results_dict: Mapping[str, Mapping[str, float]],
    title: str,
    out_path: Path,
    figsize=(9, 4),
) -> None:
    """Save a grouped Recall@k bar chart."""

    labels = list(results_dict.keys())
    x = np.arange(len(METRICS))
    width = 0.8 / len(labels)
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, results) in enumerate(results_dict.items()):
        values = [results.get(metric, 0) for metric in METRICS]
        offset = (i - len(labels) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width * 0.9, label=label, color=palette[i % len(palette)])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in METRICS], fontsize=9)
    ax.set_ylabel("Recall (%)")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 105)
    ax.axvline(x=2.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(1.0, 98, "Image to Text", ha="center", fontsize=8, color="gray")
    ax.text(4.0, 98, "Text to Image", ha="center", fontsize=8, color="gray")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: Mapping, out_path: Path) -> None:
    """Save loss and validation R@1 curves from a training history dict."""

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], marker="o", color="#4C72B0")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("InfoNCE loss")
    axes[0].set_title("Training loss")
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(epochs, history["val_i2t_R@1"], marker="o", color="#C44E52")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation I2T R@1 (%)")
    axes[1].set_title("Validation retrieval")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prefix_delta(prefix_results: Mapping[str, float], ensemble_results: Mapping[str, float], out_path: Path) -> None:
    """Save the per-metric delta between prefix and template ensemble."""

    deltas = [prefix_results[m] - ensemble_results[m] for m in METRICS]
    labels = ["I2T\nR@1", "I2T\nR@5", "I2T\nR@10", "T2I\nR@1", "T2I\nR@5", "T2I\nR@10"]
    colors = ["#55A868" if d >= 0 else "#C44E52" for d in deltas]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(np.arange(len(METRICS)), deltas, color=colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    for bar, value in zip(bars, deltas):
        va = "bottom" if value >= 0 else "top"
        offset = 0.08 if value >= 0 else -0.08
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.1f}", ha="center", va=va, fontsize=8)
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Prefix minus ensemble (percentage points)")
    ax.set_title("Prefix vs handcrafted ensemble")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_summary(vanilla_values: Sequence[float], prefix_values: Sequence[float], out_path: Path) -> None:
    """Save attribution entropy distribution and paired scatter plot."""

    vanilla_values = np.asarray(vanilla_values)
    prefix_values = np.asarray(prefix_values)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    sns.kdeplot(vanilla_values, ax=axes[0], label="Vanilla CLIP", color="#4C72B0", linewidth=2, fill=True, alpha=0.25)
    sns.kdeplot(prefix_values, ax=axes[0], label="Prefix k=32", color="#C44E52", linewidth=2, fill=True, alpha=0.25)
    axes[0].axvline(vanilla_values.mean(), color="#4C72B0", linestyle="--", linewidth=1.2)
    axes[0].axvline(prefix_values.mean(), color="#C44E52", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("Attribution entropy")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Entropy distribution")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].scatter(vanilla_values, prefix_values, alpha=0.15, s=12, color="#555555")
    lims = [min(vanilla_values.min(), prefix_values.min()), max(vanilla_values.max(), prefix_values.max())]
    axes[1].plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("Vanilla entropy")
    axes[1].set_ylabel("Prefix entropy")
    axes[1].set_title("Per-image entropy")
    axes[1].spines[["top", "right"]].set_visible(False)

    fig.suptitle("Attribution entropy: vanilla CLIP vs prefix")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_tokens(top_tokens: Mapping[str, Sequence[tuple[str, float]]], out_path: Path) -> None:
    """Save side-by-side vocabulary-level attribution bars."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = [
        ("vanilla", "Vanilla CLIP", "#4C72B0", axes[0]),
        ("prefix", "Prefix k=32", "#C44E52", axes[1]),
    ]
    for key, title, color, ax in panels:
        tokens = [token for token, _ in top_tokens[key]]
        scores = [score for _, score in top_tokens[key]]
        y = np.arange(len(tokens))
        ax.barh(y[::-1], scores, color=color, alpha=0.8)
        ax.set_yticks(y[::-1])
        ax.set_yticklabels(tokens, fontsize=9)
        ax.set_xlabel("Mean attribution")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

