"""Leave-one-out token attribution for CLIP text prompts."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from scipy import stats
from scipy.stats import entropy as scipy_entropy
from tqdm.auto import tqdm

from .modeling import encode_text_with_prefix


@torch.no_grad()
def token_attribution(model, tokenizer, image_tensor, caption: str, device: str, prefix=None):
    """Measure how much each token contributes to image-text similarity.

    Attribution is defined as:
        sim(image, full caption) - sim(image, caption with token removed)

    Positive values mean the token helped the match. Negative values mean
    removing the token increased similarity.
    """

    model.eval()
    image_tensor = image_tensor.to(device)
    img_feat = model.encode_image(image_tensor)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    full_tokens = tokenizer([caption]).to(device)
    token_ids = full_tokens[0].cpu().numpy()

    content_positions = []
    for pos, tid in enumerate(token_ids):
        if pos == 0:
            continue
        if tid >= 49406:
            break
        content_positions.append(pos)

    if prefix is None:
        txt_feat = model.encode_text(full_tokens)
    else:
        txt_feat = encode_text_with_prefix(model, full_tokens, prefix, device)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    full_sim = (img_feat @ txt_feat.T).item()

    vocab = tokenizer.encoder if hasattr(tokenizer, "encoder") else {}
    inv_vocab = {v: k for k, v in vocab.items()} if vocab else {}
    tokens_str = []
    for pos in content_positions:
        tid = int(token_ids[pos])
        token = inv_vocab.get(tid, f"tok{pos}") if inv_vocab else f"tok{pos}"
        tokens_str.append(token.replace("\u0120", " ").strip())

    attr_scores = []
    for drop_pos in content_positions:
        new_ids = [int(token_ids[0])]
        for pos in content_positions:
            if pos != drop_pos:
                new_ids.append(int(token_ids[pos]))

        eot_id = int(token_ids[content_positions[-1] + 1])
        new_ids.append(eot_id)
        while len(new_ids) < full_tokens.shape[1]:
            new_ids.append(0)
        drop_tokens = torch.tensor([new_ids[: full_tokens.shape[1]]], dtype=torch.long).to(device)

        if prefix is None:
            drop_feat = model.encode_text(drop_tokens)
        else:
            drop_feat = encode_text_with_prefix(model, drop_tokens, prefix, device)
        drop_feat = drop_feat / drop_feat.norm(dim=-1, keepdim=True)
        drop_sim = (img_feat @ drop_feat.T).item()
        attr_scores.append(full_sim - drop_sim)

    return tokens_str, np.array(attr_scores), full_sim


def attribution_entropy(attr_scores) -> float:
    """Entropy over positive attribution mass."""

    positive = np.maximum(attr_scores, 0)
    total = positive.sum()
    if total < 1e-8:
        return 0.0
    return float(scipy_entropy((positive / total) + 1e-10))


def max_token_fraction(attr_scores) -> float:
    """Share of positive attribution captured by the highest-scoring token."""

    positive = np.maximum(attr_scores, 0)
    total = positive.sum()
    return float(positive.max() / total) if total > 1e-8 else 0.0


def summarize_attribution_entropy(
    model,
    tokenizer,
    test_data,
    preprocess,
    device: str,
    prefix,
    n_eval: int = 500,
):
    """Compute entropy and max-token-fraction statistics over test images."""

    vanilla_entropy = []
    prefix_entropy = []
    vanilla_max = []
    prefix_max = []

    for idx in tqdm(range(n_eval), desc="Attribution entropy"):
        item = test_data[idx]
        caption = item["caption"][0]
        image = preprocess(item["image"].convert("RGB")).unsqueeze(0)

        _, attr_v, _ = token_attribution(model, tokenizer, image, caption, device, prefix=None)
        _, attr_p, _ = token_attribution(model, tokenizer, image, caption, device, prefix=prefix)

        vanilla_entropy.append(attribution_entropy(attr_v))
        prefix_entropy.append(attribution_entropy(attr_p))
        vanilla_max.append(max_token_fraction(attr_v))
        prefix_max.append(max_token_fraction(attr_p))

    vanilla_entropy = np.asarray(vanilla_entropy)
    prefix_entropy = np.asarray(prefix_entropy)
    vanilla_max = np.asarray(vanilla_max)
    prefix_max = np.asarray(prefix_max)

    stat, p_value = stats.wilcoxon(vanilla_entropy, prefix_entropy)
    diff = prefix_entropy - vanilla_entropy
    cohens_d = float(diff.mean() / diff.std())

    return {
        "mean_entropy_vanilla": float(vanilla_entropy.mean()),
        "mean_entropy_prefix": float(prefix_entropy.mean()),
        "delta_mean_entropy": float(prefix_entropy.mean() - vanilla_entropy.mean()),
        "median_entropy_vanilla": float(np.median(vanilla_entropy)),
        "median_entropy_prefix": float(np.median(prefix_entropy)),
        "delta_median_entropy": float(np.median(prefix_entropy) - np.median(vanilla_entropy)),
        "mean_max_token_fraction_vanilla": float(vanilla_max.mean()),
        "mean_max_token_fraction_prefix": float(prefix_max.mean()),
        "delta_max_token_fraction": float(prefix_max.mean() - vanilla_max.mean()),
        "wilcoxon_statistic": float(stat),
        "wilcoxon_p_value": float(p_value),
        "cohens_d": cohens_d,
        "vanilla_entropy_values": vanilla_entropy.tolist(),
        "prefix_entropy_values": prefix_entropy.tolist(),
    }


def aggregate_top_tokens(
    model,
    tokenizer,
    test_data,
    preprocess,
    device: str,
    prefix,
    n_vocab: int = 200,
    min_count: int = 5,
    top_n: int = 20,
):
    """Aggregate mean token attribution across examples."""

    vanilla_token_attr = defaultdict(list)
    prefix_token_attr = defaultdict(list)

    for idx in tqdm(range(n_vocab), desc="Vocabulary attribution"):
        item = test_data[idx]
        caption = item["caption"][0]
        image = preprocess(item["image"].convert("RGB")).unsqueeze(0)

        toks_v, attr_v, _ = token_attribution(model, tokenizer, image, caption, device, prefix=None)
        toks_p, attr_p, _ = token_attribution(model, tokenizer, image, caption, device, prefix=prefix)

        for token, score in zip(toks_v, attr_v):
            vanilla_token_attr[token.lower()].append(float(score))
        for token, score in zip(toks_p, attr_p):
            prefix_token_attr[token.lower()].append(float(score))

    vanilla_means = {
        token: float(np.mean(values))
        for token, values in vanilla_token_attr.items()
        if len(values) >= min_count
    }
    prefix_means = {
        token: float(np.mean(values))
        for token, values in prefix_token_attr.items()
        if len(values) >= min_count
    }
    return {
        "vanilla": sorted(vanilla_means.items(), key=lambda x: x[1], reverse=True)[:top_n],
        "prefix": sorted(prefix_means.items(), key=lambda x: x[1], reverse=True)[:top_n],
    }
