"""Retrieval evaluation for raw captions, handcrafted templates, and prefixes."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import ENSEMBLE_TEMPLATES, SINGLE_TEMPLATE
from .modeling import encode_text_with_prefix


def _first_captions(captions_list) -> list[str]:
    """Default PyTorch collation transposes the five captions into slots."""

    return list(captions_list[0])


def _recall_from_similarity(sim: torch.Tensor, k_vals: Iterable[int]) -> dict[str, float]:
    diag = sim.diag()
    i2t_ranks = (sim > diag.unsqueeze(1)).sum(dim=1).numpy()
    t2i_ranks = (sim.T > diag.unsqueeze(1)).sum(dim=1).numpy()

    results = {}
    for k in k_vals:
        results[f"i2t_R@{k}"] = float(100 * np.mean(i2t_ranks < k))
        results[f"t2i_R@{k}"] = float(100 * np.mean(t2i_ranks < k))
    return results


@torch.no_grad()
def compute_recall_at_k(
    model,
    loader,
    tokenizer,
    device: str,
    prefix: torch.Tensor | None = None,
    k_vals: Sequence[int] = (1, 5, 10),
    desc: str = "Evaluating",
) -> dict[str, float]:
    """Evaluate retrieval using raw captions or a learned soft prefix."""

    model.eval()
    all_img_feat = []
    all_txt_feat = []

    for images, captions_list in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)

        img_feat = model.encode_image(images)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        all_img_feat.append(img_feat.cpu())

        tokens = tokenizer(_first_captions(captions_list)).to(device)
        if prefix is None:
            txt_feat = model.encode_text(tokens)
        else:
            txt_feat = encode_text_with_prefix(model, tokens, prefix, device)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        all_txt_feat.append(txt_feat.cpu())

    sim = torch.cat(all_img_feat, dim=0) @ torch.cat(all_txt_feat, dim=0).T
    return _recall_from_similarity(sim, k_vals)


@torch.no_grad()
def compute_recall_single_template(
    model,
    loader,
    tokenizer,
    device: str,
    template: str = SINGLE_TEMPLATE,
    k_vals: Sequence[int] = (1, 5, 10),
    desc: str = "Single template",
) -> dict[str, float]:
    """Evaluate retrieval after wrapping every caption in one template."""

    model.eval()
    all_img_feat = []
    all_txt_feat = []

    for images, captions_list in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        img_feat = model.encode_image(images)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        all_img_feat.append(img_feat.cpu())

        prompted = [template.format(c) for c in _first_captions(captions_list)]
        tokens = tokenizer(prompted).to(device)
        txt_feat = model.encode_text(tokens)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        all_txt_feat.append(txt_feat.cpu())

    sim = torch.cat(all_img_feat, dim=0) @ torch.cat(all_txt_feat, dim=0).T
    return _recall_from_similarity(sim, k_vals)


@torch.no_grad()
def compute_recall_ensemble(
    model,
    loader,
    tokenizer,
    device: str,
    templates: Sequence[str] = ENSEMBLE_TEMPLATES,
    k_vals: Sequence[int] = (1, 5, 10),
    desc: str = "Template ensemble",
) -> dict[str, float]:
    """Average text embeddings across handcrafted templates, then normalize."""

    model.eval()
    all_img_feat = []
    all_txt_feat = []

    for images, captions_list in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        img_feat = model.encode_image(images)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        all_img_feat.append(img_feat.cpu())

        first_caps = _first_captions(captions_list)
        template_feats = []
        for template in templates:
            prompted = [template.format(c) for c in first_caps]
            tokens = tokenizer(prompted).to(device)
            template_feats.append(model.encode_text(tokens))

        txt_feat = torch.stack(template_feats, dim=0).mean(dim=0)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        all_txt_feat.append(txt_feat.cpu())

    sim = torch.cat(all_img_feat, dim=0) @ torch.cat(all_txt_feat, dim=0).T
    return _recall_from_similarity(sim, k_vals)

