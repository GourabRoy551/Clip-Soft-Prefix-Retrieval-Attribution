"""Training loop for CoOp-style soft prefixes."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .evaluation import compute_recall_at_k
from .modeling import SoftPrefix, encode_text_with_prefix, save_prefix


def infonce_loss(img_feat: torch.Tensor, txt_feat: torch.Tensor, logit_scale: torch.Tensor):
    """Symmetric CLIP-style InfoNCE loss over a batch of matched pairs."""

    logits = logit_scale * img_feat @ txt_feat.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2.0


def cosine_warmup_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Linear warmup followed by cosine decay."""

    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_soft_prefix(
    model,
    tokenizer,
    train_loader,
    val_loader,
    device: str,
    checkpoint_dir: Path,
    n_ctx: int = 32,
    epochs: int = 10,
    lr: float = 2e-3,
    weight_decay: float = 1e-2,
    warmup_epochs: int = 1,
    init_text: str = "a photo of a",
    checkpoint_name: str | None = None,
):
    """Train only the prefix vectors while CLIP stays frozen."""

    embed_dim = model.token_embedding.weight.shape[1]
    prefix_module = SoftPrefix(
        n_ctx=n_ctx,
        embed_dim=embed_dim,
        init_text=init_text,
        model=model,
        tokenizer=tokenizer,
    ).to(device)

    optimizer = torch.optim.AdamW(prefix_module.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    scheduler = cosine_warmup_scheduler(optimizer, total_steps, warmup_steps)
    logit_scale = model.logit_scale.exp().detach()

    train_losses = []
    val_i2t_r1 = []
    best_val = 0.0
    best_epoch = 0
    checkpoint_stem = checkpoint_name or f"prefix_k{n_ctx}"

    for epoch in range(1, epochs + 1):
        prefix_module.train()
        model.eval()
        epoch_losses = []

        for images, captions in tqdm(train_loader, desc=f"k={n_ctx} epoch {epoch}/{epochs}", leave=False):
            images = images.to(device)
            tokens = tokenizer(list(captions)).to(device)

            with torch.no_grad():
                img_feat = model.encode_image(images)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            txt_raw = encode_text_with_prefix(model, tokens, prefix_module(), device)
            txt_feat = txt_raw / txt_raw.norm(dim=-1, keepdim=True)
            loss = infonce_loss(img_feat, txt_feat, logit_scale)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        train_losses.append(mean_loss)

        prefix_module.eval()
        val_result = compute_recall_at_k(
            model,
            val_loader,
            tokenizer,
            device,
            prefix=prefix_module().detach(),
            k_vals=(1,),
            desc=f"k={n_ctx} validation",
        )
        val_score = val_result["i2t_R@1"]
        val_i2t_r1.append(val_score)
        print(
            f"k={n_ctx} epoch {epoch:02d}/{epochs} "
            f"loss={mean_loss:.4f} val_i2t_R@1={val_score:.2f}% "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            save_prefix(prefix_module().detach(), checkpoint_dir / f"{checkpoint_stem}_best.pt")

        if epoch % 2 == 0:
            save_prefix(prefix_module().detach(), checkpoint_dir / f"{checkpoint_stem}_epoch{epoch:02d}.pt")

    history = {
        "n_ctx": n_ctx,
        "epochs": epochs,
        "train_loss": train_losses,
        "val_i2t_R@1": val_i2t_r1,
        "best_val_i2t_R@1": best_val,
        "best_epoch": best_epoch,
    }
    return prefix_module().detach(), history

