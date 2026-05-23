"""CLIP loading and soft-prefix text encoding."""

from __future__ import annotations

import torch
import torch.nn as nn

import open_clip

from .config import MODEL_NAME, PRETRAINED


def load_frozen_clip(device: str, model_name: str = MODEL_NAME, pretrained: str = PRETRAINED):
    """Load OpenAI CLIP through open_clip and freeze both encoders."""

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model, preprocess, tokenizer


def encode_text_with_prefix(model, tokens: torch.Tensor, prefix: torch.Tensor, device: str):
    """Encode text after prepending continuous prefix vectors.

    This mirrors the notebook implementation: the prefix is inserted after
    token embedding lookup and before positional embeddings. The CLIP encoders
    remain frozen; gradients flow only into the prefix tensor during training.
    """

    tokens = tokens.to(device)
    prefix = prefix.to(device)
    batch_size = tokens.shape[0]
    n_ctx = prefix.shape[0]

    x = model.token_embedding(tokens).to(device)
    prefix_expanded = prefix.unsqueeze(0).expand(batch_size, -1, -1)
    x = torch.cat([prefix_expanded, x], dim=1)

    seq_len = min(x.shape[1], model.positional_embedding.shape[0])
    x = x[:, :seq_len, :]
    x = x + model.positional_embedding[:seq_len].unsqueeze(0)

    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x)

    eot_positions = tokens.argmax(dim=-1) + n_ctx
    eot_positions = eot_positions.clamp(max=seq_len - 1)
    x = x[torch.arange(batch_size, device=device), eot_positions]
    return x @ model.text_projection


class SoftPrefix(nn.Module):
    """Trainable prompt prefix initialised from real CLIP token embeddings."""

    def __init__(
        self,
        n_ctx: int = 16,
        embed_dim: int = 512,
        init_text: str = "a photo of a",
        model=None,
        tokenizer=None,
    ):
        super().__init__()

        if model is None or tokenizer is None:
            self.prefix = nn.Parameter(torch.randn(n_ctx, embed_dim) * 0.02)
            return

        with torch.no_grad():
            init_tokens = tokenizer(init_text)
            token_ids = init_tokens[0]
            content_ids = []
            for tid in token_ids[1:]:
                if tid.item() >= 49406:
                    break
                content_ids.append(int(tid.item()))

            device = next(model.parameters()).device
            init_tensor = torch.tensor(content_ids, dtype=torch.long, device=device)
            init_vecs = model.token_embedding(init_tensor).detach()

        if init_vecs.shape[0] >= n_ctx:
            init_data = init_vecs[:n_ctx]
        else:
            pad = torch.randn(n_ctx - init_vecs.shape[0], embed_dim, device=init_vecs.device) * 0.02
            init_data = torch.cat([init_vecs, pad], dim=0)

        self.prefix = nn.Parameter(init_data.clone())

    def forward(self) -> torch.Tensor:
        return self.prefix


def save_prefix(prefix_tensor: torch.Tensor, path) -> None:
    """Save only the learned prefix tensor, not the full CLIP model."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prefix_tensor.detach().cpu(), path)


def load_prefix(path, device: str) -> torch.Tensor:
    """Load a saved prefix tensor."""

    return torch.load(path, map_location=device)
