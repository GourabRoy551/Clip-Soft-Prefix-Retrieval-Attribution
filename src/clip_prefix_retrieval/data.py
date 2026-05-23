"""Dataset wrappers for Flickr30k and MS-COCO retrieval experiments."""

from __future__ import annotations

from io import BytesIO
from typing import Iterable

import numpy as np
import requests
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import BATCH_SIZE, EVAL_BATCH_SIZE, SEED


class Flickr30kTrainDataset(Dataset):
    """Return one random caption per image for prefix training."""

    def __init__(self, hf_split, preprocess):
        self.data = hf_split
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        cap_idx = np.random.randint(len(item["caption"]))
        caption = item["caption"][cap_idx]
        image = self.preprocess(item["image"].convert("RGB"))
        return image, caption


class Flickr30kEvalDataset(Dataset):
    """Return an image and all five captions for retrieval evaluation."""

    def __init__(self, hf_split, preprocess):
        self.data = hf_split
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        image = self.preprocess(item["image"].convert("RGB"))
        captions = item["caption"]
        return image, captions


def load_flickr30k_splits():
    """Load Flickr30k from Hugging Face and split by its internal split column."""

    flickr = load_dataset("nlphuji/flickr30k", trust_remote_code=True)
    full_data = flickr["test"]
    train_data = full_data.filter(lambda x: x["split"] == "train")
    val_data = full_data.filter(lambda x: x["split"] == "val")
    test_data = full_data.filter(lambda x: x["split"] == "test")
    return train_data, val_data, test_data


def make_flickr30k_loaders(
    preprocess,
    batch_size: int = BATCH_SIZE,
    eval_batch_size: int = EVAL_BATCH_SIZE,
):
    """Create train/validation/test loaders under the notebook protocol."""

    train_data, val_data, test_data = load_flickr30k_splits()
    train_dataset = Flickr30kTrainDataset(train_data, preprocess)
    val_dataset = Flickr30kEvalDataset(val_data, preprocess)
    test_dataset = Flickr30kEvalDataset(test_data, preprocess)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }


class COCOEvalDataset(Dataset):
    """MS-COCO validation subset with images downloaded from COCO URLs."""

    def __init__(self, hf_dataset, preprocess, indices: Iterable[int] | None = None):
        self.data = hf_dataset
        self.preprocess = preprocess
        self.indices = list(indices) if indices is not None else list(range(len(hf_dataset)))
        self.blank_downloads = 0

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        item = self.data[self.indices[idx]]
        url = item["coco_url"]

        image = None
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                break
            except Exception:
                if attempt == 2:
                    self.blank_downloads += 1
                    image = Image.new("RGB", (224, 224), (128, 128, 128))

        captions = [
            c["raw"] if isinstance(c, dict) and "raw" in c
            else c.get("caption", str(c)) if isinstance(c, dict)
            else str(c)
            for c in item["captions"]
        ]
        return self.preprocess(image), captions


def coco_collate_fn(batch):
    """Transpose COCO captions to match the Flickr30k loader shape."""

    images = torch.stack([item[0] for item in batch])
    max_caps = max(len(item[1]) for item in batch)
    padded = [item[1] + [item[1][0]] * (max_caps - len(item[1])) for item in batch]
    captions = [list(slot) for slot in zip(*padded)]
    return images, captions


def make_coco_loader(preprocess, n_eval: int = 500, batch_size: int = 16, seed: int = SEED):
    """Create the MS-COCO transfer-evaluation loader."""

    coco_raw = load_dataset("phiyodr/coco2017", trust_remote_code=True)
    coco_val = coco_raw["validation"]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(coco_val), n_eval, replace=False).tolist()
    dataset = COCOEvalDataset(coco_val, preprocess, indices=indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=coco_collate_fn,
    )
    return dataset, loader

