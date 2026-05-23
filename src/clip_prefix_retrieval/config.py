"""Project-wide constants.

The values here mirror the final Colab notebook so that the scripts can
reproduce the same experimental protocol.
"""

from pathlib import Path

SEED = 42

MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"

BATCH_SIZE = 64
EVAL_BATCH_SIZE = 32

METRICS = ("i2t_R@1", "i2t_R@5", "i2t_R@10", "t2i_R@1", "t2i_R@5", "t2i_R@10")

TEMPLATES = [
    "{}",
    "a photo of {}",
    "a picture showing {}",
    "an image of {}",
    "this is a photo of {}",
]
SINGLE_TEMPLATE = TEMPLATES[1]
ENSEMBLE_TEMPLATES = TEMPLATES

DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")

