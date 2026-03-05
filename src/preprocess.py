"""
preprocess.py — Data loading and transformation pipeline.

All configuration values (image size, normalization stats, batch size, etc.)
are read from config.yaml at the project root. No hardcoded numbers.
"""

import os
from pathlib import Path

import yaml
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Locate and parse config.yaml from the project root."""
    root = Path(__file__).resolve().parent.parent
    config_path = root / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Public API ────────────────────────────────────────────────────────────────

def load_image(path: str, image_size: int) -> Image.Image:
    """
    Load an image from disk, convert to RGB, and resize to (image_size, image_size).

    Args:
        path: Absolute or relative path to the image file.
        image_size: Target width and height in pixels.

    Returns:
        A PIL.Image.Image in RGB mode at the requested size.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return img


def get_transforms(split: str) -> transforms.Compose:
    """
    Build a torchvision transforms pipeline from config.yaml.

    Args:
        split: One of "train" or "val" (use "val" for test-time inference too).

    Returns:
        A torchvision.transforms.Compose pipeline.
    """
    cfg = _load_config()
    image_size = cfg["model"]["image_size"]
    mean = cfg["data"]["normalize"]["mean"]
    std = cfg["data"]["normalize"]["std"]
    aug = cfg["data"]["augmentation"]

    if split == "train":
        pipeline = []

        if aug.get("random_resized_crop"):
            pipeline.append(transforms.RandomResizedCrop(image_size))
        else:
            pipeline.append(transforms.Resize((image_size, image_size)))

        if aug.get("random_horizontal_flip"):
            pipeline.append(transforms.RandomHorizontalFlip())

        cj_cfg = aug.get("color_jitter")
        if cj_cfg:
            pipeline.append(
                transforms.ColorJitter(
                    brightness=cj_cfg.get("brightness", 0),
                    contrast=cj_cfg.get("contrast", 0),
                    saturation=cj_cfg.get("saturation", 0),
                    hue=cj_cfg.get("hue", 0),
                )
            )

        if aug.get("gaussian_blur"):
            pipeline.append(transforms.GaussianBlur(kernel_size=3))

        pipeline += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

        return transforms.Compose(pipeline)

    else:  # "val" or "test"
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


def build_dataset(data_dir: str, split: str) -> ImageFolder:
    """
    Construct a torch ImageFolder dataset for the given split.

    Expects the directory structure:
        data_dir/
            {split}/
                REAL/   <- class 0
                FAKE/   <- class 1

    Args:
        data_dir: Root path of the dataset (e.g. "data/cifake-real-and-ai-generated-synthetic-images").
        split:    "train" or "test".

    Returns:
        A torch.utils.data.Dataset (ImageFolder).
    """
    transform = get_transforms("train" if split == "train" else "val")
    split_dir = os.path.join(data_dir, split)
    return ImageFolder(root=split_dir, transform=transform)


def build_dataloader(
    dataset: ImageFolder,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """
    Wrap a dataset in a DataLoader.

    Args:
        dataset:     A torch Dataset (typically from build_dataset).
        batch_size:  Number of samples per batch.
        num_workers: Number of parallel data-loading workers.

    Returns:
        A torch.utils.data.DataLoader.
    """
    cfg = _load_config()
    pin_memory = cfg["data"].get("pin_memory", True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
