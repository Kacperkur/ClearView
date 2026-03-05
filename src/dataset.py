"""
dataset.py — ForensicImageDataset

Returns (image_tensor, forensic_feature_vector, label) per sample.
Supports pre-computed feature caching via .npz file to avoid re-extraction
every epoch.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.preprocess import get_transforms


class ForensicImageDataset(Dataset):
    """
    Dataset that returns (image_tensor, forensic_feature_vector, label).

    Args:
        data_dir:           Root dataset directory.
        split:              "train" or "test".
        transform:          torchvision transform pipeline. If None, uses
                            get_transforms(split) from preprocess.py.
        feature_cache_path: Path to a .npz file with pre-computed features.
                            Keys: "paths", "features".
                            If None, features are extracted on-the-fly (slow).
        feature_dim:        Expected feature vector dimension (default 46).
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        transform=None,
        feature_cache_path: Optional[str] = None,
        feature_dim: int = 46,
    ):
        self.feature_dim = feature_dim
        self.transform = transform or get_transforms(
            "train" if split == "train" else "val"
        )

        # Collect image paths and labels from class subdirectories
        split_dir = Path(data_dir) / split
        self.samples: list[tuple[str, int]] = []
        self.class_to_idx: dict[str, int] = {}

        classes = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        for idx, cls_name in enumerate(classes):
            self.class_to_idx[cls_name] = idx
            cls_dir = split_dir / cls_name
            for fname in cls_dir.iterdir():
                if fname.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    self.samples.append((str(fname), idx))

        # Load feature cache if provided
        self._cache: Optional[dict[str, np.ndarray]] = None
        if feature_cache_path and os.path.exists(feature_cache_path):
            cache = np.load(feature_cache_path, allow_pickle=True)
            paths = cache["paths"]
            features = cache["features"]
            # Build path -> feature lookup
            self._cache = {p: features[i] for i, p in enumerate(paths)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, label = self.samples[idx]

        # --- Image ---
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        # --- Forensic features ---
        if self._cache is not None and image_path in self._cache:
            feat = self._cache[image_path].astype(np.float32)
        else:
            # On-the-fly extraction (slow — use precompute_features.py first)
            from src.feature_extractor import extract_forensic_features
            feat = extract_forensic_features(image_path)

        # Guard: ensure correct length, fill with zeros if mismatched
        if len(feat) != self.feature_dim:
            padded = np.zeros(self.feature_dim, dtype=np.float32)
            n = min(len(feat), self.feature_dim)
            padded[:n] = feat[:n]
            feat = padded

        feature_tensor = torch.tensor(feat, dtype=torch.float32)

        return image_tensor, feature_tensor, label
