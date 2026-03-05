import random
import shutil
from pathlib import Path

import yaml
from torch.utils.data import ConcatDataset, DataLoader, random_split

from src.dataset import ForensicImageDataset


def build_replay_dataloader(
    new_data_dir: str,
    replay_buffer_dir: str,
    config_path: str = "config.yaml",
) -> DataLoader:
    """
    Returns a DataLoader that mixes new training data with a stratified sample
    of old data from the replay buffer.

    The replay_ratio in config.yaml controls how much old data is included.

    Example: new dataset has 10,000 images, replay_ratio=0.20
        → 10,000 new + 2,000 replay = 12,000 total per epoch

    Args:
        new_data_dir:       Root dir of the new dataset (must have train/REAL + train/FAKE)
        replay_buffer_dir:  Root dir of the replay buffer (REAL/ + FAKE/ flat structure)
        config_path:        Path to config.yaml
    """
    cfg    = yaml.safe_load(open(config_path))
    cl_cfg = cfg["continual_learning"]
    ff_cfg = cfg["forensic_features"]

    new_dataset = ForensicImageDataset(
        data_dir=new_data_dir,
        split="train",
        feature_cache_path=ff_cfg["cache_path"],
        feature_dim=ff_cfg["feature_dim"],
    )

    replay_dataset = ForensicImageDataset(
        data_dir=replay_buffer_dir,
        split="train",
        feature_cache_path=ff_cfg["cache_path"],
        feature_dim=ff_cfg["feature_dim"],
    )

    replay_size = min(
        int(len(new_dataset) * cl_cfg["replay_ratio"]),
        len(replay_dataset),
    )

    replay_subset, _ = random_split(
        replay_dataset,
        [replay_size, len(replay_dataset) - replay_size],
    )

    combined = ConcatDataset([new_dataset, replay_subset])

    print(f"New data:      {len(new_dataset):,} images")
    print(f"Replay buffer: {replay_size:,} images ({cl_cfg['replay_ratio'] * 100:.0f}% of new)")
    print(f"Total:         {len(combined):,} images per epoch")

    return DataLoader(
        combined,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )


def populate_replay_buffer(
    cifake_train_dir: str,
    replay_buffer_dir: str,
    n_per_class: int = 3000,
    seed: int = 42,
):
    """
    Builds the initial replay buffer by copying a stratified sample from
    the CIFAKE training set. Run once before first fine-tuning.

    Args:
        cifake_train_dir:   Path to cifake-224/train/  (contains REAL/ and FAKE/)
        replay_buffer_dir:  Where to write the buffer (e.g. data/replay_buffer/)
        n_per_class:        Images to sample per class (default 3,000 = ~5% of CIFAKE)
        seed:               Random seed for reproducibility
    """
    random.seed(seed)
    src_root = Path(cifake_train_dir)
    dst_root = Path(replay_buffer_dir)

    for cls in ["REAL", "FAKE"]:
        src_cls = src_root / cls
        dst_cls = dst_root / cls
        dst_cls.mkdir(parents=True, exist_ok=True)

        files = list(src_cls.glob("*"))
        if len(files) < n_per_class:
            raise ValueError(
                f"Only {len(files)} images in {src_cls}, but n_per_class={n_per_class}"
            )

        sampled = random.sample(files, n_per_class)
        for f in sampled:
            shutil.copy(f, dst_cls / f.name)

        print(f"Copied {n_per_class:,} {cls} images → {dst_cls}")

    print(f"\nReplay buffer ready at {dst_root}")
    print("Add data/replay_buffer/ to .gitignore")
