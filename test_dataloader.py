# Quick script — save as test_dataloader.py and run it
import time
import yaml
from src.dataset import ForensicImageDataset
from torch.utils.data import DataLoader
from src.preprocess import get_transforms

cfg = yaml.safe_load(open("config.yaml"))

dataset = ForensicImageDataset(
    data_dir="data/processed/train",
    split="train",
    transform=get_transforms("train"),
    feature_cache_path=cfg["forensic_features"]["cache_path"]
)

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=cfg["data"]["num_workers"],
    pin_memory=cfg["data"]["pin_memory"],
    prefetch_factor=cfg["data"]["prefetch_factor"],
    persistent_workers=cfg["data"]["persistent_workers"]
)

print(f"Dataset size: {len(dataset)} images")
start = time.time()
for i, batch in enumerate(loader):
    if i == 20: break
elapsed = time.time() - start
print(f"20 batches in {elapsed:.2f}s — {elapsed/20*1000:.0f}ms per batch")
print(f"Estimated epoch time: {(elapsed/20) * (len(dataset)/32) / 60:.1f} minutes")