# ClearView — How To Use

## What's Trained
Two EfficientNet-B3 checkpoints in `models/`:
- `baseline_efficientnet_b3.pt` — Hybrid model (image + 46-dim forensic features) — **use this one**
- `baseline_image_only.pt` — Ablation baseline, pixels only

---

## Running Evaluation on the Test Set
Open `notebooks/02_finetune.ipynb` and run cells in order. The training cells (Sections 5 and 6) will **skip automatically** if checkpoints already exist. Jump straight to **Section 8** to evaluate on the test split.

---

## Continuing Training (More Epochs on CIFAKE)

```python
from src.train import get_trainer
from src.dataset import ForensicImageDataset
from torch.utils.data import DataLoader

trainer = get_trainer()  # loads active_checkpoint from config.yaml

# Rebuild the same loaders used in Phase 2
train_loader = ...  # ForensicImageDataset, split="train"
val_loader   = ...  # ForensicImageDataset, split="val"

trainer.fine_tune(
    train_loader=train_loader,
    val_loader=val_loader,
    dataset_name="cifake_extended",
    output_checkpoint="models/finetune_v1_cifake_extended.pt",
    learning_rate=2e-6,   # ~10x lower than original LR
    epochs=10,
)
```

Update `config.yaml → continual_learning.active_checkpoint` to point to the new file if val loss improved.

---

## Training on a New Dataset

1. **Pre-compute forensic features** for the new data:
   ```bash
   python src/precompute_features.py --data_dir data/your-dataset/ --output data/processed/your_features.npz
   ```

2. **Seed the replay buffer** (run once):
   ```python
   from src.replay_buffer import populate_replay_buffer
   populate_replay_buffer("data/cifake-224/train/", "data/replay_buffer/")
   ```

3. **Fine-tune** (replay buffer automatically mixes in 20% old data):
   ```python
   from src.train import get_trainer
   from src.replay_buffer import build_replay_dataloader

   trainer     = get_trainer()
   train_loader = build_replay_dataloader("data/your-dataset/", "data/replay_buffer/")
   trainer.fine_tune(
       train_loader=train_loader,
       val_loader=val_loader,
       dataset_name="your_dataset_stage1",
       output_checkpoint="models/finetune_v1_your_dataset.pt",
       learning_rate=5e-6,
       epochs=10,
   )
   ```

---

## Checkpoint Naming Convention

| File | Dataset | Status |
|------|---------|--------|
| `baseline_efficientnet_b3.pt` | CIFAKE 224×224 | **Protected — never overwrite** |
| `baseline_image_only.pt` | CIFAKE 224×224 | **Protected — never overwrite** |
| `finetune_v1_<dataset>.pt` | New data, stage 1 | Safe to update |
| `finetune_v2_<dataset>.pt` | New data, stage 2 | Safe to update |

All runs are logged to `models/training_history.json`.

---

## Key Config Knobs (`config.yaml`)

| Key | Default | What it does |
|-----|---------|--------------|
| `continual_learning.active_checkpoint` | `baseline_efficientnet_b3.pt` | Which checkpoint `get_trainer()` loads |
| `continual_learning.replay_ratio` | `0.20` | Fraction of each batch from old data |
| `continual_learning.unfreeze_last_n_blocks` | `2` | Backbone blocks unfrozen during fine-tuning |
| `training.extra_epochs` | `10` | Additional epochs when resuming |
