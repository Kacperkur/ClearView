import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import track
import yaml

console = Console()


class ContinualTrainer:
    """
    Manages all fine-tuning runs on top of existing checkpoints.
    Never re-initializes model weights from scratch.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = yaml.safe_load(open(config_path))
        self.cl_cfg = self.cfg["continual_learning"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.history = self._load_history()
        console.print(f"[green]ContinualTrainer ready on {self.device}[/green]")

    def _load_model(self) -> nn.Module:
        """
        Loads the model from the active checkpoint.
        Raises FileNotFoundError if checkpoint is missing — never falls back to
        random initialization silently.
        """
        from src.model import HybridDetector

        checkpoint_path = self.cl_cfg["active_checkpoint"]

        if not os.path.exists(checkpoint_path):
            available = [f for f in os.listdir("models/") if f.endswith(".pt")]
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}.\n"
                f"Set continual_learning.active_checkpoint in config.yaml to a valid .pt file.\n"
                f"Available checkpoints: {available}"
            )

        model = HybridDetector(
            forensic_feature_dim=self.cfg["forensic_features"]["feature_dim"]
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)

        console.print(f"[blue]Loaded checkpoint:[/blue]  {checkpoint_path}")
        console.print(f"[blue]Checkpoint epoch:[/blue]   {checkpoint.get('epoch', 'unknown')}")
        console.print(f"[blue]Checkpoint val_loss:[/blue] {checkpoint.get('val_loss', 'unknown')}")

        return model

    def _load_history(self) -> dict:
        """Loads training history log, or creates a fresh one."""
        history_path = Path("models/training_history.json")
        if history_path.exists():
            with open(history_path) as f:
                return json.load(f)
        return {
            "runs": [],
            "created": datetime.now().isoformat(),
        }

    def _apply_freezing_strategy(self, freeze_backbone: bool, unfreeze_last_n: int):
        """
        Freezes backbone layers to prevent overwriting old knowledge.
        Only the classifier head and the last N backbone blocks are updated.
        """
        if not freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = True
            console.print(
                "[yellow]Warning: full backbone unfrozen. "
                "Only use this for large, diverse datasets.[/yellow]"
            )
            return

        # Freeze everything first
        for param in self.model.parameters():
            param.requires_grad = False

        # Always unfreeze the classifier head and forensic MLP
        for name, param in self.model.named_parameters():
            if "classifier" in name or "forensic_mlp" in name:
                param.requires_grad = True

        # Unfreeze the last N backbone blocks
        for name, param in self.model.named_parameters():
            for i in range(unfreeze_last_n):
                block_idx = 5 - i  # EfficientNet-B3 has 6 blocks (0–5)
                if f"blocks.{block_idx}" in name:
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        console.print(
            f"[blue]Trainable params:[/blue] {trainable:,} / {total:,} "
            f"({100 * trainable / total:.1f}%)"
        )

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_auc: float,
        dataset_name: str,
        output_path: str,
    ):
        """
        Saves a versioned checkpoint and appends the run to the history log.
        Raises ValueError if the output path matches a protected baseline checkpoint.
        """
        protected = ["baseline_efficientnet_b3.pt", "baseline_image_only.pt"]
        if any(p in output_path for p in protected):
            raise ValueError(
                f"Attempted to overwrite a protected baseline checkpoint: {output_path}\n"
                "Choose a versioned filename, e.g. models/finetune_v1_artifact.pt"
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "val_loss": val_loss,
                "val_auc": val_auc,
                "dataset": dataset_name,
                "date": datetime.now().isoformat(),
                "parent_checkpoint": self.cl_cfg["active_checkpoint"],
            },
            output_path,
        )

        self.history["runs"].append(
            {
                "checkpoint": output_path,
                "parent": self.cl_cfg["active_checkpoint"],
                "epoch": epoch,
                "dataset": dataset_name,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "date": datetime.now().isoformat(),
            }
        )
        with open("models/training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        console.print(f"[green]Checkpoint saved:[/green] {output_path}")
        console.print(f"[green]History updated:[/green]  models/training_history.json")

    def fine_tune(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        dataset_name: str,
        output_checkpoint: str,
        learning_rate: float,
        epochs: int = 10,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 2,
    ) -> dict:
        """
        Fine-tunes the loaded model on new data.

        Args:
            train_loader:           DataLoader with new data (replay buffer already mixed in)
            val_loader:             DataLoader for validation
            dataset_name:           Human-readable tag, e.g. "artifact_stage1"
            output_checkpoint:      Save path, e.g. "models/finetune_v1_artifact.pt"
            learning_rate:          Use values from config.yaml continual_learning.learning_rates
            epochs:                 Fine-tuning epochs — keep low (5–10) for small datasets
            freeze_backbone:        True for most fine-tuning runs
            unfreeze_last_n_blocks: How many backbone tail blocks to allow weight updates in

        Returns:
            history dict with train_loss, val_loss, val_auc per epoch
        """
        self._apply_freezing_strategy(freeze_backbone, unfreeze_last_n_blocks)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=self.cfg["training"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss(
            label_smoothing=self.cfg["training"]["label_smoothing"]
        )
        use_amp = self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val_loss = float("inf")
        best_val_auc = 0.0
        history = {"train_loss": [], "val_loss": [], "val_auc": []}

        console.rule(f"[bold blue]Fine-tuning: {dataset_name}[/bold blue]")
        console.print(
            f"Epochs: {epochs} | LR: {learning_rate:.2e} | "
            f"Freeze backbone: {freeze_backbone}"
        )

        for epoch in range(1, epochs + 1):
            # ── Train ──────────────────────────────────────────────────────────
            self.model.train()
            r_loss, n_cor, n_tot = 0.0, 0, 0

            for imgs, feats, labels in track(
                train_loader, description=f"Ep {epoch}/{epochs} train"
            ):
                imgs   = imgs.to(self.device, non_blocking=True)
                feats  = feats.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = self.model(imgs, feats)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["training"]["grad_clip_norm"]
                )
                scaler.step(optimizer)
                scaler.update()

                r_loss += loss.item() * labels.size(0)
                n_cor  += (logits.argmax(1) == labels).sum().item()
                n_tot  += labels.size(0)

            train_loss = r_loss / n_tot
            train_acc  = n_cor  / n_tot

            # ── Validate ───────────────────────────────────────────────────────
            self.model.eval()
            v_loss, v_cor, v_tot = 0.0, 0, 0
            all_probs, all_labels = [], []

            with torch.no_grad():
                for imgs, feats, labels in val_loader:
                    imgs   = imgs.to(self.device, non_blocking=True)
                    feats  = feats.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        logits = self.model(imgs, feats)
                        loss   = criterion(logits, labels)

                    v_loss += loss.item() * labels.size(0)
                    v_cor  += (logits.argmax(1) == labels).sum().item()
                    v_tot  += labels.size(0)
                    probs   = torch.softmax(logits, dim=1)[:, 1]
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            from sklearn.metrics import roc_auc_score
            val_loss = v_loss / v_tot
            val_acc  = v_cor  / v_tot
            val_auc  = roc_auc_score(all_labels, all_probs)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)

            # ── Log ────────────────────────────────────────────────────────────
            table = Table(show_header=False, box=None)
            table.add_row("Epoch",       f"{epoch}/{epochs}")
            table.add_row("Train Loss",  f"{train_loss:.4f}")
            table.add_row("Train Acc",   f"{train_acc:.4f}")
            table.add_row("Val Loss",    f"{val_loss:.4f}")
            table.add_row("Val Acc",     f"{val_acc:.4f}")
            table.add_row("Val AUC-ROC", f"{val_auc:.4f}")
            console.print(table)

            # ── Checkpoint on improvement ──────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_auc  = val_auc
                self._save_checkpoint(
                    epoch, best_val_loss, best_val_auc, dataset_name, output_checkpoint
                )
                console.print(
                    f"  [green]✓ New best val_loss={best_val_loss:.4f}[/green]"
                )

        console.print(f"\n[bold green]Fine-tuning complete.[/bold green]")
        console.print(
            f"Best Val AUC: {best_val_auc:.4f} | Best Val Loss: {best_val_loss:.4f}"
        )
        console.print(
            f"Next step: update config.yaml → "
            f"continual_learning.active_checkpoint: {output_checkpoint}"
        )
        return history
