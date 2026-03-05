"""
Entry point for all future training runs.

IMPORTANT: This file never re-initializes the model from scratch.
All training goes through ContinualTrainer which loads the active
checkpoint from config.yaml before any weight updates happen.

To run a fresh training run (new architecture experiments only),
use a separate script — do NOT modify this file to do it.
"""
from src.continual_trainer import ContinualTrainer


def get_trainer(config_path: str = "config.yaml") -> ContinualTrainer:
    """
    Returns a ContinualTrainer with the active checkpoint loaded.
    Use this in any notebook or script that needs to fine-tune the model.

    Example usage in a notebook:
        from src.train import get_trainer
        trainer = get_trainer()
        trainer.fine_tune(
            train_loader=train_loader,
            val_loader=val_loader,
            dataset_name="artifact_stage1",
            output_checkpoint="models/finetune_v1_artifact.pt",
            learning_rate=5e-6,
            epochs=10,
        )
    """
    return ContinualTrainer(config_path=config_path)
