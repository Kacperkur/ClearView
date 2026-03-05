import torch
import torch.nn as nn
import timm


class HybridDetector(nn.Module):
    """EfficientNet-B3 image embeddings fused with forensic feature MLP."""

    def __init__(self, forensic_feature_dim: int = 102, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        bb_dim = self.backbone.num_features  # 1536

        self.forensic_mlp = nn.Sequential(
            nn.Linear(forensic_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(bb_dim + 64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, image: torch.Tensor, forensic_features: torch.Tensor) -> torch.Tensor:
        return self.classifier(
            torch.cat([self.backbone(image), self.forensic_mlp(forensic_features)], dim=1)
        )


class ImageOnlyDetector(nn.Module):
    """Image-only EfficientNet-B3 baseline."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        bb_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(bb_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, image: torch.Tensor, forensic_features: torch.Tensor = None) -> torch.Tensor:
        return self.classifier(self.backbone(image))
