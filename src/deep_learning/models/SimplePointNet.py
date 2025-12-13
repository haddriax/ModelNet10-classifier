import torch
from torch import nn


class SimplePointNet(nn.Module):
    """Minimal PointNet for testing"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: [B, N, 3]
        x = self.mlp(x)  # [B, N, 1024]
        x = torch.max(x, dim=1)[0]  # [B, 1024] - max pooling
        x = self.classifier(x)  # [B, num_classes]
        return x