import torch.nn as nn
from torchvision import models

class ConvNextLit_Inference(nn.Module):
    def __init__(self, num_classes, dropout: float, weights=None):
        super().__init__()

        self.model = models.convnext_tiny(weights=weights)

        last_linear_in = self.model.classifier[-1].in_features
        self.model.classifier = nn.Sequential(
            *list(self.model.classifier[:-1]),
            nn.Dropout(p=float(dropout)),
            nn.Linear(last_linear_in, num_classes),
        )

    def forward(self, x):
        return self.model(x)