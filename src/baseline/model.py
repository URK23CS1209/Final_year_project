"""
Baseline Model Architecture for Binary Anemia Classification
------------------------------------------------------------
Implements standard CNN backbones (MobileNetV2, ResNet18/50, VGG16)
with configurable classification heads and transfer learning support.
"""

import torch
import torch.nn as nn
from torchvision import models


class AneRBCClassifier(nn.Module):
    """
    CNN Classifier for binary anemia diagnosis on RBC microscopic images.
    Outputs raw linear logits for numerical stability with BCEWithLogitsLoss.
    """

    def __init__(
        self,
        model_name: str = "mobilenet_v2",
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        num_classes: int = 1
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.num_classes = num_classes

        if self.model_name == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )

        elif self.model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )

        elif self.model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )

        elif self.model_name == "vgg16":
            weights = models.VGG16_Weights.DEFAULT if pretrained else None
            self.backbone = models.vgg16(weights=weights)
            in_features = self.backbone.classifier[6].in_features
            self.backbone.classifier[5] = nn.Dropout(p=dropout_rate)
            self.backbone.classifier[6] = nn.Linear(in_features, num_classes)

        else:
            raise ValueError(f"Unsupported model architecture: {model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        # Squeeze out last dimension if binary classification
        if self.num_classes == 1:
            out = out.view(-1)
        return out

    def freeze_backbone(self):
        """Freeze all feature extraction layers, leaving only classification head trainable."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        if self.model_name == "mobilenet_v2":
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        elif "resnet" in self.model_name:
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        elif self.model_name == "vgg16":
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all layers for end-to-end fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
