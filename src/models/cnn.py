import torch
from torch import nn
from torchvision.models import resnet18, resnet50

from src.models.modules import MaskDropout


def build_LeNet(config):
    assert config is not None
    return LeNet(config.data_shape, config.num_classes, config.dropout_rate)


def build_ResNet18(config):
    assert config is not None
    return ResNetClassifier(resnet18(weights=None), config.data_shape, config.num_classes, config.dropout_rate)


def build_ResNet50(config):
    assert config is not None
    return ResNetClassifier(resnet50(weights=None), config.data_shape, config.num_classes, config.dropout_rate)


class LeNet(nn.Module):
    """Small LeNet-style CNN with a MaskDropout layer before the classification head."""

    def __init__(self, data_shape=(3, 32, 32), num_classes=10, dropout_rate=0.):
        super().__init__()
        channels, height, width = data_shape
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # two 2x2 max pools shrink each spatial dimension by a factor of 4
        flat_features = 128 * max(height // 4, 1) * max(width // 4, 1)
        self.fc1 = nn.Linear(flat_features, 256)
        self.relu = nn.ReLU(inplace=True)
        self.do = MaskDropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.do(x)
        return self.fc2(x)


class ResNetClassifier(nn.Module):
    """Randomly initialized torchvision ResNet with a MaskDropout layer before the classification head."""

    def __init__(self, backbone, data_shape=(3, 32, 32), num_classes=10, dropout_rate=0.):
        super().__init__()
        channels = data_shape[0]
        if channels != 3:
            backbone.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.inner = backbone
        self.do = MaskDropout(dropout_rate)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.inner(x)
        x = self.do(x)
        return self.fc(x)
