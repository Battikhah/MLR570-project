"""Small CIFAR ResNets following the 6n+2 basic-block layout."""
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels),
            )

    def forward(self, x):
        import torch.nn.functional as F
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class CifarResNet(nn.Module):
    def __init__(self, depth, num_classes=10):
        super().__init__()
        if (depth - 2) % 6:
            raise ValueError("depth must be 6n+2")
        n = (depth - 2) // 6
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._layer(16, n, 1)
        self.layer2 = self._layer(32, n, 2)
        self.layer3 = self._layer(64, n, 2)
        self.linear = nn.Linear(64, num_classes)

    def _layer(self, channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        layers.extend(BasicBlock(channels, channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        import torch.nn.functional as F
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.linear(F.avg_pool2d(x, 8).flatten(1))


def resnet8(num_classes=10):
    return CifarResNet(8, num_classes)


def resnet14(num_classes=10):
    return CifarResNet(14, num_classes)


def resnet20(num_classes=10):
    return CifarResNet(20, num_classes)


def resnet32(num_classes=10):
    return CifarResNet(32, num_classes)
