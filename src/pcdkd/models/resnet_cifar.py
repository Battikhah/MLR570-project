"""CIFAR ResNets (6n+2); option-A shortcuts follow He et al. (2016), §4.2."""
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, shortcut_type="A"):
        super().__init__()
        if shortcut_type not in {"A", "B"}:
            raise ValueError("shortcut_type must be 'A' or 'B'")
        self.shortcut_type = shortcut_type
        self.in_channels = in_channels
        self.channels = channels
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Sequential()
        if shortcut_type == "B" and (stride != 1 or in_channels != channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.shortcut_type == "A":
            if self.stride != 1:
                residual = residual[:, :, :: self.stride, :: self.stride]
            channels_to_add = self.channels - self.in_channels
            if channels_to_add:
                residual = F.pad(
                    residual,
                    (0, 0, 0, 0, channels_to_add // 2, channels_to_add - channels_to_add // 2),
                )
        else:
            residual = self.shortcut(residual)
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(out + residual)


class CifarResNet(nn.Module):
    def __init__(self, depth, num_classes=10, shortcut_type="A"):
        super().__init__()
        if (depth - 2) % 6:
            raise ValueError("depth must be 6n+2")
        if shortcut_type not in {"A", "B"}:
            raise ValueError("shortcut_type must be 'A' or 'B'")
        n = (depth - 2) // 6
        self.shortcut_type = shortcut_type
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._layer(16, n, 1)
        self.layer2 = self._layer(32, n, 2)
        self.layer3 = self._layer(64, n, 2)
        self.linear = nn.Linear(64, num_classes)

    def _layer(self, channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, channels, stride, self.shortcut_type)]
        self.in_channels = channels
        layers.extend(
            BasicBlock(channels, channels, shortcut_type=self.shortcut_type)
            for _ in range(blocks - 1)
        )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.linear(F.avg_pool2d(x, 8).flatten(1))


def resnet8(num_classes=10, shortcut_type="A"):
    return CifarResNet(8, num_classes, shortcut_type)


def resnet14(num_classes=10, shortcut_type="A"):
    return CifarResNet(14, num_classes, shortcut_type)


def resnet20(num_classes=10, shortcut_type="A"):
    return CifarResNet(20, num_classes, shortcut_type)


def resnet32(num_classes=10, shortcut_type="A"):
    return CifarResNet(32, num_classes, shortcut_type)
