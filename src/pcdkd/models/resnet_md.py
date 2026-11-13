"""CRD / RepDistiller CIFAR ResNets, as packaged by mdistiller, so the published CIFAR-100
teachers load with ``load_state_dict(strict=True)``.

Architecture after HobbitLong/RepDistiller ``models/resnet.py`` (BSD-2-Clause; mdistiller copies
it), written here rather than copied: 3x3 stem, BasicBlocks with a 1x1 conv + BN ``downsample``
whenever the stride or width changes (option B), AvgPool2d(8), ``fc``. Parameter names match the
checkpoints (``layerN.i.{conv1,bn1,conv2,bn2,downsample.0,downsample.1}``, ``fc``).
"""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
        return F.relu(out + (x if self.downsample is None else self.downsample(x)))


class MdResNet(nn.Module):
    """``widths`` = (stem, layer1, layer2, layer3); x4 models use (32, 64, 128, 256)."""

    def __init__(self, depth, num_classes=100, widths=(16, 16, 32, 64)):
        super().__init__()
        if (depth - 2) % 6:
            raise ValueError("depth must be 6n+2")
        n = (depth - 2) // 6
        stem, w1, w2, w3 = widths
        self.conv1 = nn.Conv2d(3, stem, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem)
        self.layer1 = self._layer(stem, w1, n, 1)
        self.layer2 = self._layer(w1, w2, n, 2)
        self.layer3 = self._layer(w2, w3, n, 2)
        self.fc = nn.Linear(w3, num_classes)
        for m in self.modules():  # RepDistiller's init
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _layer(inplanes, planes, blocks, stride):
        return nn.Sequential(
            BasicBlock(inplanes, planes, stride),
            *(BasicBlock(planes, planes) for _ in range(blocks - 1)),
        )

    def forward(self, x, return_features=False):
        x = F.relu(self.bn1(self.conv1(x)))
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        logits = self.fc(F.avg_pool2d(f3, 8).flatten(1))
        return (logits, [f1, f2, f3]) if return_features else logits
