import torch
import torch.nn as nn
from .cbam import CBAM
import math

torch.manual_seed(42)

class ResNeXtBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, cardinality=32, bottleneck_width=4, identity_downsample=None, stride=1):
        super(ResNeXtBottleneck, self).__init__()
        D = int(math.floor(out_channels * (bottleneck_width / 64.0)))
        C = cardinality

        self.conv1 = nn.Conv2d(in_channels, D * C, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(D * C)

        self.conv2 = nn.Conv2d(D * C, D * C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm2d(D * C)

        self.conv3 = nn.Conv2d(D * C, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.identity_downsample = identity_downsample

        self.cbam = CBAM(out_channels * self.expansion)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.cbam(out)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        out += identity
        out = self.relu(out)

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class ResNeXt(nn.Module):
    def __init__(self, block, layers, image_channels=1, num_classes=7, cardinality=32, bottleneck_width=4):
        super(ResNeXt, self).__init__()
        self.in_channels = 64
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width

        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers.append(block(self.in_channels, out_channels, self.cardinality, self.bottleneck_width, identity_downsample, stride))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, self.cardinality, self.bottleneck_width))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

def resnext50(img_channels=1, num_classes=7):
    return ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], img_channels, num_classes)

def resnext101(img_channels=1, num_classes=7):
    return ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], img_channels, num_classes)
