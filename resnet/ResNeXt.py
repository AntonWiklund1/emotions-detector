import torch
import torch.nn as nn
from .cbam import CBAM

torch.manual_seed(42)

class ResNeXtBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, cardinality=32, identity_downsample=None, stride=1):
        super(ResNeXtBlock, self).__init__()
        self.cardinality = cardinality
        D = out_channels * self.cardinality // 32
        self.conv1 = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D, momentum=0.01)
        
        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=self.cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(D, momentum=0.01)
        
        self.conv3 = nn.Conv2d(D, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion, momentum=0.01)
        
        self.relu = nn.ReLU(inplace=True)
        self.identity_downsample = identity_downsample

        self.cbam = CBAM(out_channels * self.expansion)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.cbam(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        
        x = self.relu(x)

        return x

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
    def __init__(self, block, layers, image_channels, num_classes, cardinality=32):
        super(ResNeXt, self).__init__()
        self.in_channels = 64
        self.cardinality = cardinality
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion, momentum=0.01),
            )

        layers.append(block(self.in_channels, out_channels, self.cardinality, identity_downsample, stride))
        self.in_channels = out_channels * block.expansion

        for _ in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels, self.cardinality))

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

def resnext50(img_channels=1, num_classes=7, cardinality=32):
    return ResNeXt(ResNeXtBlock, [3, 4, 6, 3], img_channels, num_classes, cardinality)

def resnext101(img_channels=1, num_classes=7, cardinality=32):
    return ResNeXt(ResNeXtBlock, [3, 4, 23, 3], img_channels, num_classes, cardinality)
