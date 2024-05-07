import torch
import torch.nn as nn

class block(nn.Module):
    """
    A basic building block for a ResNet architecture.

    Parameters:
    - in_channels (int): Number of channels in the input tensor.
    - out_channels (int): Number of channels produced by the intermediate convolutions.
    - identity_downsample (nn.Module, optional): Module used to transform the identity 
      path to match dimensions with the main path, used in case of increasing dimensions or strides.
    - stride (int): Stride size for the convolutional layers, affecting the output spatial dimensions.
    """

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4  # Expands the channel size by a factor of 4 at the end of the block
        
        # First 1x1 convolution: it changes the channel dimensionality without affecting spatial dimensions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)  # Normalizes the output of conv1

        # Second 3x3 convolution: the core convolutional layer that processes the data spatially
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)  # Normalizes the output of conv2

        # Third 1x1 convolution: increases the channel depth by a factor of self.expansion
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)  # Normalizes the output of conv3

        self.relu = nn.ReLU()  # Activation function to introduce non-linearity
        
        self.identity_downsample = identity_downsample  # Optional module for adjusting dimensions

    def forward(self, x):
        identity = x  # Save the input for adding after processing

        # First convolutional layer followed by normalization and activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Second convolutional layer
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Third convolutional layer
        x = self.conv3(x)
        x = self.bn3(x)

        # If an identity_downsample module is provided, apply it to match dimensions
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity  # Add the original input to the output of the convolutional layers
        x = self.relu(x)  # Final activation function

        return x

class ResNet(nn.Module):
    """
    Implements a ResNet architecture using the defined block class as building blocks.

    Parameters:
    - block: The class used for constructing the residual blocks.
    - layers (list of int): List defining the number of blocks in each of the four layers of the network.
    - image_channels (int): Number of channels in the input image.
    - num_classes (int): Number of classes for the final classification.
    """
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 16  # Initial number of channels after the first convolution layer
        # Initial convolutional layer before the first block
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers: each layer is a sequence of blocks
        self.layer1 = self._make_layer(block, layers[0], out_channels=16, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=32, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=64, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=128, stride=2)  # Ends with 2048 channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * 4, num_classes)  # Final fully connected layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Passing the input through all the residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pooling and classification
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten the features
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        # Adjusting the identity path to match dimensions when needed
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4),
            )

        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4  # Updating in_channels for the next block

        # Adding the remaining blocks to the layer
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))  # These use a stride of 1 by default

        return nn.Sequential(*layers)

def resnet50(img_channels=1, num_classes=7):
    return ResNet(block, [3, 4, 6, 3], img_channels, num_classes)