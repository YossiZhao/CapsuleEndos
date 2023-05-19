

"""
    If you want load models from pytorch instead of building by yourself, please visit "torchvision.models".
"""


import torch
import torch.nn as nn


# Basic Block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, \
                               stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, \
                               stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(identity)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# Resnet
class Resnet(nn.Module):

    def __init__(self, basicblock, num_layers, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, \
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.in_channels = 64
        self.layer1 = self._make_layer(basicblock, 64, num_layers[0])
        self.layer2 = self._make_layer(basicblock, 128, num_layers[1], stride=2)
        self.layer3 = self._make_layer(basicblock, 256, num_layers[2], stride=2)
        self.layer4 = self._make_layer(basicblock, 512, num_layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, basicblock, out_channels, num_layers, stride=1):

        downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, \
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers.append(basicblock(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels

        for _ in range(1, num_layers):
            layers.append(basicblock(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

# resnet34 function
def resnet34(num_classes):
    num_layers = [3, 4, 6, 3]
    model = Resnet(BasicBlock, num_layers, num_classes)
    return model



















