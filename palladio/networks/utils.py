# Network options
from torchvision.models.vgg import vgg16  # noqa
from torchvision.models.alexnet import alexnet  # noqa
from torchvision.models.resnet import (  # noqa
    resnet50, resnet34, resnet101, resnet152)

from efficientnet_pytorch import EfficientNet

import torch.nn as nn



class ResNet34_conv3x3(nn.Module):
    def __init__(self):
        super(ResNet34_conv3x3, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3) #add kernel size = 1
        self.pretrained = resnet34(pretrained = True)
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pretrained(x)
        
        return x

class ResNet34_conv1x1(nn.Module):
    def __init__(self):
        super(ResNet34_conv1x1, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 1) #
        self.pretrained = resnet34(pretrained = True)
        
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pretrained(x)
        
        return x

class ResNet34_inputx3(nn.Module):
    def __init__(self):
        super(ResNet34_input3x3, self).__init__()
        self.pretrained = resnet34(pretrained = True)
        
    
    def forward(self, x):
        z = torch.cat((x, x, x), 1)
        z = self.pretrained(z)
        
        return z


def get_network(network_name, num_classes, use_pretrained, n_input_channels=3):

    if network_name == 'efficientnet-b0':
        if use_pretrained:
            net = EfficientNet.from_pretrained(
                'efficientnet-b0', in_channels=n_input_channels,
                num_classes=num_classes)
        else:
            net = EfficientNet.from_name(
                'efficientnet-b0', in_channels=n_input_channels,
                num_classes=num_classes)

    if network_name == 'efficientnet-b4':
        if use_pretrained:
            net = EfficientNet.from_pretrained(
                'efficientnet-b4', in_channels=n_input_channels,
                num_classes=num_classes)
        else:
            net = EfficientNet.from_name(
                'efficientnet-b4', in_channels=n_input_channels,
                num_classes=num_classes)

    if network_name == 'efficientnet-b7':
        if use_pretrained:
            net = EfficientNet.from_pretrained(
                'efficientnet-b7', in_channels=n_input_channels,
                num_classes=num_classes)
        else:
            net = EfficientNet.from_name(
                'efficientnet-b7', in_channels=n_input_channels,
                num_classes=num_classes)

    if network_name == 'resnet34':
        net = resnet34(pretrained=use_pretrained)
        net.fc = nn.Linear(512, num_classes)

        if n_input_channels != 3:
            net.conv1 = nn.Conv2d(
                n_input_channels, 64, kernel_size=7, stride=2, padding=3,
                bias=False)

    if network_name == 'resnet50':
        net = resnet50(pretrained=use_pretrained)
        net.fc = nn.Linear(2048, num_classes)

        if n_input_channels != 3:
            net.conv1 = nn.Conv2d(
                n_input_channels, 64, kernel_size=7, stride=2, padding=3,
                bias=False)

    if network_name == 'resnet101':
        net = resnet101(pretrained=use_pretrained)
        net.fc = nn.Linear(2048, num_classes)

        if n_input_channels != 3:
            net.conv1 = nn.Conv2d(
                n_input_channels, 64, kernel_size=7, stride=2, padding=3,
                bias=False)

    if network_name == 'resnet34_conv3x3':
        net = ResNet34_conv3x3()
        net.fc = nn.Linear(512, num_classes)
    
    if network_name == 'resnet34_conv1x1':
        net = ResNet34_conv1x1()
        net.fc = nn.Linear(512, num_classes)

    if network_name == 'resnet34_inputx3':
        net = ResNet34_inputx3()
        net.fc = nn.Linear(512, num_classes)


    # Distributed, when required
    # net = nn.DataParallel(net)

    return net
