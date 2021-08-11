# Network options
from torchvision.models.vgg import vgg16  # noqa
from torchvision.models.alexnet import alexnet  # noqa
from torchvision.models.resnet import (  # noqa
    resnet50, resnet34, resnet101, resnet152)

from efficientnet_pytorch import EfficientNet

import torch.nn as nn

pretrained = torchvision.models.alexnet(pretrained=True)
class MyAlexNet(nn.Module):
    def __init__(self, my_pretrained_model):
        super(MyAlexNet, self).__init__()
        self.pretrained = my_pretrained_model
        self.my_new_layers = nn.Sequential(nn.Linear(1000, 100),
                                           nn.ReLU(),
                                           nn.Linear(100, 2))
    
    def forward(self, x):
        x = self.pretrained(x)
        x = self.my_new_layers(x)
        return x
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

    # Distributed, when required
    # net = nn.DataParallel(net)

    return net
