import torch.nn as nn
from torchvision import models
import torch.nn.functional as functional
import time
import torch


class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        self.backend_feat = [(512, 2), (512, 2), (512, 2), (256, 2), (128, 2), (64, 2)]
        self.front_end = nn.Sequential(*(list(list(models.vgg16_bn(True).children())[0].children())[0:33]))
        self.back_end = make_layers(self.backend_feat, in_channels=512, batch_norm=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        for m in self.output_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        img_shape = x.shape
        front_end = self.front_end(x)
        back_end = self.back_end(front_end)
        output = self.output_layer(back_end)
        output = functional.interpolate(output, scale_factor=4, mode='bilinear', align_corners=False)
        return output


def make_layers(cfg, in_channels, batch_norm=False):
    layers = []
    for v, atrous in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=atrous, dilation=atrous)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
