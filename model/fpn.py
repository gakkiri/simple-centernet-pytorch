import torch
from torch import nn

Norm = nn.BatchNorm2d


class Conv1x1(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.conv = nn.Conv2d(num_in, num_out, kernel_size=1, bias=False)
        self.norm = Norm(num_out)
        self.active = nn.ReLU(True)
        self.block = nn.Sequential(self.conv, self.norm, self.active)

    def forward(self, x):
        return self.block(x)


class Conv3x3(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__()
        self.conv = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1,
                              bias=False)
        self.norm = Norm(num_out)
        self.active = nn.ReLU(True)
        self.block = nn.Sequential(self.conv, self.norm, self.active)

    def forward(self, x):
        return self.block(x)


class FPN(nn.Module):
    def __init__(self, inplanes, outplanes=512):
        super(FPN, self).__init__()

        self.laterals = nn.Sequential(*[Conv1x1(inplanes // (2 ** c), outplanes) for c in range(4)])
        self.smooths = nn.Sequential(*[Conv3x3(outplanes * c, outplanes * c) for c in range(1, 5)])
        self.pooling = nn.MaxPool2d(2)

    def forward(self, features):
        laterals = [l(features[f]) for f, l in enumerate(self.laterals)]

        map4 = laterals[3]
        map3 = laterals[2] + nn.functional.interpolate(map4, scale_factor=2,
                                                       mode="nearest")
        map2 = laterals[1] + nn.functional.interpolate(map3, scale_factor=2,
                                                       mode="nearest")
        map1 = laterals[0] + nn.functional.interpolate(map2, scale_factor=2,
                                                       mode="nearest")

        map1 = self.smooth[0](map1)
        map2 = self.smooth[1](torch.cat([map2, self.pooling(map1)], dim=1))
        map3 = self.smooth[2](torch.cat([map3, self.pooling(map2)], dim=1))
        map4 = self.smooth[3](torch.cat([map4, self.pooling(map3)], dim=1))
        return map4

