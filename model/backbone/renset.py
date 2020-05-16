"""Feature Pyramid Network (FPN) on top of ResNet. Comes with task-specific
   heads on top of it.

See:
- https://arxiv.org/abs/1612.03144 - Feature Pyramid Networks for Object
  Detection
- http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf - A Unified
  Architecture for Instance and Semantic Segmentation

"""
from torchvision import models
import torch.nn as nn
import timm


def convert_to_inplace_relu(model):
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            m.inplace = True


class ResNet(nn.Module):
    def __init__(self, slug='r50', pretrained=True):
        super().__init__()
        if not pretrained:
            print("Caution, not loading pretrained weights.")

        if slug == 'r18':
            self.resnet = models.resnet18(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif slug == 'r34':
            self.resnet = models.resnet34(pretrained=pretrained)
            num_bottleneck_filters = 512
        elif slug == 'r50':
            self.resnet = models.resnet50(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'r101':
            self.resnet = models.resnet101(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'r152':
            self.resnet = models.resnet152(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'rx50':
            self.resnet = models.resnext50_32x4d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'rx101':
            self.resnet = models.resnext101_32x8d(pretrained=pretrained)
            num_bottleneck_filters = 2048
        elif slug == 'r50d':
            self.resnet = timm.create_model('gluon_resnet50_v1d',
                                            pretrained=pretrained)
            convert_to_inplace_relu(self.resnet)
            num_bottleneck_filters = 2048
        elif slug == 'r101d':
            self.resnet = timm.create_model('gluon_resnet101_v1d',
                                            pretrained=pretrained)
            convert_to_inplace_relu(self.resnet)
            num_bottleneck_filters = 2048

        else:
            assert False, "Bad slug: %s" % slug

        self.outplanes = num_bottleneck_filters

    def forward(self, x):
        size = x.size()
        assert size[-1] % 32 == 0 and size[-2] % 32 == 0, \
            "image resolution has to be divisible by 32 for resnet"

        enc0 = self.resnet.conv1(x)
        enc0 = self.resnet.bn1(enc0)
        enc0 = self.resnet.relu(enc0)
        enc0 = self.resnet.maxpool(enc0)

        enc1 = self.resnet.layer1(enc0)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)

        return enc1, enc2, enc3, enc4

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def freeze_stages(self, stage):
        if stage >= 0:
            self.resnet.bn1.eval()
            for m in [self.resnet.conv1, self.resnet.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, stage + 1):
            layer = getattr(self.resnet, 'layer{}'.format(i))
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
