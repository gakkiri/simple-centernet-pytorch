from torch import nn


class Head(nn.Module):
    def __init__(self, num_classes=80, channel=64):
        super(Head, self).__init__()

        self.cls_head = nn.Sequential(
            nn.Conv2d(256, channel,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, num_classes,
                      kernel_size=1, stride=1, padding=0))
        self.wh_head = self.ConvReluConv(256, 2)
        self.reg_head = self.ConvReluConv(256, 2)

    def ConvReluConv(self, in_channel, out_channel, bias_fill=False, bias_value=0):
        feat_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        relu = nn.ReLU()
        out_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        if bias_fill:
            out_conv.bias.data.fill_(bias_value)
        return nn.Sequential(feat_conv, relu, out_conv)

    def forward(self, x):
        hm = self.cls_head(x).sigmoid()
        wh = self.wh_head(x).relu()
        offset = self.reg_head(x)
        return hm, wh, offset
