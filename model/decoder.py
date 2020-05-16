from torch import nn


class Decoder(nn.Module):
    def __init__(self, inplanes, bn_momentum=0.1):
        super(Decoder, self).__init__()
        self.bn_momentum = bn_momentum
        # backbone output: [b, 2048, _h, _w]
        self.inplanes = inplanes
        self.deconv_with_bias = False
        self.deconv_layers = self._make_deconv_layer(
            num_layers=3,
            num_filters=[256, 256, 256],
            num_kernels=[4, 4, 4],
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            padding = 0 if kernel == 2 else 1
            output_padding = 1 if kernel == 3 else 0
            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.deconv_layers(x)
