from torch import nn


class LeakyReLUConv2d(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, stride, padding=0, norm='None', sn=False):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.ReflectionPad2d(padding)]
        if sn:
            model += [nn.utils.spectral_norm(nn.Conv2d(inplanes, outplanes,
                                                       kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
        else:
            model += [nn.Conv2d(inplanes, outplanes,
                                kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if norm == 'Instance':
            model += [nn.InstanceNorm2d(outplanes, affine=False)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, ndf=64):
        super(Discriminator, self).__init__()

        model = []
        model += [LeakyReLUConv2d(input_dim, ndf * 2, kernel_size=3, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(ndf * 2, ndf * 2, kernel_size=3, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(ndf * 2, ndf * 2, kernel_size=3, stride=2, padding=1, norm='Instance')]
        model += [LeakyReLUConv2d(ndf * 2, ndf * 2, kernel_size=1, stride=1, padding=0)]
        model += [nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        out = x.view(-1)

        return out