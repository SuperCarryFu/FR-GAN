import torch
import torch.nn as nn
from torch.nn import Parameter
from copy import deepcopy




class SSAE(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32, norm='bn', n_blocks=6):
        super(SSAE, self).__init__()
        use_bias = norm == 'in'
        self.c1=nn.Conv2d(input_nc, ngf, kernel_size=3, padding=1,stride=2, bias=use_bias)
        self.c2=nn.Conv2d(ngf, ngf, kernel_size=3, padding=1,stride=2, bias=use_bias)
        self.c3=nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, stride=2, bias=use_bias)

        self.c4 = nn.Conv2d(ngf, ngf, kernel_size=3, padding=1, stride=1, bias=use_bias)
        self.c5 = nn.Conv2d(ngf, 3, kernel_size=3, padding=1, stride=1, bias=use_bias)
    def forward(self, x):
        # encoding
        x = self.c1(x)
        x = self.c2(x)
        latency = self.c3(x)
        latency = self.c4(latency)
        latency = self.c5(latency)


        return latency


from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
t = SSAE().to(device)
summary(t, (3, 112, 112))
print(t)

