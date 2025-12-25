# code provided by torchdiffeq repo!
# https://github.com/rtqichen/torchdiffeq/blob/master/examples/odenet_mnist.py

import torch
from torch import nn

from torchdiffeq import odeint, odeint_adjoint

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc, adjoint, solver):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.adjoint = adjoint
        self.solver = solver

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)

        '''
        odeint_adjoint: adjoint method를 사용한 ODE solver
        odeint: 일반적인 ODE solver + backpropagation 시에 일반적인 backpropagation 방식 사용 (메모리 많이 사용)
        둘 다 torchdiffeq 패키지에서 제공하는 함수

        method 인자는 ODE solver의 종류를 지정
        odeint_adjoint는 method, adjoint_method 두 가지 인자를 받는데, adjoint_method는 지정하지 않으면 method 따라간다.
        method 자체를 지정하지 않으면, 기본으로 'dopri5'가 사용된다.
        '''

        if self.adjoint:
            out = odeint_adjoint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3, method=self.solver)
        else:
            out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3, method=self.solver)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

class OfficialODENet(nn.Module):
    def __init__(self, 
                 adjoint=True,
                 solver='dopri5'):
        super(OfficialODENet, self).__init__()
        self.downsampling = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        )
        self.feature = ODEBlock(ODEfunc(64), adjoint=adjoint, solver=solver)
        self.norm = norm(64)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.downsampling(x)
        x = self.feature(x)
        x = self.norm(x)
        # 원래 공식은 여기서 ReLU가 하나 더 있었는데, 실험 동등성을 위해 제거했다. -> ResNet 구현할 때, 까먹고 빼고 구현해서..
        x = self.avg_pool(x)
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        x = x.view(-1, shape)
        out = self.fc(x)
        return out
    
    @classmethod
    def from_config(cls, cfg):
        return cls(adjoint=cfg.get('adjoint', True),
                   solver=cfg.get('solver', 'implicit_adams'))