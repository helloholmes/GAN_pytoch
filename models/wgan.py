# coding:utf-8
'''
python 3.5
pytorch 0.4.0
visdom 0.1.7
torchnet 0.0.2
auther: helloholmes
'''
import os
import torch
import torchvision as tv
import time
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from torchnet import meter

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%Y_%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)

class WGAN_NetG(BasicModule):
    def __init__(self, opt):
        # input size: (batch, opt.nz, 1, 1)
        super(WGAN_NetG, self).__init__()
        ngf = opt.ngf
        self.feature = nn.Sequential(
                    nn.ConvTranspose2d(opt.nz, ngf*8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf*8),
                    nn.ReLU(True),
                    # (batch, ngf*8, 4, 4)

                    nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf*4),
                    nn.ReLU(True),
                    # (batch, ngf*4, 8, 8)

                    nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf*2),
                    nn.ReLU(True),
                    # (batch, ngf*2, 16, 16)

                    nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                    #(batch, ngf, 32, 32)

                    nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
                    nn.Tanh(),
                    # (batch, 3, 64, 64)
                    )

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.feature(x)
        return x

class WGAN_NetD(BasicModule):
    def __init__(self, opt):
        # input size: (batch, 3, 96, 96)
        super(WGAN_NetD, self).__init__()
        ndf = opt.ndf
        self.feature = nn.Sequential(
                    nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
                    nn.BatchNorm2d(ndf),
                    nn.LeakyReLU(0.2, inplace=True),
                    # (batch, ndf, 32, 32)

                    nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf*2),
                    nn.LeakyReLU(0.2, inplace=True),
                    # (batch, ndf*2, 16, 16)

                    nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf*4),
                    nn.LeakyReLU(0.2, inplace=True),
                    # (batch, ndf*4, 8, 8)

                    nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf*8),
                    nn.LeakyReLU(0.2, inplace=True),
                    # (batch, ndf*8, 4, 4)

                    nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
                    # (batchm 1, 1, 1)
                    # no sigmoid
                    )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # return size: (batch)
        x = self.feature(x)
        return x.view(-1)