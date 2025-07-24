import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, nz=50, ngf=[512, 256, 128, 64], nc=3):
        super().__init__()

        
        layers = []
        in_channels = nz 

        # Main Blocs (ConvT -> BatchNorm -> ReLU)
        for out_channels in ngf:
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, 5, stride=2, padding=2, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            ])
            in_channels = out_channels 

        # last bloc (ConvT -> Tanh)
        layers.append(nn.ConvTranspose2d(in_channels, nc, 5, stride=2, padding=2, output_padding=1))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=[64, 128, 256, 512]):
        super().__init__()
        
        # Premiers layers (Sans BatchNorm)
        layers = [nn.Conv2d(nc, ndf[0], 5, stride=2, padding=2),
                nn.LeakyReLU(0.2, inplace=True),]
        in_channels = ndf[0]

        # Main blocs (Conv -> BatchNorm -> LeakyReLU)
        for out_channels in ndf[1:]:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 5, stride=2, padding=2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_channels = out_channels 

        # last bloc (Conv -> Sigmoid)
        layers.append(nn.Conv2d(in_channels, 1, 5, stride=2, padding=2))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

