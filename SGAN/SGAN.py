import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, nz=50, ngf=[512, 256, 128, 64], nc=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf[0], 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(ngf[0]),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf[0], ngf[1], 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(ngf[1]),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf[1], ngf[2], 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(ngf[2]),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf[2], ngf[3], 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(ngf[3]),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf[3], nc, 5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )
        # --- Version généralisée équivalente ---
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

        self.net2 = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=[64, 128, 256, 512]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf[0], 5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf[0], ndf[1], 5, stride=2, padding=2),
            nn.BatchNorm2d(ndf[1]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf[1], ndf[2], 5, stride=2, padding=2),
            nn.BatchNorm2d(ndf[2]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf[2], ndf[3], 5, stride=2, padding=2),
            nn.BatchNorm2d(ndf[3]),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf[3], 1, 5, stride=2, padding=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

