"""
Copyright (c) 2019, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
-----------------------------------------------------
The Y-shaped autoencoder model file.
"""

import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, depth=3):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(depth, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.encoder.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(0.5, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, depth=3):
        super(Decoder, self).__init__()

        self.shared = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.segmenter = nn.Sequential(

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, depth, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Tanh()
        )

        self.segmenter.apply(self.weights_init)
        self.decoder.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(0.5, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x):
        latent = self.shared(x)
        seg = self.segmenter(latent)
        rect = self.decoder(latent)

        return seg, rect


class AEModel(nn.Module):
    def __init__(self, depth=3):
        super(AEModel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(depth, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.shared = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.segmenter = nn.Sequential(

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.ConvTranspose2d(8, depth, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Tanh()
        )

        self.out_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Linear(1,1),
            nn.Sigmoid()
        )

        self.segmenter.apply(self.weights_init)
        self.decoder.apply(self.weights_init)
        self.encoder.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(0.5, 0.02)
            m.bias.data.fill_(0)

    def forward(self, x, labels_data):
        latent = self.encoder(x)
        latent = latent.reshape(-1, 2, 64, 16, 16)
        zero_abs = torch.abs(latent[:, 0]).view(latent.shape[0], -1)
        zero = zero_abs.mean(dim=1)

        one_abs = torch.abs(latent[:, 1]).view(latent.shape[0], -1)
        one = one_abs.mean(dim=1)
        # y = torch.eye(2).to(x.device)
        # y = y.index_select(dim=0, index=1 - labels_data.data.long())
        # latent = (latent * y[:, :, None, None, None]).reshape(-1, 128, 16, 16)
        latent = latent.reshape(-1,128,16,16)
        latent = self.shared(latent)
        seg = self.segmenter(latent)
        rect = self.decoder(latent)
        pred = self.out_fc(seg)

        return zero, one, seg, rect, pred



