import torch
import torch.nn as nn
from res_net.res_block import ResBlock


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels,
        resblock,
        repeat,
        useBottleneck=False,
        outputs=1000,
        kernel_size=11,
        stride=2,
        padding=3,
        pool_kernel_size=3,
        pool_stride=2,
        pool_padding=1,
    ):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.MaxPool2d(kernel_size=pool_kernel_size,
                         stride=pool_stride, padding=pool_padding),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        if useBottleneck:
            filters = [64, 256, 512, 1024, 2048]
        else:
            filters = [64, 64, 128, 256, 512]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', resblock(
            filters[0], filters[1], downsample=False))
        for i in range(1, repeat[0]):
            self.layer1.add_module('conv2_%d' % (
                i+1,), resblock(filters[1], filters[1], downsample=False))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', resblock(
            filters[1], filters[2], downsample=True))
        for i in range(1, repeat[1]):
            self.layer2.add_module('conv3_%d' % (
                i+1,), resblock(filters[2], filters[2], downsample=False))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', resblock(
            filters[2], filters[3], downsample=True))
        for i in range(1, repeat[2]):
            self.layer3.add_module('conv2_%d' % (
                i+1,), resblock(filters[3], filters[3], downsample=False))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', resblock(
            filters[3], filters[4], downsample=True))
        for i in range(1, repeat[3]):
            self.layer4.add_module('conv3_%d' % (
                i+1,), resblock(filters[4], filters[4], downsample=False))

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(filters[4], outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input).detach()
        input = self.gap(input)
        input = torch.flatten(input, start_dim=1)
        input = self.fc(input)

        return input

    def encode(self, spectrogram):
        """
        Encodes a spectrogram into a vector
        """
        return self(spectrogram)

    def encode_batch(self, spectrograms):
        """
        Encodes a batch of spectrograms into a batch of vectors
        """
        return self(spectrograms)