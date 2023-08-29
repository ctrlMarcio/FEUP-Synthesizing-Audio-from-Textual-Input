import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim
import numpy as np

import config
import models

class GANMix():
    def __init__(self, vae, generator, discriminator, generator_optimizer, discriminator_optimizer, criterion, num_workers=1):
        # models
        self.vae = vae
        self.generator = generator
        self.discriminator = discriminator

        # optimizers
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        # criterion
        self.criterion = criterion

        # workers
        self.num_workers = num_workers

    @classmethod
    def build(cls):
        # generator
        # input: config.GENERATOR_INPUT_SIZE
        # output: (batch_size, 8, 128, 107) TODO
        latents_shape = (8, 128, 107)
    
        num_workers = torch.cuda.device_count()

        vae = models.VAE()

        netG = Generator(config.GENERATOR_INPUT_SIZE, 200, latents_shape)
        netD = Discriminator(latents_shape, 200, config.GAUSSIAN_NOISE)

        netG = netG.to(config.DEVICE)
        netD = netD.to(config.DEVICE)

        if num_workers > 1:
            netG = nn.DataParallel(netG, list(range(num_workers)))
            netD = nn.DataParallel(netD, list(range(num_workers)))

        criterion = BCEWithLogitsLoss()

        optimizerD = optim.Adam(netD.parameters(
        ), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(config.BETA1, 0.999))
        optimizerG = optim.Adam(netG.parameters(
        ), lr=config.LEARNING_RATE_GENERATOR, betas=(config.BETA1, 0.999))

        return cls(vae, netG, netD, optimizerG, optimizerD, criterion, num_workers)

class Generator(nn.Module):
    # working pretty much the same way as a regular generator in a GAN
    # but taking into account that this GAN generates values that are not
    # spatially correlated, we don't use convolutions, but linear layers

    def __init__(self, input_size, hidden_size, output_shape):
        super(Generator, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_shape = output_shape

        self.network = self._build_network()

    def forward(self, x):
        x = x.to(config.DEVICE)

        x = self.network(x)

        return x        

    def _build_network(self):
        network = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size, np.prod(self.output_shape)),
        )
        return network 

    def forward(self, x):
        x = x.to(config.DEVICE)

        x = self.network(x)
        x = x.view(-1, *self.output_shape)

        return x

class Discriminator(nn.Module):
    # apply the same logic as in the Generator
    # but in reverse

    def __init__(self, input_shape, hidden_size, gaussian_noise_std):
        super(Discriminator, self).__init__()

        self.gaussian_noise_std = gaussian_noise_std
        
        self.input_shape = input_shape
        self.hidden_size = hidden_size

        self.network = self._build_network()

    def forward(self, x):
        # TODO add gaussian noise

        x = x + torch.randn_like(x) * self.gaussian_noise_std

        x = self.network(x)

        return x

    def _build_network(self):
        network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.input_shape), self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size, 1),
        )
        return network 
