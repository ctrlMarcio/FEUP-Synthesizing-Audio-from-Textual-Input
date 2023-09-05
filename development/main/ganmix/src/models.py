import torch
from torch import nn
from diffusers import AutoencoderKL
from torchsummary import summary
from torch.nn import BCEWithLogitsLoss
import torch.optim as optim

import config


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

    # factory initialize models

    @classmethod
    def build(cls):
        # Get the number of available GPUs
        num_workers = torch.cuda.device_count()

        vae = VAE()

        netG = Generator(num_workers, conv_per_layer=2, channel_multiplier=3, l1_lambda=0.0001, l2_lambda=0.0001)
        netD = Discriminator(num_workers, conv_per_layer=3, channel_multiplier=4)

        netG = netG.to(config.DEVICE)
        netD = netD.to(config.DEVICE)

        # Handle multi-GPU if desired
        if num_workers > 1:
            netG = nn.DataParallel(netG, list(range(num_workers)))
            netD = nn.DataParallel(netD, list(range(num_workers)))

        # initialize criterion
        criterion = BCEWithLogitsLoss()

        # Setup Adam optimizers for both G and D
        optimizerD = optim.Adam(netD.parameters(
        ), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(config.BETA1, 0.999))
        optimizerG = optim.Adam(netG.parameters(
        ), lr=config.LEARNING_RATE_GENERATOR, betas=(config.BETA1, 0.999))

        # Return the GANMix object
        return cls(vae, netG, netD, optimizerG, optimizerD, criterion, num_workers)


class Generator(nn.Module):
    def __init__(self, ngpu, nz=config.GENERATOR_INPUT_SIZE, ngf=config.FEATURE_MAPS_GENERATOR, l1_lambda=0.01, l2_lambda=0.01, latent_channels=config.CHANNELS_LATENT, channel_multiplier=1, dropout_prob=0.5, gaussian_noise=config.GAUSSIAN_NOISE, conv_per_layer=2):
        super(Generator, self).__init__()

        # Initialize the Generator with the provided parameters
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.channel_multiplier = channel_multiplier
        self.dropout_prob = dropout_prob
        self.latent_channels = latent_channels
        self.gaussian_noise = gaussian_noise
        self.conv_per_layer = conv_per_layer

        # Build the network layers using the private method _build_network
        self.main = self._build_network()

    def forward(self, input):
        # Move the input data to the appropriate device (CPU/GPU)
        input = input.to(config.DEVICE)

        # Perform the forward pass through the main network
        x = self.main(input)

        # crop to fit the VAE encoder output
        # acrtual size before this is torch.Size([1, 8, 131, 107])
        x = x[:, :, :128, :]

        # Decode the latent features to create a spectrogram using the VAE's decode function
        # x = self.vae.decode(x).sample

        # Return the generated spectrogram
        return x

    def elastic_net_regularization(self):
        l1_penalty = 0
        l2_penalty = 0

        for param in self.parameters():
            l1_penalty += torch.sum(torch.abs(param))
            l2_penalty += torch.sum(param ** 2)

        elastic_net_loss = self.l1_lambda * l1_penalty + self.l2_lambda * l2_penalty
        return elastic_net_loss

    def _build_network(self):
        layers = []
        in_channels = self.nz
        for out_channels, kernel_size, stride, padding in [
            (self.ngf * 4 * self.channel_multiplier, 5, 5, 1),
            (self.ngf * 2 * self.channel_multiplier, 5, (4, 3), 1),
            (self.ngf * 1 * self.channel_multiplier, 5, 4, 1),
            (self.latent_channels, 5, 3, 0)
        ]:
            layers.append(nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            # add convolutional layers
            for _ in range(self.conv_per_layer):
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                        stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(True))
            layers.append(nn.Dropout(self.dropout_prob))
            in_channels = out_channels

        # layers.append(nn.Tanh())
        # add linear activation

        return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, ngpu, channels_data=config.CHANNELS_DATA, ndf=config.FEATURE_MAPS_GENERATOR, gaussian_noise=config.GAUSSIAN_NOISE, conv_per_layer=2, channel_multiplier=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.channels_data = channels_data
        self.gaussian_noise = gaussian_noise
        self.conv_per_layer = conv_per_layer
        self.channel_multiplier = channel_multiplier

        # Create the layers of the Discriminator
        self.features = self._build_network()

    def forward(self, input):
        # Put input tensor on GPU if available
        # input = input.to(config.DEVICE)

        # Add Gaussian noise to the input data
        input = input + torch.randn_like(input) * self.gaussian_noise

        # Pass the input through the Discriminator's layers
        input = self.features(input)

        return input

    def _create_block(self, in_channels, out_channels):
        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=2, padding=1, bias=False))
        for _ in range(self.conv_per_layer - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                    stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        return nn.Sequential(*layers)

    def _build_network(self):
        return nn.Sequential(
            self._create_block(self.channels_data,
                               self.ndf * self.channel_multiplier),
            self._create_block(self.ndf * self.channel_multiplier,
                               self.ndf * 2 * self.channel_multiplier),
            self._create_block(self.ndf * 2 * self.channel_multiplier,
                               self.ndf * 4 * self.channel_multiplier),
            nn.Conv2d(self.ndf * 4 * self.channel_multiplier, 1, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(8*7, 1),
            nn.Sigmoid()
        )


def VAE():
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path=config.VAE_MODEL, subfolder=config.VAE_MODEL_SUBFOLDER).to(config.DEVICE)
    return vae


def _weights_init(m):
    # TODO DELETE THIS FUNCTION
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    vae = VAE()
    ngpu = torch.cuda.device_count()
    # Create the generator
    netG = Generator(vae, ngpu).to(config.DEVICE)

    # Handle multi-GPU if desired
    if (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    #  to ``mean=0``, ``stdev=0.02``.
    netG.apply(_weights_init)

    # Print the model
    summary(netG, (config.GENERATOR_INPUT_SIZE, 1, 1))
    print(
        f"Number of trainable parameters: {sum(p.numel() for p in netG.parameters() if p.requires_grad)}")

    # Create the Discriminator
    netD = Discriminator(ngpu).to(config.DEVICE)

    # Handle multi-GPU if desired
    if (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the ``weights_init`` function to randomly initialize all weights
    # like this: ``to mean=0, stdev=0.2``.
    netD.apply(_weights_init)

    summary(vae, (1, 513, 431), 4)

    # Print the model
    # TODO update the input size for the discriminator
    # summary(netD, (1, 513, 431), 4)
    # print(f"Number of trainable parameters: {sum(p.numel() for p in netD.parameters() if p.requires_grad)}")