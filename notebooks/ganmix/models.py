import torch
from torch import nn
from diffusers import AutoencoderKL
from torchsummary import summary

import config


# Define the Generator class
class Generator(nn.Module):
    def __init__(self, vae, ngpu, nz=config.GENERATOR_INPUT_SIZE, ngf=config.FEATURE_MAPS_GENERATOR, l2_lambda=0.01, latent_channels=config.CHANNELS_LATENT, channel_multiplier=1, dropout_prob=0.5):
        super(Generator, self).__init__()

        # Initialize the Generator with the provided parameters
        self.vae = vae
        self.ngpu = ngpu
        self.nz = nz
        self.ngf = ngf
        self.l2_lambda = l2_lambda
        self.channel_multiplier = channel_multiplier
        self.dropout_prob = dropout_prob
        self.latent_channels = latent_channels

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
        #x = self.vae.decode(x).sample

        # Return the generated spectrogram
        return x
    

    def _build_network(self):
        # Define the network architecture using a sequential container
        # Each layer is explained below

        # First transposed convolution layer
        self.conv1 = nn.ConvTranspose2d(self.nz, self.ngf * 4 * self.channel_multiplier, kernel_size=5, stride=5, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.ngf * 4 * self.channel_multiplier)
        self.relu1 = nn.LeakyReLU(True)
        self.dropout1 = nn.Dropout(self.dropout_prob)

        # Second transposed convolution layer
        self.conv2 = nn.ConvTranspose2d(self.ngf * 4 * self.channel_multiplier, self.ngf * 2 * self.channel_multiplier, kernel_size=5, stride=(4,3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.ngf * 2 * self.channel_multiplier)
        self.relu2 = nn.LeakyReLU(True)
        self.dropout2 = nn.Dropout(self.dropout_prob)

        # Third transposed convolution layer
        self.conv3 = nn.ConvTranspose2d(self.ngf * 2 * self.channel_multiplier, self.ngf * 1 * self.channel_multiplier, kernel_size=5, stride=4, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.ngf * 1 * self.channel_multiplier)
        self.relu3 = nn.LeakyReLU(True)
        self.dropout3 = nn.Dropout(self.dropout_prob)

        # Sixth transposed convolution layer
        self.conv6 = nn.ConvTranspose2d(self.ngf * 1 * self.channel_multiplier, self.latent_channels, kernel_size=5, stride=3, padding=0, bias=False)
        self.tanh = nn.Tanh()

        return nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu1,
            self.dropout1,

            self.conv2,
            self.bn2,
            self.relu2,
            self.dropout2,

            self.conv3,
            self.bn3,
            self.relu3,
            self.dropout3,

            self.conv6,
            self.tanh
        )



class Discriminator(nn.Module):
    def __init__(self, ngpu, channels_data=config.CHANNELS_DATA, ndf=config.FEATURE_MAPS_GENERATOR):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.ndf = ndf
        self.channels_data = channels_data

        # Create the layers of the Discriminator
        self.features = self._build_network()

    def forward(self, input):
        # Put input tensor on GPU if available
        #input = input.to(config.DEVICE)
        
        # Pass the input through the Discriminator's layers
        input = self.features(input)
        
        return input

    def _build_network(self):
        # Define the architecture of the Discriminator
        return nn.Sequential(
            # Convolutional layer 1
            nn.Conv2d(self.channels_data, self.ndf, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Convolutional layer 2
            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Convolutional layer 3
            nn.Conv2d(self.ndf * 2, self.ndf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Convolutional layer 4
            nn.Conv2d(self.ndf * 4, 1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Sigmoid(),

            # Flatten the output of the last convolutional layer
            nn.Flatten(),

            # Linear layer 1
            nn.Linear(8*7, 1),

            # Sigmoid activation function
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
    print(f"Number of trainable parameters: {sum(p.numel() for p in netG.parameters() if p.requires_grad)}")

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
    #summary(netD, (1, 513, 431), 4)
    #print(f"Number of trainable parameters: {sum(p.numel() for p in netD.parameters() if p.requires_grad)}")


