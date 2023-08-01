import torch
import torch.optim as optim
import torch.nn as nn

import init
import dataset
import training
import models
import config


def main():
    """
    The main function that initiates the application by calling 'init.boot()'.

    This function serves as the entry point of the application and is responsible for starting the initialization process.

    Args:
        None

    Returns:
        None
    """
    # Call the 'boot()' function from the 'init' module to initialize the application
    init.boot()

    # load the dataset
    dataloader = dataset.get_dataloader()

    # load the models
    ngpu = torch.cuda.device_count()

    vae = models.VAE()

    netG = models.Generator(vae, ngpu)
    netD = models.Discriminator(ngpu)

    netG = netG.to(config.DEVICE)
    netD = netD.to(config.DEVICE)
    
    # Handle multi-GPU if desired
    ngpu = torch.cuda.device_count()
    if (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))
        
    # initialize criterion
    criterion = torch.nn.BCELoss()

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(
    ), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(config.BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(
    ), lr=config.LEARNING_RATE_GENERATOR, betas=(config.BETA1, 0.999))

    # train the model
    training.fit(netG, netD, vae, dataloader, criterion, optimizerG, optimizerD)


if __name__ == "__main__":
    # Call the main function to start the application
    main()
