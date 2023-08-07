import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss

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

    # Get the number of available GPUs
    num_workers = torch.cuda.device_count()

    # load the dataset
    dataloader = dataset.get_dataloader(num_workers=num_workers)

    vae = models.VAE()

    netG = models.Generator(vae, num_workers)
    netD = models.Discriminator(num_workers)

    netG = netG.to(config.DEVICE)
    netD = netD.to(config.DEVICE)
    
    # Handle multi-GPU if desired
    if num_workers > 1:
        netG = nn.DataParallel(netG, list(range(num_workers)))
        netD = nn.DataParallel(netD, list(range(num_workers)))
        
    # initialize criterion
    criterion = BCEWithLogitsLoss()

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=config.LEARNING_RATE_DISCRIMINATOR, betas=(config.BETA1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(config.BETA1, 0.999))
    
    # Ask user whether to load checkpoint or start from scratch
    load_checkpoint = input("Do you want to load a checkpoint? (y/n): ").strip().lower() == 'y'
    
    # train the model
    training.fit(netG, netD, vae, dataloader, criterion, optimizerG, optimizerD, load_checkpoint=load_checkpoint)

if __name__ == "__main__":
    # Call the main function to start the application
    main()
