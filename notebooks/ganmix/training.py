from torch import nn
import time


def fit(netG, netD, dataloader, num_epochs=5):
    # Set the initial parameters
    print('Initializating the parameters...')
    # epochs
    epoch = 0

    # losses for displaying purposes
    G_losses = []
    D_losses = []

    # initialize the networks
    netG.apply(_weights_init)
    netD.apply(_weights_init)

    # Training loop
    print('Starting the training loop...')
    for epoch in range(epoch, num_epochs):
        epoch_start_time = time.time()

        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
