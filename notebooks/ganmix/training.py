import torch
from torch import nn
import time
import config
import utils


def fit(netG, netD, vae, dataloader, criterion, optimizerG, optimizerD, num_epochs=5):
    # Set the initial parameters
    print('Initializing the parameters...')
    
    # epochs
    epoch = 0

    # losses for displaying purposes
    G_losses = []
    D_losses = []

    # initialize the networks
    netG.apply(_weights_init)
    netD.apply(_weights_init)

    # save the dataloader len for displaying purposes
    dataloader_len = len(dataloader)

    # Training loop
    print("Starting Training Loop...")
    start_time = time.time()

    # For each epoch
    for epoch in range(epoch, num_epochs):
        print("\n" + "="*30)
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        epoch_start_time = time.time()
        
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            real = data[0].to(config.DEVICE)
            b_size = real.size(0)
            label = torch.full((b_size,), config.REAL_LABEL, dtype=torch.float, device=config.DEVICE)
            # Forward pass real batch through D
            embeddings = vae.encode(real)
            embeddings = embeddings.latent_dist.mode()
            output = netD(embeddings)
            output = output.view(-1)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, config.GENERATOR_INPUT_SIZE, 1, 1, device=config.DEVICE)
            # Generate fake latent features batch with G
            fake = netG(noise)
            label.fill_(config.FAKE_LABEL)
            label = label.float()
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            # fake labels are real for generator cost
            label.fill_(config.REAL_LABEL)
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Print informative and pretty information about the current training progress
            _output_stats(i, epoch, num_epochs, errD, errG, D_x, D_G_z1, D_G_z2, real, fake, dataloader_len, start_time)

            # Check how the generator is doing by saving G's output on fixed_noise
            if (epoch % config.OUTPUT_SPECTROGRAM_INTERVAL == 0) or ((epoch == num_epochs-1) and (i == dataloader_len-1)):
                with torch.no_grad():
                    fake = netG(config.FIXED_NOISE)
                    fake = fake.detach().cpu()

        epoch_elapsed_time = time.time() - epoch_start_time
        print(f"\nTime elapsed for epoch {epoch+1}: {utils.format_time(epoch_elapsed_time)}")

    print("\nTraining finished!")


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def _output_stats(i, epoch, num_epochs, errD, errG, D_x, D_G_z1, D_G_z2, real_cpu, fake, dataloader_len, start_time):
    if i > 0:
        # Output training stats
        elapsed_time = time.time() - start_time
        avg_time_per_batch = elapsed_time / (i + epoch * dataloader_len)
        remaining_time_epoch = avg_time_per_batch * (dataloader_len - i)
        remaining_time_total = avg_time_per_batch * \
            (dataloader_len * (num_epochs - epoch) - i)
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tElapsed time: %s\tTime left (epoch): %s\tTime left (total): %s'
                % (epoch, num_epochs, i, dataloader_len,
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,
                    utils.format_time(elapsed_time), utils.format_time(remaining_time_epoch), utils.format_time(remaining_time_total)))

    if i % config.OUTPUT_SPECTROGRAM_INTERVAL == 0:
        # Generate a spectrogram and display it
        utils.show_spectrogram(real_cpu.detach().cpu()[0, 0, :, :], "Real")
        utils.show_spectrogram(fake.detach().cpu()[0, 0, :, :], "Generated")
