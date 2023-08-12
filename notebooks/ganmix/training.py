import torch
from torch import nn
import time
import config
import utils
import os
import csv
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import autocast, GradScaler


def fit(netG, netD, vae, dataloader, criterion, optimizerG, optimizerD, num_epochs=config.NUM_EPOCHS, load_checkpoint=True):
    # this variable will hold if the program should start the variables from
    # scratch or load the latest checkpoint
    initialize_variables = not load_checkpoint

    # check if the checkpoint dir exists
    if os.path.exists(config.CHECKPOINT["DIR"]):
        print("Checkpoint directory exists.")
        if load_checkpoint:
            print("Loading checkpoint...")
            latest_checkpoint = max(
                [int(file.split('_')[2].split('.')[0]) for file in os.listdir(config.CHECKPOINT["DIR"]) if file.startswith(config.CHECKPOINT["BASE_NAME"])], default=None)
            if latest_checkpoint is not None:
                checkpoint_filename = _get_checkpoint_filename(
                    latest_checkpoint)
                print(f"Latest checkpoint found: {checkpoint_filename}")
                checkpoint = torch.load(checkpoint_filename)
                netG.load_state_dict(checkpoint['netG_state_dict'])
                netD.load_state_dict(checkpoint['netD_state_dict'])
                optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
                optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
                epoch = checkpoint['epoch']
                G_losses = checkpoint['G_losses']
                D_losses = checkpoint['D_losses']
                start_time = checkpoint['start_time']
                print(
                    f"Loaded checkpoint from epoch {epoch} ({checkpoint_filename})")
            else:
                print("No checkpoint found.")
                initialize_variables = True
        else:
            print("Deleting existing checkpoints...")
            # delete the checkpoints
            for file in os.listdir(config.CHECKPOINT["DIR"]):
                os.remove(os.path.join(config.CHECKPOINT["DIR"], file))
    else:
        print("Creating checkpoint directory...")
        # create the checkpoint dir
        os.makedirs(config.CHECKPOINT["DIR"])
        initialize_variables = True

    if initialize_variables:
        # initialize the variables
        print('Initializing the variables...')

        # epochs
        epoch = -1

        # losses for displaying purposes
        G_losses = []
        D_losses = []

        # initialize the networks
        netG.apply(_weights_init)
        netD.apply(_weights_init)

        # start time
        start_time = time.time()

    # save the dataloader len for displaying purposes
    dataloader_len = len(dataloader)

    # initialize the file that will log the training stats
    stats_file_path = config.STATS_DIR + f"/training_stats_{start_time}.csv"

    # to handle gradient scaling (mixed precision training)
    scaler = GradScaler()

    # Training loop
    print("Starting Training Loop...")

    # Create CSV file and write header
    with open(stats_file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'Epoch', 'Loss_D', 'Loss_G'
        ])

        # For each epoch
        for epoch in range(epoch+1, num_epochs):
            print("\n" + "="*30)
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            epoch_start_time = time.time()

            epoch_D_losses = []
            epoch_G_losses = []

            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):
                with autocast():
                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    # Train with all-real batch
                    netD.zero_grad()
                    # Format batch
                    real = data[0].to(config.DEVICE)
                    b_size = real.size(0)
                    label = torch.full((b_size,), config.REAL_LABEL,
                                    dtype=torch.float, device=config.DEVICE)
                    
                    with torch.no_grad():
                        # Forward pass real batch through D
                        embeddings = vae.encode(real)
                        embeddings = embeddings.latent_dist.mode()
                        embeddings = torch.nan_to_num(embeddings, nan=0)
                        output = netD(embeddings)
                        output = output.view(-1)

                    # Calculate loss on all-real batch
                    errD_real = criterion(output, label.detach())
                    errD_real.requires_grad = True
                    # Calculate gradients for D in backward pass
                    D_x = output.mean().item()

                    # Train with all-fake batch
                    # Generate batch of latent vectors
                    noise = torch.randn(
                        b_size, config.GENERATOR_INPUT_SIZE, 1, 1, device=config.DEVICE)
                    # Generate fake latent features batch with G
                    fake = netG(noise)
                    label.fill_(config.FAKE_LABEL)
                    label = label.float()
                    # Classify all fake batch with D
                    output = netD(fake.detach()).view(-1)
                    # Calculate D's loss on the all-fake batch
                    errD_fake = criterion(output, label)
                    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                    D_G_z1 = output.mean().item()
                    errD = errD_real + errD_fake

                scaler.scale(errD).backward()
                scaler.step(optimizerD)
                #scaler.update()

                with autocast():
                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    netG.zero_grad()
                    # fake labels are real for generator cost
                    label.fill_(config.REAL_LABEL)

                    
                    #with torch.no_grad():
                    # Since we just updated D, perform another forward pass of all-fake batch through D
                    output = netD(fake).view(-1)

                    # Calculate G's loss based on this output
                    errG = criterion(output, label)
                    # Calculate Elastic Net regularization
                    en_loss_G = netG.elastic_net_regularization()

                    # Total G loss includes the elastic net
                    errG = errG + en_loss_G

                # Calculate gradients for G
                scaler.scale(errG).backward()
                scaler.step(optimizerG)
                #scaler.update()

                with autocast():
                    D_G_z2 = output.mean().item()

                    # Save Losses for plotting later
                    G_losses.append(errG.item())
                    D_losses.append(errD.item())
                    epoch_D_losses.append(errD.item())
                    epoch_G_losses.append(errG.item())

                    # Print informative and pretty information about the current training progress
                    _output_stats(csv_writer, i, epoch, num_epochs, errD, errG, D_x,
                                D_G_z1, D_G_z2, real, fake, dataloader_len, start_time)

                    # Check how the generator is doing by saving G's output on fixed_noise
                    if (epoch % config.OUTPUT_SPECTROGRAM_INTERVAL == 0) or ((epoch == num_epochs-1) and (i == dataloader_len-1)):
                        with torch.no_grad():
                            fake = netG(config.FIXED_NOISE)
                            fake = fake.detach().cpu()

                # Update scaler for next iteration
                scaler.update()

            epoch_elapsed_time = time.time() - epoch_start_time
            print(
                f"\nTime elapsed for epoch {epoch+1}: {utils.format_time(epoch_elapsed_time)}")

            # save a spectrogram of the generated audio
            if (epoch % config.OUTPUT_SPECTROGRAM_INTERVAL == 0) or ((epoch == num_epochs-1) and (i == dataloader_len-1)):
                # save the real and the fake to compare them
                # choose a random spectrogram
                utils.save_spectrogram(real.detach()[0][0].cpu(), f"{start_time}_{epoch}_real")
                utils.save_spectrogram(fake.detach()[0][0].cpu(), f"{start_time}_{epoch}_fake")            

            # Save the checkpoint at the specified epoch interval
            if epoch % config.CHECKPOINT["EPOCH_INTERVAL"] == 0:
                checkpoint_filename = _get_checkpoint_filename(epoch)
                torch.save({
                    'epoch': epoch,
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'G_losses': G_losses,
                    'D_losses': D_losses,
                    'start_time': start_time
                }, checkpoint_filename)
                # flush the csv file
                csvfile.flush()
                print(
                    f"Checkpoint saved at epoch {epoch} to {checkpoint_filename}")

            # Save the info on the csv file at the end of each epoch
            avg_loss_D = sum(epoch_D_losses) / len(epoch_D_losses)
            avg_loss_G = sum(epoch_G_losses) / len(epoch_G_losses)
            # 'Epoch', 'Loss_D', 'Loss_G'
            csv_writer.writerow([
                epoch + 1, avg_loss_D, avg_loss_G
            ])


    print("\nTraining finished!")


def _get_checkpoint_filename(epoch):
    return os.path.join(config.CHECKPOINT["DIR"], f"{config.CHECKPOINT['BASE_NAME']}_{epoch}.pth")


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def _output_stats(csv_writer, i, epoch, num_epochs, errD, errG, D_x, D_G_z1, D_G_z2, real_cpu, fake, dataloader_len, start_time):
    if i > 0:
        # Output training stats
        elapsed_time = time.time() - start_time
        avg_time_per_batch = elapsed_time / (i + epoch * dataloader_len)
        remaining_time_epoch = avg_time_per_batch * (dataloader_len - i)
        remaining_time_total = avg_time_per_batch * \
            (dataloader_len * (num_epochs - epoch) - i)
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tElapsed time: %s\tTime left (epoch): %s\tTime left (total): %s'
              % (epoch + 1, num_epochs, i, dataloader_len,
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2,
                 utils.format_time(elapsed_time), utils.format_time(remaining_time_epoch), utils.format_time(remaining_time_total)))
