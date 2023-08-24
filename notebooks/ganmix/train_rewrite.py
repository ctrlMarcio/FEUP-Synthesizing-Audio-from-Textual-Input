from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import torch
import time
import csv

import models
import init
import config
import dataset
import utils

class Settings():
    def __init__(self, num_workers=1):
        init.boot() # folders

        # get current time
        self.start_time = time.time()
        self.dataloader = dataset.get_dataloader(num_workers=num_workers)

        # To handle gradient scaling (mixed precision training)
        self.scaler = GradScaler()

        # For file writing
        self.stats_file_path = config.STATS_DIR + f"/training_stats_{self.start_time}.csv"

def _init():
    # call the ganmix factory
    ganmix = models.GANMix.build()
    return ganmix

def _gen_fake_samples(generator, num_samples):
    # Generate a batch of fake samples from the generator using random noise as input.
    # This function is used to generate samples for the discriminator training.
    # The generated samples are returned as a tensor.
    
    noise = torch.randn(num_samples, config.GENERATOR_INPUT_SIZE, 1, 1, device=config.DEVICE)
    fake_samples = generator(noise)
    return fake_samples

def _embed_samples(vae, samples):
    embeddings = vae.encode(samples)
    embeddings = embeddings.latent_dist.mode()
    embeddings = torch.nan_to_num(embeddings, nan=0)
    return embeddings

def _train_discriminator(data, ganmix, settings):
    # Combine the generated fake samples with a batch of real samples from my dataset.
    # Label the real samples as '1' and the fake samples as '0'.
    # Pass the combined batch through the discriminator and compute the discriminator loss.
    # Backpropagate the gradients and update the discriminator's weights using the discriminator's optimizer.

    with autocast():
        # Get batch of real samples
        real_data = data.to(config.DEVICE)
        batch_size = real_data.size(0)

        # pass them through the vae encoder
        real_embeddings = _embed_samples(ganmix.vae, real_data)

        # Generate fake samples
        #with torch.no_grad():
        fake_embeddings = _gen_fake_samples(ganmix.generator, batch_size)

        # Zero out the discriminator gradients
        ganmix.discriminator_optimizer.zero_grad()

        # Compute discriminator's predictions for real and fake samples
        prediction_real = ganmix.discriminator(real_embeddings)
        # detach() is used to prevent gradients from flowing into the generator
        # this means that the generator will not be trained in this step
        prediction_fake = ganmix.discriminator(fake_embeddings.detach())

        # Define labels for real and fake samples (real: 1, fake: 0)
        real_label = torch.ones(batch_size, 1, device=config.DEVICE)
        fake_label = torch.zeros(batch_size, 1, device=config.DEVICE)

        # Compute discriminator loss for real and fake samples
        loss_real = ganmix.criterion(prediction_real, real_label)
        loss_fake = ganmix.criterion(prediction_fake, fake_label)

        # Total discriminator loss is the sum of real and fake losses
        loss_discriminator = loss_real + loss_fake

    # Backpropagate and update discriminator's weights
    settings.scaler.scale(loss_discriminator).backward()
    settings.scaler.step(ganmix.discriminator_optimizer)
    settings.scaler.update()

    return loss_discriminator

def _train_generator(ganmix, settings):
    # Generate another batch of fake samples from the generator using new random noise.
    # Label the new batch of fake samples as '1' (to encourage the generator to produce more convincing fakes).
    # Pass the batch through the discriminator and compute the generator loss.
    # Backpropagate the gradients and update the generator's weights using the generator's optimizer.

    with autocast():

        # Generate new fake samples from the generator
        #with torch.no_grad():
        fake_data = _gen_fake_samples(ganmix.generator, config.BATCH_SIZE)

        # Zero out gradients in the generator before computing new gradients
        ganmix.generator_optimizer.zero_grad()

        # Compute discriminator's predictions for the new fake samples
        prediction_fake = ganmix.discriminator(fake_data)

        # Define labels for the fake samples as real (1), aiming to fool the discriminator
        fake_label = torch.ones(config.BATCH_SIZE, 1, device=config.DEVICE)

        # Compute generator loss based on how well the fake samples fooled the discriminator
        loss_generator = ganmix.criterion(prediction_fake, fake_label)

        # Compute elastic net loss
        # loss_generator += ganmix.generator.elastic_net_regularization()

    # Backpropagate and update generator's weights
    settings.scaler.scale(loss_generator).backward()
    settings.scaler.step(ganmix.generator_optimizer)
    settings.scaler.update()

    return loss_generator

def _output_epoch_results(start_time, epoch, generator, vae, loss_discriminator_list, loss_generator_list, csv_writer, csvfile):
    # store info in the file and display it
    avg_loss_discriminator = sum(loss_discriminator_list) / len(loss_discriminator_list)
    avg_loss_generator = sum(loss_generator_list) / len(loss_generator_list)
    csv_writer.writerow([
        epoch, avg_loss_discriminator, avg_loss_generator
    ])
    csvfile.flush()
    print(f"Epoch: {epoch} | Loss D: {avg_loss_discriminator} | Loss G: {avg_loss_generator}")

    # save a spectrogram of the generated audio
    with torch.no_grad():
        fake_encodings = generator(config.FIXED_NOISE)
        # pass the fake spectrogram through the VAE decoder
        fake_spectrogram = vae.decode(fake_encodings)

    utils.save_histogram(fake_encodings.flatten().cpu().detach().numpy(), f"{start_time}_{epoch}_fake_encodings")
    utils.save_spectrogram(fake_spectrogram.sample.cpu()[0, 0, :, :], f"{start_time}_{epoch}_fake_spectrogram")


def run_train():
    # initializers
    ganmix = _init() # models
    settings = Settings(ganmix.num_workers)

    # training loop
    with open(settings.stats_file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            'Epoch', 'Loss_D', 'Loss_G'
        ])
        csvfile.flush()

        for epoch in range(config.NUM_EPOCHS):
            # Store the losses in lists to calculate the full
            loss_discriminator_list = []
            loss_generator_list = []

            for data, quote in tqdm(settings.dataloader):
                loss_discriminator = _train_discriminator(data, ganmix, settings)
                loss_discriminator_list.append(loss_discriminator.item())

                loss_generator = _train_generator(ganmix, settings)
                loss_generator_list.append(loss_generator.item())

            _output_epoch_results(settings.start_time, epoch, ganmix.generator, ganmix.vae, loss_discriminator_list, loss_generator_list, csv_writer, csvfile)

def main():
    run_train()
