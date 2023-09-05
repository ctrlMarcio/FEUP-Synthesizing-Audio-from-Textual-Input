import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

from vamgen.diffusion.model import DiffusionModel

# Define the diffusion model function for training


def train_diffusion_model(dataloader, num_epochs=50, learning_rate=0.001, max_timestep=100):
    # Check if GPU is available and use it for computation if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the U-Net and diffusion model
    diffusion_model = DiffusionModel(diffusion_steps=100, depth=5).to(device)
    unet = diffusion_model.unet.to(device)

    # Define the loss function (you may use a suitable loss for denoising)
    criterion = nn.MSELoss()

    # Define the optimizer (e.g., Adam)
    optimizer = optim.Adam(unet.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        unet.train()
        running_loss = 0.0

        # Iterate over the data batches
        for inputs, _ in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs = inputs.to(device)

            # pick a random noise level
            timestep = torch.randint(0, max_timestep, (1,)).item()

            # find the noise, according to the timestep
            noise = diffusion_model.get_noise(inputs, timestep)

            # add the noise to the inputs
            noisy_inputs = inputs + noise

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            noise_prediction = unet(noisy_inputs)
            loss = criterion(noise_prediction, noise)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    print("Training complete!")


if __name__ == "__main__":
    train_diffusion_model()
