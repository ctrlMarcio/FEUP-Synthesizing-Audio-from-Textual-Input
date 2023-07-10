from config import LOAD_CHECKPOINT
from res_net.res_net import ResNet
from res_net.res_block import ResBlock
import torch
from torchsummary import summary
from data.tools import get_loaders
from res_net.trainer import fine_tune, fit


def main():
    # Define the hyperparameters range for fine-tuning
    hyperparameters_range = {
        "kernel_size": [7, 9, 11],
        "stride": [1, 2, 3],
        "padding": [1, 2, 3],
        "pool_kernel_size": [2, 3, 4],
        "pool_stride": [1, 2],
        "pool_padding": [0, 1],
    }

    # Set the number of trials for fine-tuning
    num_trials = 5

    # Split the data into train and validation loaders
    train_loader, val_loader = get_loaders(train_percentage=0.8)

    # Perform fine-tuning with random search
    # it receives the class in the first argument
    fine_tune(ResNet, train_loader, val_loader, hyperparameters_range, num_trials, epochs=5)

    # Uncomment the following line if you want to continue training with the original settings
    # fit(resnet18, train_loader, val_loader, save_checkpoint=True, load_checkpoint=LOAD_CHECKPOINT)

if __name__ == "__main__":
    main()