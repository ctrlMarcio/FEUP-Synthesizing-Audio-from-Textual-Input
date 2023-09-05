import config
from res_net.res_net import ResNet
from res_net.res_block import ResBlock
import torch
from torchsummary import summary
from data.tools import get_loaders
from res_net.trainer import fine_tune, fit
import os


def load_trained():
    """
    This function should load the trained model
    """
    # verify if the model exists in the predefined path
    # verifying if the file exists
    if not os.path.exists(config.RES_NET.MODEL_PATH):
        # launch a warning stating that an untrained model will be used
        print("WARNING: The model was not trained yet, an untrained model will be used")
        # return an untrained model
        return ResNet(in_channels=1, resblock=ResBlock, repeat=[2, 2, 2, 2])
    else:
        # load the model
        model = torch.load(config.RES_NET.MODEL_PATH)
        # return the model
        return model



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
    fine_tune(ResNet, train_loader, val_loader,
              hyperparameters_range, num_trials, epochs=5)

    # Uncomment the following line if you want to continue training with the original settings
    # fit(resnet18, train_loader, val_loader, save_checkpoint=True, load_checkpoint=config.LOAD_CHECKPOINT)


if __name__ == "__main__":
    main()
