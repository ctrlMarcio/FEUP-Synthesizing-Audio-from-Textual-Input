# the U-Net class

import torch
from torch import nn


class UNet(nn.Module):
    """
    A U-Net model for 2D segmentation.
    This could be used for images, but also for other types of data, such as
        spectrograms.

    Most of this model is based on the original paper
    """

    def __init__(self, in_channels, out_channels, depth=5, conv_layers_per_block=2):
        """
        in_channels: int

        out_channels: int

        depth: int
            The number of layers in the U-Net.
            The number of layers in the U-Net is 2*depth + 1.
            For example, a depth of 5 would result in 11 layers.

        conv_layers_per_block: int
            The number of convolutional layers in each block of the U-Net.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.depth = depth
        self.conv_layers_per_block = conv_layers_per_block

        # initialize the device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self._contracting_path = self._make_contracting_path()
        self._bottleneck = self._make_bottleneck()
        self._expanding_path_layers = self._make_expanding_path()

    def forward(self, x):
        x = x.to(self.device)

        # implements the forward pass with concatenations
        skip_connections = []

        # contracting path
        for block in self._contracting_path:
            # create a sequential block from the list of layers
            net = nn.Sequential(*block).to(self.device)

            # apply
            x = net(x)

            # save the skip connection
            skip_connections.append(x)

        # bottleneck
        x = self._bottleneck(x)

        # expanding path
        for layer_idx in range(self.depth - 1):
            # the first layer is a transposed convolutional layer
            transposed_conv = self._expanding_path_layers[layer_idx * 2].to(
                self.device)
            x = transposed_conv(x)

            # concatenate the skip connection
            skip_connection = skip_connections.pop()
            # make sure the shapes match
            if x.shape != skip_connection.shape:
                # resize the skip connection
                skip_connection = nn.functional.interpolate(skip_connection,
                                                            size=x.shape[2:],
                                                            mode='nearest')

            # concatenate the skip connection
            x = torch.cat((x, skip_connection), dim=1)

            # the second layer is a sequential block of convolutional layers
            conv_block = self._expanding_path_layers[layer_idx * 2 + 1].to(
                self.device)
            x = conv_block(x)

        # final convolutional layer
        x = self._expanding_path_layers[-1].to(self.device)(x)

        return x

    def _make_contracting_path(self):
        """
        Create the contracting path of the U-Net.
        """
        layers = []

        # configs the used convolutional layers
        in_channels = self.in_channels
        out_channels = 64

        # create a convolutional block for each number in depth
        for _ in range(self.depth - 1):
            # create a convolutional block
            block = []

            # create the number of convolutional layers specified by
            # conv_layers_per_block
            for _ in range(self.conv_layers_per_block):
                block.append(nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels, kernel_size=3, padding="same"))
                # batch normalization
                block.append(nn.BatchNorm2d(out_channels))
                # ReLU activation
                block.append(nn.ReLU())

                # update the in_channels
                in_channels = out_channels

            # add a max pooling layer
            block.append(nn.MaxPool2d(kernel_size=2))

            # add the block to the layers
            layers.append(block)

            # double the number of channels
            out_channels *= 2

        # create a network from the layers
        return layers

    def _make_bottleneck(self):
        """
        Create the bottleneck of the U-Net.
        """
        # build the bottleneck
        layers = []

        # config the bottleneck channels
        in_channels = 64 * (2 ** (self.depth - 2))
        out_channels = in_channels * 2

        # create the number of convolutional layers specified by
        # conv_layers_per_block
        for _ in range(self.conv_layers_per_block):
            layers.append(nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=3,
                                    padding="same"))
            # batch normalization
            layers.append(nn.BatchNorm2d(out_channels))
            # ReLU activation
            layers.append(nn.ReLU())

            # update the in_channels
            in_channels = out_channels

        # add all the layers in block in the layers list
        return nn.Sequential(*layers).to(self.device)

    def _make_expanding_path(self):
        """
        Create the expanding path of the U-Net.
        This returns a list of layers, which will be used in the forward pass.
        Some layers are transposed convolutional layers, others are sequential
            blocks of convolutional layers.
        """
        layers = []

        # configs the used convolutional layers
        # the number of in channels is the number of out channels from the last
        # block in the contracting path
        in_channels = 64 * (2 ** (self.depth - 1))
        out_channels = in_channels // 2

        # create a convolutional block for each number in depth
        for _ in range(self.depth - 1):
            # add an up conv 2x2
            layers.append(nn.ConvTranspose2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=2,
                                             stride=2))

            block = []

            # create the number of convolutional layers specified by
            # conv_layers_per_block
            for _ in range(self.conv_layers_per_block):
                block.append(nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels, kernel_size=3, padding="same"))
                # batch normalization
                block.append(nn.BatchNorm2d(out_channels))
                # ReLU activation
                block.append(nn.ReLU())

                # update the in_channels
                in_channels = out_channels

            # add the block to the layers
            layers.append(nn.Sequential(*block))

            # double the number of channels
            out_channels //= 2

        # final convolutional layer
        final_layer = []
        final_layer.append(nn.Conv2d(in_channels=in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=1,
                                     padding="same"))

        final_layer.append(nn.Softmax(dim=1))

        layers.append(nn.Sequential(*final_layer))

        # create a network from the layers
        return layers


def test():
    import torch
    import torch.nn.functional as F

    # Define the input parameters
    in_channels = 3  # Number of input channels (e.g., for RGB images)
    # Number of output channels (e.g., for binary segmentation)
    out_channels = 2

    # Create a random input tensor with batch size 1 (batch size should be 1 for testing)
    # Assuming input image size of 256x256
    input_data = torch.randn((1, in_channels, 256, 256))

    # Create the UNet model
    model = UNet(in_channels, out_channels, depth=5, conv_layers_per_block=2)

    # Put the model in evaluation mode (important when using BatchNorm and Dropout layers)
    model.eval()

    # Forward pass through the model
    output = model(input_data)

    # Print the shape of the output (should match the number of output channels)
    print("Output shape:", output.shape)

    # If you want to check the probabilities for each class (e.g., for pixel-wise segmentation)
    probabilities = F.softmax(output, dim=1)

    # If you want to get the predicted class labels (e.g., for binary segmentation)
    # Here, we assume that the output channel with the highest probability is the predicted class.
    _, predicted_labels = torch.max(output, dim=1)

    # Print the shape of the probabilities and predicted labels
    print("Probabilities shape:", probabilities.shape)
    print("Predicted labels shape:", predicted_labels.shape)
