import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super(ResBlock, self).__init__()

        # Define convolutional layers and batch normalization
        if downsample:
            # If downsample is True, apply convolution with stride 2 and create
            # a shortcut path
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # If downsample is False, apply convolution with stride 1 and no
            # shortcut path
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Identity()

        # Define additional convolutional layers and batch normalization
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input_tensor):
        # Apply the shortcut path to the input
        shortcut = self.shortcut(input_tensor)
        shortcut = shortcut.detach()

        # Apply the first convolutional layer and ReLU activation, followed by
        # batch normalization
        x = nn.ReLU()(self.bn1(self.conv1(input_tensor)))
        x = x.detach()

        # Apply the second convolutional layer and ReLU activation, followed by
        # batch normalization
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = x.detach()

        # Add the shortcut to the output of the second convolutional layer
        output_tensor = x + shortcut

        # Apply ReLU activation to the output and return it
        return nn.ReLU()(output_tensor)
    
    
class ResBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=2 if downsample else 1, padding=1)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1)
        self.shortcut = nn.Sequential()
        
        if self.downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2 if self.downsample else 1),
                nn.BatchNorm2d(out_channels)
            )

        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        shortcut = shortcut.detach()

        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = nn.ReLU()(self.bn3(self.conv3(input)))
        input = input.detach()

        input = input + shortcut
        return nn.ReLU()(input)