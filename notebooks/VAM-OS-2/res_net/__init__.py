from res_net.res_net import ResNet
from res_net.res_block import ResBlock
import torch
from torchsummary import summary
from data.tools import get_loaders
from res_net.trainer import fit


def main():
    print("slkjasldj")
    # this function is the one custom made to train the resnet
    resnet18 = ResNet(1, ResBlock, [2, 2, 2, 2],
                      useBottleneck=False, outputs=10)
    resnet18.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    summary(resnet18, (1, 257, 128))

    train_loader, val_loader = get_loaders(train_percentage=0.8)

    # call the fit function
    fit(resnet18, train_loader, val_loader)

if __name__ == "__main__":
    main()