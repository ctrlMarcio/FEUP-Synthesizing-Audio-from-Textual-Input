import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


from vamgen.diffusion import train


def fit():
    # TODO all this function is about hardcoded values

    # Load your dataset and perform data transformations (if needed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add more transformations as needed (e.g., normalization, resizing, etc.)
    ])

    dataset = CIFAR10(root="./data", train=True,
                      download=True, transform=transform)

    # Create DataLoader to efficiently load and batch the data
    dataloader = DataLoader(dataset, batch_size=32,
                            shuffle=True, num_workers=4)

    # call the train_diffusion_model function
    train.train_diffusion_model(dataloader)


def generate():
    pass
