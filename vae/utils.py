import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt


def preprocess_data(batch_size):
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    kwargs = {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}

    train_dataset = MNIST(
        root="./data", train=True, download=True, transform=mnist_transform
    )
    test_dataset = MNIST(
        root="./data", train=False, download=True, transform=mnist_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )

    return train_loader, test_loader


def show_image(x, x_hat):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.array(x[0][0]), cmap="gray")
    ax1.set_title("Original Image")
    ax1.axis("off")
    ax2.imshow(x_hat[0][0], cmap="gray")
    ax2.set_title("Reconstructed Image")
    ax2.axis("off")
    plt.show()
