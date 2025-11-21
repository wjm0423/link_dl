import os
from pathlib import Path
import torch
import wandb

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)  # BASE_PATH: /Users/yhhan/git/link_dl
print(BASE_PATH)

import sys

sys.path.append(BASE_PATH)

from _01_code._99_common_utils.utils import get_num_cpu_cores, is_linux, is_windows


def get_fashion_mnist_data():
    data_path = os.path.join(BASE_PATH, "_00_data", "j_fashion_mnist")

    f_mnist_train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomCrop([28, 28], padding=4),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        v2.ToTensor(),
        v2.ConvertImageDtype(torch.float32),
        v2.Normalize(mean=[0.2860], std=[0.3530])
    ])

    f_mnist_validation_transforms = v2.Compose([
        v2.ToTensor(),
        v2.ConvertImageDtype(torch.float32),
        v2.Normalize(mean=[0.2860], std=[0.3530])
    ])

    f_mnist_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=f_mnist_train_transforms, target_transform=lambda y: torch.tensor(y))
    f_mnist_train, f_mnist_validation = random_split(f_mnist_train, [55_000, 5_000])

    f_mnist_validation.dataset.transform = f_mnist_validation_transforms

    print("Num Train Samples: ", len(f_mnist_train))
    print("Num Validation Samples: ", len(f_mnist_validation))
    print("Sample Data Shape: ", f_mnist_train[0][0].shape)  # torch.Size([1, 28, 28])
    print("Sample Data Target: ", f_mnist_train[0][1])  # 9

    num_data_loading_workers = get_num_cpu_cores()
    print("Number of Data Loading Workers:", num_data_loading_workers)

    train_data_loader = DataLoader(
        dataset=f_mnist_train, batch_size=wandb.config.batch_size, shuffle=True,
        pin_memory=True, num_workers=num_data_loading_workers
    )

    validation_data_loader = DataLoader(
        dataset=f_mnist_validation, batch_size=wandb.config.batch_size,
        pin_memory=True, num_workers=num_data_loading_workers
    )

    return train_data_loader, validation_data_loader, f_mnist_train_transforms


def get_fashion_mnist_test_data():
    data_path = os.path.join(BASE_PATH, "_00_data", "j_fashion_mnist")

    f_mnist_transforms = v2.Compose([
        v2.ToTensor(),
        v2.ConvertImageDtype(torch.float32),
        v2.Normalize(mean=[0.2860], std=[0.3530]),
    ])

    f_mnist_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=f_mnist_test_transforms)

    print("Num Test Samples: ", len(f_mnist_test))
    print("Sample Shape: ", f_mnist_test[0][0].shape)  # torch.Size([1, 28, 28])

    test_data_loader = DataLoader(dataset=f_mnist_test, batch_size=len(f_mnist_test), shuffle=False)

    return f_mnist_test, test_data_loader, f_mnist_transforms


if __name__ == "__main__":
    config = {'batch_size': 2048, }
    wandb.init(mode="disabled", config=config)

    train_data_loader, validation_data_loader, f_mnist_transforms = get_fashion_mnist_data()
    print()
    f_mnist_test_images, test_data_loader, f_mnist_transforms = get_fashion_mnist_test_data()