import os
from pathlib import Path
import torch
import wandb
from torch import nn

from torch.utils.data import DataLoader, random_split, ConcatDataset, TensorDataset
from torchvision import datasets
from torchvision.transforms import transforms, v2

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)  # BASE_PATH: /Users/yhhan/git/link_dl
print(BASE_PATH)

import sys

sys.path.append(BASE_PATH)

from _01_code._99_common_utils.utils import get_num_cpu_cores, is_linux, is_windows


def get_fashion_mnist_data():
    data_path = os.path.join(BASE_PATH, "_00_data", "j_fashion_mnist")

    f_mnist_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=v2.ToTensor(), target_transform=lambda y: torch.tensor(y))
    f_mnist_train, f_mnist_validation = random_split(f_mnist_train, [55_000, 5_000])

    f_mnist_transforms = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10)
    ])

    transformed_train_data_images = []
    transformed_train_data_labels = []
    for image, label in f_mnist_train:
        transformed_image = f_mnist_transforms(image)
        transformed_train_data_images.append(transformed_image)
        transformed_train_data_labels.append(label)

    augmented_dataset = TensorDataset(
        torch.stack(transformed_train_data_images),
        torch.tensor(transformed_train_data_labels)
    )

    f_mnist_train = ConcatDataset([f_mnist_train, augmented_dataset])

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

    f_mnist_transforms = v2.Compose([
        v2.ConvertImageDtype(torch.float),
        v2.Normalize(mean=[0.2860], std=[0.3530])
    ])

    return train_data_loader, validation_data_loader, f_mnist_transforms


def get_fashion_mnist_test_data():
    data_path = os.path.join(BASE_PATH, "_00_data", "j_fashion_mnist")

    f_mnist_test_images = datasets.FashionMNIST(data_path, train=False, download=True)
    f_mnist_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=v2.ToTensor())

    print("Num Test Samples: ", len(f_mnist_test))
    print("Sample Shape: ", f_mnist_test[0][0].shape)  # torch.Size([1, 28, 28])

    test_data_loader = DataLoader(dataset=f_mnist_test, batch_size=len(f_mnist_test))

    f_mnist_transforms = v2.Compose(
        v2.ConvertImageDtype(torch.float),
        v2.Normalize(mean=[0.2860], std=[0.3530]),
    )

    return f_mnist_test_images, test_data_loader, f_mnist_transforms


if __name__ == "__main__":
    config = {'batch_size': 2048, }
    wandb.init(mode="disabled", config=config)

    train_data_loader, validation_data_loader, f_mnist_transforms = get_fashion_mnist_data()
    print()
    f_mnist_test_images, test_data_loader, f_mnist_transforms = get_fashion_mnist_test_data()
