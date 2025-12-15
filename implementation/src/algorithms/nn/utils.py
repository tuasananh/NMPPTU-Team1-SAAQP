import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import random
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_resnet18(num_classes=10):
    """
    Create ResNet18 model adapted for CIFAR-10.
    The original ResNet18 is designed for ImageNet (224x224 images).
    We modify the first conv layer for 32x32 CIFAR images.
    """
    model = resnet18(weights=None, num_classes=num_classes)
    
    # Modify first conv layer for CIFAR-10 (32x32 images)
    # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
    # Modified: Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # Remove the max pooling layer (not needed for small images)
    model.maxpool = nn.Identity()
    
    return model

def load_cifar10(batch_size=128, num_workers=1, train_pct=1.0, test_pct=1.0):
    """
    Load CIFAR-10 dataset with standard augmentation for ResNet training.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        train_pct: Percentage of training data to use (0.0 to 1.0)
        test_pct: Percentage of test data to use (0.0 to 1.0)
    """
    # Standard augmentation for CIFAR-10 training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # No augmentation for test set
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Subsample datasets if percentage < 1.0
    if train_pct < 1.0:
        train_size = int(len(train_dataset) * train_pct)
        train_dataset, _ = torch.utils.data.random_split(
            train_dataset, 
            [train_size, len(train_dataset) - train_size],
        )
    
    if test_pct < 1.0:
        test_size = int(len(test_dataset) * test_pct)
        test_dataset, _ = torch.utils.data.random_split(
            test_dataset,
            [test_size, len(test_dataset) - test_size],
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader
