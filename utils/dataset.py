import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import matplotlib.pyplot as plt

def get_data_loaders(batch_size=32, train_dir='data/train', val_dir='data/test'):
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # Check class names
    print("Class names:", train_dataset.classes)

    # # Compute sample weights for class imbalance
    # targets = torch.tensor(train_dataset.targets)
    # class_counts = torch.bincount(targets)
    # weights = 1.0 / class_counts.float()
    # sample_weights = weights[targets]
    
    # sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    
    return train_loader, val_loader

train_loader, val_loader = get_data_loaders()