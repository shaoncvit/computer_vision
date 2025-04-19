import os
import torch
from torchvision import transforms

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Base paths
BASE_DIR = '/ssd_scratch/cvit/shaon/cv_5/Q2'
DATA_DIR = os.path.join("/ssd_scratch/cvit/shaon/cv_5", 'data')
SAVE_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create necessary directories
for dir_path in [DATA_DIR, SAVE_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Dataset statistics
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

# Base configuration
BASE_CONFIG = {
    "learning_rate": 1e-3,
    "batch_size": 128,
    "epochs": 50,
    "image_size": 32,
    "patch_size": 4,
    "embed_dim": 256,
    "depth": 6,
    "num_heads": 8,
    "mlp_ratio": 4.0,
}

# Patch size experiment configurations
PATCH_SIZE_CONFIGS = {
    "patch_size_2": {**BASE_CONFIG, "patch_size": 2},
    "patch_size_4": {**BASE_CONFIG, "patch_size": 4},
    "patch_size_8": {**BASE_CONFIG, "patch_size": 8},
}

# Hyperparameter exploration configurations
HYPERPARAMETER_CONFIGS = {
    "config1": {
        **BASE_CONFIG,
        "embed_dim": 384,
        "depth": 8,
        "num_heads": 12,
        "mlp_ratio": 4.0,
    },
    "config2": {
        **BASE_CONFIG,
        "embed_dim": 512,
        "depth": 10,
        "num_heads": 16,
        "mlp_ratio": 4.0,
    },
    "config3": {
        **BASE_CONFIG,
        "embed_dim": 256,
        "depth": 12,
        "num_heads": 8,
        "mlp_ratio": 3.0,
    },
}

# Data augmentation configurations
from torchvision import transforms

# Base transforms
BASE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# Different augmentation strategies
AUGMENTATION_CONFIGS = {
    "baseline": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]),
    
    "strong_aug": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]),
    
    "cutout": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]),
    
    "autoaugment": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]),
}

# Positional embedding configurations
POSITIONAL_EMBEDDING_CONFIGS = {
    "no_pos_embed": {
        **BASE_CONFIG,
        "pos_embed_type": "none"
    },
    "1d_learned": {
        **BASE_CONFIG,
        "pos_embed_type": "1d_learned"
    },
    "2d_learned": {
        **BASE_CONFIG,
        "pos_embed_type": "2d_learned"
    },
    "sinusoidal": {
        **BASE_CONFIG,
        "pos_embed_type": "sinusoidal"
    },
} 