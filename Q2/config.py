import torch
import os
from torchvision import transforms

# Base paths
BASE_DIR = '/ssd_scratch/cvit/shaon/cv_5/Q2'
DATA_DIR = os.path.join("/ssd_scratch/cvit/shaon/cv_5", 'data')
SAVE_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create necessary directories
for dir_path in [DATA_DIR, SAVE_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Model and training configurations
CONFIG = {
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

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset mean and std for CIFAR10
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010) 