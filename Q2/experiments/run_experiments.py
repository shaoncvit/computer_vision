import sys
import os
import torch
import wandb
from tqdm import tqdm, trange
import argparse
import json
from datetime import datetime

# Add parent directory to path to import from parent folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from train import train_model  # We'll modify train.py to make it modular

def run_patch_size_experiments():
    """Run experiments with different patch sizes"""
    results = {}
    for name, config in PATCH_SIZE_CONFIGS.items():
        print(f"\nRunning experiment: {name}")
        config = {**config, "experiment_name": f"patch_size/{name}"}
        wandb.init(
            project="differential-vit-cifar-10",
            group="patch_size_experiments",
            name=name,
            config=config,
            dir=BASE_DIR,
            reinit=True
        )
        
        test_acc = train_model(config)
        results[name] = test_acc
        wandb.finish()
    
    # Save results
    with open(os.path.join(RESULTS_DIR, 'patch_size_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

def run_hyperparameter_experiments():
    """Run experiments with different hyperparameter configurations"""
    results = {}
    for name, config in HYPERPARAMETER_CONFIGS.items():
        print(f"\nRunning experiment: {name}")
        config = {**config, "experiment_name": f"hyperparams/{name}"}
        wandb.init(
            project="differential-vit-cifar-10",
            group="hyperparameter_experiments",
            name=name,
            config=config,
            dir=BASE_DIR,
            reinit=True
        )
        
        test_acc = train_model(config)
        results[name] = test_acc
        wandb.finish()
    
    # Save results
    with open(os.path.join(RESULTS_DIR, 'hyperparameter_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

def run_augmentation_experiments():
    """Run experiments with different data augmentation techniques"""
    results = {}
    for name, transform in AUGMENTATION_CONFIGS.items():
        print(f"\nRunning experiment: {name}")
        config = {**BASE_CONFIG, "augmentation": name, "experiment_name": f"augmentation/{name}"}
        wandb.init(
            project="differential-vit-cifar-10",
            group="augmentation_experiments",
            name=name,
            config=config,
            dir=BASE_DIR,
            reinit=True
        )
        
        test_acc = train_model(config, train_transform=transform)
        results[name] = test_acc
        wandb.finish()
    
    # Save results
    with open(os.path.join(RESULTS_DIR, 'augmentation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

def run_positional_embedding_experiments():
    """Run experiments with different positional embedding types"""
    results = {}
    for name, config in POSITIONAL_EMBEDDING_CONFIGS.items():
        print(f"\nRunning experiment: {name}")
        config = {**config, "experiment_name": f"pos_embed/{name}"}
        wandb.init(
            project="differential-vit-cifar-10",
            group="positional_embedding_experiments",
            name=name,
            config=config,
            dir=BASE_DIR,
            reinit=True
        )
        
        test_acc = train_model(config)
        results[name] = test_acc
        wandb.finish()
    
    # Save results
    with open(os.path.join(RESULTS_DIR, 'positional_embedding_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ViT experiments')
    parser.add_argument('--experiment', type=str, required=True,
                      choices=['patch_size', 'hyperparams', 'augmentation', 'pos_embed'],
                      help='Which experiment to run')
    args = parser.parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.experiment == 'patch_size':
        run_patch_size_experiments()
    elif args.experiment == 'hyperparams':
        run_hyperparameter_experiments()
    elif args.experiment == 'augmentation':
        run_augmentation_experiments()
    elif args.experiment == 'pos_embed':
        run_positional_embedding_experiments() 