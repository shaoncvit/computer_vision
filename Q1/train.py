import torch
import torch.nn as nn
from models.vit import VisionTransformer
from utils.training import get_cifar10_dataloaders, train_model, plot_training_curves
from utils.visualization import (
    visualize_attention, visualize_attention_heads,
    compute_attention_rollout, visualize_attention_rollout,
    visualize_positional_embedding
)
import os
import json
import wandb
import matplotlib.pyplot as plt

# Define directories
BASE_DIR = '/ssd_scratch/cvit/shaon/cv_5/Q1'
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')

def run_patch_size_experiment():
    """
    Experiment with different patch sizes
    """
    patch_sizes = [2, 4, 8]
    results = {}
    
    wandb.init(project='vision-transformer-cifar10', name='patch_size_experiment')
    
    for patch_size in patch_sizes:
        print(f"\nRunning experiment with patch_size={patch_size}")
        
        with wandb.init(project='vision-transformer-cifar10', 
                       name=f'patch_size_{patch_size}',
                       group='patch_size_experiments'):
            # Create model
            model = VisionTransformer(
                image_size=32,
                patch_size=patch_size,
                embed_dim=256,
                num_layers=6,
                num_heads=8,
                mlp_ratio=4,
                dropout=0.1
            ).to('cuda')
            
            # Get dataloaders
            trainloader, valloader, testloader = get_cifar10_dataloaders(batch_size=128)
            
            # Train model
            history = train_model(
                model=model,
                trainloader=trainloader,
                valloader=valloader,
                testloader=testloader,
                num_epochs=50,
                lr=1e-3,
                experiment_name=f'patch_size_{patch_size}'
            )
            
            # Save results
            results[f'patch_size_{patch_size}'] = {
                'best_val_acc': history['best_val_acc'],
                'test_acc': history['test_acc'],
                'history': history
            }
    
    return results

def run_hyperparameter_experiment():
    """
    Experiment with different hyperparameters
    """
    configs = [
        {'embed_dim': 256, 'num_layers': 6, 'num_heads': 8, 'mlp_ratio': 4},
        {'embed_dim': 384, 'num_layers': 8, 'num_heads': 12, 'mlp_ratio': 4},
        {'embed_dim': 512, 'num_layers': 12, 'num_heads': 16, 'mlp_ratio': 4},
    ]
    results = {}
    
    for i, config in enumerate(configs):
        print(f"\nRunning hyperparameter experiment {i+1}")
        print(f"Config: {config}")
        
        with wandb.init(project='vision-transformer-cifar10', 
                       name=f'hyperparameter_config_{i+1}',
                       group='hyperparameter_experiments',
                       config=config):
            # Create model
            model = VisionTransformer(
                image_size=32,
                patch_size=4,
                embed_dim=config['embed_dim'],
                num_layers=config['num_layers'],
                num_heads=config['num_heads'],
                mlp_ratio=config['mlp_ratio'],
                dropout=0.1
            ).to('cuda')
            
            # Get dataloaders
            trainloader, valloader, testloader = get_cifar10_dataloaders(batch_size=128)
            
            # Train model
            history = train_model(
                model=model,
                trainloader=trainloader,
                valloader=valloader,
                testloader=testloader,
                num_epochs=50,
                lr=1e-3,
                experiment_name=f'hyperparameter_config_{i+1}'
            )
            
            # Save results
            results[f'config_{i+1}'] = {
                'config': config,
                'best_val_acc': history['best_val_acc'],
                'test_acc': history['test_acc'],
                'history': history
            }
    
    return results

def run_augmentation_experiment():
    """
    Experiment with data augmentation
    """
    with wandb.init(project='vision-transformer-cifar10', 
                   name='augmentation_experiment',
                   group='augmentation_experiments'):
        # Use best hyperparameters from previous experiment
        model = VisionTransformer(
            image_size=32,
            patch_size=4,
            embed_dim=256,
            num_layers=6,
            num_heads=8,
            mlp_ratio=4,
            dropout=0.1
        ).to('cuda')
        
        # Get dataloaders with augmentation
        trainloader, valloader, testloader = get_cifar10_dataloaders(batch_size=128, augment=True)
        
        # Train model
        history = train_model(
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            testloader=testloader,
            num_epochs=50,
            lr=1e-3,
            experiment_name='augmentation'
        )
        
        return {'augmentation': {
            'best_val_acc': history['best_val_acc'],
            'test_acc': history['test_acc'],
            'history': history
        }}

def run_positional_embedding_experiment():
    """
    Experiment with different positional embeddings
    """
    pos_embed_types = ['1d', '2d', 'sinusoidal', None]
    # pos_embed_types = ['2d', 'sinusoidal', None]
    results = {}
    
    for pos_type in pos_embed_types:
        print(f"\nRunning experiment with pos_embed_type={pos_type}")
        pos_type_str = pos_type if pos_type else 'none'
        
        with wandb.init(project='vision-transformer-cifar10', 
                       name=f'pos_embed_{pos_type_str}',
                       group='positional_embedding_experiments'):
            # Create model
            model = VisionTransformer(
                image_size=32,
                patch_size=4,
                embed_dim=256,
                num_layers=6,
                num_heads=8,
                mlp_ratio=4,
                dropout=0.1,
                pos_embed_type=pos_type
            ).to('cuda')
            
            # Visualize positional embeddings before training
            if hasattr(model, 'pos_embed') and pos_type is not None:
                try:
                    # Get positional embedding weights based on type
                    if pos_type == '1d':
                        pos_weights = model.pos_embed.pos_embedding
                    elif pos_type == '2d':
                        # Combine CLS and patch embeddings for visualization
                        cls_embed = model.pos_embed.cls_pos_embed
                        patch_embed = model.pos_embed.patch_pos_embed
                        pos_weights = torch.cat([cls_embed, patch_embed], dim=1)
                    elif pos_type == 'sinusoidal':
                        pos_weights = model.pos_embed.pe
                    
                    save_path = os.path.join(PLOTS_DIR, f'pos_embed_{pos_type_str}_init.png')
                    os.makedirs(PLOTS_DIR, exist_ok=True)
                    
                    visualize_positional_embedding(
                        pos_weights.detach().cpu(),
                        patch_size=4,
                        image_size=32,
                        save_path=save_path
                    )
                    # Log to wandb
                    wandb.log({f'pos_embed_{pos_type_str}_init': wandb.Image(save_path)})
                except Exception as e:
                    print(f"Warning: Could not visualize initial positional embeddings for {pos_type}: {e}")
            
            # Get dataloaders
            trainloader, valloader, testloader = get_cifar10_dataloaders(batch_size=128, augment=True)
            
            # Train model
            history = train_model(
                model=model,
                trainloader=trainloader,
                valloader=valloader,
                testloader=testloader,
                num_epochs=50,
                lr=1e-3,
                experiment_name=f'pos_embed_{pos_type_str}'
            )
            
            # Visualize positional embeddings after training
            if hasattr(model, 'pos_embed') and pos_type is not None:
                try:
                    # Get positional embedding weights based on type
                    if pos_type == '1d':
                        pos_weights = model.pos_embed.pos_embedding
                    elif pos_type == '2d':
                        # Combine CLS and patch embeddings for visualization
                        cls_embed = model.pos_embed.cls_pos_embed
                        patch_embed = model.pos_embed.patch_pos_embed
                        pos_weights = torch.cat([cls_embed, patch_embed], dim=1)
                    elif pos_type == 'sinusoidal':
                        pos_weights = model.pos_embed.pe
                    
                    save_path = os.path.join(PLOTS_DIR, f'pos_embed_{pos_type_str}_final.png')
                    os.makedirs(PLOTS_DIR, exist_ok=True)
                    
                    visualize_positional_embedding(
                        pos_weights.detach().cpu(),
                        patch_size=4,
                        image_size=32,
                        save_path=save_path
                    )
                    # Log to wandb
                    wandb.log({f'pos_embed_{pos_type_str}_final': wandb.Image(save_path)})
                except Exception as e:
                    print(f"Warning: Could not visualize final positional embeddings for {pos_type}: {e}")
            
            # Save results
            results[f'pos_embed_{pos_type_str}'] = {
                'best_val_acc': history['best_val_acc'],
                'test_acc': history['test_acc'],
                'history': history
            }
    
    return results

def main():
    # Create output directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Run experiments
    results = {}
    
    print("Running patch size experiments...")
    results['patch_size'] = run_patch_size_experiment()
    
    print("\nRunning hyperparameter experiments...")
    results['hyperparameter'] = run_hyperparameter_experiment()
    
    print("\nRunning augmentation experiments...")
    results['augmentation'] = run_augmentation_experiment()
    
    print("\nRunning positional embedding experiments...")
    results['positional_embedding'] = run_positional_embedding_experiment()
    
    # Save results locally
    results_file = os.path.join(RESULTS_DIR, 'experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main() 