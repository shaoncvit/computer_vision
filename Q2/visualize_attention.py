import torch
import torchvision
import torchvision.transforms as transforms
from ViTDiff import VisionTransformerDiff
from utils.visualization import (
    visualize_attention,
    compute_attention_rollout,
    visualize_attention_rollout,
    visualize_positional_embedding
)
import matplotlib.pyplot as plt
import wandb
import os
import torch.nn as nn
from config import *

def load_model(model_path, config):
    """
    Load trained ViT-Diff model
    """
    model = VisionTransformerDiff(
        image_size=config['image_size'],
        patch_size=config['patch_size'],
        in_channels=3,
        num_classes=10,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        pos_embed_type=config.get('pos_embed_type', '1d_learned')
    ).to('cuda')
    
    checkpoint = torch.load(model_path)
    
    # Handle state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if it exists
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def get_test_images(num_images=5):
    """
    Get test images from CIFAR-10
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform)
    
    images = []
    labels = []
    for i in range(num_images):
        img, label = testset[i]
        images.append(img)
        labels.append(label)
    
    return torch.stack(images), labels

def visualize_last_layer_attention(model, images, save_dir, config):
    """
    Visualize attention from CLS token to patch tokens in the last layer
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get attention maps from last layer
    with torch.no_grad():
        outputs, attentions = model(images.to('cuda'), return_attention=True)
        last_layer_attention = attentions[-1]
    
    # For each image
    for img_idx in range(images.shape[0]):
        # Get CLS token attention to patches for each head
        cls_attention = last_layer_attention[img_idx, :, 0, 1:]
        
        # Visualize each head separately
        for head_idx in range(cls_attention.shape[0]):
            fig = visualize_attention(
                images[img_idx],
                cls_attention[head_idx],
                save_path=None
            )
            wandb.log({f'last_layer_attention/image_{img_idx}/head_{head_idx}': wandb.Image(fig)})
            plt.close()
        
        # Visualize averaged attention across heads
        avg_attention = cls_attention.mean(0)
        fig = visualize_attention(
            images[img_idx],
            avg_attention,
            save_path=None
        )
        wandb.log({f'last_layer_attention/image_{img_idx}/averaged': wandb.Image(fig)})
        plt.close()

def visualize_all_layers_attention(model, images, save_dir, config):
    """
    Visualize attention from CLS token to patch tokens in all layers
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get attention maps from all layers
    with torch.no_grad():
        outputs, attentions = model(images.to('cuda'), return_attention=True)
    
    # For each image
    for img_idx in range(images.shape[0]):
        # For each layer
        for layer_idx, layer_attention in enumerate(attentions):
            # Get CLS token attention to patches (averaged across heads)
            cls_attention = layer_attention[img_idx, :, 0, 1:].mean(0)
            
            fig = visualize_attention(
                images[img_idx],
                cls_attention,
                save_path=None
            )
            wandb.log({f'all_layers_attention/image_{img_idx}/layer_{layer_idx}': wandb.Image(fig)})
            plt.close()

def visualize_attention_rollouts(model, images, save_dir, config):
    """
    Visualize attention rollout maps
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get attention maps from all layers
    with torch.no_grad():
        outputs, attentions = model(images.to('cuda'), return_attention=True)
    
    # For each image
    for img_idx in range(images.shape[0]):
        # Get attention maps for this image
        img_attentions = [attn[img_idx] for attn in attentions]
        
        # Compute attention rollout
        rollout = compute_attention_rollout(img_attentions)
        
        # Visualize rollout
        fig = visualize_attention_rollout(
            images[img_idx],
            rollout,
            save_path=os.path.join(save_dir, f'rollout_image_{img_idx}.png')
        )
        wandb.log({f'attention_rollout/image_{img_idx}': wandb.Image(fig)})
        plt.close()

def visualize_pos_embeddings(model, save_dir, config):
    """
    Visualize positional embedding similarities
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get positional embeddings based on type
    if not hasattr(model, 'pos_embed'):
        print("Model does not have positional embeddings")
        return
    
    try:
        # Get positional embeddings
        pos_embed = None
        if model.pos_embed.pos_embed_type == '1d_learned':
            pos_embed = model.pos_embed.pos_embed
        elif model.pos_embed.pos_embed_type == '2d_learned':
            pos_embed = torch.cat([
                model.pos_embed.cls_pos_embed,
                model.pos_embed.patch_pos_embed
            ], dim=1)
        elif model.pos_embed.pos_embed_type == 'sinusoidal':
            pos_embed = model.pos_embed.pos_embed
        else:
            print(f"Unsupported positional embedding type: {model.pos_embed.pos_embed_type}")
            return
        
        if pos_embed is None:
            print("Could not extract positional embeddings")
            return
        
        # Move to CPU and detach
        pos_embed = pos_embed.detach().cpu()
        
        # Visualize embeddings
        save_path = os.path.join(save_dir, 'pos_embed_visualization.png')
        fig = visualize_positional_embedding(
            pos_embed,
            patch_size=config['patch_size'],
            image_size=config['image_size'],
            save_path=save_path
        )
        
        if fig is not None:
            wandb.log({'positional_embedding/visualization': wandb.Image(fig)})
            plt.close(fig)
        
    except Exception as e:
        print(f"Error visualizing positional embeddings: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Initialize wandb
    os.environ["WANDB_DIR"] = BASE_DIR
    experiment_name = 'attention_visualization'
    wandb.init(project='differential-vit-cifar-10', name=experiment_name)
    
    # Load model configuration
    config = CONFIG  # Use the base configuration from config.py
    
    # Load trained model (update path to your best model)
    model_path = os.path.join(SAVE_DIR, '/ssd_scratch/cvit/shaon/cv_5/Q2/models/augmentation/autoaugment/best_model.pth')  # Adjust path as needed
    model = load_model(model_path, config)
    
    # Get test images
    images, labels = get_test_images(num_images=5)
    
    # Create base visualization directory
    base_dir = os.path.join('visualizations', experiment_name)
    os.makedirs(base_dir, exist_ok=True)
    
    # Run visualizations
    print("Visualizing last layer attention...")
    last_layer_dir = os.path.join(base_dir, 'last_layer_attention')
    visualize_last_layer_attention(model, images, last_layer_dir, config)
    
    print("Visualizing all layers attention...")
    all_layers_dir = os.path.join(base_dir, 'all_layers_attention')
    visualize_all_layers_attention(model, images, all_layers_dir, config)
    
    print("Visualizing attention rollout...")
    rollout_dir = os.path.join(base_dir, 'attention_rollout')
    visualize_attention_rollouts(model, images, rollout_dir, config)
    
    print("Visualizing positional embeddings...")
    pos_embed_dir = os.path.join(base_dir, 'positional_embeddings')
    visualize_pos_embeddings(model, pos_embed_dir, config)
    
    wandb.finish()

if __name__ == '__main__':
    main() 