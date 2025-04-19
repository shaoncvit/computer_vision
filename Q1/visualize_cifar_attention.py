import torch
import torchvision
import torchvision.transforms as transforms
from models.vit import VisionTransformer
from utils.visualization import (
    visualize_attention, visualize_attention_heads,
    compute_attention_rollout, visualize_attention_rollout,
    visualize_positional_embedding
)
import matplotlib.pyplot as plt
import wandb
import os
import torch.nn as nn

def load_model(model_path, config):
    """
    Load trained ViT model
    """
    model = VisionTransformer(**config).to('cuda')
    checkpoint = torch.load(model_path)
    
    # Handle DataParallel state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if it exists (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    
    # Load the processed state dict
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def get_test_images(num_images=5):
    """
    Get some test images from CIFAR-10
    """
    # Create data directory if it doesn't exist
    data_dir = '/ssd_scratch/cvit/shaon/cv_5/data'
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform)
    
    # Get a few test images
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
    Args:
        model: The ViT model
        images: Input images to visualize attention for
        save_dir: Directory to save visualizations
        config: Model configuration containing patch_size
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get attention maps from last layer
    with torch.no_grad():
        outputs, attentions = model(images.to('cuda'))
        last_layer_attention = attentions[-1]  # (batch_size, num_heads, seq_len, seq_len)
    
    # For each image
    for img_idx in range(images.shape[0]):
        # Get CLS token attention to patches for each head
        cls_attention = last_layer_attention[img_idx, :, 0, 1:]  # (num_heads, num_patches)
        
        # Visualize each head separately
        for head_idx in range(cls_attention.shape[0]):
            fig = visualize_attention(
                images[img_idx],
                cls_attention[head_idx],  # Pass the 1D attention vector directly
                save_path=None
            )
            wandb.log({f'last_layer_attention/image_{img_idx}/head_{head_idx}': wandb.Image(fig)})
            plt.close()
        
        # Visualize averaged attention across heads
        avg_attention = cls_attention.mean(0)  # (num_patches)
        fig = visualize_attention(
            images[img_idx],
            avg_attention,  # Pass the 1D attention vector directly
            save_path=None
        )
        wandb.log({f'last_layer_attention/image_{img_idx}/averaged': wandb.Image(fig)})
        plt.close()

def visualize_all_layers_attention(model, images, save_dir, config):
    """
    Visualize attention from CLS token to patch tokens in all layers
    Args:
        model: The ViT model
        images: Input images to visualize attention for
        save_dir: Directory to save visualizations
        config: Model configuration containing patch_size
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get attention maps from all layers
    with torch.no_grad():
        outputs, attentions = model(images.to('cuda'))
    
    # For each image
    for img_idx in range(images.shape[0]):
        # For each layer
        for layer_idx, layer_attention in enumerate(attentions):
            # Get CLS token attention to patches (averaged across heads)
            cls_attention = layer_attention[img_idx, :, 0, 1:].mean(0)  # (num_patches)
            
            fig = visualize_attention(
                images[img_idx],
                cls_attention,  # Pass the 1D attention vector directly
                save_path=None
            )
            wandb.log({f'all_layers_attention/image_{img_idx}/layer_{layer_idx}': wandb.Image(fig)})
            plt.close()

def visualize_attention_rollouts(model, images, save_dir, config):
    """
    Visualize attention rollout maps
    Args:
        model: The ViT model
        images: Input images to visualize attention for
        save_dir: Directory to save visualizations
        config: Model configuration containing patch_size
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get attention maps from all layers
    with torch.no_grad():
        outputs, attentions = model(images.to('cuda'))
    
    # For each image
    for img_idx in range(images.shape[0]):
        # Get attention maps for this image
        img_attentions = [attn[img_idx] for attn in attentions]
        
        # Compute attention rollout
        rollout = compute_attention_rollout(img_attentions)  # Shape: [64]
        
        # Visualize rollout
        fig = visualize_attention_rollout(
            images[img_idx],
            rollout,  # Already in correct shape [64]
            save_path=os.path.join(save_dir, f'rollout_image_{img_idx}.png')
        )
        wandb.log({f'attention_rollout/image_{img_idx}': wandb.Image(fig)})
        plt.close()

def visualize_pos_embeddings(model, save_dir, config):
    """
    Visualize positional embedding similarities
    Args:
        model: The ViT model
        save_dir: Directory to save visualizations
        config: Model configuration containing patch_size and image_size
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get positional embeddings based on type
    if not hasattr(model, 'pos_embed'):
        print("Model does not have positional embeddings")
        return
        
    if isinstance(model.pos_embed, nn.Identity):
        print("No positional embeddings to visualize (using None type)")
        return
        
    try:
        # Get positional embeddings
        pos_embed = None
        if hasattr(model.pos_embed, 'pos_embedding'):
            # 1D embeddings
            pos_embed = model.pos_embed.pos_embedding
            print("Using 1D positional embeddings")
        elif hasattr(model.pos_embed, 'patch_pos_embed'):
            # 2D embeddings
            pos_embed = torch.cat([model.pos_embed.cls_pos_embed, 
                                 model.pos_embed.patch_pos_embed], dim=1)
            print("Using 2D positional embeddings")
        elif hasattr(model.pos_embed, 'pe'):
            # Sinusoidal embeddings
            pos_embed = model.pos_embed.pe
            print("Using sinusoidal positional embeddings")
        else:
            print("Unknown positional embedding type")
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
        else:
            print("Visualization function returned None")
            
    except Exception as e:
        print(f"Error visualizing positional embeddings: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Initialize wandb with correct path
    os.environ["WANDB_DIR"] = '/ssd_scratch/cvit/shaon/cv_5/Q1/wandb'
    experiment_name = 'attention_visualization'
    wandb.init(project='vision-transformer-cifar10', name=experiment_name)
    
    # Model configuration (use the same config as your best model)
    config = {
        'image_size': 32,
        'patch_size': 4,
        'embed_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'mlp_ratio': 4,
        'dropout': 0.1,
        'pos_embed_type': '1d'  # Make sure this matches your trained model
    }
    
    # Load trained model
    model_path = '/ssd_scratch/cvit/shaon/cv_5/Q1/models_vit/augmentation/augmentation_best.pth'
    model = load_model(model_path, config)
    
    # Get test images
    images, labels = get_test_images(num_images=5)
    
    # Create base visualization directory with experiment name
    base_dir = os.path.join('visualizations', experiment_name)
    os.makedirs(base_dir, exist_ok=True)
    
    # Visualize attention maps
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