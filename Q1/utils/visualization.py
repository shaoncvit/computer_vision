import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from einops import rearrange
import os

def visualize_attention(image, attention_map, save_path=None):
    """
    Visualize attention map overlaid on the image
    Args:
        image: Input image tensor of shape [C, H, W]
        attention_map: Attention map tensor of shape [num_patches]
        save_path: Optional path to save the visualization
    """
    # Create a high-resolution figure
    plt.figure(figsize=(20, 10), dpi=200)
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0).cpu())
    plt.title('Original Image', fontsize=14)
    plt.axis('off')
    
    # Attention map
    plt.subplot(1, 2, 2)
    plt.imshow(image.permute(1, 2, 0).cpu())
    
    # Calculate grid size based on attention map size
    num_patches = attention_map.shape[0]  # Should be 64 for 32x32 image with 4x4 patches
    num_patches_side = int(np.sqrt(num_patches))  # Should be 8
    
    # Reshape attention map to 2D grid
    attention_map = attention_map.reshape(num_patches_side, num_patches_side)
    
    # Use interpolation to make attention map smoother
    plt.imshow(attention_map.cpu(), alpha=0.6, cmap='viridis', 
              interpolation='bilinear')
    plt.title('Attention Map', fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return plt.gcf()

def visualize_attention_heads(image, attention_maps, save_path=None):
    """
    Visualize attention maps from multiple heads
    Args:
        image: Input image tensor
        attention_maps: Attention maps tensor of shape [num_heads, seq_len, seq_len]
        save_path: Optional path to save the visualization
    """
    num_heads = attention_maps.shape[0]
    fig_size = int(np.ceil(np.sqrt(num_heads)))
    
    plt.figure(figsize=(20, 20))
    
    # Get attention from CLS token to patches for each head
    cls_to_patches = attention_maps[:, 0, 1:]  # [num_heads, num_patches]
    
    # Calculate grid size based on number of patches
    num_patches = cls_to_patches.shape[1]  # Should be 64 for 8x8 patches
    num_patches_side = int(np.sqrt(num_patches))  # Should be 8
    
    for i in range(num_heads):
        plt.subplot(fig_size, fig_size, i + 1)
        plt.imshow(image.permute(1, 2, 0).cpu())
        
        # Reshape attention map for this head
        attention_map = cls_to_patches[i].reshape(num_patches_side, num_patches_side)
        plt.imshow(attention_map.cpu(), alpha=0.6, cmap='viridis',
                  interpolation='bilinear')
        plt.title(f'Head {i+1}', fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return plt.gcf()  # Return the figure for wandb logging

def compute_attention_rollout(attention_maps, discard_ratio=0.9):
    """
    Compute attention rollout as described in the ViT paper
    Args:
        attention_maps: List of attention tensors from each layer [num_layers, num_heads, seq_len, seq_len]
        discard_ratio: Ratio to discard low attention weights
    Returns:
        attention_rollout: Final attention rollout map [num_patches]
    """
    # Print shapes for debugging
    print(f"Number of layers: {len(attention_maps)}")
    print(f"First layer attention shape: {attention_maps[0].shape}")
    
    # Get attention from CLS token to patches for each layer
    cls_attentions = []
    for attn in attention_maps:
        # Average across heads
        attn_averaged = attn.mean(0)  # [seq_len, seq_len]
        
        # Get CLS token attention to patches
        cls_attention = attn_averaged[0, 1:]  # [num_patches]
        
        # Normalize
        cls_attention = cls_attention / cls_attention.sum()
        
        cls_attentions.append(cls_attention.cpu())
    
    # Average across layers
    final_attention = torch.stack(cls_attentions).mean(0)  # [num_patches]
    
    # Discard low attention weights
    threshold = final_attention.max() * discard_ratio
    final_attention[final_attention < threshold] = 0
    
    # Normalize
    if final_attention.sum() > 0:
        final_attention = final_attention / final_attention.sum()
    
    return final_attention

def visualize_positional_embedding(pos_embed, patch_size=None, image_size=None, save_path=None):
    """
    Visualize positional embedding similarities and spatial patterns
    Args:
        pos_embed: Positional embeddings tensor (1, seq_len, embed_dim)
        patch_size: Size of each patch
        image_size: Size of the original image
        save_path: Path to save the visualization
    """
    # Ensure we're working with the right shape
    if pos_embed.dim() == 3:
        pos_embed = pos_embed.squeeze(0)  # Remove batch dimension if present
    
    # Compute similarity matrix
    similarity = torch.matmul(pos_embed, pos_embed.transpose(-2, -1))
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Plot similarity matrix
    ax1 = plt.subplot(131)
    sns.heatmap(similarity.numpy(), ax=ax1, cmap='viridis')
    ax1.set_title('Positional Embedding Similarities')
    
    # Plot embedding magnitudes
    ax2 = plt.subplot(132)
    magnitudes = torch.norm(pos_embed, dim=1)
    ax2.plot(magnitudes.numpy())
    ax2.set_title('Embedding Magnitudes')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Magnitude')
    
    # Plot spatial pattern if possible
    if patch_size is not None and image_size is not None:
        ax3 = plt.subplot(133)
        num_patches = (image_size // patch_size) ** 2
        
        if pos_embed.shape[0] in [num_patches, num_patches + 1]:  # Account for CLS token
            # Remove CLS token if present
            if pos_embed.shape[0] == num_patches + 1:
                patch_embeddings = pos_embed[1:]
            else:
                patch_embeddings = pos_embed
            
            # Compute average embedding magnitude for each position
            patch_magnitudes = torch.norm(patch_embeddings, dim=1)
            spatial_pattern = patch_magnitudes.view(image_size // patch_size, 
                                                  image_size // patch_size)
            
            sns.heatmap(spatial_pattern.numpy(), ax=ax3, cmap='viridis')
            ax3.set_title('Spatial Pattern')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_attention_rollout(image, attention_rollout, save_path=None):
    """
    Visualize attention rollout map
    Args:
        image: Input image tensor
        attention_rollout: Attention rollout tensor of shape [num_patches]
        save_path: Optional path to save the visualization
    """
    plt.figure(figsize=(20, 10), dpi=200)
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0).cpu())
    plt.title('Original Image', fontsize=14)
    plt.axis('off')
    
    # Attention rollout
    plt.subplot(1, 2, 2)
    plt.imshow(image.permute(1, 2, 0).cpu())
    
    # Calculate grid size based on attention rollout size
    num_patches = attention_rollout.shape[0]  # Should be 64
    num_patches_side = int(np.sqrt(num_patches))  # Should be 8
    
    # Reshape and visualize with interpolation
    attention_map = attention_rollout.reshape(num_patches_side, num_patches_side)
    plt.imshow(attention_map.cpu(), alpha=0.6, cmap='viridis',
              interpolation='bilinear')
    plt.title('Attention Rollout', fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    return plt.gcf() 