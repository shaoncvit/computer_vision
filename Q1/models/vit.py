import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .transformer import TransformerEncoder

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )

    def forward(self, x):
        # print(f"\nPatch Embedding")
        # print(f"Input image shape: {x.shape}")  # (batch, channels, height, width)
        x = self.projection(x)
        # print(f"Output patches shape: {x.shape}")  # (batch, num_patches, embed_dim)
        return x

class PositionalEmbedding1D(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

    def forward(self, x):
        return x + self.pos_embedding

class PositionalEmbedding2D(nn.Module):
    def __init__(self, height, width, embed_dim):
        super().__init__()
        # Create separate embeddings for CLS token and patches
        self.cls_pos_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.patch_pos_embed = nn.Parameter(torch.randn(1, height * width, embed_dim) * 0.02)

    def forward(self, x):
        # x shape: (batch_size, num_patches + 1, embed_dim) where first token is CLS
        # Split CLS token and patches
        cls_token, patches = x[:, :1], x[:, 1:]
        
        # Add positional embeddings
        cls_token = cls_token + self.cls_pos_embed
        patches = patches + self.patch_pos_embed
        
        # Recombine
        return torch.cat([cls_token, patches], dim=1)

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        pe = torch.zeros(seq_len, embed_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=32,  # CIFAR-10 image size
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        pos_embed_type='1d'  # '1d', '2d', 'sinusoidal', or None
    ):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Positional embedding
        if pos_embed_type == '1d':
            self.pos_embed = PositionalEmbedding1D(num_patches + 1, embed_dim)  # +1 for CLS token
        elif pos_embed_type == '2d':
            height = width = image_size // patch_size
            self.pos_embed = PositionalEmbedding2D(height, width, embed_dim)
        elif pos_embed_type == 'sinusoidal':
            self.pos_embed = SinusoidalPositionalEmbedding(num_patches + 1, embed_dim)
        else:
            self.pos_embed = nn.Identity()
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            d_model=embed_dim,
            num_heads=num_heads,
            d_ff=mlp_ratio * embed_dim,
            dropout=dropout
        )
        
        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # print(f"\nVision Transformer")
        # print(f"Input shape: {x.shape}")  # (batch, channels, height, width)
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat([cls_token, x], dim=1)
        # print(f"After adding CLS token shape: {x.shape}")  # (batch, num_patches + 1, embed_dim)
        
        # Add positional embedding
        x = self.pos_embed(x)
        # print(f"After positional embedding shape: {x.shape}")  # (batch, num_patches + 1, embed_dim)
        
        # Apply transformer
        x, attentions = self.transformer(x)
        # print(f"After transformer shape: {x.shape}")  # (batch, num_patches + 1, embed_dim)
        
        # Get CLS token output
        cls_token_output = x[:, 0]
        # print(f"CLS token output shape: {cls_token_output.shape}")  # (batch, embed_dim)
        
        # Classification
        logits = self.mlp_head(cls_token_output)
        # print(f"Output logits shape: {logits.shape}")  # (batch, num_classes)
        
        return logits, attentions 