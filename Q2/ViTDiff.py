from diff_attn import *
from Encoder import *
import math

class PositionalEmbedding(nn.Module):
    """Different types of positional embeddings"""
    def __init__(self, embed_dim, num_patches, pos_embed_type='1d_learned'):
        super().__init__()
        self.pos_embed_type = pos_embed_type
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        if pos_embed_type == 'none':
            # No positional embedding
            self.pos_embed = None
            
        elif pos_embed_type == '1d_learned':
            # 1D learned positional embedding (original ViT implementation)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            # Initialize with small random values
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            
        elif pos_embed_type == '2d_learned':
            # 2D learned positional embedding
            # Calculate grid size from num_patches
            self.grid_size = int(math.sqrt(num_patches))
            # Separate embeddings for CLS token and patches
            self.cls_pos_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.patch_pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
            
        elif pos_embed_type == 'sinusoidal':
            # Sinusoidal positional embedding from "Attention is All You Need"
            pe = torch.zeros(num_patches + 1, embed_dim)
            position = torch.arange(0, num_patches + 1, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            # Register as buffer since it's not a learnable parameter
            self.register_buffer('pos_embed', pe)
        else:
            raise ValueError(f"Unknown positional embedding type: {pos_embed_type}")
    
    def forward(self, x):
        if self.pos_embed_type == 'none':
            return x
        elif self.pos_embed_type == '2d_learned':
            # Split CLS token and patches
            cls_token, patches = x[:, :1], x[:, 1:]
            # Add positional embeddings separately
            cls_token = cls_token + self.cls_pos_embed
            patches = patches + self.patch_pos_embed
            # Recombine
            return torch.cat([cls_token, patches], dim=1)
        else:  # 1d_learned or sinusoidal
            return x + self.pos_embed

class VisionTransformerDiff(nn.Module):
    def __init__(self,
                 image_size=28,
                 patch_size=7,
                 in_channels=1,
                 num_classes=10,
                 embed_dim=64,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 pos_embed_type='1d_learned'):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = PositionalEmbedding(
            embed_dim=embed_dim,
            num_patches=num_patches,
            pos_embed_type=pos_embed_type
        )
        
        # Change from Sequential to ModuleList for attention map collection
        self.blocks = nn.ModuleList([
            TransformerEncoder(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x, return_attention=False):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        
        attention_maps = []
        if return_attention:
            for block in self.blocks:
                x, attn = block(x, return_attention=True)
                attention_maps.append(attn)
        else:
            for block in self.blocks:
                x = block(x)
        
        x = self.norm(x)
        x = x[:, 0]  # Take CLS token
        x = self.head(x)
        
        if return_attention:
            return x, attention_maps
        return x