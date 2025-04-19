import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # print(f"MLP input shape: {x.shape}")  # (batch, seq_len, d_model)
        residual = x
        x = self.net(x)
        x = self.layer_norm(x + residual)
        # print(f"MLP output shape: {x.shape}")  # (batch, seq_len, d_model)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        # print(f"\nTransformer Encoder Layer")
        # print(f"Input shape: {x.shape}")  # (batch, seq_len, d_model)
        
        # Multi-head attention
        x, attn = self.attention(x, mask)
        
        # MLP
        x = self.mlp(x)
        
        # print(f"Output shape: {x.shape}")  # (batch, seq_len, d_model)
        return x, attn

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attentions.append(attn)
        return x, attentions 