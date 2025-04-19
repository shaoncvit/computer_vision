import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Query tensor of shape (batch, heads, seq_len, d_k)
            k: Key tensor of shape (batch, heads, seq_len, d_k)
            v: Value tensor of shape (batch, heads, seq_len, d_v)
            mask: Optional mask tensor of shape (batch, 1, 1, seq_len)
        """
        d_k = q.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        # print(f"Attention scores shape: {scores.shape}")  # (batch, heads, seq_len, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # print(f"Attention weights shape: {attn.shape}")  # (batch, heads, seq_len, seq_len)
        
        # Compute output
        output = torch.matmul(attn, v)
        # print(f"Attention output shape: {output.shape}")  # (batch, heads, seq_len, d_v)
        
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        residual = x
        
        # print(f"Input shape: {x.shape}")  # (batch, seq_len, d_model)
        
        # Linear projections and reshape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # print(f"After projection shape: {q.shape}")  # (batch, seq_len, d_model)
        
        # Reshape to (batch, heads, seq_len, d_k)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # print(f"After reshape shape: {q.shape}")  # (batch, heads, seq_len, d_k)
        
        # Apply attention
        output, attn = self.attention(q, k, v, mask)
        
        # Reshape back and apply final linear layer
        output = rearrange(output, 'b h n d -> b n (h d)')
        # print(f"Before final projection shape: {output.shape}")  # (batch, seq_len, d_model)
        
        output = self.W_o(output)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        # print(f"Final output shape: {output.shape}")  # (batch, seq_len, d_model)
        
        return output, attn 