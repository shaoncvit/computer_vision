import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=28, patch_size=7, in_channels=1, embed_dim = 64):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Sequential(
            # Breaking the images into patches
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        b = x.shape[0]
        x = self.projection(x)

        # Adding a Classification Token
        cls_tokens = self.cls_token.expand(b, -1 , -1)
        x = torch.cat([cls_tokens, x], dim =1)

        # position embeddings
        x = x + self.pos_embedding

        return x


class DifferentialAttention(nn.Module):
    def __init__(self, dim, num_heads = 8, qkv_bias = False, attn_drop= 0., proj_drop=0. , lambda_init=0.8):
        super().__init__()
        assert dim % num_heads == 0,'dim should be divisible by num_heads'

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Query and Key Projections into Diff Attn
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # introducing the Lambda Parameter
        self.lambda_q1 = nn.Parameter(torch.randn(head_dim))
        self.lambda_k1 = nn.Parameter(torch.randn(head_dim))

        self.lambda_q2 = nn.Parameter(torch.randn(head_dim))
        self.lambda_k2 = nn.Parameter(torch.randn(head_dim))

        self.lambda_init = lambda_init

    def get_lambda(self):
        lambda_val = (torch.exp(self.lambda_q1 * self.lambda_k1) -
                     torch.exp(self.lambda_q2 * self.lambda_k2) +
                     self.lambda_init)
        return lambda_val.mean()

    def forward(self, x, return_attention=False):
        B , N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3,
                                  self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q1, q2, k1, k2, v = qkv[0], qkv[0], qkv[1], qkv[1], qkv[2]

        # Split into Queries and Keys
        q1, q2 = q1.chunk(2, dim= -1)
        k1, k2 = k1.chunk(2, dim=-1)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale

        attn = F.softmax(attn1, dim=-1) - (self.get_lambda() * F.softmax(attn2, dim=-1))
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x

