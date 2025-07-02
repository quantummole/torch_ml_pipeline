# -*- coding: utf-8 -*-
"""
@author: quantummole
"""

import torch
import torch.nn as nn
import torch.nn.functional as tfunc

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "must be multiple for me to make it fast"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got {embed_dim} and {num_heads})")

        self.qkv_proj = nn.Linear(embed_dim, embed_dim*3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        def forward(self, x):
            batch_size, seq_length, _ = x.size()
            qkv = self.qkv_proj(x).reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)