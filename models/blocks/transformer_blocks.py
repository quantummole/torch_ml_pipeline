# -*- coding: utf-8 -*-
"""
@author: quantummole
"""

import torch
import torch.nn as nn
import torch.nn.functional as tfunc

class ScaledDotProductKernel(nn.Module):
    def __init__(self, mask=1, bias=0):
        super(ScaledDotProductKernel, self).__init__()
        self.mask = mask
        self.bias = bias
    def forward(self, q, k):
        attn_weights = torch.matmul(q, k.T) / (q.shape[-1] ** 0.5)
        attn_weights = tfunc.softmax(attn_weights*self.mask+self.bias, dim=-1)
        return attn_weights

class ScaledRadialKernel(nn.Module):
    def __init__(self, mask=1, bias=0):
        super(ScaledRadialKernel, self).__init__()
        self.mask = mask
        self.bias = bias
    def forward(self, q, k):
        attn_weights = (torch.matmul(q, q.T) + torch.matmul(k, k.T) -2*torch.matmul(q, k.T)) / (q.shape[-1] ** 0.5)
        attn_weights = tfunc.softmax(attn_weights*self.mask+self.bias, dim=-1)
        return attn_weights

class CausalKernel(nn.Module):
    def __init__(self, kernel_func, embed_dim, bias=0):
        super(CausalKernel, self).__init__()
        mask = torch.tril(torch.ones(embed_dim, embed_dim)).unsqueeze(0)  # Create a causal mask
        self.kernel_func = kernel_func(mask, bias)
        self.embed_dim = embed_dim 
        self.bias = bias
    def forward(self, q, k):
        return self.kernel_func(q, k)
    

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, embed_dim))
        nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)
    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()
        assert embed_dim == self.embed_dim, f"Input embed_dim {embed_dim} does not match expected {self.embed_dim}"
        assert seq_length <= self.max_len, f"Sequence length {seq_length} exceeds max length {self.max_len}"
        return x + self.positional_encoding[:seq_length].unsqueeze(0)

   
class PatchifyImage(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, img_size):
        super(PatchifyImage, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.img_size = img_size
        self.linear = nn.Linear(self.in_channels*self.patch_size*self.patch_size, self.embed_dim, bias=False)
    def forward(self, x, reconstruct=False, img_size=None):
        if not reconstruct:
            batch_size, channels, height, width = x.size()
            assert channels == self.in_channels, f"Input channels {channels} does not match expected {self.in_channels}"
            assert height % self.patch_size == 0 and width % self.patch_size == 0, "Height and width must be divisible by patch size"
            Hp = height // self.patch_size
            Wp = width // self.patch_size
            x = x.view(batch_size, self.in_channels, Hp, self.patch_size, Wp, self.patch_size)
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
            patches = x.view(batch_size, -1, self.in_channels * self.patch_size * self.patch_size)  
            encoded = self.linear(patches).reshape(batch_size, Hp, Wp, self.embed_dim)
            encoded = encoded.permute(0, 3, 1, 2).contiguous()
            return encoded
        else:
            height, width = img_size
            Hp = height // self.patch_size
            Wp = width // self.patch_size
            if x.ndim == 3:
                batch_size, num_patches, embed_dim = x.size()
                assert embed_dim == self.embed_dim, f"Input embed_dim {embed_dim} does not match expected {self.embed_dim}"
                assert Hp * Wp == num_patches, f"Number of patches {num_patches} does not match expected {Hp * Wp}"
                x = x.view(batch_size, Hp, Wp, self.embed_dim)
            elif x.ndim == 4:
                batch_size, embed_dim, Hp, Wp = x.size()
                assert embed_dim == self.embed_dim, f"Input embed_dim {embed_dim} does not match expected {self.embed_dim}"
                assert Hp*self.patch_size == height and Wp*self.patch_size == width, \
                    f"Output size {Hp*self.patch_size}x{Wp*self.patch_size} does not match expected {height}x{width}"
                x = x.permute(0, 2, 3, 1).contiguous()  # Change to (batch_size, Hp, Wp, embed_dim)
            else:   
                raise ValueError(f"Input tensor must be 3D or 4D, got {x.ndim}D") 
                    
            inv_matrix = torch.pinverse(self.linear.weight.T)
            y = torch.matmul(x, inv_matrix)
            y = y.view(batch_size, Hp, Wp, self.in_channels, self.patch_size, self.patch_size)
            y = y.permute(0, 3, 1, 4, 2, 5).contiguous()
            y = y.view(batch_size, self.in_channels, height, width)
            return y

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, kernel_func, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "must be multiple for me to make it fast"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.kernel_func = kernel_func

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got {embed_dim} and {num_heads})")

        self.qkv_proj = nn.Linear(embed_dim, embed_dim*3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        def forward(self, x):
            batch_size, seq_length, _ = x.size()
            qkv = self.qkv_proj(x).reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_weights = self.kernel_func(q,k)
            attn_weights = self.attn_drop(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, self.embed_dim)
            output = self.norm(x +  self.proj_drop(self.out_proj(attn_output)))
            return output
        
class FFN(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        output = self.linear2(self.dropout(tfunc.relu(self.linear1(x))))
        output = self.norm(x + output)
        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, kernel_func, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, kernel_func, dropout)
        self.ffn = FFN(embed_dim, ffn_dim, dropout)

    def forward(self, x):
        x = self.attention(x)
        x = self.ffn(x)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, num_layers, kernel_func, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.adaptor = nn.Linear(embed_dim, embed_dim)  # Optional adapter layer

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.adaptor(self.norm(x))  # Final normalization



if __name__ == "__main__":
    from matplotlib import pyplot as plt
    ## test the patchify image
    patch_size = 16     
    embed_dim = 768
    in_channels = 1
    img_size = (320, 320)
    patchify = PatchifyImage(in_channels, patch_size, embed_dim, img_size)  
    x = torch.randn(2, in_channels, img_size[0], img_size[1])  # Batch size of 2
    encoded = patchify(x)
    assert encoded.shape == (2, embed_dim, img_size[0] // patch_size, img_size[1] // patch_size), \
        f"Encoded shape mismatch: {encoded.shape} != {(2, embed_dim, img_size[0] // patch_size, img_size[1] // patch_size)}" 
    reconstructed = patchify(encoded, reconstruct=True, img_size=img_size)
    assert reconstructed.shape == (2, in_channels, img_size[0], img_size[1]), \
        f"Reconstructed shape mismatch: {reconstructed.shape} != {(2, in_channels, img_size[0], img_size[1])}"
    assert torch.abs(reconstructed - x).sum() < 1, "Reconstruction failed"

