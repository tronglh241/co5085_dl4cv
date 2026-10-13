'''Multi-head self-attention: a from-scratch module and a PyTorch wrapper.

ScratchMSA uses only Linear, einsum and softmax. Its in_proj weight is stacked
[W_q; W_k; W_v], the same layout as nn.MultiheadAttention.in_proj_weight, so
check_msa.py can copy parameters one-to-one.
'''

import math

import torch
import torch.nn as nn
from torch.nn.utils import skip_init


class ScratchMSA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f'embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})')

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Same draw order as nn.MultiheadAttention: QKV is left empty, out_proj
        # draws Kaiming, then QKV is Xavier. A normal Linear for QKV would draw
        # extra samples first and shift the FFN and the classifier head.
        self.in_proj = skip_init(nn.Linear, embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.constant_(self.in_proj.bias, 0.0)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, x, return_attn=False):
        batch, length, _ = x.shape
        qkv = self.in_proj(x).view(batch, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scores = torch.einsum('bhnd,bhmd->bhnm', query, key) / math.sqrt(self.head_dim)
        attn = self.attn_drop(scores.softmax(dim=-1))
        mixed = torch.einsum('bhnm,bhmd->bhnd', attn, value)
        mixed = mixed.transpose(1, 2).reshape(batch, length, self.embed_dim)
        out = self.out_proj(mixed)
        if return_attn:
            return out, attn
        return out, None


class TorchMSA(nn.Module):
    '''Same call signature as ScratchMSA, implemented by nn.MultiheadAttention.'''

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )

    def forward(self, x, return_attn=False):
        # need_weights=False lets PyTorch use the fast SDPA kernels.
        out, weights = self.attn(
            x,
            x,
            x,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        if return_attn:
            return out, weights
        return out, None


def build_attention(attn_impl, embed_dim, num_heads, dropout):
    if attn_impl == 'scratch':
        return ScratchMSA(embed_dim, num_heads, dropout=dropout)
    if attn_impl == 'torch':
        return TorchMSA(embed_dim, num_heads, dropout=dropout)
    raise ValueError(f"attn_impl must be 'scratch' or 'torch', got {attn_impl!r}")
