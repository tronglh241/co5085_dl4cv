'''Shallow pre-norm encoder and a CLS classifier.

attn_impl selects ScratchMSA or TorchMSA. Every other weight is shared in
structure, so a scratch-vs-torch run differs only in the attention module.
'''

import torch
import torch.nn as nn

from .attention import build_attention


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout, attn_impl):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = build_attention(attn_impl, dim, num_heads, dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_ratio, dropout)

    def forward(self, x, return_attn=False):
        attn_out, attn = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + self.resid_drop(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x, attn


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        tokenizer,
        depth=4,
        num_heads=4,
        mlp_ratio=2,
        dropout=0.1,
        attn_impl='scratch',
        num_classes=10,
    ):
        super().__init__()
        dim = tokenizer.embed_dim
        self.tokenizer = tokenizer
        self.cls = nn.Parameter(torch.empty(1, 1, dim))
        self.pos = nn.Parameter(torch.empty(1, tokenizer.num_tokens + 1, dim))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            EncoderBlock(dim, num_heads, mlp_ratio, dropout, attn_impl)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def param_groups(self, weight_decay):
        '''AdamW groups: decay Linear/Conv weights only.

        Biases and norm scales (ndim <= 1), the CLS token and the positional
        embedding get no weight decay, as in ViT training.
        '''
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name in ('cls', 'pos'):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ]

    def forward(self, x, return_attn=False):
        tokens = self.tokenizer(x)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1) + self.pos
        attentions = []
        for block in self.blocks:
            tokens, attn = block(tokens, return_attn=return_attn)
            if return_attn:
                attentions.append(attn)
        logits = self.head(self.norm(tokens)[:, 0])
        if return_attn:
            return logits, attentions
        return logits
