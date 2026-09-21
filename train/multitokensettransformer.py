import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class DeepSetBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, mlp_ratio=4, use_batchnorm=True):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.use_batchnorm=use_batchnorm
        if use_batchnorm:
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_ratio * hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_ratio * hidden_dim, hidden_dim)
        )

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x=x+attn_out
        if self.use_batchnorm:
            x = self.norm1(x)

        # Feedforward
        ffn_out = self.ffn(x)
        x=x+ffn_out
        if self.use_batchnorm:
            x = self.norm2(x)

        return x



class MultiTokenDeepSetEncoder(nn.Module):
    """
    Input:
        [B, N, D]

    Output:
        [B, num_tokens, H]
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_tokens=1,
        num_layers=4,
        num_heads=4,
        mlp_ratio=4,
        use_batchnorm=True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.layers = nn.ModuleList([
            DeepSetBlock(
                hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_batchnorm=use_batchnorm,
            )
            for _ in range(num_layers)
        ])

        # Learned output queries
        self.query_tokens = nn.Parameter(
            torch.randn(num_tokens, hidden_dim)
        )

        self.pool_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):

        h = self.input_proj(x)

        for layer in self.layers:
            h = layer(h)

        B = h.size(0)

        queries = self.query_tokens.unsqueeze(0).expand(B, -1, -1)

        pooled, _ = self.pool_attn(
            queries,
            h,
            h
        )


        #flatten
        pooled=torch.flatten(pooled, start_dim=1)

        return pooled
