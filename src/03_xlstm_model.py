import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    D_MODEL, N_BLOCKS, NUM_HEADS, EXPAND_FACTOR, DROPOUT, SEQUENCE_LENGTH, SEED,
)

torch.manual_seed(SEED)


class CausalConv1d(nn.Module):
    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                              padding=self.padding, groups=d_model)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x[..., :-self.padding].transpose(1, 2)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class sLSTMBlock(nn.Module):
    """
    Scalar LSTM with exponential gating — vectorized.
    C_t = f_t * C_{t-1} + i_t * z_t
    """
    def __init__(self, d_model, num_heads, expand_factor, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.conv = CausalConv1d(d_model, kernel_size=4)
        self.input_proj = nn.Linear(d_model, d_model * 4)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.GroupNorm(num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

        inner_dim = d_model * expand_factor
        self.mlp_up = nn.Linear(d_model, inner_dim * 2)
        self.mlp_down = nn.Linear(inner_dim, d_model)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.5)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.mlp_up.weight, gain=0.5)
        nn.init.zeros_(self.mlp_up.bias)
        nn.init.xavier_uniform_(self.mlp_down.weight, gain=0.5)
        nn.init.zeros_(self.mlp_down.bias)

    def forward(self, x, state=None):
        batch, seq_len, _ = x.shape
        residual = x

        x_conv = self.conv(x)
        gates = self.input_proj(F.silu(x_conv))
        i_tilde, f_tilde, z, o_tilde = gates.chunk(4, dim=-1)
        i_tilde = i_tilde.view(batch, seq_len, self.num_heads, self.d_head)
        f_tilde = f_tilde.view(batch, seq_len, self.num_heads, self.d_head)
        z = z.view(batch, seq_len, self.num_heads, self.d_head)
        o_act = F.sigmoid(o_tilde.view(batch, seq_len, self.num_heads, self.d_head))
        i_gate = torch.exp(i_tilde.clamp(-8, 8))
        f_gate = torch.exp(f_tilde.clamp(-8, 8))

        F_cum = f_gate.cumprod(dim=1)
        c_t = F_cum * (i_gate * z / (F_cum + 1e-10)).cumsum(dim=1)
        n_t = F_cum * (i_gate / (F_cum + 1e-10)).cumsum(dim=1)
        h = o_act * (c_t / torch.clamp(n_t, min=1.0))
        h = h.reshape(batch, seq_len, self.d_model)

        h = h.reshape(-1, self.d_model)
        h = self.norm(h)
        h = h.reshape(batch, seq_len, self.d_model)
        h = self.out_proj(h)
        h = self.dropout(h)
        h = h + residual

        mlp_in = self.mlp_up(h)
        gate, val = mlp_in.chunk(2, dim=-1)
        h = h + self.mlp_down(self.dropout(F.silu(gate) * val))
        return h, None


class mLSTMBlock(nn.Module):
    """
    Matrix LSTM — vectorized.
    C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v_t k_t^T)
    Uses scalar gates per head.
    C_t = F_t * cumsum[(i_j / F_j) * v_j k_j^T]
    """
    def __init__(self, d_model, num_heads, expand_factor, dropout):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.conv = CausalConv1d(d_model, kernel_size=4)
        self.proj_qkv = nn.Linear(d_model, d_model * 3)
        self.gate_proj = nn.Linear(d_model, num_heads * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.GroupNorm(num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

        inner_dim = d_model * expand_factor
        self.mlp_up = nn.Linear(d_model, inner_dim * 2)
        self.mlp_down = nn.Linear(inner_dim, d_model)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_qkv.weight, gain=0.5)
        nn.init.zeros_(self.proj_qkv.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.xavier_uniform_(self.mlp_up.weight, gain=0.5)
        nn.init.zeros_(self.mlp_up.bias)
        nn.init.xavier_uniform_(self.mlp_down.weight, gain=0.5)
        nn.init.zeros_(self.mlp_down.bias)

    def forward(self, x, state=None):
        batch, seq_len, _ = x.shape
        residual = x

        x_conv = self.conv(x)
        x_act = F.silu(x_conv)

        qkv = self.proj_qkv(x_act)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.d_head)
        k = k.view(batch, seq_len, self.num_heads, self.d_head)
        v = v.view(batch, seq_len, self.num_heads, self.d_head)

        gates = self.gate_proj(x_act)
        f_gate, i_gate = gates.chunk(2, dim=-1)
        f_gate = torch.exp(f_gate.clamp(-8, 8)).view(batch, seq_len, self.num_heads, 1, 1)
        i_gate = torch.exp(i_gate.clamp(-8, 8)).view(batch, seq_len, self.num_heads, 1, 1)

        outer = torch.einsum('b t h i, b t h j -> b t h i j', v, k)

        F_cum = f_gate.cumprod(dim=1)
        discounted = i_gate * outer / (F_cum + 1e-10)
        C_t = F_cum * discounted.cumsum(dim=1)

        h = torch.einsum('b t h i j, b t h j -> b t h i', C_t, q)
        h = h.reshape(batch, seq_len, self.d_model)

        h = h.reshape(-1, self.d_model)
        h = self.norm(h)
        h = h.reshape(batch, seq_len, self.d_model)
        h = self.out_proj(h)
        h = self.dropout(h)
        h = h + residual

        mlp_in = self.mlp_up(h)
        gate, val = mlp_in.chunk(2, dim=-1)
        h = h + self.mlp_down(self.dropout(F.silu(gate) * val))
        return h, None


class XLSTMEncoder(nn.Module):
    def __init__(self, d_model, n_blocks, num_heads, expand_factor, dropout):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            if i % 2 == 0:
                self.blocks.append(mLSTMBlock(d_model, num_heads, expand_factor, dropout))
            else:
                self.blocks.append(sLSTMBlock(d_model, num_heads, expand_factor, dropout))

    def forward(self, x):
        for block in self.blocks:
            x, _ = block(x)
        return x


class XLSTMTSModel(nn.Module):
    def __init__(self, n_features, d_model=D_MODEL, n_blocks=N_BLOCKS,
                 num_heads=NUM_HEADS, expand_factor=EXPAND_FACTOR, dropout=DROPOUT):
        super().__init__()
        self.d_model = d_model

        self.input_embedding = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=SEQUENCE_LENGTH)
        self.input_dropout = nn.Dropout(dropout)

        self.encoder = XLSTMEncoder(d_model, n_blocks, num_heads, expand_factor, dropout)

        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=0.5)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.pos_encoding(x)
        x = self.input_dropout(x)
        x = self.encoder(x)
        last = x[:, -1, :]
        last = self.head_norm(last)
        logit = self.head(last)
        return logit.squeeze(-1)
