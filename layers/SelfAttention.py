import torch
import torch.nn as nn

import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau, delta
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

# class causal_convolution(nn.Module):
#     def __init__(self, c_in, c_out, ks=5, stride=1, dilation=1):
#         super(causal_convolution, self).__init__()
#         self.chomp_size = (ks - 1) * dilation
#         self.conv1 = nn.Conv1d(c_in, c_out, ks,
#                                stride=stride, padding=(ks - 1) * dilation, dilation=dilation)

#     def forward(self, x):
#         x = self.conv1(x)
#         return x[:, :, :-self.chomp_size].contiguous()


# class AttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads,
#                  d_keys=None, d_values=None, mix=False):
#         super(AttentionLayer, self).__init__()

#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)

#         self.inner_attention = attention
#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_model, d_keys * n_heads)

#         self.key_causal_convolution = causal_convolution(d_model, d_keys * n_heads)
#         self.query_causal_convolution = causal_convolution(d_model, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_model, d_values * n_heads)
#         self.out_projection = nn.Linear(d_values * n_heads, d_model)
#         self.n_heads = n_heads
#         self.mix = mix

#     def forward(self, queries, keys, values, attn_mask):
#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads
#         queries = self.query_causal_convolution(queries.permute(0, 2, 1)).view(B, L, H, -1)

#         keys = self.key_causal_convolution(keys.permute(0, 2, 1)).view(B, S, H, -1)

#         values = self.value_projection(values).view(B, S, H, -1)

#         out, attn = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask
#         )
#         if self.mix:
#             out = out.transpose(2, 1).contiguous()
#         out = out.view(B, L, -1)

#         return self.out_projection(out), attn