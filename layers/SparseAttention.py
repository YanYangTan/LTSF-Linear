import torch
import torch.nn as nn

import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask


class SparseAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(SparseAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        mask_line = torch.ones(2 * L + 1)
        stride = 2

        while (stride <= L):
            mask_line[L - stride] = 0
            mask_line[L + stride] = 0
            stride = stride * 2
        value_mask = torch.unsqueeze(mask_line[L:2 * L], dim=0)
        for i in range(1, L):
            value_mask = torch.cat([value_mask, torch.unsqueeze(mask_line[L - i:2 * L - i], dim=0)], dim=0)
        value_mask = value_mask.view(1, 1, L, S).repeat(B, H, 1, 1).cuda()

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * value_mask

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