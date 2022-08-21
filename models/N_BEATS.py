import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.blocks import GenericBasis, TrendBasis, SeasonalityBasis, NBeatsBlock

class NBeats(torch.nn.Module):
    """
    N-Beats Model.
    """

    def __init__(self, blocks: torch.nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        residuals = x
        forecast = None
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast)
            if forecast is None:
                forecast = block_forecast
            else:
                forecast += block_forecast
        return forecast


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        block_dict = {
            'g': GenericBasis,
            't': TrendBasis,
            's': SeasonalityBasis,
        }
        self.enc_in = configs.enc_in
        self.block = block_dict[configs.block_type]
        self.model = NBeats(torch.nn.ModuleList([NBeatsBlock(input_size=configs.seq_len,
                                                             theta_size=configs.seq_len + configs.pred_len,
                                                             basis_function=self.block(backcast_size=configs.seq_len,
                                                                                         forecast_size=configs.pred_len),
                                                             layers=configs.e_layers,
                                                             layer_size=configs.d_ff)
                                                 for _ in range(configs.d_layers)]))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        res = []
        for i in range(self.enc_in):
            dec_out = self.model(x_enc[:,:,i])
            res.append(dec_out)
        return torch.stack(res, dim=-1)
