import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple


class GenericBasis(nn.Module):
    """
    Generic basis function.
    """

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: torch.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(self,
                 input_size,
                 theta_size: int,
                 basis_function: nn.Module,
                 layers: int,
                 layer_size: int):
        """
        N-BEATS block.
        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features=input_size, out_features=layer_size)] +
                                    [nn.Linear(in_features=layer_size, out_features=layer_size)
                                     for _ in range(layers - 1)])
        self.basis_parameters = nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = torch.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


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
        self.enc_in = configs.enc_in
        self.model = NBeats(torch.nn.ModuleList([NBeatsBlock(input_size=configs.seq_len,
                                                             theta_size=configs.seq_len + configs.pred_len,
                                                             basis_function=GenericBasis(backcast_size=configs.seq_len,
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
