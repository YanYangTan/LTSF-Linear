import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Embed import *
import numpy as np
from math import sqrt


class Model(nn.Module):
    """Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (AAAI'21 Best Paper)

        @inproceedings{haoyietal-informer-2021,
          author    = {Haoyi Zhou and
                       Shanghang Zhang and
                       Jieqi Peng and
                       Shuai Zhang and
                       Jianxin Li and
                       Hui Xiong and
                       Wancai Zhang},
          title     = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
          booktitle = {The Thirty-Fifth {AAAI} Conference on Artificial Intelligence, {AAAI} 2021, Virtual Conference},
          volume    = {35},
          number    = {12},
          pages     = {11106--11115},
          publisher = {{AAAI} Press},
          year      = {2021},
        }

    Inputs:
        - x_enc (tensor): batch_size * backcast_length * c_in        past observation data
        - x_mark_enc (tensor): batch_size * backcast_length * 4      past observation  time
        - x_mark_dec (tensor): batch_size * period_to_forecast * 4   the timestamp of the data to be predicted

    Outputs:
       - 

    """

    def __init__(self, configs):
        super(Model, self).__init__()
        #  d_model=512, layers=3, d_ff=512, dropout=0.0, embed='fixed', freq='h', activation='gelu', distil=True, mix=True
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Encoding
        self.enc_embedding = CustomEmbedding_temp(configs.enc_in*2, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.lstm = nn.LSTM(input_size=configs.d_model,
                            hidden_size=configs.d_model,
                            num_layers=configs.n_layer,
                            bias=True,
                            batch_first=False,
                            dropout=configs.dropout)
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)
        self.relu = nn.ReLU()
        self.distribution_mu = nn.Linear(configs.d_model * configs.n_layer, configs.enc_in)
        self.distribution_presigma = nn.Linear(configs.d_model * configs.n_layer, configs.enc_in)
        self.distribution_sigma = nn.Softplus()

    def forward(self, x_enc, x_mark_enc, hidden, cell):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = enc_out.permute(1,0,2)
        output, (hidden, cell) = self.lstm(enc_out, (hidden, cell))
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        pre_sigma = self.distribution_presigma(hidden_permute)
        mu = self.distribution_mu(hidden_permute)
        sigma = self.distribution_sigma(pre_sigma)
        return mu, sigma, hidden, cell
