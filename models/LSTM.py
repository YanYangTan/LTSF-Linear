import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Embed import *
import numpy as np
from math import sqrt


class Model(nn.Module):
    """Applying LSTM to time series predictable through time-window approaches

        @inproceedings{DBLP:conf/icann/GersES01,
          author    = {Felix A. Gers and
                       Douglas Eck and
                       J{\"{u}}rgen Schmidhuber},
          editor    = {Georg Dorffner and
                       Horst Bischof and
                       Kurt Hornik},
          title     = {Applying {LSTM} to Time Series Predictable through Time-Window Approaches},
          booktitle = {Artificial Neural Networks - {ICANN} 2001, International Conference
                       Vienna, Austria, August 21-25, 2001 Proceedings},
          series    = {Lecture Notes in Computer Science},
          volume    = {2130},
          pages     = {669--676},
          publisher = {Springer},
          year      = {2001},
          url       = {https://doi.org/10.1007/3-540-44668-0\_93},
          doi       = {10.1007/3-540-44668-0\_93},
          timestamp = {Tue, 14 May 2019 10:00:49 +0200},
          biburl    = {https://dblp.org/rec/conf/icann/GersES01.bib},
          bibsource = {dblp computer science bibliography, https://dblp.org}
        }

    Inputs:
        - x_enc (tensor): batch_size * backcast_length * c_in        past observation data
        - x_mark_enc (tensor): batch_size * backcast_length * 4      past observation  time
        - x_mark_dec (tensor): batch_size * period_to_forecast * 4   the timestamp of the data to be predicted

    Outputs:
       -output (tensor):batch_size * period_to_forecast * c_in     the data to be predicted

    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Encoding
        self.embedding = CustomEmbedding_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.lstm = nn.LSTM(input_size=configs.d_model,
                            hidden_size=configs.d_model,
                            num_layers=configs.n_layer,
                            bias=True,
                            batch_first=True,
                            dropout=configs.dropout)
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)
        self.dense = nn.Linear(in_features=configs.d_model, out_features=configs.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        dec_inp = torch.zeros([x_enc.shape[0], self.pred_len + self.label_len, x_enc.shape[-1]]).float().cuda()
        input = self.embedding(torch.cat([x_enc,dec_inp],dim=1), torch.cat([x_mark_enc,x_mark_dec],dim=1))


        output, (hidden, cell) = self.lstm(input)
        output = self.dense(output)
        output=output[:,-1-self.pred_len:-1,:]

        return output
