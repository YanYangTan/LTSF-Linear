import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Embed import *
import numpy as np
from math import sqrt

class Model(nn.Module):
    """Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks

        @inproceedings{DBLP:conf/sigir/LaiCYL18,
          author    = {Guokun Lai and
                       Wei{-}Cheng Chang and
                       Yiming Yang and
                       Hanxiao Liu},
          editor    = {Kevyn Collins{-}Thompson and
                       Qiaozhu Mei and
                       Brian D. Davison and
                       Yiqun Liu and
                       Emine Yilmaz},
          title     = {Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks},
          booktitle = {The 41st International {ACM} {SIGIR} Conference on Research {\&}
                       Development in Information Retrieval, {SIGIR} 2018, Ann Arbor, MI,
                       USA, July 08-12, 2018},
          pages     = {95--104},
          publisher = {{ACM}},
          year      = {2018},
          url       = {https://doi.org/10.1145/3209978.3210006},
          doi       = {10.1145/3209978.3210006},
          timestamp = {Wed, 16 Sep 2020 13:34:22 +0200},
          biburl    = {https://dblp.org/rec/conf/sigir/LaiCYL18.bib},
          bibsource = {dblp computer science bibliography, https://dblp.org}
        }


    Inputs:
        - x_enc (tensor): batch_size * backcast_length * c_in        past observation data
        - x_mark_enc (tensor): batch_size * backcast_length * 4      past observation  time
        - x_mark_dec (tensor): batch_size * period_to_forecast * 4   the timestamp of the data to be predicted

    Outputs:
       - output (tensor):batch_size * 1 * c_in     the data to be predicted

    """

    def __init__(self, configs):
        # d_model=512, layers=3,kernel_size=5,skip=24,high_way=24,dropout=0.0, embed='fixed', freq='h'
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.P = self.seq_len
        self.d_model = configs.d_model
        self.layers = configs.n_layer
        self.Ck = configs.kernel_size
        self.skip = configs.skip
        self.hw = configs.high_way
        self.pt = int((self.seq_len - self.Ck) / self.skip)

        self.embedding = CustomEmbedding_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.conv1 = nn.Conv2d(1, self.d_model, kernel_size=(self.Ck, self.d_model))

        self.GRU1 = nn.GRU(self.d_model, self.d_model)
        self.GRUskip = nn.GRU(self.d_model, self.d_model)
        self.linear1 = nn.Linear(self.d_model + self.skip * self.d_model, self.d_model)
        self.highway = nn.Linear(self.hw, 1)


        self.dropout = nn.Dropout(p=configs.dropout)
        self.dense = nn.Linear(in_features=configs.d_model, out_features=configs.enc_in)

    def forward(self, x_enc, x_mark_enc):

        x = self.embedding(x_enc, x_mark_enc)

        batch_size = x.size(0)
        # CNN
        c = x.view(-1, 1, self.P, self.d_model)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn

        if (self.skip > 0):
            s = c[:, :, int(-self.pt * self.skip):].contiguous()
            s = s.view(batch_size, self.d_model, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.d_model)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.d_model)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if (self.hw > 0):
            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.d_model)
            res = res + z

        output = self.dense(torch.unsqueeze(res, dim=1))

        return output
