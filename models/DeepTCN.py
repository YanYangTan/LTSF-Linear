import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Embed import *
import numpy as np
from math import sqrt
import torch.nn as nn


class ResidualTCN(nn.Module):
    def __init__(self, d, n_residue=35, k=2, **kwargs):
        super(ResidualTCN, self).__init__(**kwargs)
        self.conv1 = nn.Conv1d(in_channels=n_residue,out_channels=n_residue,kernel_size=k,dilation=d) #nn.Conv1D(in_channels=n_residue, channels=n_residue, kernel_size=k, dilation=d)
        self.bn1 = nn.BatchNorm1d(n_residue)#nn.BatchNorm()
        self.conv2 = nn.Conv1d(in_channels=n_residue,out_channels=n_residue,kernel_size=k,dilation=d)#nn.Conv1D(in_channels=n_residue, channels=n_residue, kernel_size=k, dilation=d)
        self.bn2 = nn.BatchNorm1d(n_residue)#nn.BatchNorm()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0,2,1)

        out = self.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))
        result = self.relu(out + x[:, :, -out.shape[2]:])
        result = result.permute(0,2,1)
        return result


class futureResidual(nn.Module):
    def __init__(self, xDim=64, **kwargs):
        super(futureResidual, self).__init__(**kwargs)
        self.fc1 = nn.Linear(in_features=xDim,out_features=xDim)#nn.Dense(xDim, flatten=False)
        self.bn1 = nn.BatchNorm1d(xDim)#nn.BatchNorm(axis=2)
        self.fc2 = nn.Linear(in_features=xDim,out_features=xDim)#nn.Dense(units=xDim, flatten=False)
        self.bn2 = nn.BatchNorm1d(xDim)#nn.BatchNorm(axis=2)
        self.relu = nn.ReLU()

    def forward(self,enc_out,dec_inp):

        out = self.fc1(dec_inp).permute(0,2,1)

        out = self.relu(self.bn1(out)).permute(0,2,1)
        out = self.fc2(out).permute(0,2,1)
        out = self.bn2(out).permute(0,2,1)

        return self.relu(torch.cat([enc_out, out], dim=2))




class Model(nn.Module):
    """Probabilistic Forecasting with Temporal Convolutional Neural Network

        @article{DBLP:journals/ijon/ChenKCW20,
          author    = {Yitian Chen and
                       Yanfei Kang and
                       Yixiong Chen and
                       Zizhuo Wang},
          title     = {Probabilistic forecasting with temporal convolutional neural network},
          journal   = {Neurocomputing},
          volume    = {399},
          pages     = {491--501},
          year      = {2020},
          url       = {https://doi.org/10.1016/j.neucom.2020.03.011},
          doi       = {10.1016/j.neucom.2020.03.011},
          timestamp = {Mon, 15 Jun 2020 16:52:59 +0200},
          biburl    = {https://dblp.org/rec/journals/ijon/ChenKCW20.bib},
          bibsource = {dblp computer science bibliography, https://dblp.org}
        }

    Inputs:
        - x_enc (tensor): batch_size * backcast_length * c_in        past observation data
        - x_mark_enc (tensor): batch_size * backcast_length * 4      past observation  time
        - x_mark_dec (tensor): batch_size * period_to_forecast * 4   the timestamp of the data to be predicted

    Outputs:
       - output (tensor):batch_size * period_to_forecast * c_in     the data to be predicted

    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.c_in = configs.enc_in
        self.encoder = nn.ModuleList()
        self.outputLayer = nn.Sequential()
        for d in configs.dilations:
            self.encoder.append(ResidualTCN(d=d, n_residue=configs.d_model))
        self.enc_embedding = CustomEmbedding_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.dec_embedding = CustomEmbedding_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        self.decoder = (futureResidual(xDim=configs.d_model))
        self.dense = nn.Linear(in_features=2*configs.d_model,out_features=configs.enc_in)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        inputSeries = self.enc_embedding(x_enc, x_mark_enc)
        #inputSeries = inputSeries.permute(1,0,2)
        for subTCN in self.encoder:
            inputSeries = subTCN(inputSeries)
        inputSeries = torch.mean(input=inputSeries,dim=1)
        enc_out=torch.unsqueeze(inputSeries,dim=1).repeat(1,x_mark_dec.shape[1],1)
        dec_inp = torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[-1]]).float().cuda()
        dec_inp = self.dec_embedding(dec_inp, x_mark_dec)
        output = self.decoder(enc_out,dec_inp)
        output = self.dense(output)
        return output


