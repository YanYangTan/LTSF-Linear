import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.Embed import *
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask
from layers.SparseAttention import SparseAttention
from layers.SelfAttention import FullAttention,AttentionLayer
from layers.Transformer_EncDec import *

# class EncoderStack(nn.Module):
#     def __init__(self, encoders, inp_lens):
#         super(EncoderStack, self).__init__()
#         self.encoders = nn.ModuleList(encoders)
#         self.inp_lens = inp_lens

#     def forward(self, x, attn_mask=None):
#         # x [B, L, D]
#         x_stack = [];
#         attns = []
#         for i_len, encoder in zip(self.inp_lens, self.encoders):
#             inp_len = x.shape[1] // (2 ** i_len)
#             x_s, attn = encoder(x[:, -inp_len:, :])
#             x_stack.append(x_s);
#             attns.append(attn)
#         x_stack = torch.cat(x_stack, -2)

#         return x_stack, attns


class Model(nn.Module):
    """Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting

        @inproceedings{DBLP:conf/nips/LiJXZCWY19,
          author    = {Shiyang Li and
                       Xiaoyong Jin and
                       Yao Xuan and
                       Xiyou Zhou and
                       Wenhu Chen and
                       Yu{-}Xiang Wang and
                       Xifeng Yan},
          editor    = {Hanna M. Wallach and
                       Hugo Larochelle and
                       Alina Beygelzimer and
                       Florence d'Alch{\'{e}}{-}Buc and
                       Emily B. Fox and
                       Roman Garnett},
          title     = {Enhancing the Locality and Breaking the Memory Bottleneck of Transformer
                       on Time Series Forecasting},
          booktitle = {Advances in Neural Information Processing Systems 32: Annual Conference
                       on Neural Information Processing Systems 2019, NeurIPS 2019, December
                       8-14, 2019, Vancouver, BC, Canada},
          pages     = {5244--5254},
          year      = {2019},
          url       = {https://proceedings.neurips.cc/paper/2019/hash/6775a0635c302542da2c32aa19d86be0-Abstract.html},
          timestamp = {Tue, 28 Sep 2021 10:22:48 +0200},
          biburl    = {https://dblp.org/rec/conf/nips/LiJXZCWY19.bib},
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
        self.label_len = configs.label_len 
        self.pred_len = configs.pred_len

        # Encoding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(SparseAttention(False, attention_dropout=configs.dropout, output_attention=False),
                                   configs.d_model, configs.n_heads, mix=False),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],

            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(SparseAttention(True, attention_dropout=configs.dropout, output_attention=False),
                                   configs.d_model, configs.n_heads, mix=configs.mix),
                    AttentionLayer(FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
                                   configs.d_model,configs.n_heads, mix=False),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.projection = nn.Linear(configs.d_model, configs.enc_in, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # dec_inp = torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[-1]]).float().cuda()
        # dec_inp = torch.cat([x_enc[:, -self.label_len:, :], dec_inp], dim=1).float().cuda()
        # x_dec = dec_inp
        # x_mark_dec = torch.cat([x_mark_enc[:, -self.label_len:, :], x_mark_dec], dim=1)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        output = self.projection(dec_out)[:, -self.pred_len:, :]

        return output
