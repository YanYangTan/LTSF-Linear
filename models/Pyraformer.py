from distutils.command.config import config
import torch
import torch.nn as nn
from layers.Pyraformer_Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
from layers.Pyraformer_Layers import EncoderLayer, Predictor
from layers.Pyraformer_Layers import get_mask, refer_points
from layers.Embed import DataEmbedding



class Model(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device('cpu')
            
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.dropout)

        self.mask, self.all_size = get_mask(configs.seq_len, configs.window_size, configs.factor, device)
        self.indexes = refer_points(self.all_size, configs.window_size, device)

        self.layers = nn.ModuleList([
            EncoderLayer(configs.d_model, configs.d_ff, configs.n_heads, configs.d_k, configs.d_v, dropout=configs.dropout, \
                normalize_before=False) for i in range(configs.e_layers)
            ])

        self.conv_layers = eval(configs.CSCM)(configs.d_model, configs.window_size, configs.d_bottleneck)

        self.predictor = Predictor(4 * configs.d_model, configs.pred_len * configs.enc_in)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
            enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """
        """
        seq_enc = self.enc_embedding(x_enc, x_mark_enc)

        mask = self.mask.repeat(len(seq_enc), 1, 1).to(x_enc.device)
        seq_enc = self.conv_layers(seq_enc)

        for i in range(len(self.layers)):
            seq_enc, _ = self.layers[i](seq_enc, mask)

        indexes = self.indexes.repeat(seq_enc.size(0), 1, 1, seq_enc.size(2)).to(seq_enc.device)
        indexes = indexes.view(seq_enc.size(0), -1, seq_enc.size(2))
        all_enc = torch.gather(seq_enc, 1, indexes)
        seq_enc = all_enc.view(seq_enc.size(0), self.all_size[0], -1)

        enc_output = seq_enc[:, -1, :]
        enc_output = self.predictor(enc_output).view(enc_output.size(0), self.pred_len, -1)

        return enc_output
