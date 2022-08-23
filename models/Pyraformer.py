import torch, math
import torch.nn as nn
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos,DataEmbedding_wo_pos_temp,DataEmbedding_wo_temp

from layers.Pyraformer_Layers import Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct
from layers.Pyraformer_Layers import EncoderLayer
from layers.Embed import DataEmbedding


def get_mask(input_size, window_size, inner_size, device):
    """Get the attention mask of PAM-Naive"""
    # Get the size of all layers
    all_size = []
    all_size.append(input_size)
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)

    seq_length = sum(all_size)
    mask = torch.zeros(seq_length, seq_length, device=device)

    # get intra-scale mask
    inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = max(i - inner_window, start)
            right_side = min(i + inner_window + 1, start + all_size[layer_idx])
            mask[i, left_side:right_side] = 1

    # get inter-scale mask
    for layer_idx in range(1, len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = (start - all_size[layer_idx - 1]) + (i - start) * window_size[layer_idx - 1]
            if i == ( start + all_size[layer_idx] - 1):
                right_side = start
            else:
                right_side = (start - all_size[layer_idx - 1]) + (i - start + 1) * window_size[layer_idx - 1]
            mask[i, left_side:right_side] = 1
            mask[left_side:right_side, i] = 1

    mask = (1 - mask).bool()

    return mask, all_size


def refer_points(all_sizes, window_size, device):
    """Gather features from PAM's pyramid sequences"""
    input_size = all_sizes[0]
    indexes = torch.zeros(input_size, len(all_sizes), device=device)

    for i in range(input_size):
        indexes[i][0] = i
        former_index = i
        for j in range(1, len(all_sizes)):
            start = sum(all_sizes[:j])
            inner_layer_idx = former_index - (start - all_sizes[j - 1])
            former_index = start + min(inner_layer_idx // window_size[j - 1], all_sizes[j] - 1)
            indexes[i][j] = former_index

    indexes = indexes.unsqueeze(0).unsqueeze(3)

    return indexes.long()


def get_subsequent_mask(input_size, window_size, predict_step, truncate):
    """Get causal attention mask for decoder."""
    if truncate:
        mask = torch.zeros(predict_step, input_size + predict_step)
        for i in range(predict_step):
            mask[i][:input_size+i+1] = 1
        mask = (1 - mask).bool().unsqueeze(0)
    else:
        all_size = []
        all_size.append(input_size)
        for i in range(len(window_size)):
            layer_size = math.floor(all_size[i] / window_size[i])
            all_size.append(layer_size)
        all_size = sum(all_size)
        mask = torch.zeros(predict_step, all_size + predict_step)
        for i in range(predict_step):
            mask[i][:all_size+i+1] = 1
        mask = (1 - mask).bool().unsqueeze(0)

    return mask


def get_q_k(input_size, window_size, stride, device):
    """
    Get the index of the key that a given query needs to attend to.
    """
    second_length = input_size // stride
    second_last = input_size - (second_length - 1) * stride
    third_start = input_size + second_length
    third_length = second_length // stride
    third_last = second_length - (third_length - 1) * stride
    max_attn = max(second_last, third_last)
    fourth_start = third_start + third_length
    fourth_length = third_length // stride
    full_length = fourth_start + fourth_length
    fourth_last = third_length - (fourth_length - 1) * stride
    max_attn = max(third_last, fourth_last)

    max_attn += window_size + 1
    mask = torch.zeros(full_length, max_attn, dtype=torch.int32, device=device) - 1

    for i in range(input_size):
        mask[i, 0:window_size] = i + torch.arange(window_size) - window_size // 2
        mask[i, mask[i] > input_size - 1] = -1

        mask[i, -1] = i // stride + input_size
        mask[i][mask[i] > third_start - 1] = third_start - 1
    for i in range(second_length):
        mask[input_size+i, 0:window_size] = input_size + i + torch.arange(window_size) - window_size // 2
        mask[input_size+i, mask[input_size+i] < input_size] = -1
        mask[input_size+i, mask[input_size+i] > third_start - 1] = -1

        if i < second_length - 1:
            mask[input_size+i, window_size:(window_size+stride)] = torch.arange(stride) + i * stride
        else:
            mask[input_size+i, window_size:(window_size+second_last)] = torch.arange(second_last) + i * stride

        mask[input_size+i, -1] = i // stride + third_start
        mask[input_size+i, mask[input_size+i] > fourth_start - 1] = fourth_start - 1
    for i in range(third_length):
        mask[third_start+i, 0:window_size] = third_start + i + torch.arange(window_size) - window_size // 2
        mask[third_start+i, mask[third_start+i] < third_start] = -1
        mask[third_start+i, mask[third_start+i] > fourth_start - 1] = -1

        if i < third_length - 1:
            mask[third_start+i, window_size:(window_size+stride)] = input_size + torch.arange(stride) + i * stride
        else:
            mask[third_start+i, window_size:(window_size+third_last)] = input_size + torch.arange(third_last) + i * stride

        mask[third_start+i, -1] = i // stride + fourth_start
        mask[third_start+i, mask[third_start+i] > full_length - 1] = full_length - 1
    for i in range(fourth_length):
        mask[fourth_start+i, 0:window_size] = fourth_start + i + torch.arange(window_size) - window_size // 2
        mask[fourth_start+i, mask[fourth_start+i] < fourth_start] = -1
        mask[fourth_start+i, mask[fourth_start+i] > full_length - 1] = -1

        if i < fourth_length - 1:
            mask[fourth_start+i, window_size:(window_size+stride)] = third_start + torch.arange(stride) + i * stride
        else:
            mask[fourth_start+i, window_size:(window_size+fourth_last)] = third_start + torch.arange(fourth_last) + i * stride

    return mask


def get_k_q(q_k_mask):
    """
    Get the index of the query that can attend to the given key.
    """
    k_q_mask = q_k_mask.clone()
    for i in range(len(q_k_mask)):
        for j in range(len(q_k_mask[0])):
            if q_k_mask[i, j] >= 0:
                k_q_mask[i, j] = torch.where(q_k_mask[q_k_mask[i, j]] ==i )[0]

    return k_q_mask


class Model(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device('cpu')
            
        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        
        self.mask, self.all_size = get_mask(configs.seq_len, configs.window_size, configs.factor, device)
        self.indexes = refer_points(self.all_size, configs.window_size, device)

        self.layers = nn.ModuleList([
            EncoderLayer(configs.d_model, 
                         configs.d_ff, 
                         configs.n_heads, 
                         configs.d_k, 
                         configs.d_v, 
                         dropout=configs.dropout,
                normalize_before=False) for i in range(configs.e_layers)
            ])

        self.conv_layers = eval(configs.CSCM)(configs.d_model, configs.window_size, configs.d_bottleneck)

        self.projection = nn.Linear(4*configs.d_model, configs.pred_len * configs.c_out, bias=False)


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
        enc_output = self.projection(enc_output).view(enc_output.size(0), self.pred_len, -1)

        return enc_output


