import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.StockBlockLayer import StockBlockLayer

class Model(nn.Module):
    """Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting

        @inproceedings{DBLP:conf/nips/CaoWDZZHTXBTZ20,
          author    = {Defu Cao and
                       Yujing Wang and
                       Juanyong Duan and
                       Ce Zhang and
                       Xia Zhu and
                       Congrui Huang and
                       Yunhai Tong and
                       Bixiong Xu and
                       Jing Bai and
                       Jie Tong and
                       Qi Zhang},
          editor    = {Hugo Larochelle and
                       Marc'Aurelio Ranzato and
                       Raia Hadsell and
                       Maria{-}Florina Balcan and
                       Hsuan{-}Tien Lin},
          title     = {Spectral Temporal Graph Neural Network for Multivariate Time-series
                       Forecasting},
          booktitle = {Advances in Neural Information Processing Systems 33: Annual Conference
                       on Neural Information Processing Systems 2020, NeurIPS 2020, December
                       6-12, 2020, virtual},
          year      = {2020},
          url       = {https://proceedings.neurips.cc/paper/2020/hash/cdf6581cb7aca4b7e19ef136c6e601a5-Abstract.html},
          timestamp = {Fri, 22 Jan 2021 13:30:41 +0100},
          biburl    = {https://dblp.org/rec/conf/nips/CaoWDZZHTXBTZ20.bib},
          bibsource = {dblp computer science bibliography, https://dblp.org}
        }

    Inputs:
        - x (tensor): batch_size * backcast_length * c_in        past observation data


    Outputs:
       - forecast (tensor):batch_size * period_to_forecast * c_in     the data to be predicted

    """

    def __init__(self,configs):
        super(Model, self).__init__()
        # default : stack_cnt=2, multi_layer=5, dropout_rate=0.5,leaky_rate=0.2
        self.unit = configs.enc_in
        self.stack_cnt = configs.stack_cnt
        self.alpha = configs.leaky_rate
        self.time_step = configs.seq_len
        self.horizon = configs.pred_len
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.GRU = nn.GRU(self.time_step, self.unit)
        self.multi_layer = configs.multi_layer
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [StockBlockLayer(self.time_step, self.unit, self.multi_layer, stack_cnt=i) for i in range(self.stack_cnt)])
        self.fc = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), self.horizon),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=configs.dropout)

    def get_laplacian(self, graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)

        return multi_order_laplacian  # 4 * c_in * c_in

    def latent_correlation_layer(self, x):
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)  # batch_size * c_in * c_in
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))
        mul_L = self.cheb_polynomial(laplacian)
        return mul_L, attention

    def self_graph_attention(self, input):
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc  B*L*c_in
        mul_L, attention = self.latent_correlation_layer(x_enc)

        X = x_enc.unsqueeze(1).permute(0, 1, 3, 2).contiguous()  # B*1*c_in * L

        result = []
        for stack_i in range(self.stack_cnt):
            forecast, X = self.stock_block[stack_i](X, mul_L)
            result.append(forecast)
        forecast = result[0] + result[1]  # B c_in L

        forecast = self.fc(forecast).permute(0, 2, 1)  # B c_in output_len
        return forecast
