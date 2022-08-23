from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DeepAR
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from utils.loss import gaussian_log_likehood
import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_HiddenCell(Exp_Basic):
    def __init__(self, args):
        super(Exp_HiddenCell, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'DeepAR':DeepAR
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        self.modelShortY_list=[
            'DeepAR'
        ]
        return model

    def _get_data(self, flag):
        if self.args.model in  self.modelShortY_list:
            data_set, data_loader = data_provider(self.args, flag, ShortY=True)
        else:
            data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = gaussian_log_likehood
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        val_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            loss = torch.zeros(1, device=self.device)
            hidden = self.init_hidden(batch_x.shape[0])
            cell = self.init_cell(batch_x.shape[0])
            for t in range(self.args.seq_len - 1):
                if (t == 0):
                    zeros_ = torch.zeros_like(batch_x[:, 0, :]).to(self.device)
                    mu, sigma, hidden, cell = self.model(
                        torch.cat([zeros_, batch_x[:, t, :]], dim=1).unsqueeze_(1).clone(), batch_x_mark[:, t:t + 1, :],
                        hidden, cell)

                else:
                    mu, sigma, hidden, cell = self.model(torch.cat([mu, batch_x[:, t, :]], dim=1).unsqueeze_(1).clone(),
                                                         batch_x_mark[:, t:t + 1, :], hidden,
                                                         cell)

                loss += criterion(mu, sigma, batch_x[:, t + 1, :])

            val_loss.append(loss.item())
        total_loss = np.average(val_loss)
        self.model.train()
        return total_loss


    def init_hidden(self, input_size, layers=3, hidden_dim=512):
        return torch.zeros(layers, input_size, hidden_dim, device=self.device)

    def init_cell(self, input_size, layers=3, hidden_dim=512):
        return torch.zeros(layers, input_size, hidden_dim, device=self.device)

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                loss = torch.zeros(1, device=self.device)
                hidden = self.init_hidden(batch_x.shape[0])
                cell = self.init_cell(batch_x.shape[0])

                for t in range(self.args.seq_len- 1):
                    # if z_t is missing, replace it by output mu from the last time step
                    if (t == 0):
                        zeros_ = torch.zeros_like(batch_x[:, 0, :]).to(self.device)
                        mu, sigma, hidden, cell = self.model(
                            torch.cat([zeros_, batch_x[:, t, :]], dim=1).unsqueeze_(1).clone(), batch_x_mark[:, t:t + 1, :],
                            hidden, cell)

                    else:
                        mu, sigma, hidden, cell = self.model(
                            torch.cat([mu, batch_x[:, t, :]], dim=1).unsqueeze_(1).clone(), batch_x_mark[:, t:t + 1, :],
                            hidden,
                            cell)

                    loss += criterion(mu, sigma, batch_x[:, t + 1, :])
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                hidden = self.init_hidden(batch_x.shape[0])
                cell = self.init_cell(batch_x.shape[0])
                for t in range(self.args.seq_len - 1):
                    if (t == 0):
                        zeros_ = torch.zeros_like(batch_x[:, 0, :]).to(self.device)
                        mu, sigma, hidden, cell = self.model(
                            torch.cat([zeros_, batch_x[:, t, :]], dim=1).unsqueeze_(1).clone(),
                            batch_x_mark[:, t:t + 1, :],
                            hidden, cell)

                    else:
                        mu, sigma, hidden, cell = self.model(
                            torch.cat([mu, batch_x[:, t, :]], dim=1).unsqueeze_(1).clone(),
                            batch_x_mark[:, t:t + 1, :], hidden,
                            cell)
                gaussian = torch.distributions.normal.Normal(mu, sigma)
                pred = gaussian.sample()  # not scaled
                samples = torch.zeros(batch_x.shape[0], self.args.pred_len, batch_x.shape[-1],
                                      device=self.device)
                mus = torch.zeros(batch_x.shape[0], self.args.pred_len, batch_x.shape[-1],
                                  device=self.device)
                sigmas = torch.zeros(batch_x.shape[0], self.args.pred_len, batch_x.shape[-1],
                                     device=self.device)
                samples[:, 0, :] = pred
                mus[:, 0, :] = mu
                sigmas[:, 0, :] = sigma
                for t in range(self.args.pred_len - 1):
                    mu, sigma, hidden, cell = self.model(
                        torch.cat([mu, pred], dim=1).unsqueeze_(1).clone(),
                        batch_x_mark[:, t:t + 1, :], hidden,
                        cell)
                    gaussian = torch.distributions.normal.Normal(mu, sigma)
                    pred = gaussian.sample()  # not scaled
                    samples[:, t + 1, :] = pred
                    mus[:, t + 1, :] = mu
                    sigmas[:, t + 1, :] = sigma

                outputs = samples.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # preds = np.concatenate(preds, axis=0)
        # trues = np.concatenate(trues, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                hidden = self.init_hidden(batch_x.shape[0])
                cell = self.init_cell(batch_x.shape[0])
                for t in range(self.args.seq_len - 1):

                    if (t == 0):
                        zeros_ = torch.zeros_like(batch_x[:, 0, :]).to(self.device)
                        mu, sigma, hidden, cell = self.model(
                            torch.cat([zeros_, batch_x[:, t, :]], dim=1).unsqueeze_(1).clone(),
                            batch_x_mark[:, t:t + 1, :],
                            hidden, cell)

                    else:
                        mu, sigma, hidden, cell = self.model(
                            torch.cat([mu, batch_x[:, t, :]], dim=1).unsqueeze_(1).clone(),
                            batch_x_mark[:, t:t + 1, :], hidden,
                            cell)

                gaussian = torch.distributions.normal.Normal(mu, sigma)
                pred = gaussian.sample()  # not scaled
                samples = torch.zeros(batch_x.shape[0], self.args.pred_len, batch_x.shape[-1],
                                      device=self.device)
                mus = torch.zeros(batch_x.shape[0], self.args.pred_len, batch_x.shape[-1],
                                  device=self.device)
                sigmas = torch.zeros(batch_x.shape[0], self.args.pred_len, batch_x.shape[-1],
                                     device=self.device)
                samples[:, 0, :] = pred
                mus[:, 0, :] = mu
                sigmas[:, 0, :] = sigma
                for t in range(self.args.pred_len - 1):
                    mu, sigma, hidden, cell = self.model(
                        torch.cat([mu, pred], dim=1).unsqueeze_(1).clone(),
                        batch_y_mark[:, t:t + 1, :], hidden,
                        cell)
                    gaussian = torch.distributions.normal.Normal(mu, sigma)
                    pred = gaussian.sample()  # not scaled
                    samples[:, t + 1, :] = pred
                    mus[:, t + 1, :] = mu
                    sigmas[:, t + 1, :] = sigma
                preds.append(samples[0].detach().cpu().numpy())
        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return