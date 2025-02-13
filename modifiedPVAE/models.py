# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Sequence, List
# 假设你已经有了以下这些类 (可以从之前的代码中复制过来)
from base.common import Conv2D, Linear, ResDenseLayer, get_act_fn, FactorizedReduce, Cell, DeConv2D
from base.distributions import Poisson
from main.vae import PoissonVAE, _build_conv_enc
from base.distributions import (
	dists, softclamp, softclamp_upper,
	Normal, Laplace, Poisson, Categorical,
)
import random
class ModifiedPoissonVAE(PoissonVAE): 
    def __init__(self, cfg, **kwargs):
        super(ModifiedPoissonVAE, self).__init__(cfg, **kwargs) 
        self.rnn_decoder = nn.LSTM(
            input_size=100,  # 每个时间步的输入是单个 spike (0 或 1)
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
        )
        self.cnn = nn.Sequential(
         nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1),  
         nn.ReLU(inplace=True),
         nn.ConvTranspose2d(2, 1, kernel_size=4, stride=2, padding=1),  
        #  nn.ReLU(inplace=True),
        #  nn.Conv2d(2, 1, kernel_size=3, padding=1),  # 输出通道数为 1 (灰度图像)
     )
        self.fc_out = nn.Linear(cfg.hidden_size, cfg.input_sz ** 2)
        self.shape = (-1, 1, cfg.input_sz, cfg.input_sz)
        # 使用 sigmoid 函数，将输出归一化到 0-1 之间。
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, spike_data):
        # 1. 编码图像，获取预测的 spike count (z_pred)
        dist, log_dr = self.infer(x)  # 使用父类的 infer 方法, dist 是一个 Poisson 分布
        z_pred = dist.mean  # (batch_size, n_latents), n_latents = neurons_nums

        # 2. 根据 z_pred 动态截取 spike_data
        z_pred = z_pred.long()  # 转换为整数
        truncated_spikes = []
        for i in range(z_pred.size(0)):  # 遍历批次中的每个样本
            truncated_sample = []
            for j in range(z_pred.size(1)):  # 遍历每个神经元
                count = 0
                truncated_neuron = []
                for k in range(spike_data.size(2)):  # 遍历 spike 序列  spike_data: (batch,  n_neurons, seq_len)
                    # 修改判断条件：
                    if count < z_pred[i, j] or len(truncated_neuron) == 0:
                        if spike_data[i,j,k] > 0:
                            count += 1
                        truncated_neuron.append(spike_data[i, j, k].unsqueeze(0))
                    else:
                        break
                # 截断后的数据使用 stack 拼接
                truncated_sample.append(torch.cat(truncated_neuron)) # 堆叠
            truncated_spikes.append(torch.stack(truncated_sample))

        # 3. 使用 RNN 解码器
        y = self.decode_all_CNN(truncated_spikes)
        return y, z_pred, dist, log_dr  # 返回重建图像和预测的 spike counts

    # spike 值一个一个输入，需要把 LSTM 初始化中的input_size设置为 1
    def decode(self, z):
        # z: list[Tensor(n_neurons, seq_len)]
        batch_size = len(z)
        hidden = None
        outputs = []
        # 初始化 RNN 的隐藏状态 (如果需要)
        for i in range(batch_size):
            h0 = torch.zeros(self.rnn_decoder.num_layers, 1, self.rnn_decoder.hidden_size).to(z[i].device)
            c0 = torch.zeros(self.rnn_decoder.num_layers, 1, self.rnn_decoder.hidden_size).to(z[i].device)
            hidden = (h0, c0)
            neuron_outputs = []
            for j in range(z[i].size(0)):  # batch 中每一个样本的每一个神经元进行循环
                # z[i][j]: (seq_len,) -> (seq_len, 1) -> (1, seq_len, 1)
                rnn_out, hidden = self.rnn_decoder(z[i][j].float().unsqueeze(-1).unsqueeze(0), hidden)
                # rnn_out: (1, seq_len, hidden_size)
                # 取最后一个时间步的输出
                out = self.fc_out(rnn_out[:, -1, :])  # (1, input_sz * input_sz)
                neuron_outputs.append(out)
            # 在这里，不要直接 cat， 而是先 reshape
            neuron_outputs = torch.cat(neuron_outputs).reshape(1, self.cfg.n_latents, self.cfg.input_sz, self.cfg.input_sz) # (1, n_neurons, input_sz, input_sz)
            outputs.append(neuron_outputs)
        outputs = torch.cat(outputs) # (batch_size, n_neurons, input_sz, input_sz)

        # 修改：
        outputs = torch.mean(outputs, dim=1, keepdim=True) # (batch_size, 1, input_sz, input_sz)
        return self.sigmoid(outputs) # 使用 sigmoid 函数

    # spike 值通过填充后一次性输入，需要把 LSTM 初始化中的input_size设置为 100
    def decode_all(self, z):
        # z: list[Tensor(n_neurons, seq_len_i)]
        batch_size = len(z)
        # 1. 将 list of tensors 转换为 tensor
        # 由于每个 spike 序列的长度可能不同，我们需要先找到最长的序列长度
        max_len = 0
        for i in range(batch_size):
          for j in range(z[i].size(0)):
            if len(z[i][j]) > max_len:
              max_len = len(z[i][j])

        # 然后进行填充
        padded_spikes = []
        for i in range(batch_size):
            padded_sample = []
            for j in range(z[i].size(0)):
                padding_len = max_len - len(z[i][j])
                padding = torch.zeros(padding_len, device=z[i].device) # 使用 0 填充
                padded_neuron = torch.cat([z[i][j], padding])
                padded_sample.append(padded_neuron)
            padded_spikes.append(torch.stack(padded_sample)) # (n_neurons, max_len)

        # (batch_size, n_neurons, max_len)
        padded_spikes = torch.stack(padded_spikes)

        # 2. 将输入维度调整为 (batch_size, sequence_length, n_neurons)
        #    这是为了适应 LSTM 的 batch_first=True 设置
        z = padded_spikes.permute(0, 2, 1).float()
        # 3. 初始化 LSTM 的隐藏状态 
        h0 = torch.zeros(self.rnn_decoder.num_layers, batch_size, self.rnn_decoder.hidden_size).to(z.device)
        c0 = torch.zeros(self.rnn_decoder.num_layers, batch_size, self.rnn_decoder.hidden_size).to(z.device)
        hidden = (h0, c0)

        # 4. 通过 LSTM 解码
        #    z: (batch_size, sequence_length, n_neurons)
        #    rnn_out: (batch_size, sequence_length, hidden_size)
        rnn_out, hidden = self.rnn_decoder(z, hidden)

        # 5. 使用全连接层处理所有时间步
        #    rnn_out: (batch_size, sequence_length, hidden_size)
        #    output: (batch_size, sequence_length, input_sz * input_sz)
        output = self.fc_out(rnn_out)

        # 6. 调整输出形状
        #    (batch_size, sequence_length, input_sz * input_sz)
        #    -> (batch_size, sequence_length, 1, input_sz, input_sz)
        #    -> (batch_size, 1, input_sz, input_sz)  (通过取平均值)
        output = output.view(-1, max_len, 1, self.cfg.input_sz, self.cfg.input_sz)
        output = torch.mean(output, dim=1, keepdim=False) # 在 sequence 维度上求平均

        return self.sigmoid(output)

    # spike 值通过填充后一次性输入RNN后再经过一个 CNN 重建为图像，需要把 LSTM 初始化中的input_size设置为 100
    def decode_all_CNN(self, z):
        # z: list[Tensor(n_neurons, seq_len_i)]
        batch_size = len(z)
        # 1. 将 list of tensors 转换为 tensor
        # 由于每个 spike 序列的长度可能不同，我们需要先找到最长的序列长度
        max_len = 0
        for i in range(batch_size):
          for j in range(z[i].size(0)):
            if len(z[i][j]) > max_len:
              max_len = len(z[i][j])

        # 然后进行填充
        padded_spikes = []
        for i in range(batch_size):
            padded_sample = []
            for j in range(z[i].size(0)):
                padding_len = max_len - len(z[i][j])
                padding = torch.zeros(padding_len, device=z[i].device) # 使用 0 填充
                padded_neuron = torch.cat([z[i][j], padding])
                padded_sample.append(padded_neuron)
            padded_spikes.append(torch.stack(padded_sample)) # (n_neurons, max_len)

        # (batch_size, n_neurons, max_len)
        padded_spikes = torch.stack(padded_spikes)

        # 2. 将输入维度调整为 (batch_size, sequence_length, n_neurons)
        #    这是为了适应 LSTM 的 batch_first=True 设置
        z = padded_spikes.permute(0, 2, 1).float()
        # 3. 初始化 LSTM 的隐藏状态 (这次是正确的！)
        h0 = torch.zeros(self.rnn_decoder.num_layers, batch_size, self.rnn_decoder.hidden_size).to(z.device)
        c0 = torch.zeros(self.rnn_decoder.num_layers, batch_size, self.rnn_decoder.hidden_size).to(z.device)
        hidden = (h0, c0)

        # 4. 通过 LSTM 解码
        #    z: (batch_size, sequence_length, n_neurons)
        #    rnn_out: (batch_size, sequence_length, hidden_size)
        rnn_out, hidden = self.rnn_decoder(z, hidden)
        # print(rnn_out.shape)
         # 5. 调整 RNN 输出的形状，以适应 CNN 的输入
        #    rnn_out: (batch_size, sequence_length, hidden_size)
        #    -> (batch_size, hidden_size, sequence_length)  # 调整维度顺序
        rnn_out = rnn_out.permute(0, 2, 1)
        #    -> (batch_size, hidden_size, sequence_length, 1) # 增加一个维度
        # print(rnn_out.shape)
        rnn_out = rnn_out.unsqueeze(-1)
        (B, C, W, H) = rnn_out.shape
        rnn_out = rnn_out.view(B, 4, 8, 8)
        # print(rnn_out.shape)
        # 6. 使用 CNN
        output = self.cnn(rnn_out)
        # print(output.shape)
        # 7. 调整输出形状
        return self.sigmoid(output)
    
    def infer(self, x, t = 0, ablate = None):
      """
      这里需要重写 infer 函数，因为父类 BaseVAE 中的 infer 涉及到 self.temp
      """
      log_r = self.log_rate.expand(len(x), -1)
      log_dr = self.encode(x)
      log_dr = softclamp_upper(log_dr, 10.0)
      if ablate is not None:
          log_dr[:, ablate] = 0.0
      dist = self.Dist(
          log_rate=log_r + log_dr,
          n_exp=self.n_exp,
          temp=t,
      )
      return dist, log_dr
    def loss_kl(self, log_dr):
        log_r = self.log_rate.expand(len(log_dr), -1)
        f = 1 + torch.exp(log_dr) * (log_dr - 1)
        kl = torch.exp(log_r) * f
        return kl

    def _init_enc(self):
        normalize_dim = 1
        if self.cfg.enc_type == 'conv':
            # stem
            if self.cfg.dataset in ['vH16', 'CIFAR16', 'BALLS']:
                padding = 1
            elif self.cfg.dataset == 'MNIST':
                padding = 'valid'
            else:
                raise ValueError(self.cfg.dataset)
            kws = dict(
                in_channels=1,
                out_channels=self.cfg.n_ch,
                kernel_size=3,
                padding=padding,
                reg_lognorm=True,
            )
            self.stem = Conv2D(**kws)
            # sequential
            self._kws_conv['n_nodes'] = 2
            self.enc = _build_conv_enc(
                nch=self.cfg.n_ch,
                kws=self._kws_conv,
                dataset=self.cfg.dataset,
            )
            # final fc step
            self.fc_enc = Linear(
                in_features=self.enc[-1].dim,
                out_features=self._enc_out_channel(),
                normalize=False,
                normalize_dim=normalize_dim,
                bias=False,
            )

        elif self.cfg.enc_type == 'mlp':
            self.enc = ResDenseLayer(
                dim=self.cfg.input_sz ** 2,
                expand=self.cfg.n_ch,
            )
            self.fc_enc = Linear(
                in_features=self.cfg.input_sz ** 2,
                out_features=self._enc_out_channel(),
                normalize=self.cfg.enc_norm,
                normalize_dim=normalize_dim,
                bias=self.cfg.enc_bias,
            )

        elif self.cfg.enc_type == 'lin':
            self.fc_enc = Linear(
                in_features=self.cfg.input_sz ** 2,
                out_features=self._enc_out_channel(),
                normalize=self.cfg.enc_norm,
                normalize_dim=normalize_dim,
                bias=self.cfg.enc_bias,
            )

        else:
            raise ValueError(self.cfg.enc_type)

        return

    def _init_dec(self):
      pass

    def _init_norm(self, regul_list: List[str] = None):
        regul_list = regul_list if regul_list else [
            'enc', 'dec']
        self.all_lognorm = []
        for child_name, child in self.named_children():
            for m in child.modules():
                cond = (
                    isinstance(m, (Conv2D, DeConv2D))
                    and child_name in regul_list
                    and m.lognorm.requires_grad
                )
                if cond:
                    self.all_lognorm.append(m.lognorm)
        return
    def _enc_out_channel(self):
      return self.cfg.n_latents

    def _dec_in_channel(self):
      return self.cfg.n_latents
    def get_rng(
            x: Union[int, np.random.Generator, random.Random] = 42,
            use_np: bool = True, ):
        if isinstance(x, int):
            if use_np:
                return np.random.default_rng(seed=x)
            else:
                return random.Random(x)
        elif isinstance(x, (np.random.Generator, random.Random)):
            return x
        else:
            print('Warning, invalid random state. returning default')
            return np.random.default_rng(seed=42)
    def _init_prior(self):
        rng = self.get_rng(self.cfg.seed)
        kws = {'size': (1, self.cfg.n_latents)}
        if self.cfg.prior_log_dist == 'cte':
            log_rate = np.ones(kws['size'])
            log_rate *= self.cfg.prior_clamp
        elif self.cfg.prior_log_dist == 'uniform':
            kws.update(dict(
                low=-6.0,
                high=self.cfg.prior_clamp,
            ))
            log_rate = rng.uniform(**kws)
        elif self.cfg.prior_log_dist == 'normal':
            s = np.abs(np.log(np.abs(
                self.cfg.prior_clamp)))
            kws.update(dict(loc=0.0, scale=s))
            log_rate = rng.normal(**kws)
        else:
            raise NotImplementedError(
                self.cfg.prior_log_dist)

        log_rate = torch.tensor(
            data=log_rate,
            dtype=torch.float,
        )
        log_rate[log_rate > 6.0] = 0.0

        self.log_rate = nn.Parameter(
            data=log_rate,
            requires_grad=True
        )
        return