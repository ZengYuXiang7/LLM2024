# coding : utf-8
# Author : yuxiang Zeng

import torch


class BiRNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BiRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.transfer = torch.nn.Linear(input_dim, hidden_dim)
        self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, x):
        dnn_seq = x[:, 1:]
        one_hot_encoded = torch.nn.functional.one_hot(dnn_seq.long(), num_classes=6).to(torch.float64)
        x = self.transfer(one_hot_encoded)
        # LSTM returns output and a tuple of (hidden state, cell state)
        out, (hn, cn) = self.lstm(x)
        # hn 的形状是 (num_layers * num_directions, batch_size, hidden_dim)
        # 对于单层双向LSTM, 我们需要取最后两个隐藏状态
        hn_fwd = hn[-2, :, :]  # 前向的最后隐藏状态
        hn_bwd = hn[-1, :, :]  # 后向的最后隐藏状态
        # 将前向和后向的最后隐藏状态沿着最后一个维度拼接
        hn_combined = torch.cat((hn_fwd, hn_bwd), dim=1)  # 形状: (batch_size, hidden_dim * 2)
        return hn_combined