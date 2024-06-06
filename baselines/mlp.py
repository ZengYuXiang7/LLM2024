# coding : utf-8
# Author : yuxiang Zeng

# coding : utf-8
# Author : yuxiang Zeng

import torch
from utils.config import get_config


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(MLP, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.NeuCF = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),  # FFN
            torch.nn.LayerNorm(self.hidden_dim),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(self.hidden_dim, self.hidden_dim // 2),  # FFN
            torch.nn.LayerNorm(self.hidden_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(self.hidden_dim // 2, output_dim)  # y
        )

    def forward(self, _, x):
        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        outputs = self.NeuCF(x)
        outputs = torch.sigmoid(outputs)
        return outputs



if __name__ == '__main__':
    # Build a random graph
    args = get_config()
    features = torch.randn(256, 6)
    mlp = MLP(6, 128, 1, args)
    embeds = mlp(None, features)
    print(embeds.shape)
