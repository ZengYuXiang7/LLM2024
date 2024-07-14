# coding : utf-8
# Author : yuxiang Zeng


import torch

from modules.chatgpt import NAS_ChatGPT
from modules.pred_layer import Predictor
import pickle

class InductiveModel(torch.nn.Module):
    def __init__(self, args):
        super(InductiveModel, self).__init__()
        self.args = args
        self.dim = args.rank
        self.dnn_embedding = torch.nn.Embedding(6, self.dim)
        self.predictor = Predictor(self.dim * 6, self.dim, 1)

        if args.llm:
            self.info_encoder = torch.nn.Linear(5, self.dim)
            try:
                with open(f'./agu/{args.device_name}.pkl', 'rb') as f:
                    self.aug_data = torch.tensor(pickle.load(f)).float().unsqueeze(0)
            except:
                import ast
                llm = NAS_ChatGPT(args)
                self.aug_data = ast.literal_eval(llm.get_device_more_info(args.device_name))
                self.aug_data = torch.tensor(self.aug_data).float().unsqueeze(0)
                print("增强数据: ", self.aug_data)
            self.predictor = Predictor(self.dim * 7, self.dim, 1)


    def forward(self, adj, key):
        if self.args.llm:
            y = self.embed_with_llm(key)
        return y

    # Ablation
    def only_embeds(self, key):
        dnn_embeds = self.dnn_embedding(key.long()).reshape(key.shape[0], -1)
        y = self.predictor(dnn_embeds)
        return y

    def embed_with_llm(self, key):
        dnn_embeds = self.dnn_embedding(key.long()).reshape(key.shape[0], -1)
        # 广播增强数据到批次大小
        batch_size = key.shape[0]
        aug_data_repeated = self.aug_data.expand(batch_size, -1)

        info_embeds = self.info_encoder(aug_data_repeated)
        embeds = torch.cat([dnn_embeds, info_embeds], dim=1)
        y = self.predictor(embeds)
        return y

