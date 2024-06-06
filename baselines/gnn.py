# coding : utf-8
# Author : yuxiang Zeng

import torch
import dgl
from dgl.nn.pytorch import SAGEConv

from utils.config import get_config


class GraphSAGEConv(torch.nn.Module):
    def __init__(self, input_dim, dim, order, args):
        super(GraphSAGEConv, self).__init__()
        self.args = args
        self.dim = dim
        self.order = order
        self.layers = torch.nn.ModuleList([dgl.nn.pytorch.GraphConv(dim if i != 0 else input_dim, dim) for i in range(order)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(dim) for _ in range(order)])
        self.acts = torch.nn.ModuleList([torch.nn.ELU() for _ in range(order)])
        self.dropout = torch.nn.Dropout(0.10)
        self.pred_layers = torch.nn.Linear(9 * dim, 1)

    def forward(self, graph, features):
        # Reshape features from (batch_size, num_nodes, feature_dim) to (total_num_nodes, feature_dim)
        batch_size, num_nodes, feature_dim = features.shape
        total_num_nodes = batch_size * num_nodes
        features = features.view(total_num_nodes, feature_dim)

        g, g.ndata['L0'] = graph, features
        feats = g.ndata['L0']
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(g, feats)
            feats = norm(feats)
            feats = act(feats)
            g.ndata[f'L{i + 1}'] = feats

        # # 获取批处理图中每个图的全局节点
        # global_nodes = []
        # start = 0
        # for num_nodes in range(batch_size):
        #     global_nodes.append(start)  # 每个图的全局节点在批处理图中对应的位置
        #     start += 9

        # 提取全局节点的嵌入向量，形状为 (batch_size, dim)
        # print(g.ndata[f'L{self.order}'].shape)
        # embeds = g.ndata[f'L{self.order}'][global_nodes]
        # embeds = g.ndata[f'L{self.order}'].view(batch_size, num_nodes, self.dim)[:, 0, :]
        embeds = g.ndata[f'L{self.order}'].view(batch_size, num_nodes * self.dim)
        # embeds = torch.sum(g.ndata[f'L{self.order}'].view(batch_size, num_nodes, self.dim), dim=1)
        y = self.pred_layers(embeds)
        return y


if __name__ == '__main__':
    # Build a random graph
    args = get_config()
    num_nodes, num_edges = 100, 200
    src_nodes = torch.randint(0, num_nodes, (num_edges,))
    dst_nodes = torch.randint(0, num_nodes, (num_edges,))
    print(src_nodes.shape, dst_nodes.shape)

    graph = dgl.graph((src_nodes, dst_nodes))
    graph = dgl.add_self_loop(graph)

    # Demo test
    bs = 32
    features = torch.randn(num_nodes, 64)
    graph_gcn = GraphSAGEConv(64, 128, 2, args)
    # print(graph_gcn)
    embeds = graph_gcn(graph, features)
    print(embeds.shape)
