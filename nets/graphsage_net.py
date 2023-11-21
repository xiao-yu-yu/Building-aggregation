import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

"""
    GraphSAGE: 
    William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
    https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

from layers.graphsage_layer import GraphSageLayer
from layers.mlp_readout_layer import MLPPredictor


class GraphSageNet(nn.Module):
    """
    Grahpsage network with multiple GraphSageLayer layers
    """

    def __init__(self, net_params):
        super().__init__()

        in_dim = net_params['in_dim']  # node_dim (feat is an integer)
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        dropout = net_params['dropout']
        n_classes = net_params['n_classes']
        e_features = net_params['e_features']

        aggregator_type = net_params['sage_aggregator']
        n_layers = net_params['L']
        batch_norm = net_params['batch_norm']
        residual = net_params['residual']
        # self.readout = net_params['readout']
        self.n_classes = n_classes
        self.device = net_params['device']

        self.layers =nn.ModuleList()
        self.layers.append(GraphSageLayer(in_dim, hidden_dim, F.relu, dropout, aggregator_type, batch_norm, residual))
        for _ in range(n_layers-2):
            self.layers.append(
                GraphSageLayer(hidden_dim, hidden_dim, F.relu, dropout, aggregator_type, batch_norm, residual))
        self.layers.append(GraphSageLayer(hidden_dim, out_dim, F.relu, dropout, aggregator_type, batch_norm, residual))
        self.pred = MLPPredictor(e_features+1, n_classes)

    def forward(self, g, h):

        # graphsage
        for conv in self.layers:
            h = conv(g, h)

        # output
        h = self.pred(g, h)

        return h


    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight[0]=weight[0]*2
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss = criterion(pred, label)

        return loss

    # def loss(self, pred, label):
    #     criterion = nn.CrossEntropyLoss()
    #     loss = criterion(pred, label)
    #
    #     return loss