import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

import dgl
import numpy as np

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""

from layers.gcn_layer import GCNLayer
from layers.mlp_readout_layer import MLPPredictor


class GCNNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        dropout = net_params['dropout']
        n_classes = net_params['n_classes']
        n_layers =net_params['L']
        e_features=net_params['e_features']

        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes=n_classes
        self.device = net_params['device']

        self.layers =nn.ModuleList()
        # Input layer
        self.layers.append(GCNLayer(in_dim,hidden_dim,F.relu,dropout,self.batch_norm,self.residual))

        # Hidden layers
        for _ in range(n_layers-2):
            self.layers.append(GCNLayer(hidden_dim,hidden_dim,F.relu,dropout,self.batch_norm,self.residual))

        self.layers.append(GCNLayer(hidden_dim,out_dim,F.relu,dropout,self.batch_norm,self.residual))
        self.pred = MLPPredictor(e_features+1, n_classes)

    def forward(self, g, h):

        # GCN
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
    #
    #     criterion = nn.CrossEntropyLoss()
    #     loss = criterion(pred, label)
    #
    #     return loss