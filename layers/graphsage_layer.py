# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# import dgl.function as fn
# from dgl.nn.pytorch import SAGEConv
#
# """
#     GraphSAGE:
#     William L. Hamilton, Rex Ying, Jure Leskovec, Inductive Representation Learning on Large Graphs (NeurIPS 2017)
#     https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
# """
#
#
# class GraphSageLayer(nn.Module):
#
#     def __init__(self, in_feats, out_feats, activation, dropout,
#                  aggregator_type, batch_norm, residual=False,
#                  bias=True, dgl_builtin=False):
#         super().__init__()
#         self.in_channels = in_feats
#         self.out_channels = out_feats
#         self.aggregator_type = aggregator_type
#         self.batch_norm = batch_norm
#         self.residual = residual
#         self.dgl_builtin = dgl_builtin
#
#         if in_feats != out_feats:
#             self.residual = False
#
#         self.dropout = nn.Dropout(p=dropout)
#
#         if dgl_builtin == False:
#             self.nodeapply = NodeApply(in_feats, out_feats, activation, dropout,
#                                        bias=bias)
#             if aggregator_type == "maxpool":
#                 self.aggregator = MaxPoolAggregator(in_feats, in_feats,
#                                                     activation, bias)
#             elif aggregator_type == "lstm":
#                 self.aggregator = LSTMAggregator(in_feats, in_feats)
#             else:
#                 self.aggregator = MeanAggregator()
#         else:
#             self.sageconv = SAGEConv(in_feats, out_feats, aggregator_type,
#                                      dropout, activation=activation)
#
#         if self.batch_norm:
#             self.batchnorm_h = nn.BatchNorm1d(out_feats)
#
#     def forward(self, g, h):
#         h_in = h  # for residual connection
#
#         if self.dgl_builtin == False:
#             # h = self.dropout(h)
#             g.ndata['h'] = h
#             # g.update_all(fn.copy_u(src='h', out='m'),
#             #             self.aggregator,
#             #             self.nodeapply)
#             if self.aggregator_type == 'maxpool':
#                 g.ndata['h'] = self.aggregator.linear(g.ndata['h'])
#                 g.ndata['h'] = self.aggregator.activation(g.ndata['h'])
#                 g.update_all(fn.copy_u('h', 'm'), fn.max('m', 'c'), self.nodeapply)
#             elif self.aggregator_type == 'lstm':
#                 g.update_all(fn.copy_u(src='h', out='m'),
#                              self.aggregator,
#                              self.nodeapply)
#             else:
#                 g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'c'), self.nodeapply)
#             h = g.ndata['h']
#         else:
#             h = self.sageconv(g, h)
#
#         if self.batch_norm:
#             h = self.batchnorm_h(h)
#
#         if self.residual:
#             h = h_in + h  # residual connection
#
#         return h
#
#     def __repr__(self):
#         return '{}(in_channels={}, out_channels={}, aggregator={}, residual={})'.format(self.__class__.__name__,
#                                                                                         self.in_channels,
#                                                                                         self.out_channels,
#                                                                                         self.aggregator_type,
#                                                                                         self.residual)
#
#
# """
#     Aggregators for GraphSage
# """
#
#
# class Aggregator(nn.Module):
#     """
#     Base Aggregator class.
#     """
#
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, node):
#         neighbour = node.mailbox['m']
#         c = self.aggre(neighbour)
#         return {"c": c}
#
#     def aggre(self, neighbour):
#         # N x F
#         raise NotImplementedError
#
#
# class MeanAggregator(Aggregator):
#     """
#     Mean Aggregator for graphsage
#     """
#
#     def __init__(self):
#         super().__init__()
#
#     def aggre(self, neighbour):
#         mean_neighbour = torch.mean(neighbour, dim=1)
#         return mean_neighbour
#
#
# class MaxPoolAggregator(Aggregator):
#     """
#     Maxpooling aggregator for graphsage
#     """
#
#     def __init__(self, in_feats, out_feats, activation, bias):
#         super().__init__()
#         self.linear = nn.Linear(in_feats, out_feats, bias=bias)
#         self.activation = activation
#
#     def aggre(self, neighbour):
#         neighbour = self.linear(neighbour)
#         if self.activation:
#             neighbour = self.activation(neighbour)
#         maxpool_neighbour = torch.max(neighbour, dim=1)[0]
#         return maxpool_neighbour
#
#
# class LSTMAggregator(Aggregator):
#     """
#     LSTM aggregator for graphsage
#     """
#
#     def __init__(self, in_feats, hidden_feats):
#         super().__init__()
#         self.lstm = nn.LSTM(in_feats, hidden_feats, batch_first=True)
#         self.hidden_dim = hidden_feats
#         self.hidden = self.init_hidden()
#
#         nn.init.xavier_uniform_(self.lstm.weight,
#                                 gain=nn.init.calculate_gain('relu'))
#
#     def init_hidden(self):
#         """
#         Defaulted to initialite all zero
#         """
#         return (torch.zeros(1, 1, self.hidden_dim),
#                 torch.zeros(1, 1, self.hidden_dim))
#
#     def aggre(self, neighbours):
#         """
#         aggregation function
#         """
#         # N X F
#         rand_order = torch.randperm(neighbours.size()[1])
#         neighbours = neighbours[:, rand_order, :]
#
#         (lstm_out, self.hidden) = self.lstm(neighbours.view(neighbours.size()[0], neighbours.size()[1], -1))
#         return lstm_out[:, -1, :]
#
#     def forward(self, node):
#         neighbour = node.mailbox['m']
#         c = self.aggre(neighbour)
#         return {"c": c}
#
#
# class NodeApply(nn.Module):
#     """
#     Works -> the node_apply function in DGL paradigm
#     """
#
#     def __init__(self, in_feats, out_feats, activation, dropout, bias=True):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         self.linear = nn.Linear(in_feats * 2, out_feats, bias)
#         self.activation = activation
#
#     def concat(self, h, aggre_result):
#         bundle = torch.cat((h, aggre_result), 1)
#         bundle = self.linear(bundle)
#         return bundle
#
#     def forward(self, node):
#         h = node.data['h']
#         c = node.data['c']
#         bundle = self.concat(h, c)
#         bundle = F.normalize(bundle, p=2, dim=1)
#         if self.activation:
#             bundle = self.activation(bundle)
#         return {"h": bundle}

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch import SAGEConv

class GraphSageLayer(nn.Module):
    def __init__(self,in_feats,out_feats,activation,dropout,aggregator_type,batch_norm,residual=False,bias=True):
        super().__init__()
        self.in_channels = in_feats
        self.out_channels = out_feats
        # 聚合器类型可以是'mean'或'gcn'，默认为'mean'
        self.aggregator_type = aggregator_type
        self.batch_norm = batch_norm
        self.residual = residual

        if in_feats != out_feats:
            self.residual = False

        self.dropout = nn.Dropout(p=dropout)

        # 使用SAGEConv层进行节点特征聚合
        self.sageconv = SAGEConv(in_feats, out_feats, aggregator_type,dropout,activation=activation)
        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_feats)


    def forward(self, g, h):
        h_in=h

        # 使用SAGEConv进行节点特征聚合
        h = self.sageconv(g, h)
        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.residual:
            h = h_in + h  # residual connection

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, aggregator={}, residual={})'.format(self.__class__.__name__,
                                              self.in_channels,
                                              self.out_channels, self.aggregator_type, self.residual)
