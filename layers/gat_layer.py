import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GATConv

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""


class GATLayer(nn.Module):
    """
    Parameters
    ----------
    in_dim :
        Number of input features.
    out_dim :
        Number of output features.
    num_heads : int
        Number of heads in Multi-Head Attention.
    dropout :
        Required for dropout of attn and feat in GATConv
    batch_norm :
        boolean flag for batch_norm layer.
    residual :
        If True, use residual connection inside this layer. Default: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.

    Using dgl builtin GATConv by default:
    https://github.com/graphdeeplearning/benchmarking-gnns/commit/206e888ecc0f8d941c54e061d5dffcc7ae2142fc
    """

    def __init__(self, in_dim, out_dim, num_heads, dropout, batch_norm, residual=False, activation=F.elu):
        super().__init__()
        self.residual = residual
        self.activation = activation
        self.batch_norm = batch_norm

        if in_dim != (out_dim * num_heads):
            self.residual = False

        if dgl.__version__ < "0.5":
            self.gatconv = GATConv(in_dim, out_dim, num_heads, dropout, dropout)
        else:
            self.gatconv = GATConv(in_dim, out_dim, num_heads, dropout, dropout, allow_zero_in_degree=True)

        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_dim * num_heads)

    def forward(self, g, h):
        h_in = h  # for residual connection

        h = self.gatconv(g, h).flatten(1)

        if self.batch_norm:
            h = self.batchnorm_h(h)

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h  # residual connection

        return h