import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    MLP Layer used after graph vector representation
"""


class MLPPredictor(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W =nn.Linear(input_dim,output_dim)

    def apply_edges(self,edges):
        h_u=edges.src['h']
        h_v=edges.dst['h']
        h_e=edges.data['feat']
        similarity=F.cosine_similarity(h_u,h_v,dim=-1)
        similarity=similarity.unsqueeze(1)
        y=self.W(torch.cat([similarity,h_e],dim=-1))
        return {'score':y}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


