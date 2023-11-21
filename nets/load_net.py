"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.gcn_net import GCNNet
from nets.gat_net import GATNet
from nets.graphsage_net import GraphSageNet
# from nets.mlp_net import MLPNet


def GCN(net_params):
    return GCNNet(net_params)


def GAT(net_params):
    return GATNet(net_params)


def GraphSage(net_params):
    return GraphSageNet(net_params)


def MLP(net_params):
    return MLPNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GCN': GCN,
        'GAT': GAT,
        'GraphSage': GraphSage,
        # 'MLP': MLP,
    }

    return models[MODEL_NAME](net_params)