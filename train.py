import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl
import numpy as np
import torch.nn.functional as F
from metrics import accuracy_Building as accuracy
from tqdm import tqdm
from sklearn import metrics


def train_epoch(model, optimizer, device, graph, node_feat, edge_feat, train_mask, labels, epoch):

    model.train()
    epoch_loss=0
    epoch_train_acc=0

    node_feat=graph.ndata['feat'].to(device)
    edge_feat= graph.edata['feat'].to(device)
    pre_edge=model(graph,node_feat)
    loss=model.loss(pre_edge[train_mask],labels[train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_loss = loss.detach().item()
    epoch_train_acc = accuracy(pre_edge[train_mask], labels[train_mask])

    return epoch_loss, epoch_train_acc, optimizer



def evaluate_network(model, optimizer, device, graph, node_feat, edge_feat, mask, labels, epoch,write_edge_name,output_button=False):

    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0

    with torch.no_grad():

        graph = graph.to(device)
        node_feat = graph.ndata['feat'].to(device)
        edge_feat = graph.edata['feat'].to(device)
        pre_edge = model(graph, node_feat)

        loss = model.loss(pre_edge[mask], labels[mask])
        epoch_test_loss = loss.detach().item()
        epoch_test_acc = accuracy(pre_edge[mask], labels[mask])
    if output_button==True:
        result = pre_edge.max(1)[1].type_as(labels).detach().cpu().numpy()
        return epoch_test_loss, epoch_test_acc,result,pre_edge
    else:

            # result=pre_edge.max(1)[1].type_as(labels).detach().cpu().numpy()
            # label = labels.detach().cpu().numpy()
            # mask = torch.nonzero(mask).squeeze().cpu().numpy()
            # pred_edge_txt = open(write_edge_name, "w")
            # for i in range(len(mask)):
            #     pred_edge_txt.write(str(mask[i]) + "," + str(result[mask[i]]) + "," + str(label[mask[i]]) + '\n')
            # pred_edge_txt.close()
        return epoch_test_loss, epoch_test_acc
