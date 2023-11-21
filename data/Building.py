import torch
import pickle
import torch.utils.data
import time
import os
import numpy as np

import dgl
import json
import networkx as nx
from scipy import sparse as sp

import itertools


class BuildingDataset(torch.utils.data.Dataset):
    def __init__(self, DATASET_NAME='Building', path="data/Building/"):
        self.name = DATASET_NAME
        self.data = np.load("./data/Building/all_data_sz1.npy", allow_pickle=True)
        self.g, self.labels = None, None
        self.train_masks, self.stopping_masks, self.val_masks, self.test_mask = None, None, None, None
        self.num_classes, self.n_feats = None, None

        self._load()

    def _load(self):
        t0 = time.time()
        print("[I] Loading Building ...")
        features = torch.FloatTensor(np.array(self.data[0]))
        self.labels = torch.LongTensor(np.array(self.data[1]))
        #节点特征组合
        featureDict = dict(area=0, peri=1, circularity=2, mean_radius=3,
                           orientation=4, concavity_area=5, concavity_peri=6, MBRFullness=7,
                           Elongation=8, WALL_orien=9,degree=10,shifting_degree_l=11,shifting_degree_w=12,density=13)
        featureIndex = []
        featureGroup = ['area','peri','mean_radius'
            ,'WALL_orien','orientation'
            ,'concavity_area','circularity', 'Elongation', 'MBRFullness'
            ,'shifting_degree_l','shifting_degree_w','density']
        for featureName in featureGroup:
            featureIndex.append(int(featureDict[featureName]))

        select_features = torch.empty(len(features), len(featureIndex))

        for dataIndex in range(len(features)):
            select_features[dataIndex]=features[dataIndex][featureIndex]
        # area, peri, circularity, mean_radius, orientation, concavity_area, concavity_peri, MBRFullness, Elongation, WALL_orien, degree, shifting_degree_l, shifting_degree_w


        self.edge = np.array((self.data[2]))
        edgefeature= torch.FloatTensor(np.array(self.data[3]))

        #边特征组合
        edge_featureDict = dict(min_dis=0, mean_dis=1, vis_dis=2, convex_dis=3,weight_dis=4)
        edge_featureIndex = []
        edge_featureGroup = ['min_dis','mean_dis','vis_dis']
        for edge_featureName in edge_featureGroup:
            edge_featureIndex.append(int(edge_featureDict[edge_featureName]))

        edge_select_features = torch.empty(len(edgefeature), len(edge_featureIndex))

        for edge_dataIndex in range(len(edgefeature)):
            edge_select_features[edge_dataIndex] = edgefeature[edge_dataIndex][edge_featureIndex]




        self.train_masks = torch.zeros(len(edgefeature), dtype=torch.bool)
        self.test_mask = torch.zeros(len(edgefeature), dtype=torch.bool)
        self.val_masks = torch.zeros(len(edgefeature), dtype=torch.bool)
        self.train_masks[self.data[4]]=True
        self.test_mask[self.data[5]] = True
        self.val_masks[self.data[6]] = True


        self.n_feats = select_features.shape[1]
        self.e_feats = edge_select_features.shape[1]
        self.num_classes = len(set(self.data[1]))


        self.g = dgl.DGLGraph()
        self.g.add_nodes(len(self.data[0]))
        src,dst=[],[]
        for i in range(len(self.edge)):
            src.append(self.edge[i][0])
            dst.append(self.edge[i][1])
        src = np.array(src)
        dst = np.array(dst)
        self.g.add_edges(src,dst)
        self.g.ndata['feat']=select_features       # available select_features
        self.g.edata['feat']=edge_select_features
        print("[I] Finished loading after {:.4f}s".format(time.time()-t0))

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.g

    def __len__(self):
        return 1

    def save(self):
        pass

    def load(self):
        pass

    def has_cache(self):
        pass