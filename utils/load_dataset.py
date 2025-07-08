import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Amazon, Actor
import os
from .utils import random_splits
import torch as th
import numpy as np
import networkx as nx
from .hetero_utils import preprocess_data
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset,AmazonCoBuyPhotoDataset,AmazonCoBuyComputerDataset
from .datasets import Dataset
import torch_geometric.transforms as T


def load_data(name,train_rate=0.6,val_rate=0.2,seed=0,self_loop=False):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = f'{current_dir}/data'
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(data_path, name, transform=T.NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(data_path, name, transform=T.NormalizeFeatures())
    elif name in ['computers', 'photo']:
        dataset = Amazon(data_path, name, transform=T.NormalizeFeatures())
    elif name in ['actor']:
        dataset = Actor(f'{data_path}/actor')
    data = dataset[0]
    feat = data.x
    label = data.y
    edge_index = data.edge_index

    n_feat = feat.shape[1]
    n_classes = np.unique(label).shape[0]

    edge_index = edge_index
    feat = feat

    n_node = feat.shape[0]
    lbl1 = th.ones(n_node * 2)
    lbl2 = th.zeros(n_node * 2)
    lbl = th.cat((lbl1, lbl2))
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/

    percls_trn = int(round(train_rate * len(label) / n_classes))
    val_lb = int(round(val_rate * len(label)))

    train_mask, val_mask, test_mask = random_splits(label, n_classes, percls_trn, val_lb, seed=seed)

    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)
    test_mask = th.BoolTensor(test_mask)

    # if name in ['cora','citeseer','pubmed']:
    #     train_mask,val_mask,test_mask=data['train_mask'],data['val_mask'],data['test_mask']

    g = dgl.graph(data=(data.edge_index[0], data.edge_index[1]))

    g = dgl.remove_self_loop(g)
    if self_loop:
        g = dgl.add_self_loop(g)

    print("""----Data statistics------'
          #Edges %d
          #Classes %d
          #Train samples %d
          #Val samples %d
          #Test samples %d""" %
          (g.num_edges(), n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    return g, feat, label, train_mask, val_mask, test_mask, feat.shape[-1], n_classes, g.num_edges()





