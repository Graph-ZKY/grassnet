import numpy as np
import torch
import argparse
import numpy as np
import scipy.sparse as sp
import torch
import random
import networkx as nx
import dgl
from dgl import DGLGraph
from dgl.data import *
import hashlib

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_planetoid_splits(g_nodes,labels, num_classes, percls_trn=20, val_lb=500, seed=12134):
    index = [i for i in range(0, labels.shape[0])]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    # print(seed)
    # raise Exception
    for c in range(num_classes):
        class_idx = np.where(labels.cpu() == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]
    # print(test_idx)

    train_mask = index_to_mask(train_idx, size=g_nodes)
    val_mask = index_to_mask(val_idx, size=g_nodes)
    test_mask = index_to_mask(test_idx, size=g_nodes)

    return train_mask,val_mask,test_mask


def new_random_planetoid_splits(g_nodes,labels,num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (labels == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        train_mask = index_to_mask(train_index, size=g_nodes)
        val_mask = index_to_mask(rest_index[:val_lb], size=g_nodes)
        test_mask = index_to_mask(
            rest_index[val_lb:], size=g_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        train_mask = index_to_mask(train_index, size=g_nodes)
        val_mask = index_to_mask(val_index, size=g_nodes)
        test_mask = index_to_mask(rest_index, size=g_nodes)

    return train_mask,val_mask,test_mask

def preprocess_data(dataset, train_ratio,seed=0):
    random.seed(seed)

    if dataset in ['cora', 'citeseer', 'pubmed']:

        edge = np.loadtxt('../low_freq/{}.edge'.format(dataset), dtype=int).tolist()
        feat = np.loadtxt('../low_freq/{}.feature'.format(dataset))
        labels = np.loadtxt('../low_freq/{}.label'.format(dataset), dtype=int)
        train = np.loadtxt('../low_freq/{}.train'.format(dataset), dtype=int)
        val = np.loadtxt('../low_freq/{}.val'.format(dataset), dtype=int)
        test = np.loadtxt('../low_freq/{}.test'.format(dataset), dtype=int)
        nclass = len(set(labels.tolist()))
        print(dataset, nclass)

        U = [e[0] for e in edge]
        V = [e[1] for e in edge]
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        g = dgl.to_bidirected(g)

        feat = normalize_features(feat)
        feat = torch.FloatTensor(feat)
        labels = torch.LongTensor(labels)
        train = torch.LongTensor(train)
        val = torch.LongTensor(val)
        test = torch.LongTensor(test)

        return g, nclass, feat, labels, train, val, test


    elif 'syn' in dataset:
        edge = np.loadtxt('../syn/{}.edge'.format(dataset), dtype=int).tolist()
        labels = np.loadtxt('../syn/{}.lab'.format(dataset), dtype=int)
        features = np.loadtxt('../syn/{}.feat'.format(dataset), dtype=float)

        n = labels.shape[0]
        idx = [i for i in range(n)]
        random.shuffle(idx)
        idx_train = np.array(idx[:100])
        idx_test = np.array(idx[100:])

        U = [e[0] for e in edge]
        V = [e[1] for e in edge]
        g = dgl.graph((U, V))

        c1 = 0
        c2 = 0
        lab = labels.tolist()
        for e in edge:
            if lab[e[0]] == lab[e[1]]:
                c1 += 1
            else:
                c2 += 1
        print(c1 / len(edge), c2 / len(edge))
        with open('log.txt', 'a') as f:
            f.write(str(c1 / len(edge)) + '    ' + str(c2 / len(edge)))
            f.write('\n')

        # normalization will make features degenerated
        # features = normalize_features(features)
        features = torch.FloatTensor(features)

        nclass = 2
        labels = torch.LongTensor(labels)
        train = torch.LongTensor(idx_train)
        test = torch.LongTensor(idx_test)
        print(dataset, nclass)

        return g, nclass, features, labels, train, train, test


    elif dataset in ['film']:
        graph_adjacency_list_file_path = './high_freq/{}/out1_graph_edges.txt'.format(dataset)
        graph_node_features_and_labels_file_path = './high_freq/{}/out1_node_feature_label.txt'.format(dataset)

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint16)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        row, col = np.where(adj.todense() > 0)

        U = row.tolist()
        V = col.tolist()
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)

        features = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])],
                            dtype=float)
        labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])], dtype=int)

        n = labels.shape[0]
        idx = [i for i in range(n)]
        random.shuffle(idx)  # 55555555555555555555555555
        r0 = int(n * train_ratio)
        r1 = int(n * 0.6)
        r2 = int(n * 0.8)

        idx_train = np.array(idx[:r0])
        idx_val = np.array(idx[r1:r2])
        idx_test = np.array(idx[r2:])

        features = normalize_features(features)
        features = torch.FloatTensor(features)

        nclass = 5
        labels = torch.LongTensor(labels)
        train = torch.LongTensor(idx_train)
        val = torch.LongTensor(idx_val)
        test = torch.LongTensor(idx_test)
        print(dataset, nclass)

        return g, nclass, features, labels, train, val, test


    # datasets in Geom-GCN
    elif dataset in ['cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel']:

        graph_adjacency_list_file_path = './high_freq/{}/out1_graph_edges.txt'.format(dataset)
        graph_node_features_and_labels_file_path = './high_freq/{}/out1_node_feature_label.txt'.format(dataset)

        G = nx.DiGraph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        features = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

        features = normalize_features(features)

        g = DGLGraph(adj)
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)

        n = len(labels.tolist())
        idx = [i for i in range(n)]
        random.shuffle(idx)  # 55555555555555555555555555
        r0 = int(n * train_ratio)
        r1 = int(n * 0.6)
        r2 = int(n * 0.8)
        train = np.array(idx[:r0])
        val = np.array(idx[r1:r2])
        test = np.array(idx[r2:])

        nclass = len(set(labels.tolist()))
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)
        train = torch.LongTensor(train)
        val = torch.LongTensor(val)
        test = torch.LongTensor(test)
        print(dataset, nclass)

        return g, nclass, features, labels, train, val, test


    # datasets in FAGCN
    elif dataset in ['new_chameleon', 'new_squirrel']:
        edge = np.loadtxt('./high_freq/{}/edges.txt'.format(dataset), dtype=int)
        labels = np.loadtxt('./high_freq/{}/labels.txt'.format(dataset), dtype=int).tolist()
        features = np.loadtxt('./high_freq/{}/features.txt'.format(dataset), dtype=float)

        U = [e[0] for e in edge]
        V = [e[1] for e in edge]
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.remove_self_loop(g)

        n = len(labels)
        idx = [i for i in range(n)]
        random.shuffle(idx)
        r0 = int(n * train_ratio)
        r1 = int(n * 0.6)
        r2 = int(n * 0.8)
        train = np.array(idx[:r0])
        val = np.array(idx[r1:r2])
        test = np.array(idx[r2:])

        features = normalize_features(features)
        features = torch.FloatTensor(features)

        nclass = 3
        labels = torch.LongTensor(labels)
        train = torch.LongTensor(train)
        val = torch.LongTensor(val)
        test = torch.LongTensor(test)
        print(dataset, nclass)

        return g, nclass, features, labels, train, val, test

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')
