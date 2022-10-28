# This code is a TF 2.X refactor of the original GraphSAGE implementation:
# https://github.com/williamleif/GraphSAGE

# Additional updates to implement OGB datasets + Spektral wrapper
# https://ogb.stanford.edu/
# https://github.com/danielegrattarola/spektral

import os
import networkx as nx
from networkx.readwrite import json_graph
import json
import numpy as np
import tensorflow as tf

from ogb.nodeproppred import NodePropPredDataset
from spektral.datasets.ogb import OGB
from spektral.datasets import Reddit
from spektral.transforms import GCNFilter
from spektral.datasets.utils import DATASET_FOLDER


###########################
# NX Datasets
###########################
class NXDataset(object):
    '''
    This object is a wrapper for networkx-based datasets that implements the 
    filter/batching and model configuration components of GraphSAGE-Sparse

    path -- path/to/data
    layer_infos -- model layer config dictionaries
    max_degree -- max_degree filter for graph
    kernel -- GraphSAGE kernel to use (overrides whatever is in layer_infos)
    batch_size -- batch_size for training
    reg_weight -- l2 regularization weight
    ftrain -- fraction of data to use for training
    fval -- fraction of data to use for validation
    shuffle - whether to shuffle data before splits
    '''

    def __init__(self, path, layer_infos, max_degree=128, kernel='gcn', 
                 batch_size=256, reg_weight=1e-9, ftrain=.75, fval=.15,
                 shuffle=False):
        self.path = path
        for i in range(len(layer_infos)):
            layer_infos[i]['agg_type'] = kernel
        self.layer_infos = layer_infos
        self.max_degree = max_degree
        self.kernel = kernel
        self.batch_size = batch_size
        self.reg_weight = reg_weight
        self.k = len(layer_infos)
        self.ftrain = ftrain
        self.fval = fval
        self.shuffle = shuffle

    def load(self):
        gfile = os.path.join(self.path,'G.json')
        G = json_graph.node_link_graph(json.load(open(gfile)))

        ffile = os.path.join(self.path,'feats.npy')
        feats = np.load(ffile)
        # as in original GS, pad features with dummy zero vector to serve as 
        #  the 'null' feature
        self.feats = np.vstack([feats, np.zeros((feats.shape[1],))])

        ifile = os.path.join(self.path,'id_map.json')
        id_map = json.load(open(ifile))

        self.id_map = {k:int(v) for k,v in id_map.items()}
        self.rev_map = {self.id_map[k]:k for k in self.id_map.keys()}
        self.G = nx.relabel_nodes(G, id_map)

        cfile = os.path.join(self.path,'class_map.json')
        class_map = json.load(open(cfile))
        self.class_map = {k:v for k,v in class_map.items()}

    def get_adj(self):
        # the adjacency matrix has shape (V+1) x max_degree
        # nodes with no adjacencies are mapped as adjacent to a 'null' node: V
        V = len(self.id_map)
        self.adj = V*np.ones((V+1, self.max_degree), dtype=np.int32)
        self.deg = np.zeros((V,), dtype=np.int32)
        for nodeid in list(self.G.nodes):
            neighbors = np.array([neighbor for neighbor in list(self.G.neighbors(nodeid))])
            self.deg[nodeid] = len(neighbors)
            if len(neighbors) == 0:
                continue
            elif len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            self.adj[nodeid, :] = neighbors


class NodeDataset(NXDataset):
    '''
    Node prediction dataset

    final_info -- final layer config dictionary
    '''

    def __init__(self, path, layer_infos, final_info, max_degree=128, kernel='gcn', 
                 batch_size=256, reg_weight=1e-9, ftrain=.75, fval=.15):
        super().__init__(path, layer_infos, max_degree=max_degree, kernel=kernel, 
                         batch_size=batch_size, reg_weight=reg_weight, 
                         ftrain=ftrain, fval=fval)
        self.final_info = final_info

        self.load()

        self.get_adj()

        self.get_splits()

    def get_splits(self):
        # labels
        labels = np.array([self.class_map[self.rev_map[nodeid]] for nodeid in 
                           list(self.G.nodes())]).astype('float')

        # create splits
        V = len(self.id_map)
        nodes = np.arange(V)
        N_train, N_val = np.int(self.ftrain*V), np.int(self.fval*V)
        N_test = V - N_train - N_val

        if self.shuffle:
            nodes = np.random.permutation(nodes).astype(np.int32)
        labels = labels[nodes]
        self.train, self.val, self.test = nodes[:N_train], \
            nodes[N_train:-N_test], nodes[-N_test:]
        self.labels_train, self.labels_val, self.labels_test = labels[:N_train], \
            labels[N_train:-N_test], labels[-N_test:]


class LinkDataset(NXDataset):
    '''
    Link prediction dataset

    negative_sample_size -- no. of negative samples per batch
    '''

    def __init__(self, path, layer_infos, max_degree=128, kernel='gcn', 
                 batch_size=256, reg_weight=1e-9, ftrain=.75, fval=.15, 
                 negative_sample_size=5):
        super().__init__(path, layer_infos, max_degree=max_degree, kernel=kernel, 
                         batch_size=batch_size, reg_weight=reg_weight, 
                         ftrain=ftrain, fval=fval)
        self.negative_sample_size = negative_sample_size

        self.load()

        self.get_adj()

        self.get_splits()

    def get_splits(self):
        wfile = os.path.join(self.path,'walks.txt')
        with open(wfile) as fp:
            self.context_pairs = [list(map(lambda x: int(x), line.split())) 
                                  for line in fp.readlines()]

        # create splits
        N = len(self.context_pairs)
        N_train, N_val = np.int(self.ftrain*N), np.int(self.fval*N)
        N_test = N - N_train - N_val

        if self.shuffle:
            edges = np.random.permutation(self.context_pairs).astype(np.int32)
        else:
            edges = self.context_pairs
        edges = np.array(edges)

        self.train, self.val, self.test = edges[:N_train,:], \
            edges[N_train:-N_test,:], edges[-N_test:,:]

        # label dataset is unused; only provided for consistency with model.fit() API
        self.labels_train = np.zeros((N_train, 1))
        self.labels_val = np.zeros((N_val, 1))
        self.labels_test = np.zeros((N_test, 1))


###########################
# OGB/Spektral Datasets
###########################
class OGBArxivDataset(object):
    '''
    This object is a wrapper for the OGB-Arxiv node prediction
    dataset provided by Spektral

    layer_infos -- model layer config dictionaries
    final_info -- final layer config dictionary
    max_degree -- max_degree filter for graph
    kernel -- GraphSAGE kernel to use (overrides whatever is in layer_infos)
    batch_size -- batch_size for training
    reg_weight -- l2 regularization weight
    '''

    def __init__(self, layer_infos, final_info, max_degree=128, kernel='gcn', 
                 batch_size=256, reg_weight=1e-9):
        for i in range(len(layer_infos)):
            layer_infos[i]['agg_type'] = kernel
        self.layer_infos = layer_infos
        self.final_info = final_info
        self.max_degree = max_degree
        self.kernel = kernel
        self.batch_size = batch_size
        self.reg_weight = reg_weight
        self.k = len(layer_infos)

        # load data
        ogb_dataset = NodePropPredDataset('ogbn-arxiv')
        dataset = OGB(ogb_dataset, transform=[GCNFilter()])
        graph = dataset[0]
        feats, adj, y = graph.x, graph.a, graph.y
        # as in original GS, pad features with dummy zero vector to serve as 
        #  the 'null' feature
        self.feats = np.vstack([feats, np.zeros((feats.shape[1],))])

        # the adjacency matrix has shape (V+1) x max_degree
        # nodes with no adjacencies are mapped as adjacent to the 'null' node V
        V = adj.shape[0]
        self.adj = V*np.ones((V+1, self.max_degree), dtype=np.int32)
        for nodeid in np.arange(V):
            neighbors = np.where(adj[nodeid].todense()>0)[1]
            if len(neighbors) == 0:
                continue
            elif len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            self.adj[nodeid, :] = neighbors

        # labels
        labels = tf.squeeze(tf.one_hot(y, depth=np.max(y)+1)).numpy()
        self.final_info['num_classes'] = labels.shape[1]

        # apply splits provided by dataset
        idx = ogb_dataset.get_idx_split()
        self.train, self.val, self.test = idx['train'].astype(np.int32), \
                idx['valid'].astype(np.int32), idx['test'].astype(np.int32)
        self.labels_train, self.labels_val, self.labels_test = labels[self.train], \
            labels[self.val], labels[self.test]


class LRedditDataset(object):
    '''
    This object is a wrapper for the Reddit link prediction
    dataset provided by Spektral

    layer_infos -- model layer config dictionaries
    max_degree -- max_degree filter for graph
    kernel -- GraphSAGE kernel to use (overrides whatever is in layer_infos)
    batch_size -- batch_size for training
    reg_weight -- l2 regularization weight
    negative_sample_size -- no. of negative samples per batch
    p -- fraction of random walk training data to use
    '''

    def __init__(self, layer_infos, max_degree=128, kernel='gcn', 
                 batch_size=256, reg_weight=1e-9, negative_sample_size=5, 
                 shuffle=False, p=1.):
        for i in range(len(layer_infos)):
            layer_infos[i]['agg_type'] = kernel
        self.layer_infos = layer_infos
        self.max_degree = max_degree
        self.kernel = kernel
        self.batch_size = batch_size
        self.reg_weight = reg_weight
        self.negative_sample_size = negative_sample_size
        self.k = len(layer_infos)
        self.shuffle = shuffle
        self.p = np.clip(p, 0, 1)

        # load data
        rdataset = Reddit()
        graph = rdataset.graphs[0]
        feats, adj = graph.x, graph.a
        # as in original GS, pad features with dummy zero vector to serve as 
        #  the 'null' feature
        self.feats = np.vstack([feats, np.zeros((feats.shape[1],))])

        # the adjacency matrix has shape (V+1) x max_degree
        # nodes with no adjacencies are mapped as adjacent to the 'null' node V
        V = adj.shape[0]
        self.adj = V*np.ones((V+1, self.max_degree), dtype=np.int32)
        for nodeid in np.arange(V):
            neighbors = np.where(adj[nodeid].todense()>0)[1]
            if len(neighbors) == 0:
                continue
            elif len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            self.adj[nodeid, :] = neighbors

        # process walks to use node ids, if necessary
        prefix = os.path.join(DATASET_FOLDER,'GraphSage/reddit')
        walk_path = os.path.join(prefix,'reddit-processed-walks.txt')
        if not os.path.exists(walk_path):
            print('processing reddit-walks.txt')
            idfile = os.path.join(prefix,'reddit-id_map.json')
            id_map = json.load(open(idfile))

            orig_walk_path = os.path.join(prefix,'reddit-walks.txt')
            with open(orig_walk_path, 'r') as fi:
                with open(walk_path, 'w') as fo:
                    for i, line in enumerate(fi):
                        nodes = line.strip().split()
                        fo.write(str(id_map[nodes[0]]) + ' ' + str(id_map[nodes[1]]) + '\n')

        # read processed walks to get context pairs; these are the train edges
        # no val/test specified for lreddit
        self.train = np.loadtxt(walk_path, delimiter=' ').astype(int)
        if self.p < 1:
            lsize = len(self.train)
            if self.shuffle:
                sample = np.random.choice(lsize, int(self.p*lsize), replace=False)
            else:
                sample = np.arange(int(self.p*lsize)).astype(int)
            self.train = self.train[sample]

        # label dataset is unused; only provided for consistency with model.fit() API
        self.labels_train = np.zeros((len(self.train), 1))


class NRedditDataset(object):
    '''
    This object is a wrapper for the Reddit node prediction
    dataset provided by Spektral

    layer_infos -- model layer config dictionaries
    final_info -- final layer config dictionary
    max_degree -- max_degree filter for graph
    kernel -- GraphSAGE kernel to use (overrides whatever is in layer_infos)
    batch_size -- batch_size for training
    reg_weight -- l2 regularization weight
    '''

    def __init__(self, layer_infos, final_info, max_degree=128, kernel='gcn', 
                 batch_size=256, reg_weight=1e-9):
        for i in range(len(layer_infos)):
            layer_infos[i]['agg_type'] = kernel
        self.layer_infos = layer_infos
        self.final_info = final_info
        self.max_degree = max_degree
        self.kernel = kernel
        self.batch_size = batch_size
        self.reg_weight = reg_weight
        self.k = len(layer_infos)

        # load data
        rdataset = Reddit()
        self.final_info['num_classes'] = rdataset.n_labels
        graph = rdataset.graphs[0]
        feats, adj, labels = graph.x, graph.a, graph.y
        # as in original GS, pad features with dummy zero vector to serve as 
        #  the 'null' feature
        self.feats = np.vstack([feats, np.zeros((feats.shape[1],))])

        # the adjacency matrix has shape (V+1) x max_degree
        # nodes with no adjacencies are mapped as adjacent to the 'null' node V
        V = adj.shape[0]
        self.adj = V*np.ones((V+1, self.max_degree), dtype=np.int32)
        for nodeid in np.arange(V):
            neighbors = np.where(adj[nodeid].todense()>0)[1]
            if len(neighbors) == 0:
                continue
            elif len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            self.adj[nodeid, :] = neighbors

        # apply splits provided by dataset
        self.train = np.where(rdataset.mask_tr)[0].astype(np.int32)
        self.val = np.where(rdataset.mask_va)[0].astype(np.int32)
        self.test = np.where(rdataset.mask_te)[0].astype(np.int32)
        self.labels_train, self.labels_val, self.labels_test = labels[self.train], \
            labels[self.val], labels[self.test]