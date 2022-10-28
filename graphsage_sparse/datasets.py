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
import numba
from numba import njit
import scipy as sp
import pandas as pd
import tensorflow as tf

from ogb.nodeproppred import NodePropPredDataset
from spektral.datasets.ogb import OGB
from spektral.datasets import Reddit
from spektral.transforms import GCNFilter, AdjToSpTensor
from spektral.datasets.utils import DATASET_FOLDER

###########################
# Standalone Numba funcs
###########################
@njit(numba.int32[:,:](numba.int32[:,:], numba.int32[:], numba.int32))
def nbdfilt(x, nbd, sample_size):
    if len(x) > sample_size:
        a = x[0][0]  # base node for this split, in sampled coords
        b = nbd[a]  # a in original coords
        ind = np.random.choice(len(x), sample_size, replace=False)
        x = x[ind]
        if not b in x[:,1]:  # make sure self loops are present
            x[0] = [a,b]
    return x


@njit()
def nbdwrap(rc, indices, nbd, sample_size):
    rs = np.split(rc, indices)
    rs = [nbdfilt(rsi, nbd, sample_size) for rsi in rs]
    return rs


###########################
# Baseline class
###########################
class Dataset(object):
    '''
    This object is a wrapper for datasets that implement the 
    filter/batching and model configuration components of GraphSAGE-Sparse

    layer_infos -- model layer config dictionaries
    kernel -- GraphSAGE kernel to use (overrides whatever is in layer_infos)
    batch_size -- batch_size for training
    max_degree -- max_degree filter for graph (=-1 for no filter)
    sample_size -- controls minibatch size
    symm -- whether to symmeterize adjacency matrix
    reg_weight -- l2 regularization weight
    '''

    def __init__(self, layer_infos, kernel='gcn', batch_size=256, 
                 max_degree=-1, sample_size=15, symm=False, reg_weight=1e-9):
        self.kernel = kernel
        self.max_degree = max_degree
        self.sample_size = np.int32(sample_size)
        self.symm = symm
        self.batch_size = batch_size
        self.reg_weight = reg_weight
        for i in range(len(layer_infos)):
            layer_infos[i]['agg_type'] = kernel
        self.layer_infos = layer_infos
        self.k = len(layer_infos)

    def preprocess(self):
        if self.max_degree > 0:
            self.a = self.deg_filter(self.a, self.max_degree)

        if self.symm:
            # note this can increase max_degree
            self.a = self.symmetrize(self.a)

        self.a_e = self.add_eye(self.a)

    def add_eye(self, a):
        '''
        A <- A+I
        '''
        return a + sp.sparse.identity(a.shape[0])

    def symmetrize(self, a):
        '''
        A <- A+A^T, unit entries
        '''
        a = a + a.T
        return sp.sparse.csr_matrix((np.ones(a.data.shape), (a.nonzero())), shape=a.shape)

    def scale(self, a):
        '''
        A <- D^{-1}A
        '''
        d = sp.sparse.spdiags(1/sp.sparse.csr_matrix.sum(a, axis=1).T, 
                              [0], a.shape[0], a.shape[1])
        return d*a

    def power(self, a, k):
        '''
        A <- I + A + A^2 + ... + A^k
        Gives #paths of length <= k from i to j
        '''
        ak = sp.sparse.identity(a.shape[0])
        ap = ak
        for _ in range(k):
            ak = ak*a
            ap += ak
        return ap

    def deg_filter(self, a, max_deg):
        '''
        Apply max_deg filter to matrix a
        Assumes i->j adjacency
        '''
        v_old = -1
        start_ind = 0
        r, c = a.nonzero()
        ind = np.zeros(len(r), dtype=np.int32)
        for (i, v) in enumerate(r):
            if v_old != v:
                # save off previous set
                tind = np.arange(start_ind, i)
                if len(tind) > max_deg:
                    tind = np.random.choice(tind, max_deg, replace=False)
                ind[tind] = 1
                start_ind = i
                v_old = v
        tind = ind[start_ind:]
        if len(tind) > max_deg:
            tind = np.random.choice(tind, max_deg, replace=False)
        ind[tind] = 1

        ind = ind.astype(bool)
        return sp.sparse.csr_matrix((a.data[ind], (r[ind], c[ind])), shape=a.shape)


###########################
# Node prediction datasets
###########################
class NodeDataset(Dataset):
    '''
    Wrapper for node prediction datasets

    final_info -- model final layer config dictionary
    '''

    def __init__(self, layer_infos, final_info, kernel='gcn', batch_size=256, 
                 max_degree=-1, sample_size=15, symm=False, reg_weight=1e-9):
        super().__init__(layer_infos, kernel=kernel,
                         batch_size=batch_size, max_degree=max_degree,
                         sample_size=sample_size, symm=symm, reg_weight=reg_weight)
        self.final_info = final_info
        self.pad_size = self.batch_size*self.sample_size**self.k

    def get_gen(self, mode='train'):
        '''
        Wrapper to build generator and corresponding type_spec
        '''
        m = self.feats.shape[1]

        if mode == 'val':
            x = self.idx_va
        elif mode == 'test':
            x = self.idx_te
        else:
            x = self.idx_tr

        def gen():
            '''
            Generator.  For each batch, return the label matrix, feature matrix, 
              and kernel matrix associated with the k-neighborhood of the batch
            b_s - array, batch in sampled coordinates
            f_s - array, feature matrix induced by sample
            a_ks - csr_matrix, kernel matrix subgraph induced by sample
            y_s - array, labels induced by sample
            '''
            for i in range(0, len(x), self.batch_size):
                batch = x[i:i+self.batch_size]

                # sample indices and build maparr
                nbd = batch.astype(np.int32)
                for _ in range(self.k):
                    tnbd = nbd
                    r, c = self.a_e[tnbd].nonzero()
                    rc = np.transpose(np.array([r,c]))
                    indices = np.unique(rc[:,0], return_index=True)[1]
                    rs = nbdwrap(rc, indices, tnbd, self.sample_size)
                    rc = np.concatenate(rs, axis=0)
                    nbd = pd.unique(rc[:,1]).astype(np.int32)
                r = tnbd[rc[:,0]]
                c = rc[:,1]
                nbdi = np.arange(len(nbd))
                maparr = sp.sparse.csr_matrix((nbdi+1, (nbd,np.zeros(len(nbd)))), 
                                              shape=(np.max(nbd)+1,1))

                # a_ks
                rr = maparr[r].data-1
                cc = maparr[c].data-1
                if self.kernel == 'gcn':
                    # A_{ks} <- D_{ks}^{-1} A_{ks}
                    d = self.scale(
                        sp.sparse.csr_matrix(
                            (np.ones(rr.shape), (rr,cc)), 
                            shape=(self.pad_size, self.pad_size))).data
                    a_ks = tf.cast(tf.sparse.SparseTensor(
                        indices=np.array([rr,cc]).T, values=d,
                        dense_shape=(self.pad_size, self.pad_size)), tf.float32)
                else:
                    a_ks = tf.cast(tf.sparse.SparseTensor(
                        indices=np.array([rr,cc]).T, values=np.ones(rr.shape),
                        dense_shape=(self.pad_size, self.pad_size)), tf.float32)

                # b_s
                b_s = maparr[batch].data-1

                # f_s
                f_s = np.zeros((self.pad_size, m), dtype=self.feats.dtype)
                f_s[:len(nbd)] = self.feats[nbd]

                # y_s
                y_s = self.labels[batch]

                yield (b_s, f_s, a_ks), y_s

        # type_spec
        type_spec = ((tf.TensorSpec(shape=(self.batch_size,), dtype=tf.int32), 
                      tf.TensorSpec(shape=(self.pad_size, m), dtype=tf.float32), 
                      tf.SparseTensorSpec(shape=(self.pad_size, self.pad_size), dtype=tf.float32)), 
                     tf.TensorSpec(shape=(self.batch_size, self.final_info['num_classes']), 
                                   dtype=tf.float32))

        return (gen, type_spec)


class OGBArxivDataset(NodeDataset):
    '''
    OGB-Arxiv node prediction dataset, as provided by Spektral
    '''

    def __init__(self, layer_infos, final_info, kernel='gcn', batch_size=256, 
                 max_degree=-1, sample_size=15, symm=False, reg_weight=1e-9):
        super().__init__(layer_infos, final_info, kernel=kernel,
                         batch_size=batch_size, max_degree=max_degree,
                         sample_size=sample_size, symm=symm, reg_weight=reg_weight)

        # load dataset
        self.load()

        # preprocess
        self.preprocess()

        # get generators
        self.train, self.type_spec = self.get_gen()
        self.val, _ = self.get_gen(mode='val')
        self.test, _ = self.get_gen(mode='test')

    def load(self):
        # load data
        nppdataset = NodePropPredDataset('ogbn-arxiv')
        self.final_info['num_classes'] = nppdataset.num_classes
        dataset = OGB(nppdataset, transform=[GCNFilter(), AdjToSpTensor()])
        graph = dataset[0]
        self.feats, self.a, y = graph.x, graph.a, graph.y

        # apply splits provided by dataset
        idx = nppdataset.get_idx_split()
        self.idx_tr, self.idx_va, self.idx_te = idx['train'], idx['valid'], idx['test']

        # need to clip for generator's type_spec
        self.idx_tr = self.idx_tr[:-np.mod(len(self.idx_tr), self.batch_size)]
        self.idx_va = self.idx_va[:-np.mod(len(self.idx_va), self.batch_size)]
        self.idx_te = self.idx_te[:-np.mod(len(self.idx_te), self.batch_size)]

        # labels
        self.labels = tf.squeeze(tf.one_hot(y, depth=np.max(y)+1)).numpy()

        # lengths
        self.train_len = len(self.idx_tr)
        self.val_len = len(self.idx_va)
        self.test_len = len(self.idx_te)


class NRedditDataset(NodeDataset):
    '''
    Reddit node prediction dataset, as provided by Spektral
    '''

    def __init__(self, layer_infos, final_info, kernel='gcn', batch_size=256, 
                 max_degree=-1, sample_size=15, symm=False, reg_weight=1e-9):
        super().__init__(layer_infos, final_info, kernel=kernel,
                         batch_size=batch_size, max_degree=max_degree,
                         sample_size=sample_size, symm=symm, reg_weight=reg_weight)

        # load dataset
        self.load()

        # preprocess
        self.preprocess()

        # get generators
        self.train, self.type_spec = self.get_gen()
        self.val, _ = self.get_gen(mode='val')
        self.test, _ = self.get_gen(mode='test')

    def load(self):
        # load data
        rdataset = Reddit()
        self.final_info['num_classes'] = rdataset.n_labels
        graph = rdataset.graphs[0]
        self.feats, self.a, self.labels = graph.x, graph.a, graph.y

        # apply splits provided by dataset
        self.idx_tr = np.where(rdataset.mask_tr)[0]
        self.idx_va = np.where(rdataset.mask_va)[0]
        self.idx_te = np.where(rdataset.mask_te)[0]

        # need to clip for generator's type_spec
        self.idx_tr = self.idx_tr[:-np.mod(len(self.idx_tr), self.batch_size)]
        self.idx_va = self.idx_va[:-np.mod(len(self.idx_va), self.batch_size)]
        self.idx_te = self.idx_te[:-np.mod(len(self.idx_te), self.batch_size)]

        # lengths
        self.train_len = len(self.idx_tr)
        self.val_len = len(self.idx_va)
        self.test_len = len(self.idx_te)


###########################
# Link prediction datasets
###########################
class LinkDataset(Dataset):
    '''
    Wrapper for link prediction datasets

    negative_sample_size -- num. of negative samples per batch
    '''

    def __init__(self, layer_infos, kernel='gcn', batch_size=256, 
                 max_degree=-1, sample_size=15, symm=False, reg_weight=1e-9,
                 negative_sample_size=5):
        super().__init__(layer_infos, kernel=kernel, batch_size=batch_size, 
                         max_degree=max_degree, sample_size=sample_size, symm=symm, 
                         reg_weight=reg_weight)
        self.negative_sample_size = negative_sample_size
        self.pad_size = (2*self.batch_size+self.negative_sample_size)*self.sample_size**self.k

    def get_gen(self, mode='train'):
        '''
        Wrapper to build generator and corresponding type_spec
        '''
        n = self.a.shape[0]
        m = self.feats.shape[1]

        if mode == 'val':
            x = self.x1_val
            y = self.x2_val
        elif mode == 'test':
            x = self.x1_test
            y = self.x2_test
        else:
            x = self.x1_train
            y = self.x2_train

        def gen():
            '''
            Generator.  For each batch pair, sample the associated negative 
              sample batch, form the combined batch, then obtain the corresonding 
              label, feature and kernel matrices associated with the k-neighborhood 
              of the combined batch
            bx, by, bn - arrays, batch of x, y, and negative samples in 
              their respective sampled coordinates
            f_s - array, feature matrix induced by combined sample
            a_ks - csr_matrix, kernel matrix subgraph induced by combined sample
            y_s - array, dummy output matrix for compatibility
            '''
            for i in range(0, len(x), self.batch_size):
                batchx = x[i:i+self.batch_size]
                batchy = y[i:i+self.batch_size]
                # negative samples
                batchn, _, _ = tf.random.log_uniform_candidate_sampler(
                    true_classes=np.expand_dims(batchy, 1),
                    num_true=1,
                    num_sampled=self.negative_sample_size,
                    unique=False,
                    range_max=n)
                batchn = batchn.numpy()
                batch = pd.unique(np.concatenate([batchx, batchy, batchn]))

                # sample indices and build maparr
                nbd = batch.astype(np.int32)
                for _ in range(self.k):
                    tnbd = nbd
                    r, c = self.a_e[tnbd].nonzero()
                    rc = np.transpose(np.array([r,c]))
                    indices = np.unique(rc[:,0], return_index=True)[1]
                    rs = nbdwrap(rc, indices, tnbd, self.sample_size)
                    rc = np.concatenate(rs, axis=0)
                    nbd = pd.unique(rc[:,1]).astype(np.int32)
                r = tnbd[rc[:,0]]
                c = rc[:,1]
                nbdi = np.arange(len(nbd))
                maparr = sp.sparse.csr_matrix((nbdi+1, (nbd,np.zeros(len(nbd)))), 
                                              shape=(np.max(nbd)+1,1))

                # a_ks
                rr = maparr[r].data-1
                cc = maparr[c].data-1
                if self.kernel == 'gcn':
                    # A_{ks} <- D_{ks}^{-1} A_{ks}
                    d = self.scale(
                        sp.sparse.csr_matrix(
                            (np.ones(rr.shape), (rr,cc)), 
                            shape=(self.pad_size, self.pad_size))).data
                    a_ks = tf.cast(tf.sparse.SparseTensor(
                        indices=np.array([rr,cc]).T, values=d,
                        dense_shape=(self.pad_size, self.pad_size)), tf.float32)
                else:
                    a_ks = tf.cast(tf.sparse.SparseTensor(
                        indices=np.array([rr,cc]).T, values=np.ones(rr.shape),
                        dense_shape=(self.pad_size, self.pad_size)), tf.float32)

                # bx, by, bn
                bx = maparr[batchx].data-1
                by = maparr[batchy].data-1
                bn = maparr[batchn].data-1

                # f_s
                f_s = np.zeros((self.pad_size, m), dtype=self.feats.dtype)
                f_s[:len(nbd)] = self.feats[nbd]

                # y_s
                y_s = np.zeros(self.batch_size)

                yield (bx, by, bn, f_s, a_ks), y_s

        # type_spec
        type_spec = ((tf.TensorSpec(shape=(self.batch_size,), dtype=tf.int32),
                      tf.TensorSpec(shape=(self.batch_size,), dtype=tf.int32),
                      tf.TensorSpec(shape=(self.negative_sample_size,), dtype=tf.int32),
                      tf.TensorSpec(shape=(self.pad_size, m), dtype=tf.float32),
                      tf.SparseTensorSpec(shape=(self.pad_size, self.pad_size), dtype=tf.float32)),
                     tf.TensorSpec(shape=(self.batch_size,), dtype=tf.float32))

        return (gen, type_spec)


class BTERDataset(LinkDataset):
    '''
    BTER link prediction dataset

    train -- fraction of data to use for training
    val -- fraction of data to use for validation
    shuffle - whether or not to shuffle the data
    '''

    def __init__(self, path, layer_infos, kernel='gcn', batch_size=256,
                 max_degree=-1, sample_size=15, symm=False, reg_weight=1e-9,
                 train=.75, val=.15, shuffle=False, negative_sample_size=5):
        super().__init__(layer_infos, kernel=kernel, batch_size=batch_size, 
                         max_degree=max_degree, sample_size=sample_size, symm=symm, 
                         reg_weight=reg_weight, negative_sample_size=negative_sample_size)
        self.path = path
        self.train = train
        self.val = val
        self.shuffle = shuffle

        # load dataset
        self.load()

        # preprocess
        self.preprocess()

        # get generators
        self.train, self.type_spec = self.get_gen()
        self.val, _ = self.get_gen(mode='val')
        self.test, _ = self.get_gen(mode='test')

    def load(self):
        # load data
        gfile = os.path.join(self.path,'G.json')
        g = json_graph.node_link_graph(json.load(open(gfile)))

        # need to convert nodes back to ints
        ifile = os.path.join(self.path,'id_map.json')
        id_map = json.load(open(ifile))
        id_map = {k:int(v) for k,v in id_map.items()}
        g = nx.relabel_nodes(g, id_map)

        self.a = nx.linalg.graphmatrix.adjacency_matrix(g)

        ffile = os.path.join(self.path,'feats.npy')
        self.feats = np.load(ffile)

        wfile = os.path.join(self.path,'walks.txt')
        with open(wfile) as fp: 
            pairs = np.array([list(map(lambda x: int(x), line.split()))
                              for line in fp.readlines()])

        # create splits
        N = len(pairs)
        N_train, N_val = np.int(self.train*N), np.int(self.val*N)
        N_test = N - N_train - N_val
        if self.shuffle:
            pairs = pairs[np.random.permutation(np.arange(len(pairs)))]
        self.x1_train, self.x2_train = pairs[:N_train,0], pairs[:N_train,1]
        self.x1_val, self.x2_val = pairs[N_train:-N_test,0], pairs[N_train:-N_test,1]
        self.x1_test, self.x2_test = pairs[-N_test:,0], pairs[-N_test:,1]

        # need to clip for generator's type_spec
        self.x1_train = self.x1_train[:-np.mod(len(self.x1_train), self.batch_size)]
        self.x2_train = self.x2_train[:-np.mod(len(self.x2_train), self.batch_size)]
        self.x1_val = self.x1_val[:-np.mod(len(self.x1_val), self.batch_size)] 
        self.x2_val = self.x2_val[:-np.mod(len(self.x2_val), self.batch_size)] 
        self.x1_test = self.x1_test[:-np.mod(len(self.x1_test), self.batch_size)] 
        self.x2_test = self.x2_test[:-np.mod(len(self.x2_test), self.batch_size)]

        # lengths
        self.train_len = len(self.x1_train)
        self.val_len = len(self.x1_val)
        self.test_len = len(self.x1_test)


class LRedditDataset(LinkDataset):
    '''
    Reddit node prediction dataset, as provided by Spektral

    p -- fraction of random walk training data to use
    shuffle -- whether or not to shuffle the data
    '''

    def __init__(self, layer_infos, kernel='gcn', batch_size=256, 
                 max_degree=-1, sample_size=15, symm=False, reg_weight=1e-9,
                 shuffle=False, negative_sample_size=5, p=.1):
        super().__init__(layer_infos, kernel=kernel, batch_size=batch_size, 
                         max_degree=max_degree, sample_size=sample_size, symm=symm, 
                         reg_weight=reg_weight, negative_sample_size=negative_sample_size)
        self.p = np.clip(p,0,1)
        self.shuffle = shuffle

        # load dataset
        self.load()

        # preprocess
        self.preprocess()

        # get generators
        self.train, self.type_spec = self.get_gen() # no val/test for lreddit

    def load(self):
        rdataset = Reddit()
        graph = rdataset.graphs[0]
        self.feats, self.a = graph.x, graph.a

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
        pairs = np.loadtxt(walk_path, delimiter=' ').astype(int)
        if self.p < 1:
            lsize = len(pairs)
            if self.shuffle:
                sample = np.random.choice(lsize, int(self.p*lsize), replace=False)
            else:
                sample = np.arange(int(self.p*lsize)).astype(int)
            pairs = pairs[sample]
        self.x1_train, self.x2_train = pairs[:,0], pairs[:,1]

        # need to clip for generator's type_spec
        self.x1_train = self.x1_train[:-np.mod(len(self.x1_train), self.batch_size)]
        self.x2_train = self.x2_train[:-np.mod(len(self.x2_train), self.batch_size)]

        # lengths
        self.train_len = len(self.x1_train)
        self.val_len = 0
        self.test_len = 0