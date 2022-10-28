# This code is a TF 2.X refactor of the original GraphSAGE implementation:
# https://github.com/williamleif/GraphSAGE

import tensorflow as tf
from tensorflow.keras.layers import Layer


class NegativeSampler(Layer):
    """ Perform negative sampling of labels for input to loss function
    """
    def __init__(self, neg_sample_size, range_max, num_true=1, unique=False, **kwargs):
        super(NegativeSampler, self).__init__()
        self.neg_sample_size = neg_sample_size
        self.range_max = range_max
        self.num_true = num_true
        self.unique = unique
        
    def call(self, inputs):
        neg_samples, _, _ = tf.random.log_uniform_candidate_sampler(
            true_classes = tf.cast(inputs, tf.int64),
            num_true = self.num_true,
            num_sampled = self.neg_sample_size,
            unique = self.unique,
            range_max = self.range_max,
            name='neg_sample')
        
        return tf.cast(neg_samples, tf.int32)

class UniformSampler(Layer):
    """
    Implements batch sampling (Alg 2. of Hamilton, et al. 2017)
    Schematic: let B be inital batch of samples
    - B 
    - [B, N_k(B)]
    - [B, N_{k-1}(B), N_k(B), N_{k-1}N_k(B)]
    - [B, N_{k-2}(B), N_{k-1}(B), N_{k-2}N_{k-1}(B), 
       N_k(B), N_{k-2}N_k(B), N_{k-1}N_k(B), N_{k-2}N_{k-1}N_k(B)]
    Subsets are reduced in aggregation stage of model call
    """
    def __init__(self, adj_info, layer_infos, **kwargs):
        super(UniformSampler, self).__init__()
        self.adj_info = adj_info
        self.num_adj = self.adj_info.shape[-1]
        self.layer_infos = layer_infos
        
    def call(self, inputs):
        samples = [inputs]
        for k in range(len(self.layer_infos)):
            t = len(self.layer_infos) - k - 1 # go from h_{k-1} to h_0
            num_samples = self.layer_infos[t]['sample_size']
            new_samples = []
            for sample in samples:
                indices = tf.gather(tf.argsort(tf.random.uniform([self.num_adj])),
                                    tf.range(num_samples))
                # sample adj before sampling neighbors - saves memory
                adj = tf.gather(self.adj_info, indices, axis=-1)
                new_samples.append(tf.gather(adj, sample, axis=0))
            samples = [v for p in zip(samples, new_samples) for v in p]
            
        return samples
