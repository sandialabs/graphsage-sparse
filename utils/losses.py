# This code is a TF 2.X refactor of the original GraphSAGE implementation:
# https://github.com/williamleif/GraphSAGE

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops


class Xent(tf.keras.losses.Loss):
    '''
    Custom cross entropy loss
    '''

    def __init__(self, neg_sample_size, neg_sample_weights=1.0, **kwargs):
        super(Xent, self).__init__()
        self.neg_sample_size = neg_sample_size
        self.neg_sample_weights = neg_sample_weights

    def call(self, y_true, y_pred):
        # y_true is unused, included for compatibility
        aff = tf.gather(y_pred, 0, axis=1)
        neg_aff = tf.gather(y_pred, np.arange(self.neg_sample_size)+1, axis=1)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_aff),
                                                                logits=neg_aff)
        loss = tf.reduce_mean(true_xent) + self.neg_sample_weights*tf.reduce_mean(
            tf.reduce_sum(negative_xent, axis=1))
        return loss


class Skipgram(tf.keras.losses.Loss):
    '''
    Custom skipgram loss
    '''

    def __init__(self):
        super(Skipgram, self).__init__()

    def call(self, y_true, y_pred):
        # y_true is unused, included for compatibility
        aff = y_pred[:,:1]
        neg_aff = y_pred[:,1:]
        neg_cost = tf.math.log(tf.reduce_sum(tf.math.exp(neg_aff), axis=1))
        loss = tf.reduce_mean(aff - neg_cost)
        return loss


class Hinge(tf.keras.losses.Loss):
    '''
    Custom hinge loss
    '''

    def __init__(self, margin=.1):
        super(Hinge, self).__init__()
        self.margin = margin

    def call(self, y_true, y_pred):
        # y_true is unused, included for compatibility
        aff = y_pred[:,:1]
        neg_aff = y_pred[:,1:]
        loss = tf.reduce_mean(tf.nn.relu(tf.subtract(neg_aff, aff - self.margin)))
        return loss


class SigmoidCrossEntropy(tf.keras.losses.Loss):
    '''
    Sigmoid cross-entropy with logits.
    Used when class labels are not mutually exclusive
    '''

    def __init__(self):
        super(SigmoidCrossEntropy, self).__init__()

    def call(self, y_true, y_pred):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        loss = tf.reduce_mean(loss, axis=1)
        return loss


class SoftmaxCrossEntropy(tf.keras.losses.Loss):
    '''
    Softmax cross-entropy with logits.
    Used when class labels are mutally exclusive
    '''

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def call(self, y_true, y_pred):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)

        return loss


class MRR(tf.keras.metrics.Metric):
    ''' Computes mean rank of affinity score vs. negative affinity scores
    '''

    def __init__(self, **kwargs):
        super(MRR, self).__init__()
        self.total_rank = self.add_weight(name='tr', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true is unused, included for compatibility
        _, indices_of_ranks = tf.nn.top_k(y_pred, k=y_pred.shape[1])
        _, ranks = tf.nn.top_k(-indices_of_ranks, k=y_pred.shape[1])
        ranks = tf.math.divide(1.0, tf.cast(ranks[:,0]+1, dtype=tf.float32))
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, ranks.dtype)
            sample_weight = tf.broadcast_to(sample_weight, ranks.shape)
            ranks = tf.multiply(sample_weight, ranks)
        self.total_rank.assign_add(tf.reduce_sum(ranks))
        self.count.assign_add(tf.cast(array_ops.size(ranks), dtype=tf.float32))

    def result(self):
        return tf.math.divide(self.total_rank, self.count)

    def reset_states(self):
        self.total_rank.assign(0)
        self.count.assign(0)

