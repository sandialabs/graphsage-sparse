# This code is a TF 2.X refactor of the original GraphSAGE implementation:
# https://github.com/williamleif/GraphSAGE

# Additional updates to implement OGB datasets + Spektral wrapper
# https://ogb.stanford.edu/
# https://github.com/danielegrattarola/spektral

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization
from tensorflow.keras.regularizers import l2


class GCNAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used for self vector and neighbor vectors.
    """

    def __init__(self, input_shape, output_dim, dropout=0., use_norm=True, 
                 use_bias=True, act=tf.nn.relu, reg_weight=1e-9, **kwargs):
        super(GCNAggregator, self).__init__()
        self.dropout = dropout
        self.act = act
        self.layer = Dense(output_dim, input_shape=input_shape, 
                           use_bias=use_bias, kernel_regularizer=l2(reg_weight))
        self.un = use_norm
        if self.un:
            self.bn = BatchNormalization()

    def call(self, inputs, training=None):
        # x: features, a: (D+I)^{-1}(A+I)
        x, a = inputs
        x = tf.sparse.sparse_dense_matmul(a, x)
        x = self.act(self.layer(x))
        # suggsted to apply dropout after bn to not leak statistics
        if self.un:
            x = self.bn(x)
        if training:
            x = tf.nn.dropout(x, rate=self.dropout)

        return x
