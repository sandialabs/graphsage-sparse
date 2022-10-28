# This code is a TF 2.X refactor of the original GraphSAGE implementation:
# https://github.com/williamleif/GraphSAGE

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LSTMCell, RNN
from tensorflow.keras.regularizers import l2


class MeanAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, input_dim, output_dim, neigh_dim=None, 
                 use_bias=False, use_norm=True, act=tf.nn.relu, concat=False, 
                 dropout=0, reg_weight=1e-9, **kwargs):
        super(MeanAggregator, self).__init__()

        self.concat = concat
        self.act = act
        if neigh_dim is None:
            neigh_dim = input_dim
        self.dropout = dropout
        self.un = use_norm
        self.neigh_layer = Dense(output_dim, input_shape=(None, neigh_dim), 
                                 use_bias=use_bias, 
                                 kernel_regularizer=l2(reg_weight))
        self.self_layer = Dense(output_dim, input_shape=(None, input_dim), 
                                use_bias=use_bias,
                                kernel_regularizer=l2(reg_weight))

    def call(self, inputs, training=None):
        self_vecs, neigh_vecs = inputs
        self_dims = self_vecs.shape
        self_vecs = tf.reshape(self_vecs, [-1, self_dims[-1]])
        neigh_dims = neigh_vecs.shape
        neigh_vecs = tf.reshape(neigh_vecs, [-1, neigh_dims[-2], neigh_dims[-1]])
        neigh_vecs = tf.reduce_mean(neigh_vecs, axis=1)
        neigh_vecs = self.neigh_layer(neigh_vecs)
        self_vecs = self.self_layer(self_vecs)

        if not self.concat:
            output = tf.add_n([self_vecs, neigh_vecs])
        else:
            output = tf.concat([self_vecs, neigh_vecs], axis=1)

        output_dims = self_dims.as_list()
        output_dims[0] = -1
        output_dims[-1] = output.shape[-1]
        output = tf.reshape(output, output_dims)
        output = self.act(output)
        # externally signal training state
        if training:
            self.training = True
        else:
            self.training = False
        return output


class GCNAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used for self vector and neighbor vectors.
    """

    def __init__(self, input_dim, output_dim, dropout=0., 
                 use_bias=False, use_norm=True, act=tf.nn.relu, reg_weight=1e-9, **kwargs):
        super(GCNAggregator, self).__init__()

        self.dropout = dropout
        self.act = act
        self.un = use_norm
        self.layer = Dense(output_dim, input_shape=(None, input_dim), 
                           use_bias=use_bias, 
                           kernel_regularizer=l2(reg_weight))

    def call(self, inputs, training=None):
        self_vecs, neigh_vecs = inputs
        self_dims = self_vecs.shape
        self_vecs = tf.reshape(self_vecs, [-1, self_dims[-1]])
        neigh_dims = neigh_vecs.shape
        neigh_vecs = tf.reshape(neigh_vecs, [-1, neigh_dims[-2], neigh_dims[-1]])
        means = tf.reduce_mean(tf.concat([neigh_vecs, 
                                          tf.expand_dims(self_vecs, axis=1)],
                                         axis=1), axis=1)

        output = self.layer(means)
        output_dims = self_dims.as_list()
        output_dims[0] = -1
        output_dims[-1] = output.shape[-1]
        output = tf.reshape(output, output_dims)
        output = self.act(output)
        # externally signal training state
        if training:
            self.training = True
        else:
            self.training = False
        return output


class MaxPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """

    def __init__(self, input_dim, output_dim, num_neighbors, 
                 hidden_dim, neigh_dim=None, dropout=0., 
                 use_bias=False, use_norm=True, act=tf.nn.relu, concat=False, 
                 reg_weight=1e-9, **kwargs):
        super(MaxPoolingAggregator, self).__init__()

        self.concat = concat
        self.act = act
        if neigh_dim is None:
            neigh_dim = input_dim
        self.num_neighbors = num_neighbors
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.un = use_norm
        self.mlp_layer = Dense(hidden_dim, input_shape=(None, neigh_dim), 
                               use_bias=use_bias, activation=tf.nn.relu, 
                               kernel_regularizer=l2(reg_weight))
        self.neigh_layer = Dense(output_dim, input_shape=(None, hidden_dim), 
                                 use_bias=use_bias, 
                                 kernel_regularizer=l2(reg_weight))
        self.self_layer = Dense(output_dim, input_shape=(None, input_dim), 
                                use_bias=use_bias, 
                                kernel_regularizer=l2(reg_weight))

    def call(self, inputs, training=None):
        self_vecs, neigh_vecs = inputs
        self_dims = self_vecs.shape
        self_vecs = tf.reshape(self_vecs, [-1, self_dims[-1]])
        neigh_dims = neigh_vecs.shape

        # [nodes * num_neighbors] x [neigh_dim]
        neigh_vecs = tf.reshape(neigh_vecs, [-1, neigh_dims[-1]])
        # [nodes * num_neighbors] x [hidden_dim]
        neigh_vecs = self.mlp_layer(neigh_vecs)
        # [nodes] x [num_neighbors] x [hidden_dim]
        neigh_vecs = tf.reshape(neigh_vecs, [-1, self.num_neighbors, 
                                             self.hidden_dim])
        # [nodes] x [hidden_dim]
        neigh_vecs = tf.reduce_max(neigh_vecs, axis=1)
        neigh_vecs = self.neigh_layer(neigh_vecs)
        self_vecs = self.self_layer(self_vecs)

        if not self.concat:
            output = tf.add_n([self_vecs, neigh_vecs])
        else:
            output = tf.concat([self_vecs, neigh_vecs], axis=1)
        output_dims = self_dims.as_list()
        output_dims[0] = -1
        output_dims[-1] = output.shape[-1]
        output = tf.reshape(output, output_dims)
        output = self.act(output)
        # externally signal training state
        if training:
            self.training = True
        else:
            self.training = False
        return output


class MeanPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """

    def __init__(self, input_dim, output_dim, num_neighbors, 
                 hidden_dim, neigh_dim=None, dropout=0., 
                 use_bias=False, use_norm=True, act=tf.nn.relu, concat=False, 
                 reg_weight=1e-9, **kwargs):
        super(MeanPoolingAggregator, self).__init__()

        self.concat = concat
        self.act = act
        if neigh_dim is None:
            neigh_dim = input_dim
        self.num_neighbors = num_neighbors
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.un = use_norm
        self.mlp_layer = Dense(hidden_dim, input_shape=(None, neigh_dim), 
                               use_bias=use_bias, activation=tf.nn.relu, 
                               kernel_regularizer=l2(reg_weight))
        self.neigh_layer = Dense(output_dim, input_shape=(None, hidden_dim), 
                                 use_bias=use_bias, 
                                 kernel_regularizer=l2(reg_weight))
        self.self_layer = Dense(output_dim, input_shape=(None, input_dim), 
                                use_bias=use_bias, 
                                kernel_regularizer=l2(reg_weight))

    def call(self, inputs, training=None):
        self_vecs, neigh_vecs = inputs
        self_dims = self_vecs.shape
        self_vecs = tf.reshape(self_vecs, [-1, self_dims[-1]])
        neigh_dims = neigh_vecs.shape

        # [nodes * num_neighbors] x [neigh_dim]
        neigh_vecs = tf.reshape(neigh_vecs, [-1, neigh_dims[-1]])
        # [nodes * num_neighbors] x [hidden_dim]
        neigh_vecs = self.mlp_layer(neigh_vecs)
        # [nodes] x [num_neighbors] x [hidden_dim]
        neigh_vecs = tf.reshape(neigh_vecs, [-1, self.num_neighbors, 
                                             self.hidden_dim])
        # [nodes] x [hidden_dim]
        neigh_vecs = tf.reduce_mean(neigh_vecs, axis=1)
        neigh_vecs = self.neigh_layer(neigh_vecs)
        self_vecs = self.self_layer(self_vecs)

        if not self.concat:
            output = tf.add_n([self_vecs, neigh_vecs])
        else:
            output = tf.concat([self_vecs, neigh_vecs], axis=1)
        output_dims = self_dims.as_list()
        output_dims[0] = -1
        output_dims[-1] = output.shape[-1]
        output = tf.reshape(output, output_dims)
        output = self.act(output)
        # externally signal training state
        if training:
            self.training = True
        else:
            self.training = False
        return output


class TwoMaxLayerPoolingAggregator(Layer):
    """ Aggregates via pooling over two MLP functions.
    """

    def __init__(self, input_dim, output_dim, num_neighbors, 
                 hidden_dim_1, hidden_dim_2, neigh_dim=None, dropout=0., 
                 use_bias=False, use_norm=True, act=tf.nn.relu, concat=False, 
                 reg_weight=1e-9, **kwargs):
        super(TwoMaxLayerPoolingAggregator, self).__init__()

        self.concat = concat
        self.act = act
        if neigh_dim is None:
            neigh_dim = input_dim
        self.num_neighbors = num_neighbors
        self.dropout = dropout
        self.hidden_dim_2 = hidden_dim_2
        self.un = use_norm
        self.mlp_layer_0 = Dense(hidden_dim_1, input_shape=(None, neigh_dim), 
                                 use_bias=use_bias, activation=tf.nn.relu, 
                                 kernel_regularizer=l2(reg_weight))
        self.mlp_layer_1 = Dense(hidden_dim_2, input_shape=(None, hidden_dim_1), 
                                 use_bias=use_bias, activation=tf.nn.relu, 
                                 kernel_regularizer=l2(reg_weight))
        self.neigh_layer = Dense(output_dim, input_shape=(None, hidden_dim_2), 
                                 use_bias=use_bias, 
                                 kernel_regularizer=l2(reg_weight))
        self.self_layer = Dense(output_dim, input_shape=(None, input_dim), 
                                use_bias=use_bias, 
                                kernel_regularizer=l2(reg_weight))

    def call(self, inputs, training=None):
        self_vecs, neigh_vecs = inputs
        self_dims = self_vecs.shape
        self_vecs = tf.reshape(self_vecs, [-1, self_dims[-1]])
        neigh_dims = neigh_vecs.shape

        # [nodes * num_neighbors] x [neigh_dim]
        neigh_vecs = tf.reshape(neigh_vecs, [-1, neigh_dims[-1]])
        # [nodes * num_neighbors] x [hidden_dim_1]
        neigh_vecs = self.mlp_layer_0(neigh_vecs)
        # [nodes * num_neighbors] x [hidden_dim_2]
        neigh_vecs = self.mlp_layer_1(neigh_vecs)
        # [nodes] x [num_neighbors] x [hidden_dim_2]
        neigh_vecs = tf.reshape(neigh_vecs, [-1, self.num_neighbors, 
                                             self.hidden_dim_2])
        # [nodes] x [hidden_dim_2]
        neigh_vecs = tf.reduce_max(neigh_vecs, axis=1)
        neigh_vecs = self.neigh_layer(neigh_vecs)
        self_vecs = self.self_layer(self_vecs)

        if not self.concat:
            output = tf.add_n([self_vecs, neigh_vecs])
        else:
            output = tf.concat([self_vecs, neigh_vecs], axis=1)
        output_dims = self_dims.as_list()
        output_dims[0] = -1
        output_dims[-1] = output.shape[-1]
        output = tf.reshape(output, output_dims)
        output = self.act(output)
        # externally signal training state
        if training:
            self.training = True
        else:
            self.training = False
        return output


class SeqAggregator(Layer):
    """ Aggregates via a standard LSTM.
    """

    def __init__(self, input_dim, output_dim, hidden_dim, 
                 dropout=0., use_bias=False, use_norm=True, act=tf.nn.relu, 
                 concat=False, reg_weight=1e-9, **kwargs):
        super(SeqAggregator, self).__init__()

        self.concat = concat
        self.act = act
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.un = use_norm
        self.cell = LSTMCell(hidden_dim)
        self.rnn = RNN(self.cell, time_major=False)
        self.neigh_layer = Dense(output_dim, input_shape=(None, hidden_dim), 
                                 use_bias=use_bias, 
                                 kernel_regularizer=l2(reg_weight))
        self.self_layer = Dense(output_dim, input_shape=(None, input_dim), 
                                use_bias=use_bias, 
                                kernel_regularizer=l2(reg_weight))

    def call(self, inputs, training=None):
        self_vecs, neigh_vecs = inputs
        self_dims = self_vecs.shape
        self_vecs = tf.reshape(self_vecs, [-1, self_dims[-1]])
        neigh_dims = neigh_vecs.shape
        neigh_vecs = tf.reshape(neigh_vecs, [-1, neigh_dims[-2], neigh_dims[-1]])
        neigh_vecs = self.rnn(neigh_vecs)
        neigh_vecs = self.neigh_layer(neigh_vecs)
        self_vecs = self.self_layer(self_vecs)

        if not self.concat:
            output = tf.add_n([self_vecs, neigh_vecs])
        else:
            output = tf.concat([self_vecs, neigh_vecs], axis=1)
        output_dims = self_dims.as_list()
        output_dims[0] = -1
        output_dims[-1] = output.shape[-1]
        output = tf.reshape(output, output_dims)
        output = self.act(output)
        # externally signal training state
        if training:
            self.training = True
        else:
            self.training = False
        return output
