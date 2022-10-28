# This code is a TF 2.X refactor of the original GraphSAGE implementation:
# https://github.com/williamleif/GraphSAGE

# Additional updates to implement OGB datasets + Spektral wrapper
# https://ogb.stanford.edu/
# https://github.com/danielegrattarola/spektral

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from .aggregators import *


def buildModel(layer_infos, type_spec, reg_weight=1e-9):
    '''
    Implementation of unsupervised GraphSAGE using Keras functional API
    NOTE:  shaping assumes that model is fit under distributed policy,
      even for only 1 device
    '''
    indsx = tf.keras.Input(shape=list(type_spec[0][0].shape)[1:], dtype=tf.int32)
    indsy = tf.keras.Input(shape=list(type_spec[0][1].shape)[1:], dtype=tf.int32)
    indsn = tf.keras.Input(shape=list(type_spec[0][2].shape)[1:], dtype=tf.int32)
    feats = tf.keras.Input(shape=list(type_spec[0][3].shape)[1:], dtype=tf.float32)
    adj = tf.keras.Input(shape=list(type_spec[0][4].shape)[1:], sparse=True, dtype=tf.float32)

    # define layers
    layers = []
    for k in range(len(layer_infos)):
        agg_type = layer_infos[k]['agg_type']
        output_dim = layer_infos[k]['agg_dim']
        use_bias = layer_infos[k]['use_bias']

        # use_norm = layer_infos[k]['use_norm']

        ################################################
        # modification for consistency with benchmarks:
        # only use normalization for final layer
        if k == len(layer_infos)-1:
            use_norm = False
        else:
            use_norm = True
        ################################################

        if 'agg_dropout' in layer_infos[k].keys():
            dropout = layer_infos[k]['agg_dropout']
        else:
            dropout = 0

        if k == 0:
            input_shape = feats.shape
        else:
            dim_mult = 1
            if 'concat' in layer_infos[k-1].keys():
                if layer_infos[k-1]['concat']:
                    dim_mult = 2
            input_shape = (feats.shape[0], dim_mult*layer_infos[k-1]['agg_dim'])

        if k == len(layer_infos)-1:
            act = lambda x:x
        else:
            act = tf.nn.relu

        if agg_type == 'gcn':
            layers.append(GCNAggregator(input_shape, output_dim, act=act, use_norm=use_norm,
                                        use_bias=use_bias, dropout=dropout, reg_weight=reg_weight))
        else:
            raise Exception("Unknown aggregator: ", layer_infos[k].agg_type)

    # call
    f = feats
    for layer in layers:
        f = layer([f, adj])
    outx = tf.gather(f, indsx)
    outy = tf.gather(f, indsy)
    outn = tf.gather(f, indsn)

    score = tf.expand_dims(tf.reduce_sum(outx*outy, axis=1), axis=1)
    neg_cost = tf.matmul(outx, tf.transpose(outn))
    outputs = tf.concat([score, neg_cost], axis=1)

    model = Model(inputs=[indsx, indsy, indsn, feats, adj], outputs=outputs)

    return model


def buildSupModel(layer_infos, final_info, type_spec, reg_weight=1e-9):
    '''
    Implementation of supervised GraphSAGE using Keras functional API
    NOTE:  shaping assumes that model is fit under distributed policy,
      even for only 1 device
    '''
    inds = tf.keras.Input(shape=list(type_spec[0][0].shape)[1:], dtype=tf.int32)
    feats = tf.keras.Input(shape=list(type_spec[0][1].shape)[1:], dtype=tf.float32)
    adj = tf.keras.Input(shape=list(type_spec[0][2].shape)[1:], sparse=True, dtype=tf.float32)

    # define layers
    layers = []
    for k in range(len(layer_infos)):
        agg_type = layer_infos[k]['agg_type']
        output_dim = layer_infos[k]['agg_dim']
        use_bias = layer_infos[k]['use_bias']
        use_norm = layer_infos[k]['use_norm']
        if 'agg_dropout' in layer_infos[k].keys():
            dropout = layer_infos[k]['agg_dropout']
        else:
            dropout = 0

        if k == 0:
            input_shape = feats.shape
        else:
            dim_mult = 1
            if 'concat' in layer_infos[k-1].keys():
                if layer_infos[k-1]['concat']:
                    dim_mult = 2
            input_shape = (feats.shape[0], dim_mult*layer_infos[k-1]['agg_dim'])

        if agg_type == 'gcn':
            layers.append(GCNAggregator(input_shape, output_dim, act=tf.nn.relu,
                                        use_bias=use_bias, use_norm=use_norm, 
                                        dropout=dropout, reg_weight=reg_weight))
        else:
            raise Exception("Unknown aggregator: ", layer_infos[k].agg_type)

    # final output layer
    dim_mult = 1
    if 'concat' in layer_infos[k-1].keys():
        if layer_infos[k-1]['concat']:
            dim_mult = 2
    input_shape = (inds.shape[0], dim_mult*layer_infos[-1]['agg_dim'])
    output_dim = final_info['num_classes']
    final_layer = Dense(output_dim, input_shape=input_shape, activation=lambda x:x,
                        kernel_regularizer=l2(reg_weight))

    # call
    f = feats
    for layer in layers:
        f = layer([f, adj])
    f = tf.gather(f, inds)
    outputs = final_layer(f)

    model = Model(inputs=[inds, feats, adj], outputs=outputs)

    return model