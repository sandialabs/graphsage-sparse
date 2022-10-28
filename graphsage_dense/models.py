# This code is a TF 2.X refactor of the original GraphSAGE implementation:
# https://github.com/williamleif/GraphSAGE

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.regularizers import l2
from .aggregators import *
from .samplers import NegativeSampler, UniformSampler


# Alternate model definition scheme to use Keras functional API ##
def buildEmbed(inputs, features, sample, layers, normalizers):
    '''
    Function to define embedding component of computation graph.
    For use as part of buildModel, buildSupModel
    '''
    # sample to get convolution supports
    print(inputs)
    samples = sample(inputs)
    print(samples)
    features = [tf.nn.embedding_lookup(features, sample) for sample in samples]
    features.reverse()  # for popping later

    # perform convolution for each set of inputs
    # at each stage:  [A_k, A'_k, B_k, B'_k, C_k, C'_k, ...] 
    #  -> [A_{k+1}, B_{k+1}, C_{k+1}, ...]
    # see Sample class definition for more details
    ind = 0
    for k, layer in enumerate(layers):
        next_features = []
        for i in range(2**(len(layers)-k-1)):
            self_vecs, neigh_vecs = features.pop(), features.pop()
            outputs = layer([self_vecs, neigh_vecs])

            # need to handle batch normalization and dropout here
            # suggsted to apply dropout after bn to not leak statistics
            # if layer.un:
            #     outputs = normalizers[ind](outputs)

            ################################################
            # modification for consistency with benchmarks:
            # only use normalization for final layer
            if k == len(layers)-1:
                outputs = normalizers[ind](outputs)
            ################################################

            if layer.training:
                outputs = tf.nn.dropout(outputs, rate=layer.dropout)
            ind += 1

            next_features.append(outputs)
        features = next_features
        features.reverse()

    features = features.pop()
    features = tf.reshape(features, [-1, features.shape[-1]])

    return features


def buildModel(features, adj_info, layer_infos, neg_sample_size, 
               bilinear_weights=False, reg_weight=1e-9):
    '''
    Implementation of unsupervised GraphSAGE using Keras functional API
    '''
    features = tf.cast(tf.convert_to_tensor(features), tf.float32)
    adj_info = tf.cast(tf.convert_to_tensor(adj_info), tf.int32)

    # build sample layers
    sample = UniformSampler(adj_info, layer_infos)
    neg_sample = NegativeSampler(neg_sample_size, len(features))

    layers = []
    for k in range(len(layer_infos)):
        output_dim = layer_infos[k]['agg_dim']
        agg_type = layer_infos[k]['agg_type']
        use_bias = layer_infos[k]['use_bias']
        use_norm = layer_infos[k]['use_norm']
        if 'agg_dropout' in layer_infos[k].keys():
            dropout = layer_infos[k]['agg_dropout']
        else:
            dropout = 0
        concat = False
        if 'concat' in layer_infos[k].keys():
            concat = layer_infos[k]['concat']

        if k == 0:
            input_dim = features.shape[1]
        else:
            dim_mult = 1
            if 'concat' in layer_infos[k-1].keys():
                if layer_infos[k-1]['concat']:
                    dim_mult = 2
            input_dim = dim_mult*layer_infos[k-1]['agg_dim']

        if k == len(layer_infos)-1:
            act = lambda x:x
        else:
            act = tf.nn.relu

        if agg_type == 'mean':
            layers.append(MeanAggregator(input_dim, output_dim, act=act, 
                                         concat=concat, dropout=dropout, 
                                         use_bias=use_bias, use_norm=use_norm, 
                                         reg_weight=reg_weight))
        elif agg_type =='seq':
            hidden_size = layer_infos[k]['agg_size']
            layers.append(SeqAggregator(input_dim, output_dim, hidden_size, 
                                        act=act, concat=concat, dropout=dropout, 
                                        use_bias=use_bias, use_norm=use_norm, 
                                        reg_weight=reg_weight))
        elif agg_type == 'maxpool':
            hidden_size = layer_infos[k]['agg_size']
            num_samples = layer_infos[k]['sample_size']
            layers.append(MaxPoolingAggregator(input_dim, output_dim, num_samples, 
                                               hidden_size, act=act, concat=concat, 
                                               dropout=dropout, use_bias=use_bias, 
                                               use_norm=use_norm, reg_weight=reg_weight))
        elif agg_type == 'meanpool':
            hidden_size = layer_infos[k]['agg_size']
            num_samples = layer_infos[k]['sample_size']
            layers.append(MeanPoolingAggregator(input_dim, output_dim, num_samples, 
                                                hidden_size, act=act, 
                                                concat=concat, dropout=dropout, 
                                                use_bias=use_bias, use_norm=use_norm, 
                                                reg_weight=reg_weight))
        elif agg_type == 'gcn':
            layers.append(GCNAggregator(input_dim, output_dim, act=act, 
                                        dropout=dropout, use_bias=use_bias, 
                                        use_norm=use_norm, reg_weight=reg_weight))
        else:
            raise Exception("Unknown aggregator: ", layer_infos[k].agg_type)

    # need to handle BatchNormalization at the model level
    #   since each layer is generates multiply-shaped outputs
    normalizers = [BatchNormalization() for _ in range(2**(len(layers))-1)]

    # build affinity layer (if needed)
    if bilinear_weights:
        dim_mult = 1
        if 'concat' in layer_infos[k-1].keys():
            if layer_infos[k-1]['concat']:
                dim_mult = 2
        output_dim = dim_mult*layer_infos[-1]['agg_dim']
        affinity = Dense(output_dim, input_shape=(None, output_dim), 
                         use_bias=False, activation=lambda x:x, 
                         kernel_regularizer=l2(reg_weight))

    # chain layers to create model output
    inputs = tf.keras.Input(shape=(2,), dtype=tf.int32)
    inputs0 = tf.gather(inputs, 0, axis=1)
    inputs1 = tf.gather(inputs, 1, axis=1)

    # get embeddings for both sets of inputs and negative samples
    features0 = buildEmbed(inputs0, features, sample, layers, normalizers)
    features1 = buildEmbed(inputs1, features, sample, layers, normalizers)
    neg_inputs = neg_sample(tf.expand_dims(inputs1, axis=1))
    neg_features = buildEmbed(neg_inputs, features, sample, layers, normalizers)

    # compute affinities for loss
    # concatenate as [affinity, neg_cost] for loss downstream
    if bilinear_weights:
        features0 = affinity(features0)

    score = tf.expand_dims(tf.reduce_sum(features1*features0, axis=1), axis=1)
    neg_cost = tf.matmul(features0, tf.transpose(neg_features))

    outputs = tf.concat([score, neg_cost], axis=1)
    model = Model(inputs=inputs, outputs=outputs)

    return model


def buildSupModel(features, adj_info, layer_infos, final_info, reg_weight=1e-9):
    '''
    Implementation of supervised GraphSAGE using Keras functional API
    '''
    features = tf.cast(tf.convert_to_tensor(features), tf.float32)
    adj_info = tf.cast(tf.convert_to_tensor(adj_info), tf.int32)

    # build sample layers
    sample = UniformSampler(adj_info, layer_infos)

    layers = []
    for k in range(len(layer_infos)):
        output_dim = layer_infos[k]['agg_dim']
        agg_type = layer_infos[k]['agg_type']
        use_bias = layer_infos[k]['use_bias']
        use_norm = layer_infos[k]['use_norm']
        if 'agg_dropout' in layer_infos[k].keys():
            dropout = layer_infos[k]['agg_dropout']
        else:
            dropout = 0
        concat = False
        if 'concat' in layer_infos[k].keys():
            concat = layer_infos[k]['concat']

        if k == 0:
            input_dim = features.shape[1]
        else:
            dim_mult = 1
            if 'concat' in layer_infos[k-1].keys():
                if layer_infos[k-1]['concat']:
                    dim_mult = 2
            input_dim = dim_mult*layer_infos[k-1]['agg_dim']

        if k == len(layer_infos)-1:
            act = lambda x:x
        else:
            act = tf.nn.relu

        if agg_type == 'mean':
            layers.append(MeanAggregator(input_dim, output_dim, act=act, 
                                         concat=concat, dropout=dropout, 
                                         use_bias=use_bias, use_norm=use_norm, 
                                         reg_weight=reg_weight))
        elif agg_type =='seq':
            hidden_size = layer_infos[k]['agg_size']
            layers.append(SeqAggregator(input_dim, output_dim, hidden_size, 
                                        act=act, concat=concat, dropout=dropout, 
                                        use_bias=use_bias, use_norm=use_norm, 
                                        reg_weight=reg_weight))
        elif agg_type == 'maxpool':
            hidden_size = layer_infos[k]['agg_size']
            num_samples = layer_infos[k]['sample_size']
            layers.append(MaxPoolingAggregator(input_dim, output_dim, num_samples, 
                                               hidden_size, act=act, 
                                               concat=concat, dropout=dropout, 
                                               use_bias=use_bias, use_norm=use_norm, 
                                               reg_weight=reg_weight))
        elif agg_type == 'meanpool':
            hidden_size = layer_infos[k]['agg_size']
            num_samples = layer_infos[k]['sample_size']
            layers.append(MeanPoolingAggregator(input_dim, output_dim, num_samples, 
                                                hidden_size, act=act, 
                                                concat=concat, dropout=dropout, 
                                                use_bias=use_bias, use_norm=use_norm, 
                                                reg_weight=reg_weight))
        elif agg_type == 'gcn':
            layers.append(GCNAggregator(input_dim, output_dim, act=act,
                                        use_bias=use_bias, use_norm=use_norm, 
                                        dropout=dropout, reg_weight=reg_weight))
        else:
            raise Exception("Unknown aggregator: ", layer_infos[k].agg_type)

    # need to handle BatchNormalization at the model level
    #   since each layer is generates multiply-shaped outputs
    normalizers = [BatchNormalization() for _ in range(2**(len(layers))-1)]

    # final output layer
    dim_mult = 1
    if 'concat' in layer_infos[k-1].keys():
        if layer_infos[k-1]['concat']:
            dim_mult = 2
    input_dim = dim_mult*layer_infos[-1]['agg_dim']
    output_dim = final_info['num_classes']
    final_layer = Dense(output_dim, input_shape=(None, input_dim),
                        activation=lambda x:x, kernel_regularizer=l2(reg_weight))

    # chain layers to create model output
    inputs = tf.keras.Input(shape=(1,), dtype=tf.int32)

    # get embeddings
    features = buildEmbed(inputs, features, sample, layers, normalizers)

    # compute logits for loss
    outputs = final_layer(features)

    model = Model(inputs=inputs, outputs=outputs)

    return model
