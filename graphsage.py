import sys, os
import argparse
import numpy as np
import tensorflow as tf
import time

projdir = os.getcwd()
sys.path.insert(0, projdir)

import utils.callbacks as callbacks
import utils.losses as losses
import graphsage_dense.models as dense_models
import graphsage_dense.datasets as dense_datasets
import graphsage_sparse.models as sparse_models
import graphsage_sparse.datasets as sparse_datasets
from spektral.datasets.utils import DATASET_FOLDER

###########################
# Globals
###########################
UNSUP_DNAMES = ['bter', 'lreddit']
SUP_DNAMES = ['arxiv', 'nreddit']
BTER_PATH = DATASET_FOLDER + '/bter'


###########################
# Strategies
###########################
def get_strategy(config):
    '''
    Get distribution strategy for specified accelerator
    '''
    numrep = config['numrep']
    platform = config['platform']
    if platform == 'gpu':
        devices = ['/gpu:%d'%d for d in range(numrep)]
        strategy = tf.distribute.MirroredStrategy(devices=devices)
    elif platform == 'gc':
        from tensorflow.python import ipu
        cfg = ipu.utils.IPUConfig()
        cfg.auto_select_ipus = numrep
        cfg.configure_ipu_system()
        strategy = ipu.ipu_strategy.IPUStrategy()
    return strategy


###########################
# Configuration
###########################
def get_layer_infos(config):
    '''
    Build layer_infos specification and add to config dictionary
    '''
    layer_infos = []
    for _ in range(config['depth']):
        layer_info = {'agg_dropout': config['dropout'], 
                      'agg_dim': config['agg_dim'],
                      'concat': config['concat'],
                      'use_norm': config['use_norm'], 
                      'use_bias': config['use_bias']}
        if config['algorithm'] == 'dense':  # dense samples are per-layer
            layer_info['sample_size'] = config['sample_size']
        layer_infos.append(layer_info)
    config['layer_infos'] = layer_infos
    if config['dname'] in SUP_DNAMES:
        config['final_info'] = {'dropout': config['dropout']}
    return config


###########################
# Datasets & Preprocessing
###########################
def get_dataset_fn(mode, dataset):
    '''
    Tensorflow dataset generator
    '''
    if mode == 'train':
        gen = dataset.train
    elif mode == 'val':
        gen = dataset.val
    elif mode == 'test':
        gen = dataset.test

    def dataset_fn(input_context):
        tf_dataset = tf.data.Dataset.from_generator(
            gen, output_signature=dataset.type_spec)
        tf_dataset = tf_dataset.shard(
            input_context.num_input_pipelines, input_context.input_pipeline_id)
        tf_dataset = tf_dataset.cache().repeat().prefetch(
            tf.data.experimental.AUTOTUNE)
        return tf_dataset
    return dataset_fn


def get_dataset(config):
    '''
    Load dataset class corresponding to specified experiment
    '''
    # common params
    alg = config['algorithm']
    dname = config['dname']
    layer_infos = config['layer_infos']
    max_degree = config['max_degree']
    sample_size = config['sample_size']
    batch_size = config['batch_size']
    kernel = config['kernel']

    if alg == 'dense':
        if dname == 'bter':
            dataset = dense_datasets.LinkDataset(
                BTER_PATH, layer_infos, max_degree=max_degree, 
                kernel=kernel, batch_size=batch_size, 
                negative_sample_size=config['negative_sample_size'])
        elif dname == 'lreddit':
            dataset = dense_datasets.LRedditDataset(
                layer_infos, max_degree=max_degree, 
                kernel=kernel, batch_size=batch_size, 
                negative_sample_size=config['negative_sample_size'],
                p=config['pt'])
        elif dname == 'arxiv':
            dataset = dense_datasets.OGBArxivDataset(
                layer_infos, config['final_info'], 
                max_degree=max_degree, kernel=kernel,
                batch_size=batch_size)
        elif dname == 'nreddit':
            dataset = dense_datasets.NRedditDataset(
                layer_infos, config['final_info'], 
                max_degree=max_degree, kernel=kernel, 
                batch_size=batch_size)
    elif alg == 'sparse':
        if dname == 'bter':
            dataset = sparse_datasets.BTERDataset(
                BTER_PATH, layer_infos, max_degree=max_degree, 
                kernel=kernel, batch_size=batch_size, sample_size=sample_size,
                negative_sample_size=config['negative_sample_size'])
        elif dname == 'lreddit':
            dataset = sparse_datasets.LRedditDataset(
                layer_infos, max_degree=max_degree, 
                kernel=kernel, batch_size=batch_size, sample_size=sample_size,
                negative_sample_size=config['negative_sample_size'],
                p=config['pt'])
        elif dname == 'arxiv':
            dataset = sparse_datasets.OGBArxivDataset(
                layer_infos, config['final_info'], 
                max_degree=max_degree, kernel=kernel, sample_size=sample_size,
                batch_size=batch_size)
        elif dname == 'nreddit':
            dataset = sparse_datasets.NRedditDataset(
                layer_infos, config['final_info'], 
                max_degree=max_degree, kernel=kernel, sample_size=sample_size,
                batch_size=batch_size)
    return dataset


def get_tf_datasets(config, dataset, strategy):
    '''
    Convert dataset class to corresponding tf.data.Dataset object
    '''
    platform = config['platform']
    algorithm = config['algorithm']
    numrep = config['numrep']
    batch_size = config['batch_size']

    if algorithm == 'dense':
        # update effective batch_size, based on platform
        # we also update steps_per_epoch for each split below
        if platform == 'gpu':
            eff_bs = batch_size*numrep
        elif platform == 'gc':
            eff_bs = batch_size

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (dataset.train, dataset.labels_train))
        train_dataset = train_dataset.batch(eff_bs, drop_remainder=True)
        train_steps_per_epoch = len(train_dataset)
        if platform == 'gc':
            train_steps_per_epoch /= numrep
        train_steps_per_epoch = train_steps_per_epoch - train_steps_per_epoch%numrep
        train_dataset = train_dataset.cache().repeat().prefetch(
            tf.data.experimental.AUTOTUNE)

        if 'val' in dir(dataset):
            val_dataset = tf.data.Dataset.from_tensor_slices(
                (dataset.val, dataset.labels_val))
            val_dataset = val_dataset.batch(eff_bs, drop_remainder=True)
            val_steps_per_epoch = len(val_dataset)
            if platform == 'gc':
                val_steps_per_epoch /= numrep
            val_steps_per_epoch = val_steps_per_epoch - val_steps_per_epoch%numrep
            val_dataset = val_dataset.cache().repeat().prefetch(
                tf.data.experimental.AUTOTUNE)
        else:
            val_dataset = None
            val_steps_per_epoch = -1

        if 'test' in dir(dataset):
            test_dataset = tf.data.Dataset.from_tensor_slices(
                (dataset.test, dataset.labels_test))
            test_dataset = test_dataset.batch(eff_bs, drop_remainder=True)
            test_steps_per_epoch = len(test_dataset)
            if platform == 'gc':
                test_steps_per_epoch /= numrep
            test_steps_per_epoch = test_steps_per_epoch - test_steps_per_epoch%numrep
            test_dataset = test_dataset.cache().repeat().prefetch(
                tf.data.experimental.AUTOTUNE)
        else:
            test_dataset = None
            test_steps_per_epoch = -1

    elif algorithm == 'sparse':
        # batch_size is handled implicitly by the generators
        # thus, we update steps_per_epoch manually below
        train_dataset_fn = get_dataset_fn('train', dataset)
        train_dataset = strategy.distribute_datasets_from_function(train_dataset_fn)
        train_steps_per_epoch = dataset.train_len//(numrep*batch_size)
        train_steps_per_epoch = train_steps_per_epoch - train_steps_per_epoch%numrep

        if 'val' in dir(dataset):
            val_dataset_fn = get_dataset_fn('val', dataset)
            val_dataset = strategy.distribute_datasets_from_function(val_dataset_fn)
            val_steps_per_epoch = dataset.val_len//(numrep*batch_size)
            val_steps_per_epoch = val_steps_per_epoch - val_steps_per_epoch%numrep
        else:
            val_dataset = None
            val_steps_per_epoch = -1

        if 'test' in dir(dataset):
            test_dataset_fn = get_dataset_fn('test', dataset)
            test_dataset = strategy.distribute_datasets_from_function(test_dataset_fn)
            test_steps_per_epoch = dataset.test_len//(numrep*batch_size)
            test_steps_per_epoch = test_steps_per_epoch - test_steps_per_epoch%numrep
        else:
            test_dataset = None
            test_steps_per_epoch = -1

    return train_dataset, train_steps_per_epoch, val_dataset, val_steps_per_epoch, \
        test_dataset, test_steps_per_epoch


###########################
# Driver
###########################
def run(config):
    # get strategy
    strategy = get_strategy(config)

    # complete model configuration based on chosen params
    config = get_layer_infos(config)

    # preprocess data and obtain associated tensorflow datasets
    dataset = get_dataset(config)
    train_dataset, train_steps_per_epoch, val_dataset, val_steps_per_epoch, \
        test_dataset, test_steps_per_epoch = get_tf_datasets(config, dataset, strategy)

    # build and compile model
    with strategy.scope():
        # build model
        if config['algorithm'] == 'dense':
            if config['dname'] in UNSUP_DNAMES:
                model = dense_models.buildModel(
                    dataset.feats, dataset.adj, dataset.layer_infos,
                    dataset.negative_sample_size)
            elif config['dname'] in SUP_DNAMES:
                model = dense_models.buildSupModel(
                    dataset.feats, dataset.adj, dataset.layer_infos,
                    dataset.final_info)
        elif config['algorithm'] == 'sparse':
            if config['dname'] in UNSUP_DNAMES:
                model = sparse_models.buildModel(
                    dataset.layer_infos, dataset.type_spec)
            elif config['dname'] in SUP_DNAMES:
                model = sparse_models.buildSupModel(
                    dataset.layer_infos, dataset.final_info, dataset.type_spec)

        optimizer = tf.keras.optimizers.Adam(lr=config['learning_rate'])
        if config['dname'] in UNSUP_DNAMES:
            loss = losses.Xent(dataset.negative_sample_size)
            metrics = [losses.MRR()]
        elif config['dname'] in SUP_DNAMES:
            loss = losses.SoftmaxCrossEntropy()
            metrics = ['acc']

        if config['algorithm'] == 'dense':
            steps_per_execution = train_steps_per_epoch
        elif config['algorithm'] == 'sparse':
            # TODO revisit this for additional optimization
            steps_per_execution = config['numrep']

        model.compile(loss=loss, optimizer=optimizer, run_eagerly=False,
                      metrics=metrics, steps_per_execution=steps_per_execution)

    # train
    tcb = callbacks.TimeCallback()
    cbacks = [tcb]
    if config['log_dir']:
        tbp = callbacks.get_tboardCallback(config['log_dir'])
        cbacks.append(tbp)
    if val_dataset:
        escb = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=config['patience'])
        cbacks.append(escb)
        t0 = time.time()
        history = model.fit(
            train_dataset, steps_per_epoch=train_steps_per_epoch,
            validation_data=val_dataset, validation_steps=val_steps_per_epoch,
            epochs=config['epochs'], callbacks=cbacks)
        t1 = time.time()
    else:
        t0 = time.time()
        history = model.fit(
            train_dataset, steps_per_epoch=train_steps_per_epoch,
            epochs=config['epochs'], callbacks=cbacks)
        t1 = time.time()

    # evaluate
    print('total time: %.2fs'%(t1-t0))
    print('warmup time: %.2fs'%tcb.times[0])
    print('mean post time: %.2fs'%np.mean(tcb.times[1:]))
    print('total steps: %d'%(config['numrep']*train_steps_per_epoch))
    print('loss: '+','.join([str(val) for val in history.history['loss']]))

    if test_dataset:
        if config['dname'] in UNSUP_DNAMES:
            # MRR
            out = model.predict(test_dataset, steps=test_steps_per_epoch, verbose=1)
            _, ior = tf.nn.top_k(out, k=out.shape[1])
            _, ranks = tf.nn.top_k(-ior, k=out.shape[1])
            ranks = tf.math.divide(1.0, tf.cast(ranks[:,0]+1, dtype=tf.float32))
            print('test mrr: %.2f'%tf.reduce_mean(ranks))
        elif config['dname'] in SUP_DNAMES:
            # evaluate
            results = model.evaluate(test_dataset, steps=test_steps_per_epoch)
            print('test accuracy: %.2f'%results[1])


###########################
# Main
###########################

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-n',
                    '--numrep',
                    default=1,
                    type=int,
                    help='number of devices')
parser.add_argument('-a',
                    '--algorithm',
                    default='dense',
                    choices=['dense', 'sparse'],
                    type=str,
                    help='algorithm')
parser.add_argument('-d',
                    '--dataset',
                    default='bter',
                    choices=['bter', 'lreddit', 'nreddit', 'arxiv'],
                    type=str,
                    help='dataset')
parser.add_argument('-p',
                    '--platform',
                    default='gpu',
                    choices=['gpu', 'gc'],
                    type=str,
                    help='platform')
parser.add_argument('-b',
                    '--batch_size',
                    default=128,
                    type=int,
                    help='batch size')
parser.add_argument('-e',
                    '--epochs',
                    default=5,
                    type=int,
                    help='epochs')
parser.add_argument('-ad',
                    '--agg_dim',
                    default=256,
                    type=int,
                    help='layer aggregation dimension')
parser.add_argument('-de',
                    '--depth',
                    default=2,
                    type=int,
                    help='num. layers')
parser.add_argument('-m',
                    '--max_degree',
                    default=100,
                    type=int,
                    help='max node degree (-1 for no limit)')
parser.add_argument('-nss',
                    '--negative_sample_size',
                    default=5,
                    type=int,
                    help='negative sample size')
parser.add_argument('-s',
                    '--sample_size',
                    default=15,
                    type=int,
                    help='neighborhood sample size')
parser.add_argument('-do',
                    '--dropout',
                    default=.5,
                    type=float,
                    help='dropout')
parser.add_argument('-pa',
                    '--patience',
                    type=int,
                    default=20,
                    help='early stopping patience')
parser.add_argument('-pt',
                    '--ptrain',
                    type=float,
                    default=1.0,
                    help='training data fraction for large datasets')
parser.add_argument('-co',
                    '--concat',
                    type=bool,
                    default=False,
                    help='layer concatenation option')
parser.add_argument('-un',
                    '--use_norm',
                    type=bool,
                    default=True,
                    help='layer normalization option')
parser.add_argument('-ub',
                    '--use_bias',
                    type=bool,
                    default=True,
                    help='layer bias vector option')
parser.add_argument('-k',
                    '--kernel',
                    type=str,
                    default='gcn',
                    help='layer kernel option')
parser.add_argument('-lr',
                    '--learning_rate',
                    type=float,
                    default=1e-3,
                    help='learning rate')
parser.add_argument('-ld',
                    '--log_dir',
                    type=str,
                    default=None,
                    help='logging directory')
args = parser.parse_args()

config = {}
config['numrep'] = args.numrep
config['algorithm'] = args.algorithm
config['dname'] = args.dataset
config['platform'] = args.platform
config['batch_size'] = args.batch_size
config['epochs'] = args.epochs
config['agg_dim'] = args.agg_dim
config['depth'] = args.depth
config['max_degree'] = args.max_degree
config['negative_sample_size'] = args.negative_sample_size
config['sample_size'] = args.sample_size
config['dropout'] = args.dropout
config['patience'] = args.patience
config['pt'] = args.ptrain
config['concat'] = args.concat
config['use_norm'] = args.use_norm
config['use_bias'] = args.use_bias
config['kernel'] = args.kernel
config['learning_rate'] = args.learning_rate
config['log_dir'] = args.log_dir

if config['max_degree']<0 and config['algorithm']=='dense':
    raise Exception('Must choose max_degree >0 for dense implementation')

run(config)