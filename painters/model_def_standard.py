import os

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.datasets.cifar import load_batch
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.data_utils import get_file

import pedl
from pedl.data import ArrayBatchLoader
from pedl.frameworks.keras import KerasTrial
from pedl.frameworks.keras.data import KerasDataAdapter
from pedl.frameworks.util import elementwise_mean
from sklearn.metrics import roc_auc_score

from train_cnn import _cnn, IMGS_DIM_3D, BATCH_SIZE, SOFTMAX_SIZE
from data_provider import load_organized_data_info, train_val_dirs_generators
from validation import _create_pairs_generator, IMGS_DIM_1D
from utils import pairs_dot

import objgraph

def categorical_error(y_true, y_pred):
    return 1. - categorical_accuracy(y_true, y_pred)


class PainterTrial(KerasTrial):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.kernel_size = pedl.get_hyperparameter("kernel_size")
        self.dropout = pedl.get_hyperparameter("dropout")
        self.pool_size = pedl.get_hyperparameter("pool_size")
        self.L2_REG = pedl.get_hyperparameter("L2_REG")
        self.lr = pedl.get_hyperparameter("lr")
        self.my_batch_size = pedl.get_hyperparameter("batch_size")
        self.data_info = load_organized_data_info(IMGS_DIM_1D)

    # def session_config(self, hparams):
    #     if hparams.get("disable_CPU_parallelism", False):
    #         return tf.ConfigProto(intra_op_parallelism_threads=1,
    #                               inter_op_parallelism_threads=1)
    #     else:
    #         return tf.ConfigProto()

    def build_model(self, hparams):
        model = _cnn(IMGS_DIM_3D)
        return model

    def optimizer(self):
        adam = Adam(lr=self.lr)
        return adam

    def loss(self):
        return categorical_crossentropy

    def batch_size(self):
        return self.my_batch_size

    def roc_auc(self, y_true, y_pred):
        pass

    def validation_metrics(self):
        return {}

    def vec_to_int(self, batch_labels):
        batch_int_labels = []
        for y in batch_labels:
            batch_int_labels.append(np.where(y==1)[0][0])
        return batch_int_labels


    # HACK: compute the validation metrics in a customized way
    def compute_validation_metrics(self, step_id):
        assert self.validation_data_adapter is not None
        assert self.model is not None

        self.validation_data_adapter.start(is_validation=True)
        validation_iterator = self.validation_data_adapter.get_iterator()
        assert validation_iterator is not None

        X_val_embedded = None
        y_val = None
        num_inputs = 0

        objgraph.show_growth(limit=50)
        for X_batch, y_batch in validation_iterator:
            X_embedded = self.model.predict(X_batch)
            if X_val_embedded is None: X_val_embedded = X_embedded
            else: X_val_embedded = np.concatenate((X_val_embedded, X_embedded), axis=0)
            if y_val is None: y_val = y_batch
            else: y_val = np.concatenate((y_val, y_batch), axis=0)

            num_inputs += len(X_batch)

        y_val = self.vec_to_int(y_val)
        batches_val = _create_pairs_generator(
            X_val_embedded, y_val, lambda u, v: [u, v],
            num_groups=32,
            batch_size=10000)

        y_pred, y_true = np.array([]), np.array([])
        for X, y in batches_val:
            y_pred = np.hstack((y_pred, pairs_dot(X)))
            y_true = np.hstack((y_true, y))
        roc_auc = roc_auc_score(y_true, y_pred)
        objgraph.show_growth(limit=50)

        return {"num_inputs": num_inputs, "validation_metrics": {'roc_auc': roc_auc}}



def make_data_loaders(experiment_config, hparams):
    data_info = load_organized_data_info(IMGS_DIM_3D[1])
    dir_tr = data_info['dir_tr']
    dir_val = data_info['dir_val']

    gen_tr, gen_val = train_val_dirs_generators(BATCH_SIZE, dir_tr, dir_val)

    gen_tr = KerasDataAdapter(gen_tr, workers=16, use_multiprocessing=True)
    gen_val = KerasDataAdapter(gen_val, workers=16, use_multiprocessing=True)

    return (gen_tr, gen_val)




