from os.path import join

import math
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    BatchNormalization, Conv2D, Dense, Flatten, Activation, MaxPooling2D,
    Dropout)
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam, Adadelta
from keras.regularizers import l2

import pedl

from data_provider import load_organized_data_info
from data_provider import train_val_dirs_generators
from config import *

K.set_learning_phase(1)
K.set_image_data_format('channels_first')


def _cnn(imgs_dim):

    model = Sequential()

    dropout = pedl.get_hyperparameter("dropout")
    batch_size = pedl.get_hyperparameter("batch_size")
    kernel_size = pedl.get_hyperparameter("kernel_size")
    pool_size = pedl.get_hyperparameter("pool_size")
    l2_reg = pedl.get_hyperparameter("l2_reg")
    initializer = pedl.get_hyperparameter("initializer")

    model.add(_convolutional_layer(nb_filter=16, input_shape=imgs_dim))
    _maybe_add_batch_normalization(model)
    model.add(PReLU(alpha_initializer=initializer))
    model.add(_convolutional_layer(nb_filter=16))
    _maybe_add_batch_normalization(model)
    model.add(PReLU(alpha_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(rate=dropout))

    model.add(_convolutional_layer(nb_filter=32))
    _maybe_add_batch_normalization(model)
    model.add(PReLU(alpha_initializer=initializer))
    model.add(_convolutional_layer(nb_filter=32))
    _maybe_add_batch_normalization(model)
    model.add(PReLU(alpha_initializer=initializer))
    model.add(_convolutional_layer(nb_filter=32))
    _maybe_add_batch_normalization(model)
    model.add(PReLU(alpha_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(rate=dropout))

    model.add(_convolutional_layer(nb_filter=64))
    _maybe_add_batch_normalization(model)
    model.add(PReLU(alpha_initializer=initializer))
    model.add(_convolutional_layer(nb_filter=64))
    _maybe_add_batch_normalization(model)
    model.add(PReLU(alpha_initializer=initializer))
    model.add(_convolutional_layer(nb_filter=64))
    _maybe_add_batch_normalization(model)
    model.add(PReLU(alpha_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(rate=dropout))

    model.add(_convolutional_layer(nb_filter=128))
    _maybe_add_batch_normalization(model)
    model.add(PReLU(alpha_initializer=initializer))
    model.add(_convolutional_layer(nb_filter=128))
    _maybe_add_batch_normalization(model)
    model.add(PReLU(alpha_initializer=initializer))
    model.add(_convolutional_layer(nb_filter=128))
    _maybe_add_batch_normalization(model)
    model.add(PReLU(alpha_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(rate=dropout))

    model.add(_convolutional_layer(nb_filter=256))
    _maybe_add_batch_normalization(model)
    model.add(PReLU(alpha_initializer=initializer))
    model.add(_convolutional_layer(nb_filter=256))
    _maybe_add_batch_normalization(model)
    model.add(PReLU(alpha_initializer=initializer))
    model.add(_convolutional_layer(nb_filter=256))
    _maybe_add_batch_normalization(model)
    model.add(PReLU(alpha_initializer=initializer))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(rate=dropout))

    model.add(Flatten())
    model.add(_dense_layer(output_dim=PENULTIMATE_SIZE))
    _maybe_add_batch_normalization(model)
    model.add(PReLU(alpha_initializer=initializer))
    model.add(Dropout(rate=dropout))
    model.add(_dense_layer(output_dim=SOFTMAX_SIZE))
    _maybe_add_batch_normalization(model)
    model.add(Activation(activation='softmax'))

    return model


def _simple_cnn(imgs_dim):
    dropout = pedl.get_hyperparameter("dropout")
    kernel_size = pedl.get_hyperparameter("kernel_size")
    pool_size = pedl.get_hyperparameter("simple_cnn_pool_size")

    model = Sequential()
    model.add(
        Conv2D(32, (kernel_size, kernel_size), padding="same", input_shape=imgs_dim)
    )
    model.add(Activation("relu"))
    model.add(Conv2D(32, (kernel_size, kernel_size)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout))

    model.add(Conv2D(64, (kernel_size, kernel_size), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (kernel_size, kernel_size)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, (kernel_size, kernel_size), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(128, (kernel_size, kernel_size)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(SOFTMAX_SIZE))
    model.add(Activation("softmax"))

    return model


def _convolutional_layer(nb_filter, input_shape=None):
    if input_shape:
        return _first_convolutional_layer(nb_filter, input_shape)
    else:
        return _intermediate_convolutional_layer(nb_filter)


def _first_convolutional_layer(nb_filter, input_shape):
    kernel_size = pedl.get_hyperparameter("kernel_size")
    l2_reg = pedl.get_hyperparameter("l2_reg")
    initializer = pedl.get_hyperparameter("initializer")
    return Conv2D(filters=nb_filter, kernel_size=(kernel_size, kernel_size), input_shape=input_shape,
        padding='same', kernel_initializer=initializer, kernel_regularizer=l2(l=l2_reg))


def _intermediate_convolutional_layer(nb_filter):
    kernel_size = pedl.get_hyperparameter("kernel_size")
    l2_reg = pedl.get_hyperparameter("l2_reg")
    initializer = pedl.get_hyperparameter("initializer")
    return Conv2D(
        filters=nb_filter, kernel_size=(kernel_size, kernel_size), padding='same',
        kernel_initializer=initializer, kernel_regularizer=l2(l=l2_reg))


def _dense_layer(output_dim):
    l2_reg = pedl.get_hyperparameter("l2_reg")
    initializer = pedl.get_hyperparameter("initializer")
    return Dense(units=output_dim, kernel_regularizer=l2(l=l2_reg), kernel_initializer=initializer)

def _maybe_add_batch_normalization(model):
    if pedl.get_hyperparameter("batch_normalization"):
        model.add(BatchNormalization(axis=1))

def load_trained_cnn_feature_maps_layer(model_path):
    return _load_trained_cnn_layer(model_path, LAST_FEATURE_MAPS_LAYER)


def load_trained_cnn_penultimate_layer(model_path):
    return _load_trained_cnn_layer(model_path, PENULTIMATE_LAYER)


def load_trained_cnn_softmax_layer(model_path):
    return _load_trained_cnn_layer(model_path, SOFTMAX_LAYER)


def _load_trained_cnn_layer(model_path, layer_index):
    model = model_from_json(model_path)
    dense_output = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[layer_index].output])
    return lambda X: dense_output([X, 0])[0]


if __name__ == '__main__':
    _train_model()
