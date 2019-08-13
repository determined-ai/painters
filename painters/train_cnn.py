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

batch_size = pedl.get_hyperparameter("batch_size")
kernel_size = pedl.get_hyperparameter("kernel_size")
dropout = pedl.get_hyperparameter("dropout")
pool_size = pedl.get_hyperparameter("pool_size")
l2_reg = pedl.get_hyperparameter("l2_reg")


def _cnn(imgs_dim):

    model = Sequential()

    model.add(_convolutional_layer(nb_filter=16, input_shape=imgs_dim))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=16))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(rate=dropout))

    model.add(Flatten())
    model.add(_dense_layer(output_dim=PENULTIMATE_SIZE))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(Dropout(rate=dropout))
    model.add(_dense_layer(output_dim=SOFTMAX_SIZE))
    model.add(BatchNormalization())
    model.add(Activation(activation='softmax'))

    return model


def _convolutional_layer(nb_filter, input_shape=None):
    if input_shape:
        return _first_convolutional_layer(nb_filter, input_shape)
    else:
        return _intermediate_convolutional_layer(nb_filter)


def _first_convolutional_layer(nb_filter, input_shape):
    return Conv2D(filters=nb_filter, kernel_size=(kernel_size, kernel_size), input_shape=input_shape,
        padding='same', kernel_initializer=W_INIT, kernel_regularizer=l2(l=l2_reg))


def _intermediate_convolutional_layer(nb_filter):
    return Conv2D(
        filters=nb_filter, kernel_size=(kernel_size, kernel_size), padding='same',
        kernel_initializer=W_INIT, kernel_regularizer=l2(l=l2_reg))


def _dense_layer(output_dim):
    return Dense(units=output_dim, kernel_regularizer=l2(l=l2_reg), kernel_initializer=W_INIT)


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
