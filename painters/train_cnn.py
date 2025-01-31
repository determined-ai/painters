from os.path import join

import math
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D
from keras.layers import Flatten, Dropout
from normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam, Adadelta
from keras.regularizers import l2

from data_provider import load_organized_data_info
from data_provider import train_val_dirs_generators
from config import *

K.set_learning_phase(1)
K.set_image_data_format('channels_first')


def _train_model():
    data_info = load_organized_data_info(IMGS_DIM_3D[1])
    dir_tr = data_info['dir_tr']
    dir_val = data_info['dir_val']

    gen_tr, gen_val = train_val_dirs_generators(BATCH_SIZE, dir_tr, dir_val)
    model = _cnn(IMGS_DIM_3D)

    model.fit_generator(
        generator=gen_tr,
        epochs=MAX_EPOCHS,
        steps_per_epoch=300,
        validation_data=gen_val,
        validation_steps=math.ceil(data_info['num_val'] / BATCH_SIZE),
        validation_freq=10,
        callbacks=[ModelCheckpoint(CNN_MODEL_FILE, save_best_only=True)],
        workers=16,
        use_multiprocessing=True,
        verbose=1)


def _cnn(imgs_dim, compile_=True):

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

    if compile_:
        model.add(Dropout(rate=dropout))
        # Output: a vector of size (# of classes).
        model.add(_dense_layer(output_dim=SOFTMAX_SIZE))
        model.add(BatchNormalization())
        model.add(Activation(activation='softmax'))
        return compile_model(model)
        # return model

    return model


def _convolutional_layer(nb_filter, input_shape=None):
    if input_shape:
        return _first_convolutional_layer(nb_filter, input_shape)
    else:
        return _intermediate_convolutional_layer(nb_filter)


def _first_convolutional_layer(nb_filter, input_shape):
    return Conv2D(filters=nb_filter, kernel_size=(kernel_size, kernel_size), input_shape=input_shape,
        padding='same', kernel_initializer=W_INIT, kernel_regularizer=l2(l=L2_REG))


def _intermediate_convolutional_layer(nb_filter):
    return Conv2D(
        filters=nb_filter, kernel_size=(kernel_size, kernel_size), padding='same',
        kernel_initializer=W_INIT, kernel_regularizer=l2(l=L2_REG))


def _dense_layer(output_dim):
    return Dense(units=output_dim, kernel_regularizer=l2(l=L2_REG), kernel_initializer=W_INIT)


def compile_model(model):
    adam = Adam(lr=lr)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=adam,
        metrics=['accuracy'])
    return model


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
