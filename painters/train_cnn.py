from os.path import join

import math
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D
from keras.layers import Flatten, BatchNormalization, Dropout
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.regularizers import l2

from data_provider import MODELS_DIR
from data_provider import load_organized_data_info
from data_provider import train_val_dirs_generators

K.set_learning_phase(1)

IMGS_DIM_3D = (3, 256, 256)
CNN_MODEL_FILE = join(MODELS_DIR, 'cnn.h5')
MAX_EPOCHS = 500
BATCH_SIZE = 96
L2_REG = 0.003
W_INIT = 'he_normal'
LAST_FEATURE_MAPS_LAYER = 46
LAST_FEATURE_MAPS_SIZE = (128, 8, 8)
PENULTIMATE_LAYER = 51
PENULTIMATE_SIZE = 2048
SOFTMAX_LAYER = 55
SOFTMAX_SIZE = 1057


def _train_model():
    data_info = load_organized_data_info(IMGS_DIM_3D[1])
    print("data_info loaded")
    dir_tr = data_info['dir_tr']
    dir_val = data_info['dir_val']

    gen_tr, gen_val = train_val_dirs_generators(BATCH_SIZE, dir_tr, dir_val)
    print("generators loaded")
    model = _cnn(IMGS_DIM_3D)

    print(dir_tr, data_info['num_tr'], dir_val, data_info['num_val'])
    model.fit_generator(
        generator=gen_tr,
        epochs=MAX_EPOCHS,
        steps_per_epoch=math.ceil(data_info['num_tr'] / BATCH_SIZE),
        validation_data=gen_val,
        validation_steps=math.ceil(data_info['num_val'] / BATCH_SIZE),
        callbacks=[ModelCheckpoint(CNN_MODEL_FILE, save_best_only=True)],
        verbose=1)


def _cnn(imgs_dim, compile_=True):
    model = Sequential()

    model.add(_convolutional_layer(nb_filter=16, input_shape=imgs_dim))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=16))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=32))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=64))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=128))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(_convolutional_layer(nb_filter=256))
    model.add(BatchNormalization(axis=1))
    model.add(PReLU(alpha_initializer=W_INIT))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.5))

    model.add(Flatten())
    model.add(_dense_layer(output_dim=PENULTIMATE_SIZE))
    model.add(BatchNormalization())
    model.add(PReLU(alpha_initializer=W_INIT))

    if compile_:
        model.add(Dropout(rate=0.5))
        model.add(_dense_layer(output_dim=SOFTMAX_SIZE))
        model.add(BatchNormalization())
        model.add(Activation(activation='softmax'))
        return compile_model(model)

    return model


def _convolutional_layer(nb_filter, input_shape=None):
    if input_shape:
        return _first_convolutional_layer(nb_filter, input_shape)
    else:
        return _intermediate_convolutional_layer(nb_filter)


def _first_convolutional_layer(nb_filter, input_shape):
    return Conv2D(
        filters=nb_filter, kernel_size=(3, 3), input_shape=input_shape,
        padding='same', kernel_initializer=W_INIT, kernel_regularizer=l2(l=L2_REG))


def _intermediate_convolutional_layer(nb_filter):
    return Conv2D(
        filters=nb_filter, kernel_size=(3, 3), padding='same',
        kernel_initializer=W_INIT, kernel_regularizer=l2(l=L2_REG))


def _dense_layer(output_dim):
    return Dense(units=output_dim, kernel_regularizer=l2(l=L2_REG), kernel_initializer=W_INIT)


def compile_model(model):
    adam = Adam(lr=0.000074)

    import tensorflow as tf
    from keras.utils import multi_gpu_model
    with tf.device('/device:CPU:0'):
        origin_model = model
    model = multi_gpu_model(origin_model, gpus=4)

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
    model = load_model(model_path)
    dense_output = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[layer_index].output])
    # output in test mode = 0
    return lambda X: dense_output([X, 0])[0]


if __name__ == '__main__':
    _train_model()
