import keras
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
import tensorflow as tf

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

import pedl
from pedl.frameworks.keras import KerasTrial
from pedl.frameworks.keras.data import KerasDataAdapter

from train_cnn import _cnn, _simple_cnn
from data_provider import load_organized_data_info, train_val_dirs_generators
from validation import _create_pairs_generator, IMGS_DIM_1D
from utils import pairs_dot
from config import IMGS_DIM_3D, NUM_CLASSES


class PainterTrial(KerasTrial):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.lr = pedl.get_hyperparameter("lr")
        self.my_batch_size = pedl.get_hyperparameter("batch_size")
        self.data_info = load_organized_data_info(IMGS_DIM_3D[1])

    def build_model(self, hparams):
        if pedl.get_hyperparameter("model_type") == "original":
            model = _cnn(IMGS_DIM_3D)
        elif pedl.get_hyperparameter("model_type") == "vgg19":
            model = keras.applications.vgg19.VGG19(
                include_top=True,
                weights=None,
                input_shape=IMGS_DIM_3D,
                classes=NUM_CLASSES
            )
        elif pedl.get_hyperparameter("model_type") == "resnet50":
            model = keras.applications.resnet.ResNet50(
                include_top=True,
                weights=None,
                input_shape=IMGS_DIM_3D,
                classes=NUM_CLASSES
            )
        elif pedl.get_hyperparameter("model_type") == "simple":
            model = _simple_cnn(IMGS_DIM_3D)
        else:
            assert False

        print(model.summary())
        return model

    def optimizer(self):
        adam = Adam(lr=self.lr)
        return adam

    def loss(self):
        return categorical_crossentropy

    def batch_size(self):
        return self.my_batch_size

    # This is an abstract function which the trial class needs for instantiation.
    def validation_metrics(self):
        return {}

    # HACK: compute the validation metrics in a customized way.
    def compute_validation_metrics(self, step_id):
        assert self.validation_data_adapter is not None
        assert self.model is not None

        self.validation_data_adapter.start(is_validation=True)
        validation_iterator = self.validation_data_adapter.get_iterator()
        assert validation_iterator is not None

        # TODO: preallocate numpy arrays
        X_val_embedded = None
        y_val = None

        # Shape of X_batch: (96=batch_size, 3, 256, 256 (dims of a single image)).
        # Shape of y_batch: (96=batch_size, number of classes).
        num_inputs = 0
        for X_batch, y_batch in validation_iterator:
            # Shape of X_embedded: (96=batch_size, number of classes).
            X_embedded = self.model.predict(X_batch)

            # Shape of X_val_embedded: (iteration * batch_size, number of classes).
            if X_val_embedded is None:
                X_val_embedded = X_embedded
            else:
                X_val_embedded = np.concatenate((X_val_embedded, X_embedded), axis=0)

            # Shape of y_val: number of (iteration * batch_size, number of classes).
            if y_val is None:
                y_val = y_batch
            else:
                y_val = np.concatenate((y_val, y_batch), axis=0)
            num_inputs += len(X_batch)

        self.validation_data_adapter.stop()

        # Calculate categorical cross entropy for validation data,
        # which is the same loss as used in training
        y_pred_vec = tf.convert_to_tensor(X_val_embedded)
        y_val_vec = tf.convert_to_tensor(y_val)
        cce = categorical_crossentropy(y_pred_vec, y_val_vec)
        with tf.Session().as_default():
            cce = cce.eval()
        cce = np.mean(cce)

        # Calculate accuracy of single image artist classification.
        # Class prediction is the class with max prob in vector
        # prediction of each painting.
        y_val = np.argmax(y_val, axis=1)
        y_pred = np.argmax(X_val_embedded, axis=1)
        single_painting_acc = accuracy_score(y_val, y_pred)

        # Pairwise evaluation: whether they are from the same artist
        batches_val = _create_pairs_generator(
            X_val_embedded, y_val, lambda u, v: [u, v],
            num_groups=32,
            batch_size=1000000)

        y_pred, y_true = np.array([]), np.array([])
        for X, y in batches_val:
            y_pred = np.hstack((y_pred, pairs_dot(X)))
            y_true = np.hstack((y_true, y))
        roc_auc = roc_auc_score(y_true, y_pred)

        return {"num_inputs": num_inputs,
                "validation_metrics": {'roc_auc': roc_auc,
                                       'categorical_crossentropy': cce,
                                       'single_painting_accuracy': single_painting_acc}}

def make_data_loaders(experiment_config, hparams):
    # multi_crop improves training, but was not used for author's submission
    data_info = load_organized_data_info(
        IMGS_DIM_3D[1], multi_crop=pedl.get_hyperparameter("multi_crop"))
    dir_tr = data_info['dir_tr']
    dir_val = data_info['dir_val']

    gen_tr, gen_val = train_val_dirs_generators(
        pedl.get_hyperparameter("batch_size"), dir_tr, dir_val)

    gen_tr = KerasDataAdapter(gen_tr, workers=4, use_multiprocessing=True)
    gen_val = KerasDataAdapter(gen_val, workers=4, use_multiprocessing=True)

    return (gen_tr, gen_val)
