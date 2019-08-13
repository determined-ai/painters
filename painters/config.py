from os.path import join, dirname
import pedl

# Data directories
DATA_DIR = '/home/chris/painters/data_check'
TEST_DIR = join(DATA_DIR, 'test')
TRAIN_DIR = join(DATA_DIR, 'train')
TRAIN_INFO_FILE = join(DATA_DIR, 'train_info.csv')
SUBMISSION_INFO_FILE = join(DATA_DIR, 'submission_info.csv')
ORGANIZED_DATA_INFO_FILE = 'organized_data_info_.json'
MODELS_DIR = join(dirname(dirname(__file__)), 'models')
MISC_DIR = join(dirname(dirname(__file__)), 'misc')


IMGS_DIM_3D = (3, 256, 256)
CNN_MODEL_FILE = join(MODELS_DIR, 'cnn.h5')
MAX_EPOCHS = 500
W_INIT = 'he_normal'
LAST_FEATURE_MAPS_LAYER = 46
LAST_FEATURE_MAPS_SIZE = (128, 8, 8)
PENULTIMATE_LAYER = 51
PENULTIMATE_SIZE = 2048
SOFTMAX_LAYER = 55
NUM_CLASSES = SOFTMAX_SIZE = 1584
MODEL_NAME = 'cnn_2_9069_vl.h5'

LAYER_SIZES = {
    'feature_maps': LAST_FEATURE_MAPS_SIZE,
    'penultimate': PENULTIMATE_SIZE,
    'softmax': SOFTMAX_SIZE
}
