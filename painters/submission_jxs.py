# # create submission file from model checkpoint stored in pedl
#
# from keras.models import model_from_json
#
# def create_submission():
#     model = get_model()
#     test_data = load_test_image()
#     predictions = predict(model, test_data)
#     write_submission(predictions)
#
#
# def get_model():
#     model = model_from_json('model.json')
#     model.load_weights('weights.h5')
#     return model
#
#
# def load_test_image():
#     pass
#
#
# def predict(model, data):
#     pass
#
#
# def write_submission(prediction):
#     pass
#
#
# if __name__ == '__main__':
#     create_submission()


from os.path import join

import numpy as np
from data_provider import SUBMISSION_INFO_FILE, DATA_DIR
from data_provider import load_organized_data_info
from utils import append_to_file
from utils import read_lines_in_batches
from keras.models import model_from_json
from os.path import basename

IMGS_DIM_1D = 256
SUBMISSION_FILE = join(DATA_DIR, 'submission.csv')
BATCH_SIZE = 10000
FILES_TO_AVG = {}


def _create_submission_file_avg_cnns():
    data_info = load_organized_data_info(IMGS_DIM_1D)
    model = _get_model()
    X_test, names = _average_embedded_test_data(model, data_info)
    features_lookup = {n: f for n, f in zip(names, X_test)}
    _create_submission_file(
        BATCH_SIZE, features_lookup, _calculate_batch_prediction_dot)


def _get_model():
    with open('../models/model.json', 'r') as json_file:
        model = model_from_json(json_file.read())
    model.load_weights('../models/weights.h5')
    return model


def _average_embedded_test_data(model, data_info):
    X_test, y_test = None, None

    data_info = load_organized_data_info(IMGS_DIM_1D)
    dir_te, num_te = data_info['dir_te'], data_info['num_te']
    dir_tr = data_info['dir_tr']
    from data_provider import testing_generator, init_directory_generator
    gen = testing_generator(dir_tr=dir_tr)
    gen_test = init_directory_generator(
        gen, dir_te, BATCH_SIZE, class_mode='sparse', shuffle_=False)

    num_full_epochs = num_te // BATCH_SIZE
    last_batch_size = num_te - (num_full_epochs * BATCH_SIZE)

    for i in range(num_full_epochs + 1):
        X_batch, y_batch = next(gen_test)

        if i == num_full_epochs:
            X_batch = X_batch[:last_batch_size]

        if X_test is None:
            X_test = model.predict(X_batch)
        else:
            X_test = np.vstack((X_test, model.predict(X_batch)))

    names = [basename(p) for p in gen_test.filenames]
    return X_test, names


def _calculate_batch_prediction_dot(lines, features_lookup):
    y_pred, submission_indices = [], []

    for line in lines:
        submission_indices.append(line[0])
        image_feature_a = features_lookup[line[1]]
        image_feature_b = features_lookup[line[2]]
        y_pred.append(np.dot(image_feature_a, image_feature_b))

    return y_pred, submission_indices


def _create_submission_file(batch_size, features_lookup, batch_predict_func):
    append_to_file(["index,sameArtist\n"], SUBMISSION_FILE)
    for batch in read_lines_in_batches(SUBMISSION_INFO_FILE, batch_size):
        y_pred, indices = batch_predict_func(batch, features_lookup)
        lines = ["{:s},{:f}\n".format(i, p) for i, p in zip(indices, y_pred)]
        append_to_file(lines, SUBMISSION_FILE)


def _average_submission_files():
    lines_gens, weights = [], []

    for file_name, weight in FILES_TO_AVG.items():
        file_path = join(DATA_DIR, file_name)
        lines_gen = read_lines_in_batches(file_path, batch_size=BATCH_SIZE)
        lines_gens.append(lines_gen)
        weights.append(weight)

    append_to_file(["index,sameArtist\n"], SUBMISSION_FILE)

    while True:
        try:
            _average_write_next_batch(lines_gens, weights)
        except StopIteration:
            return


def _average_write_next_batch(lines_gens, weights):
    separated_lines = [next(lg) for lg in lines_gens]
    merged_lines = zip(*separated_lines)

    result_lines = []

    for same_example_lines in merged_lines:
        example_index = same_example_lines[0][0]
        preds = [float(l[1]) for l in same_example_lines]
        pred_avg = sum(w * p for w, p in zip(weights, preds)) / sum(weights)
        result_lines.append("{:s},{:f}\n".format(example_index, pred_avg))

    append_to_file(result_lines, SUBMISSION_FILE)


if __name__ == '__main__':
    _create_submission_file_avg_cnns()