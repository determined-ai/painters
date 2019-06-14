import numpy as np
from sklearn.metrics import roc_auc_score

from cnn_embedding import get_embedded_train_val_split
from data_provider import pairs_generator, load_organized_data_info
from utils import pairs_dot


def _softmax_dot():
    data_info = load_organized_data_info(IMGS_DIM_1D)
    X_avg, y_val = _average_embedded_val_data(data_info)

    batches_val = _create_pairs_generator(
        X_avg, y_val, lambda u, v: [u, v],
        num_groups=32,
        batch_size=1000000)

    y_pred, y_true = np.array([]), np.array([])
    for X, y in batches_val:
        y_pred = np.hstack((y_pred, pairs_dot(X)))
        y_true = np.hstack((y_true, y))

    print("Validation AUC: {:.4f}".format(roc_auc_score(y_true, y_pred)))


def _average_embedded_val_data(data_info):
    X_avg, y_val =\
        np.zeros((data_info['num_val'], data_info['num_distinct_cls'])), None

    for model, weight in CNNS_WEIGHTS.items():
        print("Model: {:s}".format(model))
        split = get_embedded_train_val_split('softmax', model_name=model)
        _, _, _, X_val, y_val, _ = split
        X_avg += weight * X_val

    X_avg /= sum(CNNS_WEIGHTS.values())
    return X_avg, y_val


def _create_pairs_generator(X, y, pairs_func, num_groups, batch_size):
    return pairs_generator(
        X, y,
        batch_size=batch_size,
        pair_func=pairs_func,
        num_groups=num_groups)


def compute_validation_metrics(self, step_id: StepID) -> Dict[str, Any]:
    assert self.validation_data_adapter is not None
    assert self.model is not None

    metrics = []

    self.validation_data_adapter.start(is_validation=True)
    # validation generator
    validation_iterator = self.validation_data_adapter.get_iterator()
    assert validation_iterator is not None

    progbar_length = None
    if self.validation_data_adapter.is_finite():
        progbar_length = len(self.validation_data_adapter)
    progbar = Progbar(target=progbar_length, interval=0.5)
    progbar_idx = 0

    num_inputs = 0


    y_preds = []
    y_trues = []
    # while True:
        # (X1, X2, y1, y2) = validation_iterator.next()
    y1, y2 = 0, 1
    X1, X2 = np.random.rand(256, 256), np.random.rand(256, 256)
    softmax1, softmax2 = self.model.predict(X1), self.model.predict(X2)
    y_pred = np.dot(softmax1, softmax2)
    y_preds.append(y_pred)
    y_trues.append(y1==y2)
    roc_auc = roc_auc_score(y_trues, y_preds)
    return {"num_inputs": num_inputs, "validation_metrics": {'val_auc_roc': roc_auc, 'loss': 0.5}}


    # for batch_data, batch_labels in validation_iterator:
    #     # test_on_batch always includes the validation loss as the
    #     # first metric it returns, followed by any additional metrics
    #     # that were requested. If no validation metrics other than the
    #     # loss are requested, the loss is returned as a scalar value.
    #     # Otherwise, the metric values are returned in a list.
    #     with self.session.graph.as_default():
    #         # call self.model twice instead of test_on_batch
    #         # also do the loss on the test batch
    #         metrics_values = self.model.test_on_batch(batch_data, batch_labels)
    #
    #     if not self.t_metrics_funcs and not self.v_metrics_funcs:
    #         metrics_values = [metrics_values]

    #     check_len(metrics_values, 1 + len(self.t_metrics_names) + len(self.v_metrics_names))
    #     v_metrics_values = [metrics_values[0]] + metrics_values[1 + len(self.t_metrics_names):]
    #     metrics.append(v_metrics_values)
    #
    #     num_inputs += len(batch_data)
    #
    #     progbar_idx += 1
    #     progbar.update(progbar_idx)
    #
    # self.validation_data_adapter.stop()
    #
    # check_gt(len(metrics), 0)
    #
    # # The "loss" will be the first element in the v_metrics_values list,
    # # followed by user-specified validation metrics.
    # v_metrics_names = ["loss"] + self.v_metrics_names
    # v_metrics_reducers = [elementwise_mean]  # type: List[Reducer]
    # v_metrics_reducers += self.v_metrics_reducers
    # reduced_metrics = [
    #     reducer([b[idx] for b in metrics]) for idx, reducer in enumerate(v_metrics_reducers)
    # ]
    # check_eq_len(v_metrics_names, reduced_metrics)
    # named_metrics = dict(zip(v_metrics_names, reduced_metrics))
    #
    # return {"num_inputs": num_inputs, "validation_metrics": named_metrics}
