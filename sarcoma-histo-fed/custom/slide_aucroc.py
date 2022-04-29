import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score


def aggregate_to_slides(slides, prediction, actual):
    df = (
        pd.DataFrame(
            {"slides": slides, "prediction": prediction, "actual": actual},
            columns=["slides", "prediction", "actual"],
        )
        .groupby("slides")
        .agg(
            pred_slide=("prediction", lambda x: np.vstack(x).mean(axis=0)),
            actual_slide=("actual", lambda x: np.vstack(x).mean(axis=0)),
        )
    )

    pred_aggregate = np.stack(df["pred_slide"].to_numpy())
    actual_aggregate = np.stack(df["actual_slide"].to_numpy())

    return pred_aggregate, actual_aggregate


def compute_roc_auc(dataset, model):
    files = list(dataset.map(lambda files, pixels, labels: files).as_numpy_iterator())
    pixels = dataset.map(lambda files, pixels, labels: pixels)
    labels = list(dataset.map(lambda files, pixels, labels: labels).as_numpy_iterator())
    y_pred = model.predict(pixels).tolist()

    # labels are shaped sorta weird [[[label0]],[[label1]],[[label2]],...]
    # so we'll remove one level of nesting...am I missing something??
    labels = [l[0] for l in labels]
    pred, actual = aggregate_to_slides(files, y_pred, labels)

    return roc_auc_score(actual, pred, multi_class="ovr")


class SlideROCCallback(tf.keras.callbacks.Callback):
    def __init__(self, train, valid, num_epoch_per_auc_calc):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.train = train
        self.valid = valid
        self.num_epoch_per_auc_calc = num_epoch_per_auc_calc

    def on_epoch_end(self, epoch, log={}):
        if epoch % self.num_epoch_per_auc_calc != 0:
            return
        train_roc = compute_roc_auc(self.train, self.model)
        valid_roc = compute_roc_auc(self.valid, self.model)

        self.logger.info(
            "\nSlide-level train ROC-AUC: %.4f\nSlide-level valid ROC-AUC: %.4f\n"
            % (train_roc, valid_roc)
        )
