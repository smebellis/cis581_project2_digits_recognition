import LearnNet
import numpy as np


def make_nnet_error_rate(out_enc):
    def nnet_error_rate(y_true, y_pred):
        y_pred_label = np.argmax(y_pred, axis=0).reshape(-1, 1)
        y_true_label = out_enc.inverse_transform(y_true.T).reshape(-1, 1)
        return LearnNet.error_rate(y_true_label, y_pred_label)

    return nnet_error_rate
