import numpy as np
import matplotlib.pyplot as plt
import LearnNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import ParameterGrid, KFold, train_test_split


def nnet_error_rate(y_true, y_pred):
    y_pred_label = np.argmax(y_pred, axis=0).reshape(-1, 1)
    y_true_label = out_enc.inverse_transform(y_true.T).reshape(-1, 1)
    return LearnNet.error_rate(y_true_label, y_pred_label)


if __name__ == "__main__":

    dataset_train = np.loadtxt("optdigits_train.dat")  # Load training dataset
    dataset_test = np.loadtxt("optdigits_test.dat")  # Load testing dataset
    dataset_trial = np.loadtxt("optdigits_trial.dat")  # Load trial dataset

    m, n = (
        dataset_train.shape[0],
        dataset_train.shape[1] - 1,
    )  # Get number of samples and features

    X = dataset_train[:, :-1].reshape(m, n)  # Extract features from training dataset
    y = dataset_train[:, -1].reshape(m, 1)  # Extract labels from training dataset

    out_enc = LabelBinarizer()  # Initialize label binarizer for one-hot encoding
    y_ohe = out_enc.fit_transform(y)  # One-hot encode training labels

    K = y_ohe.shape[1]  # Number of unique classes in the dataset

    m_test = dataset_test.shape[0]  # Get number of samples in the test dataset

    X_test = dataset_test[:, :-1].reshape(
        m_test, n
    )  # Extract features from test dataset
    y_test = dataset_test[:, -1].reshape(m_test, 1)  # Extract labels from test dataset

    y_test_ohe = out_enc.transform(y_test)  # One-hot encode test labels

    m_trial = dataset_trial.shape[0]  # Get number of samples in the trial dataset

    X_trial = dataset_trial[:, :-1].reshape(
        m_trial, n
    )  # Extract features from trial dataset
    y_trial = dataset_trial[:, -1].reshape(
        m_trial, 1
    )  # Extract labels from trial dataset

    y_trial_ohe = out_enc.transform(y_trial)  # One-hot encode trial labels

    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    #############################################
    #           Multi Output Perceptron         #
    #           1024 Inputs, 10 outputs         #
    #           Gradient Descent                #
    #   Learning Rate [4^0, 4^1, 4^2, 4^3, 4^4] #
    #############################################

    error_logs = {}

    best_lr = None
    best_test_err_rate = float("inf")

    nnet_metric = LearnNet.NNetMetric(f=nnet_error_rate)

    nnet = LearnNet.NNet(nunits=[1024, K])

    learning_rate = [4**0, 4**1, 4**2, 4**3, 4**4]

    for lr in learning_rate:
        opt = LearnNet.NNetGDOptimizer(metric=nnet_metric, max_iters=50, learn_rate=lr)

        best_nnet = nnet.fit(X, y_ohe, X_test, y_test_ohe, optimizer=opt, verbose=1)

        train_err = np.array(opt.train_err)
        test_err = np.array(opt.test_err)

        error_logs[lr] = {
            "train_err": train_err,
            "test_err": test_err,
        }

        final_test_err_rate = test_err[-1, 1]
        if final_test_err_rate < best_test_err_rate:
            best_test_err_rate = final_test_err_rate
            best_lr = lr

    print(f"Best Learning Rate --> {best_lr}")
    print(f"Best Test error rate --> {best_test_err_rate}")
    breakpoint()
