import math
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from sklearn.preprocessing import LabelBinarizer

import LearnNet


def perceptron(
    X: np.ndarray, y_ohe: np.ndarray, X_test: np.ndarray, y_test_ohe: np.ndarray, K: int
):
    results = {}
    summary = []

    best_test_error = float("inf")
    best_train_error = float("inf")
    best_train_loss = float("inf")
    best_test_loss = float("inf")

    nnet_metric = LearnNet.NNetMetric(f=nnet_error_rate)

    nnet = LearnNet.NNet(nunits=[1024, K])

    learning_rate_exp = [0, 1, 2, 3, 4]

    for lr_exp in learning_rate_exp:

        lr = 4**lr_exp

        if lr not in results:
            results[lr] = []

        opt = LearnNet.NNetGDOptimizer(metric=nnet_metric, max_iters=50, learn_rate=lr)

        best_nnet = nnet.fit(
            X,
            y_ohe,
            X_test,
            y_test_ohe,
            optimizer=opt,
            verbose=1,
        )

        train_err = np.array(opt.train_err)
        test_err = np.array(opt.test_err)

        # Get the final errors (last iteration)
        final_train_error = train_err[-1, 1]
        final_test_error = test_err[-1, 1]
        final_train_loss = train_err[-1, 0]
        final_test_loss = test_err[-1, 0]

        # Get the best (minimum) errors during training
        best_train_error = np.min(train_err[:, 1])
        best_train_loss = np.min(train_err[:, 0])
        best_test_error = np.min(test_err[:, 1])
        best_test_loss = np.min(test_err[:, 0])

        run_result = {
            "final_train_error": final_train_error,
            "final_test_error": final_test_error,
            "final_train_loss": final_train_loss,
            "final_test_loss": final_test_loss,
            "best_train_error": best_train_error,
            "best_test_error": best_test_error,
            "best_train_loss": best_train_loss,
            "best_test_loss": best_test_loss,
            "train_err_curve": train_err,
            "test_err_curve": test_err,
        }
        results[lr].append(run_result)
        summary.append(
            {
                "lr": lr,
                "final_train_error": final_train_error,
                "final_test_error": final_test_error,
            }
        )
    return results, summary


def multi_layer_nn(
    X: np.ndarray,
    y_ohe: np.ndarray,
    X_test: np.ndarray,
    y_test_ohe: np.ndarray,
    m: int,
    n: int,
    K: int,
) -> dict:

    results = {}

    best_test_error = float("inf")
    best_train_error = float("inf")
    best_train_loss = float("inf")
    best_test_loss = float("inf")

    nnet_metric = LearnNet.NNetMetric(f=nnet_error_rate)

    hidden_layer_units = [4**2, 4**3, 4**4]

    hidden_layers = [1, 2, 3, 4]

    learning_rate_exp = [-2, -1, 0, 1, 2]

    for units, layers, lr_exp in product(
        hidden_layer_units, hidden_layers, learning_rate_exp
    ):

        lr = 4**lr_exp

        if lr not in results:
            results[lr] = []

        nunits = LearnNet.make_nunits(n, K, layers, units)

        nnet_time = LearnNet.time_nnet(nunits)

        R = min(1000, math.ceil(LearnNet.MAX_TIME / (m * nnet_time)))

        opt = LearnNet.NNetGDOptimizer(metric=nnet_metric, max_iters=R, learn_rate=lr)

        nnet = LearnNet.NNet(nunits=nunits)

        best_nnet = nnet.fit(
            X,
            y_ohe,
            X_test,
            y_test_ohe,
            optimizer=opt,
            verbose=1,
        )

        train_err = np.array(opt.train_err)
        test_err = np.array(opt.test_err)

        # Get the final errors (last iteration)
        final_train_error = train_err[-1, 1]
        final_test_error = test_err[-1, 1]
        final_train_loss = train_err[-1, 0]
        final_test_loss = test_err[-1, 0]

        # Get the best (minimum) errors during training
        best_train_error = np.min(train_err[:, 1])
        best_train_loss = np.min(train_err[:, 0])
        best_test_error = np.min(test_err[:, 1])
        best_test_loss = np.min(test_err[:, 0])

        fold_result = {
            "final_train_error": final_train_error,
            "final_test_error": final_test_error,
            "final_train_loss": final_train_loss,
            "final_test_loss": final_test_loss,
            "best_train_error": best_train_error,
            "best_test_error": best_test_error,
            "best_train_loss": best_train_loss,
            "best_test_loss": best_test_loss,
            "train_err": train_err,
            "test_err": test_err,
        }
        results[lr].append(fold_result)
    return train_err, test_err


def two_layer_nn(
    X: np.ndarray,
    y_ohe: np.ndarray,
    X_test: np.ndarray,
    y_test_ohe: np.ndarray,
    m: int,
    n: int,
    K: int,
) -> dict:

    results = {}

    best_test_error = float("inf")
    best_train_error = float("inf")
    best_train_loss = float("inf")
    best_test_loss = float("inf")

    nnet_metric = LearnNet.NNetMetric(f=nnet_error_rate)

    first_layer_units = [4**3, 4**4]
    second_layer_units = [4**2, 4**3]

    learning_rate_exp = [-3, -2, -1, 0, 1]

    for lr_exp in learning_rate_exp:

        lr = 4**lr_exp

        if lr not in results:
            results[lr] = []

        for first in first_layer_units:
            for second in second_layer_units:
                if second >= first:
                    continue  # ensure the second layer is smaller

                nunits = [n, first, second, K]  # explicitly define the layer sizes

                nnet_time = LearnNet.time_nnet(nunits)

                R = min(1000, math.ceil(LearnNet.MAX_TIME / (m * nnet_time)))

                opt = LearnNet.NNetGDOptimizer(
                    metric=nnet_metric, max_iters=R, learn_rate=lr
                )

                nnet = LearnNet.NNet(nunits=nunits)

                best_nnet = nnet.fit(
                    X,
                    y_ohe,
                    X_test,
                    y_test_ohe,
                    optimizer=opt,
                    verbose=1,
                )

                train_err = np.array(opt.train_err)
                test_err = np.array(opt.test_err)

                # Get the final errors (last iteration)
                final_train_error = train_err[-1, 1]
                final_test_error = test_err[-1, 1]
                final_train_loss = train_err[-1, 0]
                final_test_loss = test_err[-1, 0]

                # Get the best (minimum) errors during training
                best_train_error = np.min(train_err[:, 1])
                best_train_loss = np.min(train_err[:, 0])
                best_test_error = np.min(test_err[:, 1])
                best_test_loss = np.min(test_err[:, 0])

                fold_result = {
                    "final_train_error": final_train_error,
                    "final_test_error": final_test_error,
                    "final_train_loss": final_train_loss,
                    "final_test_loss": final_test_loss,
                    "best_train_error": best_train_error,
                    "best_test_error": best_test_error,
                    "best_train_loss": best_train_loss,
                    "best_test_loss": best_test_loss,
                    "train_err": train_err,
                    "test_err": test_err,
                }
                results[lr].append(fold_result)

    return results


def nnet_error_rate(y_true, y_pred):
    y_pred_label = np.argmax(y_pred, axis=0).reshape(-1, 1)
    y_true_label = out_enc.inverse_transform(y_true.T).reshape(-1, 1)
    return LearnNet.error_rate(y_true_label, y_pred_label)


def plot_err_loss(train_err: np.ndarray, test_err: np.ndarray, plot: str):
    if plot == "loss":
        plt.plot(train_err[:, 0])
        plt.plot(test_err[:, 0])
        plt.savefig("loss.png")
    else:
        plt.plot(train_err[:, 1])
        plt.plot(test_err[:, 1])
        plt.show()


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

    kf = KFold(n_splits=3, random_state=42, shuffle=True)

    #############################################
    #           Multi Output Perceptron         #
    #           1024 Inputs, 10 outputs         #
    #           Gradient Descent                #
    #   Learning Rate [4^0, 4^1, 4^2, 4^3, 4^4] #
    #############################################

    fold_train_errors = []
    fold_val_errors = []

    for i, (train_idx, val_idx) in enumerate(kf.split(X)):

        # 1. Split into training and validation subsets
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold_ohe, y_val_fold_ohe = y_ohe[train_idx], y_ohe[val_idx]

        # 2. Train the model using the fold's train set

        results, summary = perceptron(
            X_train_fold,
            y_train_fold_ohe,
            X_val_fold,
            y_val_fold_ohe,
            K,
        )

    #############################################
    #   Deep Neural Network                     #
    #   Hidden Layer Units: [4^2, 4^3, 4^4]     #
    #   Hidden Layers: [1, 2, 3, 4]             #
    #   Learning Rate [-1, -1, 0, 1, 2]         #
    #############################################

    # mlp_train_err, mlp_test_err = multi_layer_nn(
    # X, y_ohe, X_test, y_test_ohe, m, n, K, kf
    # )

    ###############################################
    #   Deep Neural Network                       #
    #   Hidden Layer Units: [4^4, 4^3, 4^3, 4^2]  #
    #   Hidden Layers: [2]                        #
    #   Learning Rate [-3, -2, -1, 0, 1]          #
    ###############################################

    # two_layer_train_err, two_layer_test_err = two_layer_nn(
    #     X, y_ohe, X_test, y_test_ohe, m, n, K, kf
    # )

    breakpoint()
    # -----------------------------------
    # 4. Analyze and report results (turn into a function)
    # -----------------------------------
    for lr, folds in perceptron_results.items():
        print(f"\nLearning Rate: {lr}")
        for i, fold_res in enumerate(folds):
            print(f"  Fold {i + 1}:")
            print(f"    Final Train Error: {fold_res['final_train_error']:.4f}")
            print(f"    Final Test Error:  {fold_res['final_test_error']:.4f}")
            print(f"    Best Train Error:  {fold_res['best_train_error']:.4f}")
            print(f"    Best Test Error:   {fold_res['best_test_error']:.4f}")
