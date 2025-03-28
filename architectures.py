import math
from itertools import product

import numpy as np

import LearnNet
from helper import nnet_error_rate


def perceptron(
    X: np.ndarray,
    y_ohe: np.ndarray,
    X_test: np.ndarray,
    y_test_ohe: np.ndarray,
    m: int,
    n: int,
    K: int,
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
            verbose=0,
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


def perceptron_single_lr(
    X: np.ndarray,
    y_ohe: np.ndarray,
    X_test: np.ndarray,
    y_test_ohe: np.ndarray,
    K: int,
    lr: float,
    max_iters: int = 50,
):

    nnet_metric = LearnNet.NNetMetric(f=nnet_error_rate)
    nnet = LearnNet.NNet(nunits=[1024, K])

    opt = LearnNet.NNetGDOptimizer(metric=nnet_metric, max_iters=50, learn_rate=lr)

    best_nnet = nnet.fit(
        X,
        y_ohe,
        X_test,
        y_test_ohe,
        optimizer=opt,
        verbose=0,
    )

    train_err = np.array(opt.train_err)
    test_err = np.array(opt.test_err)

    results = {"train_err_curve": train_err, "test_err_curve": test_err}

    return results


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
    summary = []

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
            verbose=0,
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
            "hidden_units": units,
            "hidden_layers": layers,
            "lr": lr,
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


def multi_single_lr(
    X: np.ndarray,
    y_ohe: np.ndarray,
    X_test: np.ndarray,
    y_test_ohe: np.ndarray,
    m: int,
    n: int,
    K: int,
    hidden_units,
    hidden_layers,
    lr,
) -> dict:

    results = {}

    nnet_metric = LearnNet.NNetMetric(f=nnet_error_rate)

    nunits = LearnNet.make_nunits(n, K, hidden_layers, hidden_units)

    nnet = LearnNet.NNet(nunits=nunits)

    opt = LearnNet.NNetGDOptimizer(metric=nnet_metric, max_iters=50, learn_rate=lr)

    best_nnet = nnet.fit(
        X,
        y_ohe,
        X_test,
        y_test_ohe,
        optimizer=opt,
        verbose=0,
    )

    train_err = np.array(opt.train_err)
    test_err = np.array(opt.test_err)

    results = {"train_err_curve": train_err, "test_err_curve": test_err}
    return results


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
    summary = []
    best_test_error = float("inf")
    best_train_error = float("inf")
    best_train_loss = float("inf")
    best_test_loss = float("inf")

    nnet_metric = LearnNet.NNetMetric(f=nnet_error_rate)

    first_layer_options = [4**4, 4**3]  # 256, 64
    second_layer_options = [4**3, 4**2]  # 64, 16
    learning_rate_exp = [-3, -2, -1, 0, 1]

    for first_units, second_units, lr_exp in product(
        first_layer_options, second_layer_options, learning_rate_exp
    ):
        # Enforce second layer < first layer
        if second_units >= first_units:
            continue

        lr = 4**lr_exp
        if lr not in results:
            results[lr] = []

            nunits = [
                n,
                first_units,
                second_units,
                K,
            ]  # explicitly define the layer sizes

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
                verbose=0,
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
                "first_units": first_units,
                "second_units": second_units,
                "lr_exp": lr_exp,
                "lr": lr,
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


def two_layer_single_lr(
    X: np.ndarray,
    y_ohe: np.ndarray,
    X_test: np.ndarray,
    y_test_ohe: np.ndarray,
    m: int,
    n: int,
    K: int,
    first_units: int,
    second_units: int,
    lr: float,
) -> dict:

    # Define your metric (or accept it as a parameter)
    nnet_metric = LearnNet.NNetMetric(f=nnet_error_rate)

    # Explicitly define the layer sizes: [input_dim, first_hidden, second_hidden, output_dim]
    nunits = [n, first_units, second_units, K]

    # Estimate time for one iteration
    nnet_time = LearnNet.time_nnet(nunits)

    # Determine the max iterations you can run under your time constraint
    R = min(1000, math.ceil(LearnNet.MAX_TIME / (m * nnet_time)))

    # Create the optimizer
    opt = LearnNet.NNetGDOptimizer(metric=nnet_metric, max_iters=R, learn_rate=lr)

    # Build and fit the network
    nnet = LearnNet.NNet(nunits=nunits)
    best_nnet = nnet.fit(
        X,
        y_ohe,
        X_test,
        y_test_ohe,
        optimizer=opt,
        verbose=0,  # or 1 if you want to see training logs
    )

    # Convert logs to arrays: shape (iterations, 2) => (loss, error)
    train_err = np.array(opt.train_err)
    test_err = np.array(opt.test_err)

    # Return the curves
    return {"train_err_curve": train_err, "test_err_curve": test_err}
