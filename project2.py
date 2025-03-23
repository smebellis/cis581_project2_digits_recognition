import math
import time
from collections import defaultdict, Counter
from itertools import product
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from sklearn.preprocessing import LabelBinarizer

import LearnNet

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="overflow encountered in exp"
)


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


def summarize_results(results: dict) -> dict:

    summary = {}

    for lr_value, runs in results.items():

        # Collect metrics on each run
        final_train_errors = [run["final_train_error"] for run in runs]
        final_test_errors = [run["final_test_error"] for run in runs]
        final_train_losses = [run["final_train_loss"] for run in runs]
        final_test_losses = [run["final_test_loss"] for run in runs]

        # Compute averages
        avg_train_err = np.mean(final_train_errors)
        avg_test_err = np.mean(final_test_errors)
        avg_train_loss = np.mean(final_train_losses)
        avg_test_loss = np.mean(final_test_losses)

        summary[lr_value] = {
            "avg_train_error": avg_train_err,
            "avg_test_error": avg_test_err,
            "avg_train_loss": avg_train_loss,
            "avg_test_loss": avg_test_loss,
        }

    return summary


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


def plot_training_curves(results: dict, lr_value: float):
    """
    Plots the training and test loss/error vs. iteration for the specified learning rate.

    Parameters
    ----------
    results : dict
        The dictionary returned by perceptron(...).
    lr_value : float
        The learning rate key for which to plot the curves.
    """

    run_data = results[lr_value][0]

    train_curve = run_data["train_err_curve"]  # shape (iterations, 2)
    test_curve = run_data["test_err_curve"]

    iterations = range(len(train_curve))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot the LOSS curves
    ax1.plot(iterations, train_curve[:, 0], label="Train Loss")
    ax1.plot(iterations, test_curve[:, 0], label="Test Loss")
    ax1.set_title(f"Loss Curves (LR = {lr_value})")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Plot the ERROR curves
    ax2.plot(iterations, train_curve[:, 1], label="Train Error")
    ax2.plot(iterations, test_curve[:, 1], label="Test Error")
    ax2.set_title(f"Error Curves (LR = {lr_value})")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Error")
    ax2.legend()

    plt.show()


if __name__ == "__main__":
    starting_time = time.time()
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

    # Initialize the fold_train_errors, fold_val_errors
    fold_train_errors = defaultdict(list)
    fold_val_errors = defaultdict(list)
    fold_train_losses = defaultdict(list)
    fold_val_losses = defaultdict(list)

    best_lr_perceptron = None
    lowest_val_error_perceptron = float("inf")

    print("=== Cross-Validation: Multi-Output Perceptron ===")

    for i, (train_idx, val_idx) in enumerate(kf.split(X)):

        # 1. Split into training and validation subsets
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold_ohe, y_val_fold_ohe = y_ohe[train_idx], y_ohe[val_idx]

        # 2. Train the model using the fold's train set

        perceptron_results, perceptron_summary = perceptron(
            X_train_fold,
            y_train_fold_ohe,
            X_val_fold,
            y_val_fold_ohe,
            K,
        )

        # 3. Summarize the results (averages across runs for each LR)
        avg_metrics = summarize_results(perceptron_results)

        # 4. Store each LR's average final metrics for this fold
        for lr_value, metrics in avg_metrics.items():
            fold_train_errors[lr_value].append(metrics["avg_train_error"])
            fold_val_errors[lr_value].append(metrics["avg_test_error"])
            fold_train_losses[lr_value].append(metrics["avg_train_loss"])
            fold_val_losses[lr_value].append(metrics["avg_test_loss"])

    # 5. After all folds are done, compute final averages across folds
    print("=== Perceptron CV Summary ===")
    for lr_value in fold_train_errors.keys():
        mean_train_err = np.mean(fold_train_errors[lr_value])
        mean_val_err = np.mean(fold_val_errors[lr_value])
        mean_train_loss = np.mean(fold_train_losses[lr_value])
        mean_val_loss = np.mean(fold_val_losses[lr_value])

        print(
            f"LR = {lr_value}: "
            f"Avg Train Error = {mean_train_err:.4f}, "
            f"Avg Val Error = {mean_val_err:.4f}, "
            f"Avg Train Loss = {mean_train_loss:.4f}, "
            f"Avg Val Loss = {mean_val_loss:.4f}"
        )

        if mean_val_err < lowest_val_error_perceptron:
            lowest_val_error_perceptron = mean_val_err
            best_lr_perceptron = lr_value

    print(
        f"\nBest LR from CV = {best_lr_perceptron:}, with avg val error = {lowest_val_error_perceptron:.4f}"
    )

    perceptron_best = {
        "lr": best_lr_perceptron,
        "lowest_val_error": lowest_val_error_perceptron,
    }

    print("=== Time to Complete Perceptron Run ===")
    end_time_perceptron = time.time()
    total_seconds_perceptron = end_time_perceptron - starting_time

    minutes = int(total_seconds_perceptron // 60)
    seconds = int(total_seconds_perceptron % 60)

    print(f"\n🕒 Total Run Time: {minutes} minutes {seconds} seconds")
    #############################################
    #   Deep Neural Network                     #
    #   Hidden Layer Units: [4^2, 4^3, 4^4]     #
    #   Hidden Layers: [1, 2, 3, 4]             #
    #   Learning Rate [-1, -1, 0, 1, 2]         #
    #############################################

    # Reset the errors
    fold_train_errors = defaultdict(list)
    fold_val_errors = defaultdict(list)
    fold_train_losses = defaultdict(list)
    fold_val_losses = defaultdict(list)
    best_config_for_lr = defaultdict(list)

    lowest_val_error_multi = float("inf")
    best_lr_multi = None

    print("=== Cross-Validation: Multi-Layer NN (Uniform) ===")

    for i, (train_idx, val_idx) in enumerate(kf.split(X)):

        # 1. Split into training and validation subsets
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold_ohe, y_val_fold_ohe = y_ohe[train_idx], y_ohe[val_idx]

        # 2. Train the model using the fold's train set

        multi_results, multi_summary = multi_layer_nn(
            X_train_fold,
            y_train_fold_ohe,
            X_val_fold,
            y_val_fold_ohe,
            m,
            n,
            K,
        )

        # 3. Summarize the results (averages across runs for each LR)
        avg_metrics = summarize_results(multi_results)

        # 4. Store each LR's average final metrics for this fold
        for lr_value, run_list in multi_results.items():
            best_run = min(run_list, key=lambda r: r["final_test_error"])

            fold_train_errors[lr_value].append(best_run["final_train_error"])
            fold_val_errors[lr_value].append(best_run["final_test_error"])
            fold_train_losses[lr_value].append(best_run["final_train_loss"])
            fold_val_losses[lr_value].append(best_run["final_test_loss"])

            best_config_for_lr[lr_value].append(
                (best_run["hidden_units"], best_run["hidden_layers"])
            )

    # 5. After all folds are done, compute final averages across folds
    print("=== Multi-Layer NN CV Summary ===")
    for lr_value in fold_train_errors.keys():
        mean_train_err = np.mean(fold_train_errors[lr_value])
        mean_val_err = np.mean(fold_val_errors[lr_value])
        mean_train_loss = np.mean(fold_train_losses[lr_value])
        mean_val_loss = np.mean(fold_val_losses[lr_value])

        print(
            f"LR = {lr_value}: "
            f"Avg Train Error = {mean_train_err:.4f}, "
            f"Avg Val Error = {mean_val_err:.4f}, "
            f"Avg Train Loss = {mean_train_loss:.4f}, "
            f"Avg Val Loss = {mean_val_loss:.4f}"
        )

        if mean_val_err < lowest_val_error_multi:
            lowest_val_error_multi = mean_val_err
            best_lr_multi = lr_value

    print(
        f"\nBest LR from CV = {best_lr_multi}, with avg val error = {lowest_val_error_multi:.4f}"
    )

    multi_best = {"lr": best_lr_multi, "lowest_val_error": lowest_val_error_multi}
    configs_chosen = best_config_for_lr[best_lr_multi]
    print(f"Configs chosen for LR={best_lr_multi} across folds: {configs_chosen}")

    combo_counts = Counter(configs_chosen)
    most_common_combo, count = combo_counts.most_common(1)[0]
    print(
        f"Most common combo for LR={best_lr_multi}: {most_common_combo}, chosen {count} times"
    )

    print("=== Time to Complete Multi-Layer Run ===")
    end_time_multi = time.time()
    total_seconds_multi = end_time_multi - starting_time

    minutes = int(total_seconds_multi // 60)
    seconds = int(total_seconds_multi % 60)

    print(f"\n🕒 Total Run Time: {minutes} minutes {seconds} seconds")
    ###############################################
    #   Two-Layer NN                              #
    #   Hidden Layer Units: [4^4, 4^3, 4^3, 4^2]  #
    #   Hidden Layers: [2]                        #
    #   Learning Rate [-3, -2, -1, 0, 1]          #
    ###############################################

    # Reset the errors
    fold_train_errors = defaultdict(list)
    fold_val_errors = defaultdict(list)
    fold_train_losses = defaultdict(list)
    fold_val_losses = defaultdict(list)
    best_config_for_lr = defaultdict(list)

    best_config_for_lr = defaultdict(
        list
    )  # store best (first_units, second_units) for each LR
    best_run_train_err = defaultdict(list)  # store best run's final train error
    best_run_val_err = defaultdict(list)  # store best run's final test error
    best_run_train_loss = defaultdict(list)
    best_run_val_loss = defaultdict(list)

    lowest_val_error_two = float("inf")
    best_lr_two = None

    print("=== Cross-Validation: Two-Layer NN ===")

    for i, (train_idx, val_idx) in enumerate(kf.split(X)):

        # 1. Split into training and validation subsets
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold_ohe, y_val_fold_ohe = y_ohe[train_idx], y_ohe[val_idx]

        # 2. Train the model using the fold's train set

        two_results, two_summary = two_layer_nn(
            X_train_fold,
            y_train_fold_ohe,
            X_val_fold,
            y_val_fold_ohe,
            m,
            n,
            K,
        )

        # 3. Summarize the results (averages across runs for each LR)
        avg_metrics = summarize_results(two_results)

        # 4. Store each LR's average final metrics for this fold
        for (
            lr_value,
            metrics,
        ) in avg_metrics.items():  # TODO: Change this to two_results
            fold_train_errors[lr_value].append(metrics["avg_train_error"])
            fold_val_errors[lr_value].append(metrics["avg_test_error"])
            fold_train_losses[lr_value].append(metrics["avg_train_loss"])
            fold_val_losses[lr_value].append(metrics["avg_test_loss"])

    # 5. After all folds are done, compute final averages across folds
    print("=== Two-Layer NN CV Summary ===")
    for lr_value in fold_train_errors.keys():
        mean_train_err = np.mean(fold_train_errors[lr_value])
        mean_val_err = np.mean(fold_val_errors[lr_value])
        mean_train_loss = np.mean(fold_train_losses[lr_value])
        mean_val_loss = np.mean(fold_val_losses[lr_value])

        print(
            f"LR = {lr_value}: "
            f"Avg Train Error = {mean_train_err:.4f}, "
            f"Avg Val Error = {mean_val_err:.4f}, "
            f"Avg Train Loss = {mean_train_loss:.4f}, "
            f"Avg Val Loss = {mean_val_loss:.4f}"
        )

        if mean_val_err < lowest_val_error_two:
            lowest_val_error_two = mean_val_err
            best_lr_two = lr_value

    print(
        f"\nBest LR from CV = {best_lr_two:.4f}, with avg val error = {lowest_val_error_two:.4f}"
    )

    two_layer_best = {"lr": best_lr_two, "lowest_val_error": lowest_val_error_two}

    print("=== Time to Complete Two-Layer Run ===")
    end_time_two = time.time()
    total_seconds_two = end_time_two - starting_time

    minutes = int(total_seconds_two // 60)
    seconds = int(total_seconds_two % 60)

    print(f"\n🕒 Total Run Time: {minutes} minutes {seconds} seconds")

    # -------------------------------------------------------#
    #   Model Selection based on Misclassification Error     #
    # -------------------------------------------------------#

    all_architectures = [
        ("perceptron", perceptron_best["lr"], perceptron_best["lowest_val_error"]),
        ("multi_layer", multi_best["lr"], multi_best["lowest_val_error"]),
        ("two_layer", two_layer_best["lr"], two_layer_best["lowest_val_error"]),
    ]

    # Pick the architecture with the smallest validation error
    best_arch, best_lr, best_err = min(all_architectures, key=lambda x: x[2])

    print(f"=== Best Architecture Overall ===")
    print(f"Architecture: {best_arch}, LR = {best_lr}, Avg Val Error = {best_err:.4f}")

    end_time = time.time()
    total_seconds = end_time - starting_time

    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    print(f"\n🕒 Total Run Time: {minutes} minutes {seconds} seconds")

    breakpoint()
    if best_arch == "perceptron":
        final_run = perceptron_single_lr(X, y_ohe, X_test, y_test_ohe, K, lr=best_lr)
    elif best_arch == "multi_layer":
        # Suppose you found a certain best hidden_units & hidden_layers from the CV
        # If you tried multiple combos, you’d have stored them as well
        final_run = multi_single_lr(
            X,
            y_ohe,
            X_test,
            y_test_ohe,
            m,
            n,
            K,
            hidden_units=...,
            hidden_layers=...,
            lr=best_lr,
        )
    elif best_arch == "two_layer":
        final_run = two_layer_single_lr(
            X,
            y_ohe,
            X_test,
            y_test_ohe,
            m,
            n,
            K,
            first_units=...,
            second_units=...,
            lr=best_lr,
        )

    train_curve = final_run["train_err_curve"]
    test_curve = final_run["test_err_curve"]

    iterations = range(len(train_curve))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(iterations, train_curve[:, 0], label="Train Loss")
    ax1.plot(iterations, test_curve[:, 0], label="Test Loss")
    ax1.set_title(f"Loss Curves (LR = {best_lr:.4f})")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(iterations, train_curve[:, 1], label="Train Error")
    ax2.plot(iterations, test_curve[:, 1], label="Test Error")
    ax2.set_title(f"Error Curves (LR = {best_lr:.4f})")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Error")
    ax2.legend()

    plt.show()
