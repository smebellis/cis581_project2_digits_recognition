import math
from collections import defaultdict

import numpy as np
import pandas as pd

import LearnNet
from utils import make_nnet_error_rate

# Size of Subset for debugging
SIZE = 5


def train_network(
    X, y_ohe, X_test, y_test_ohe, layer_sizes, lr, max_iters, out_enc, debug=False
):
    nnet_metric = LearnNet.NNetMetric(f=make_nnet_error_rate(out_enc))
    nnet = LearnNet.NNet(nunits=layer_sizes)
    opt = LearnNet.NNetGDOptimizer(
        metric=nnet_metric, max_iters=max_iters, learn_rate=lr
    )

    if debug:

        X = X[:SIZE]
        y_ohe = y_ohe[:SIZE]
        X_test = X_test[:SIZE]
        y_test_ohe = y_test_ohe[:SIZE]

    best_nnet = nnet.fit(X, y_ohe, X_test, y_test_ohe, optimizer=opt, verbose=0)

    train_err = np.array(opt.train_err)
    test_err = np.array(opt.test_err)

    results = {
        "trained_model": best_nnet,
        "final_train_error": train_err[-1, 1],
        "final_test_error": test_err[-1, 1],
        "final_train_loss": train_err[-1, 0],
        "final_test_loss": test_err[-1, 0],
        "best_train_error": np.min(train_err[:, 1]),
        "best_test_error": np.min(test_err[:, 1]),
        "best_train_loss": np.min(train_err[:, 0]),
        "best_test_loss": np.min(test_err[:, 0]),
        "train_err_curve": train_err,
        "test_err_curve": test_err,
    }

    return results


def evaluate_models(
    X, y_ohe, X_test, y_test_ohe, m, configs, out_enc, max_iters=50, debug=False
):
    results = []

    if debug:
        X = X[:SIZE]
        y_ohe = y_ohe[:SIZE]
        X_test = X_test[:SIZE]
        y_test_ohe = y_test_ohe[:SIZE]

    for config in configs:
        layer_sizes, lr = config["layer_sizes"], config["lr"]

        nnet_time = LearnNet.time_nnet(layer_sizes)
        adjusted_max_iters = min(1000, math.ceil(LearnNet.MAX_TIME / (m * nnet_time)))

        run_result = train_network(
            X, y_ohe, X_test, y_test_ohe, layer_sizes, lr, adjusted_max_iters, out_enc
        )

        results.append(
            {
                "layer_sizes": layer_sizes,
                "lr": lr,
                **run_result,
            }
        )

    return results


def train_cv(X, y_ohe, kf, configs, out_enc, arch_name=None, debug=False):
    fold_train_errors = defaultdict(list)
    fold_val_errors = defaultdict(list)
    fold_train_losses = defaultdict(list)
    fold_val_losses = defaultdict(list)
    best_config_for_lr = defaultdict(list)

    if debug:
        X = X[:SIZE]
        y_ohe = y_ohe[:SIZE]

    for train_idx, val_idx in kf.split(X):

        # Split data into folds
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold_ohe, y_val_fold_ohe = y_ohe[train_idx], y_ohe[val_idx]

        # Train neural network models on current fold
        fold_results = evaluate_models(
            X_train_fold,
            y_train_fold_ohe,
            X_val_fold,
            y_val_fold_ohe,
            len(train_idx),
            configs,
            out_enc,
        )

        # Store fold metrics
        for run_result in fold_results:
            lr_value = run_result["lr"]

            fold_train_errors[lr_value].append(run_result["final_train_error"])
            fold_val_errors[lr_value].append(run_result["final_test_error"])
            fold_train_losses[lr_value].append(run_result["final_train_loss"])
            fold_val_losses[lr_value].append(run_result["final_test_loss"])

            config = run_result.get("layer_sizes")
            if config:
                best_config_for_lr[lr_value].append(config)

    # Compute average metrics across all folds
    summary_metrics = {}
    lowest_val_error = float("inf")
    best_lr = None

    for lr_value in fold_train_errors.keys():
        mean_train_err = np.mean(fold_train_errors[lr_value])
        mean_val_err = np.mean(fold_val_errors[lr_value])
        mean_train_loss = np.mean(fold_train_losses[lr_value])
        mean_val_loss = np.mean(fold_val_losses[lr_value])

        summary_metrics[lr_value] = {
            "mean_train_err": mean_train_err,
            "mean_val_err": mean_val_err,
            "mean_train_loss": mean_train_loss,
            "mean_val_loss": mean_val_loss,
        }

        if mean_val_err < lowest_val_error:
            lowest_val_error = mean_val_err
            best_lr = lr_value

    best_configs_chosen = best_config_for_lr[best_lr]

    # Save the best metrics for the final neural network (final training error and losses)
    best_train_err = summary_metrics[best_lr]["mean_train_err"]
    best_train_loss = summary_metrics[best_lr]["mean_train_loss"]
    best_val_loss = summary_metrics[best_lr]["mean_val_loss"]

    # Convert the summary_metrics dict into a DataFrame
    df_summary = (
        pd.DataFrame.from_dict(summary_metrics, orient="index")
        .reset_index()
        .rename(columns={"index": "Learning Rate"})
    )
    if arch_name is not None:
        df_summary["Architecture"] = arch_name

    return {
        "summary_metrics": summary_metrics,
        "summary_df": df_summary,
        "best_lr": best_lr,
        "lowest_val_error": lowest_val_error,
        "best_train_error": best_train_err,
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "configs_chosen": best_configs_chosen,
    }


def evaluate_trial_dataset(
    final_nnet, X_trial, y_trial, arch_name, results_save_path=None
):

    if results_save_path is None:
        results_save_path = f"results/trial_dataset_results_{arch_name}.csv"

    # Forward pass to get predicted probabilities
    y_pred_probs = final_nnet.forwardprop(
        X_trial.T
    )  # assuming shape (features, samples)

    # Predicted labels
    y_pred_labels = np.argmax(y_pred_probs, axis=0)

    # True labels
    y_true_labels = y_trial.flatten().astype(int)

    # Prepare results for tabulation
    df_results = pd.DataFrame(
        {
            "Example #": np.arange(1, len(y_trial) + 1),
            "True Label": y_true_labels,
            "Predicted Label": y_pred_labels,
            "Correct Prediction": (y_true_labels == y_pred_labels),
        }
    )

    # Display results
    print(f"\nTrial Dataset Evaluation: {arch_name}")
    print(df_results)

    # Save results as CSV
    df_results.to_csv(results_save_path, index=False)
    print(f"\n Trial dataset evaluation results saved to '{results_save_path}'.")
