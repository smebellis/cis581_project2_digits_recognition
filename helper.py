import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
import random

from train import train_network, evaluate_trial_dataset
from utils import make_nnet_error_rate
from plots import plot_learning_curves, plot_output_weights, plot_random_hidden_units
import LearnNet


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


def print_cv_summary(arch_name, cv_results, duration):
    print(f"\n=== {arch_name.upper()} CV Summary ===")
    for lr, metrics in cv_results["summary_metrics"].items():
        print(
            f"LR = {lr}: "
            f"Avg Train Error = {metrics['mean_train_err']:.4f}, "
            f"Avg Val Error = {metrics['mean_val_err']:.4f}, "
            f"Avg Train Loss = {metrics['mean_train_loss']:.4f}, "
            f"Avg Val Loss = {metrics['mean_val_loss']:.4f}"
        )
    print(f"\nBest LR: {cv_results['best_lr']}")
    print(f"Best Configurations: {cv_results['configs_chosen']}")
    print(f" {arch_name} CV completed in {duration/60:.2f} minutes.")


def tabulate_cv_errors(
    cv_results_dict,
    csv_save_path="cv_results_summary.csv",
    plot_save_path="cv_results_plot.png",
):
    records = []

    # Collect data from each architecture
    for arch_name, cv_result in cv_results_dict.items():
        for lr, metrics in cv_result["summary_metrics"].items():
            records.append(
                {
                    "Architecture": arch_name,
                    "Learning Rate": lr,
                    "Avg CV Test Misclassification Error": metrics["mean_val_err"],
                }
            )

    # Create DataFrame
    df = pd.DataFrame(records)

    # Tabulate results
    pivot_df = df.pivot(
        index="Learning Rate",
        columns="Architecture",
        values="Avg CV Test Misclassification Error",
    ).round(4)

    print("\n Average CV Test Misclassification Errors:")
    print(pivot_df)

    # Save the tabulated results as CSV
    pivot_df.to_csv(csv_save_path)
    print(f"\n CV results summary saved to '{csv_save_path}'.")

    return df


def tabulate_final_results(train_curve, test_curve, save_path="final_results.csv"):
    """
    Tabulates and prints the final training/test proxy error (loss)
    and misclassification error clearly.
    """
    final_results = {
        "Dataset": ["Training", "Test"],
        "Proxy Error (Loss)": [train_curve[-1, 0], test_curve[-1, 0]],
        "Misclassification Error": [train_curve[-1, 1], test_curve[-1, 1]],
    }

    df_results = pd.DataFrame(final_results).round(4)

    print("\n Final Neural Network Performance:")
    print(df_results)

    df_results.to_csv(save_path, index=False)
    print(f"\n Final results saved to '{save_path}'.")


def generate_learning_curves(
    train_network_fn,
    X,
    y_ohe,
    X_test,
    y_test_ohe,
    best_architecture_parameters,
    out_enc,
    train_sizes=None,
    max_iters=1000,
    debug=False,
):

    if train_sizes is None:
        train_sizes = [10, 40, 100, 200, 400, 800, 1600]

    results = []

    for m_current in train_sizes:
        # Subset the data
        X_subset = X[:m_current]
        y_subset_ohe = y_ohe[:m_current]

        # Train model on the current subset
        final_run_subset = train_network_fn(
            X_subset,
            y_subset_ohe,
            X_test,
            y_test_ohe,
            best_architecture_parameters["layer_sizes"],
            best_architecture_parameters["best_lr"],
            max_iters=max_iters,
            out_enc=out_enc,
            debug=debug,
        )

        train_loss = final_run_subset["train_err_curve"][-1, 0]
        train_error = final_run_subset["train_err_curve"][-1, 1]
        test_loss = final_run_subset["test_err_curve"][-1, 0]
        test_error = final_run_subset["test_err_curve"][-1, 1]

        results.append(
            {
                "m": m_current,
                "train_error": train_error,
                "test_error": test_error,
                "train_loss": train_loss,
                "test_loss": test_loss,
            }
        )

    # Create a DataFrame from the collected results
    df_results = pd.DataFrame(results)

    plot_learning_curves(df_results)

    return df_results


def run_experiment(
    final_run,
    X_trial,
    y_trial,
    out_enc,
    X,
    y_ohe,
    X_test,
    y_test_ohe,
    best_architecture_parameters,
    debug=False,
):

    if debug:

        return None
    else:
        # ----------------------------#
        #   Evaluate Trial Dataset    #
        # ----------------------------#
        evaluate_trial_dataset(
            final_nnet=final_run["trained_model"],
            X_trial=X_trial,
            y_trial=y_trial,
            out_enc=out_enc,
            results_save_path="trial_dataset_results.csv",
        )

        # ----------------------------#
        #   Learning Curve            #
        # ----------------------------#
        df_results = generate_learning_curves(
            train_network_fn=train_network,
            X=X,
            y_ohe=y_ohe,
            X_test=X_test,
            y_test_ohe=y_test_ohe,
            best_architecture_parameters=best_architecture_parameters,
            out_enc=out_enc,
            train_sizes=None,
            max_iters=1000,
            debug=debug,
        )

        # --------------------------------------#
        #   Weight Parameter Interpretation     #
        # --------------------------------------#

        plot_output_weights(final_run)

        # 4) Plot random hidden units from the first hidden layer
        plot_random_hidden_units(
            final_run,
            layer_idx=1,
            num_units=10,
            reshape_size=(32, 32),
            cmap="gray",
        )

        return df_results
