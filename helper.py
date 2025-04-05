import numpy as np
import pandas as pd
from collections import Counter

from train import train_network, evaluate_trial_dataset

from plots import (
    plot_learning_curves,
    plot_output_weights,
    plot_random_hidden_units_from_architectures,
    plot_and_save_final_curves,
)


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
    csv_save_path="results/cv_results_summary.csv",
    plot_save_path="results/cv_results_plot.png",
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

    # Save the results as CSV
    pivot_df.to_csv(csv_save_path)
    print(f"\n CV results summary saved to '{csv_save_path}'.")

    return df


def tabulate_final_results(
    train_curve, test_curve, model_name, save_path="results/final_results.csv"
):

    print(f"\nFinal Neural Network Performance for model: {model_name}")

    final_results = {
        "Dataset": ["Training", "Test"],
        "Proxy Error (Loss)": [train_curve[-1, 0], test_curve[-1, 0]],
        "Misclassification Error": [train_curve[-1, 1], test_curve[-1, 1]],
    }

    df_results = pd.DataFrame(final_results).round(4)

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
    arch_name,
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

    plot_learning_curves(df_results, arch_name)

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
    model_name,
    debug=False,
):

    if debug:
        return None
    else:
        arch_name = best_architecture_parameters["architecture"]
        # ----------------------------#
        #   Evaluate Trial Dataset    #
        # ----------------------------#
        evaluate_trial_dataset(
            final_nnet=final_run["trained_model"],
            X_trial=X_trial,
            y_trial=y_trial,
            results_save_path=None,
            arch_name=model_name,
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
            arch_name=arch_name,
            train_sizes=None,
            max_iters=1000,
            debug=debug,
        )

        # --------------------------------------#
        #   Weight Parameter Interpretation     #
        # --------------------------------------#

        # If the architecture is "perceptron", visualize output-layer weights
        if arch_name.lower() == "perceptron":
            plot_output_weights(
                final_run,
                filename=f"results/final_perceptron_output_weights_{arch_name}.png",
                cmap="gray",
                ncols=5,
                reshape_size=(32, 32),
                arch_name=arch_name,
            )

        else:
            # For deep networks, plot 10 random units from the first hidden layer
            plot_random_hidden_units_from_architectures(
                {arch_name: final_run},
                layer_idx=1,  # first hidden layer
                num_units=10,
                reshape_size=(32, 32),
                cmap="gray",
            )

        return df_results


def get_best_params_and_final_runs(
    architectures, X, y_ohe, X_test, y_test_ohe, out_enc, debug=False
):
    results = {}
    for model_name, cv_result in architectures:
        best_lr = cv_result["best_lr"]
        # Use Counter to determine the most common configuration from the list of configs
        best_config = Counter(
            tuple(config) for config in cv_result["configs_chosen"]
        ).most_common(1)[0][0]
        best_params = {
            "architecture": model_name,
            "best_lr": best_lr,
            "layer_sizes": best_config,
            "validation_error": cv_result["lowest_val_error"],
        }

        print(f"\nFinal training for {model_name} with best parameters:")
        print(f"  Best LR: {best_lr}")
        print(f"  Layer Sizes: {best_config}")
        print(f"  Validation Error: {cv_result['lowest_val_error']:.4f}")

        final_run_raw = train_network(
            X,
            y_ohe,
            X_test,
            y_test_ohe,
            best_config,
            best_lr,
            max_iters=1000,
            out_enc=out_enc,
            debug=debug,
        )

        # If train_network returns a tuple, convert it to a dictionary.
        if isinstance(final_run_raw, tuple):
            final_run = {
                "trained_model": final_run_raw[0],
                "train_err_curve": final_run_raw[1],
                "test_err_curve": final_run_raw[2],
            }
        else:
            final_run = final_run_raw

        # Extract training and testing error curves from the final run
        train_curve = final_run["train_err_curve"]
        test_curve = final_run["test_err_curve"]

        # Define a unique file path for each model's final training plot
        plot_save_path = (
            f"results/final_training_plots_{model_name.replace(' ', '_')}.png"
        )
        # Plot and save the final training curves and loss curves
        plot_and_save_final_curves(
            train_curve, test_curve, best_params, save_path=plot_save_path
        )

        csv_save_path = f"results/final_results_{model_name.replace(' ', '_')}.csv"
        tabulate_final_results(
            train_curve, test_curve, model_name, save_path=csv_save_path
        )

        results[model_name] = {"best_params": best_params, "final_model": final_run}

    return results
