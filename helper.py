import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


def make_nnet_error_rate(out_enc):
    def nnet_error_rate(y_true, y_pred):
        y_pred_label = np.argmax(y_pred, axis=0).reshape(-1, 1)
        y_true_label = out_enc.inverse_transform(y_true.T).reshape(-1, 1)
        return LearnNet.error_rate(y_true_label, y_pred_label)

    return nnet_error_rate


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


def tabulate_and_plot_cv_errors(
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

    # Plotting results
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=df,
        x="Learning Rate",
        y="Avg CV Test Misclassification Error",
        hue="Architecture",
    )
    plt.title(
        "Average CV Test Misclassification Error by Architecture and Learning Rate"
    )
    plt.ylabel("Avg CV Test Misclassification Error")
    plt.xlabel("Learning Rate")
    plt.legend(title="Architecture")
    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(plot_save_path)
    print(f"CV results plot saved to '{plot_save_path}'.")

    plt.show()


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


def plot_and_save_final_curves(
    train_curve,
    test_curve,
    best_architecture_parameters,
    save_path="final_training_plots.png",
):
    """
    Plots training/test loss and error curves, then saves the plots.
    """
    iterations = np.arange(len(train_curve))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(iterations, train_curve[:, 0], label="Train Loss")
    ax1.plot(iterations, test_curve[:, 0], label="Test Loss")
    ax1.set_title(
        f"Loss Curves ({best_architecture_parameters['architecture']}, LR={best_architecture_parameters['best_lr']})"
    )
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(iterations, train_curve[:, 1], label="Train Error")
    ax2.plot(iterations, test_curve[:, 1], label="Test Error")
    ax2.set_title(
        f"Error Curves ({best_architecture_parameters['architecture']}, LR={best_architecture_parameters['best_lr']})"
    )
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Error")
    ax2.legend()

    plt.tight_layout()

    # Save plots
    plt.savefig(save_path)
    print(f"\n Training plots saved to '{save_path}'.")

    plt.show()
