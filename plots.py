import matplotlib.pyplot as plt
import numpy as np
import math
import random
import seaborn as sns

PLOT = False


def plot_learning_curves(
    df_results,
    arch_name,
    filename_prefix="results/",
    plot=PLOT,
    misclass_file="results/learning_curve_misclassification_error.png",
    proxy_file="results/learning_curve_proxy_error.png",
):

    # Plot misclassification error
    plt.figure(figsize=(10, 5))
    plt.plot(df_results["m"], df_results["train_error"], "o-", label="Training Error")
    plt.plot(df_results["m"], df_results["test_error"], "s-", label="Test Error")
    plt.title(f"{arch_name} Learning Curve (Misclassification Error)")
    plt.xlabel("Number of Training Examples (m)")
    plt.ylabel("Misclassification Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_{arch_name}_misclass_error.png", dpi=300)
    if plot:
        plt.show()

    # Plot proxy error (loss)
    plt.figure(figsize=(10, 5))
    plt.plot(df_results["m"], df_results["train_loss"], "o-", label="Training Loss")
    plt.plot(df_results["m"], df_results["test_loss"], "s-", label="Test Loss")
    plt.title(f"{arch_name} Learning Curve (Proxy Error/Loss)")
    plt.xlabel("Number of Training Examples (m)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_{arch_name}_proxy_error.png", dpi=300)
    if plot:
        plt.show()


def plot_and_save_final_curves(
    train_curve,
    test_curve,
    best_architecture_parameters,
    plot=PLOT,
    save_path="results/final_training_plots.png",
):
    """
    Plots training/test loss and error curves, then saves the plots.
    """
    iterations = np.arange(len(train_curve))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Architecture: {best_architecture_parameters['architecture']}")

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

    if PLOT:
        plt.show()


def plot_output_weights(
    final_run,
    filename="results/final_perceptron_output_weights.png",
    cmap="gray",
    ncols=5,
    reshape_size=(32, 32),
    plot=PLOT,
    arch_name="Perceptron",
):
    K = final_run["trained_model"].nunits[-1]

    # Calculate number of rows needed based on number of columns
    nrows = math.ceil(K / ncols)

    # Create the grid of subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3 * nrows))
    fig.suptitle(f"Architecture: {arch_name}", fontsize=16)
    # Flatten the axes array
    axes = axes.flatten()

    # Loop over each output unit, reshape its weights, and plot
    for out_unit in range(K):
        # skip bias (first column)
        w = final_run["trained_model"].layer[-1].W[out_unit, 1:]
        ax = axes[out_unit]
        ax.imshow(w.reshape(reshape_size), cmap=cmap)
        ax.set_title(f"Output unit {out_unit}")
        ax.axis("off")  # Hide axis ticks and labels

    # Hide any extra subplots that are not used
    for i in range(K, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    if PLOT:
        plt.show()


def plot_random_hidden_units(
    final_run,
    layer_idx=1,
    num_units=10,
    reshape_size=(32, 32),
    cmap="gray",
    plot=PLOT,
    filename="results/hidden_units.png",
):

    # Retrieve the weight matrix from the specified hidden layer
    W = final_run["trained_model"].layer[layer_idx].W  # shape: (n_out, n_in+1)

    # Number of units in this layer
    n_out = W.shape[0]

    # Randomly select 'num_units' distinct units
    random_units = random.sample(range(n_out), min(num_units, n_out))

    # Create a grid for plotting
    ncols = 5
    nrows = math.ceil(num_units / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()

    for i, unit_idx in enumerate(random_units):
        ax = axes[i]
        # Skip the bias (column 0), reshape the rest
        w = W[unit_idx, 1:].reshape(reshape_size)
        ax.imshow(w, cmap=cmap)
        ax.set_title(f"Hidden unit {unit_idx}")
        ax.axis("off")

    # Turn off extra axes if num_units < nrows*ncols
    for i in range(len(random_units), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300)

    # Show the plot
    if PLOT:
        plt.show()


def plot_random_hidden_units_from_architectures(
    architecture_final_runs,
    layer_idx=1,
    num_units=10,
    reshape_size=(32, 32),
    cmap="gray",
    plot=PLOT,
    filename_prefix="results/hidden_units",
):

    if isinstance(architecture_final_runs, list):
        architecture_final_runs = {
            f"Model {i+1}": run for i, run in enumerate(architecture_final_runs)
        }

    # List of architecture names
    arch_names = list(architecture_final_runs.keys())

    # Iterate over each architecture
    for arch_name in arch_names:
        final_run = architecture_final_runs[arch_name]

        # Extract weights
        W = final_run["trained_model"].layer[layer_idx].W
        n_out = W.shape[0]

        # Sample the hidden units
        sampled_units = random.sample(range(n_out), min(num_units, n_out))

        # Create a new figure for this architecture
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
        fig.suptitle(f"Architecture: {arch_name}", fontsize=16)

        # Flatten axes for easier iteration
        axes = axes.flatten()

        for i, unit_idx in enumerate(sampled_units):
            ax = axes[i]
            # Reshape the weights (skipping the bias if needed)
            w = W[unit_idx, 1:].reshape(reshape_size)
            ax.imshow(w, cmap=cmap)
            ax.axis("off")
            ax.set_title(f"Unit {unit_idx}")

        # Turn off any remaining unused subplots if fewer than 10 units
        for i in range(len(sampled_units), 10):
            axes[i].axis("off")

        # Save or show
        if filename_prefix:
            plt.savefig(f"{filename_prefix}_{arch_name}.png", dpi=300)
        if PLOT:
            plt.show()


def plot_cv_errors(df, plot=PLOT, plot_save_path="results/cv_results_plot.png"):
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
    plt.xlabel("Learning Rate")
    plt.ylabel("Avg CV Test Misclassification Error")
    plt.legend(title="Architecture")
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(plot_save_path, dpi=300)
    print(f"CV results plot saved to '{plot_save_path}'.")
    if PLOT:
        plt.show()


def plot_cv_errors_by_arch(
    df,
    plot=PLOT,
    col_learning_rate="Learning Rate",
    col_architecture="Architecture",
    col_cv_error="mean_val_err",
    col_train_proxy="mean_train_loss",
    col_test_proxy="mean_val_loss",
    col_final_nn_error="mean_train_err",
    plot_save_path="results/cv_results_plot_2.png",
):
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

    # Plot 1: Avg CV Test Misclassification Error
    sns.barplot(
        data=df, x=col_learning_rate, y=col_cv_error, hue=col_architecture, ax=axs[0, 0]
    )
    axs[0, 0].set_title("Avg CV Test Misclassification Error")
    axs[0, 0].set_xlabel(col_learning_rate)
    axs[0, 0].set_ylabel(col_cv_error)

    # Plot 2: Training Proxy Error
    sns.barplot(
        data=df,
        x=col_learning_rate,
        y=col_train_proxy,
        hue=col_architecture,
        ax=axs[0, 1],
    )
    axs[0, 1].set_title("Training Proxy Error")
    axs[0, 1].set_xlabel(col_learning_rate)
    axs[0, 1].set_ylabel(col_train_proxy)

    # Plot 3: Test Proxy Error
    sns.barplot(
        data=df,
        x=col_learning_rate,
        y=col_test_proxy,
        hue=col_architecture,
        ax=axs[1, 0],
    )
    axs[1, 0].set_title("Test Proxy Error")
    axs[1, 0].set_xlabel(col_learning_rate)
    axs[1, 0].set_ylabel(col_test_proxy)

    # Plot 4: Final NN Misclassification Error
    sns.barplot(
        data=df,
        x=col_learning_rate,
        y=col_final_nn_error,
        hue=col_architecture,
        ax=axs[1, 1],
    )
    axs[1, 1].set_title("Final NN Misclassification Error")
    axs[1, 1].set_xlabel(col_learning_rate)
    axs[1, 1].set_ylabel(col_final_nn_error)

    # Remove duplicate legends from subplots (only keep one overall legend)
    axs[0, 1].get_legend().remove()
    axs[1, 0].get_legend().remove()
    axs[1, 1].get_legend().remove()

    # Add a common legend
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title=col_architecture, loc="upper right")

    # Set an overall title for the figure
    fig.suptitle(
        "CV & Final Network Errors by Architecture and Learning Rate", fontsize=16
    )

    # Adjust layout to make room for the super title
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(plot_save_path, dpi=300)
    print(f"CV results plot saved to '{plot_save_path}'.")
    if PLOT:
        plt.show()
