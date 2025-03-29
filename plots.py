import matplotlib.pyplot as plt
import numpy as np
import math
import random
import seaborn as sns


def plot_learning_curves(
    df_results,
    misclass_file="learning_curve_misclassification_error.png",
    proxy_file="learning_curve_proxy_error.png",
):

    # Plot misclassification error
    plt.figure(figsize=(10, 5))
    plt.plot(df_results["m"], df_results["train_error"], "o-", label="Training Error")
    plt.plot(df_results["m"], df_results["test_error"], "s-", label="Test Error")
    plt.title("Learning Curve (Misclassification Error)")
    plt.xlabel("Number of Training Examples (m)")
    plt.ylabel("Misclassification Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(misclass_file, dpi=300)
    plt.show()

    # Plot proxy error (loss)
    plt.figure(figsize=(10, 5))
    plt.plot(df_results["m"], df_results["train_loss"], "o-", label="Training Loss")
    plt.plot(df_results["m"], df_results["test_loss"], "s-", label="Test Loss")
    plt.title("Learning Curve (Proxy Error/Loss)")
    plt.xlabel("Number of Training Examples (m)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(proxy_file, dpi=300)
    plt.show()


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


def plot_output_weights(
    final_run,
    filename="final_perceptron_output_weights.png",
    cmap="gray",
    ncols=5,
    reshape_size=(32, 32),
):
    K = final_run["trained_model"].nunits[-1]

    # Calculate number of rows needed based on number of columns
    nrows = math.ceil(K / ncols)

    # Create the grid of subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3 * ncols, 3 * nrows))

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
    plt.show()


def plot_random_hidden_units(
    final_run,
    layer_idx=1,
    num_units=10,
    reshape_size=(32, 32),
    cmap="gray",
    filename=None,
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

    # Optionally save the figure
    if filename:
        plt.savefig(filename, dpi=300)

    # Show the plot
    plt.show()


def plot_cv_errors(df, plot_save_path="cv_results_plot.png"):
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
    plt.show()
