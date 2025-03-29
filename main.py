import os
import pickle
import time
import warnings
import math
import random
from collections import Counter
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer

import LearnNet
from helper import (
    plot_and_save_final_curves,
    print_cv_summary,
    tabulate_and_plot_cv_errors,
    tabulate_final_results,
    plot_output_weights,
)
from train import evaluate_trial_dataset, train_cv, train_network

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="overflow encountered in exp"
)

RESULTS_FILE = "best_model_results.pkl"

# Change to False for final run
# True means it runs on 100 samples.
DEBUG = True

if __name__ == "__main__":

    # Load datasets
    dataset_train = np.loadtxt("optdigits_train.dat")
    dataset_test = np.loadtxt("optdigits_test.dat")
    dataset_trial = np.loadtxt("optdigits_trial.dat")

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
    n_trial = dataset_trial.shape[1] - 1
    X_trial = dataset_trial[:, :-1].reshape(
        m_trial, n_trial
    )  # Extract features from trial dataset
    y_trial = dataset_trial[:, -1].reshape(
        m_trial, 1
    )  # Extract labels from trial dataset
    y_trial_ohe = out_enc.transform(y_trial)  # One-hot encode trial labels

    # Load results from Pickle File
    if os.path.exists(RESULTS_FILE):
        print("\n Saved results found. Loading...")
        with open(RESULTS_FILE, "rb") as f:
            saved_results = pickle.load(f)

        best_architecture_parameters = saved_results["best_params"]
        final_run = saved_results["final_run"]
        out_enc = saved_results["out_enc"]

    else:
        print("\n No saved results found. Starting training...")

        kf = KFold(n_splits=3, random_state=42, shuffle=True)

        # Define architecture configs
        perceptron_configs = [
            {"layer_sizes": [n, K], "lr": 4**i} for i in [0, 1, 2, 3, 4]
        ]
        multi_layer_configs = [
            {"layer_sizes": LearnNet.make_nunits(n, K, layers, units), "lr": 4**lr_exp}
            for units, layers, lr_exp in product(
                [16, 64, 256], [1, 2, 3, 4], [-2, -1, 0, 1, 2]
            )
        ]
        two_layer_configs = [
            {"layer_sizes": [n, first_units, second_units, K], "lr": 4**lr_exp}
            for first_units, second_units, lr_exp in product(
                [256, 64], [64, 16], [-3, -2, -1, 0, 1]
            )
            if second_units < first_units
        ]

        overall_start_time = time.time()

        # ---------------------#
        #   Perceptron Model   #
        # ---------------------#

        # Cross-validation: Perceptron
        print("\n Starting Perceptron CV...")
        start_time = time.time()
        perceptron_cv = train_cv(X, y_ohe, kf, perceptron_configs, out_enc, debug=DEBUG)
        perceptron_duration = time.time() - start_time
        print_cv_summary("Perceptron", perceptron_cv, perceptron_duration)

        # ---------------------#
        #   Multi-Layer NN     #
        # ---------------------#

        # Cross-validation: Multi-Layer NN
        print("\n Starting Multi-Layer NN CV...")
        start_time = time.time()
        multi_cv = train_cv(X, y_ohe, kf, multi_layer_configs, out_enc, debug=DEBUG)
        multi_duration = time.time() - start_time
        print_cv_summary("Multi-Layer NN", multi_cv, multi_duration)

        # ---------------------#
        #   Two-Layer NN       #
        # ---------------------#

        # Cross-validation: Two-Layer NN
        print("\n Starting Two-Layer NN CV...")
        start_time = time.time()
        two_layer_cv = train_cv(X, y_ohe, kf, two_layer_configs, out_enc, debug=DEBUG)
        two_layer_duration = time.time() - start_time
        print_cv_summary("Two-Layer NN", two_layer_cv, two_layer_duration)

        overall_duration = time.time() - overall_start_time
        print(
            f"\n Total CV runtime for all architectures: {overall_duration/60:.2f} minutes."
        )

        # ----------------------#
        #   CV Error Curves     #
        # ----------------------#

        cv_results_all = {
            "Perceptron": perceptron_cv,
            "Multi-Layer NN": multi_cv,
            "Two-Layer NN": two_layer_cv,
        }

        tabulate_and_plot_cv_errors(
            cv_results_all,
            csv_save_path="cv_results_summary.csv",
            plot_save_path="cv_results_plot.png",
        )

        # -------------------------------------------------------#
        #   Model Selection based on Misclassification Error     #
        # -------------------------------------------------------#

        # Final model selection based on validation error
        architectures = [
            ("perceptron", perceptron_cv),
            ("multi_layer", multi_cv),
            ("two_layer", two_layer_cv),
        ]

        best_arch, best_cv = min(
            architectures, key=lambda arch: arch[1]["lowest_val_error"]
        )

        best_lr = best_cv["best_lr"]
        # best_config = Counter(best_cv["configs_chosen"]).most_common(1)[0][0]
        best_config = Counter(
            tuple(config) for config in best_cv["configs_chosen"]
        ).most_common(1)[0][0]

        print("\n === Best Architecture Selected ===")
        print(f"Architecture: {best_arch}")
        print(f"Best Learning Rate: {best_lr}")
        print(f"Best Layer Sizes: {best_config}")
        print(f"Avg Validation Error: {best_cv['lowest_val_error']:.4f}")

        # Return all best parameters clearly
        best_architecture_parameters = {
            "architecture": best_arch,
            "best_lr": best_lr,
            "layer_sizes": best_config,
            "validation_error": best_cv["lowest_val_error"],
        }

        # Final Training
        print("\n Starting Final Training...")
        final_run = train_network(
            X,
            y_ohe,
            X_test,
            y_test_ohe,
            best_config,
            best_lr,
            max_iters=1000,
            out_enc=out_enc,
            debug=DEBUG,
        )

        # Save results to disk
        saved_results = {
            "best_params": best_architecture_parameters,
            "final_run": final_run,
            "cv_results_all": cv_results_all,
            "out_enc": out_enc,
        }

        with open(RESULTS_FILE, "wb") as f:
            pickle.dump(saved_results, f)

        print("\n Results saved successfully!")

        # Plotting final run
        train_curve, test_curve = (
            final_run["train_err_curve"],
            final_run["test_err_curve"],
        )
        iterations = np.arange(len(train_curve))

        # ----------------------------#
        #   Tabulate Error / loss     #
        # ----------------------------#

        tabulate_final_results(train_curve, test_curve, save_path="final_results.csv")

        plot_and_save_final_curves(
            train_curve,
            test_curve,
            best_architecture_parameters,
            save_path="final_training_plots.png",
        )

        total_runtime = time.time() - overall_start_time
        print(
            f"\n Total Runtime (CV + Final Training): {total_runtime/60:.2f} minutes."
        )

        # Display best parameters
        print("\n === Best Architecture Parameters ===")
        for key, value in best_architecture_parameters.items():
            print(f"{key.capitalize()}: {value}")

    # ----------------------------#
    #   Evaluate Trial Dataset    #
    # ----------------------------#

    if DEBUG:
        pass
    else:

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

        train_sizes = [10, 40, 100, 200, 400, 800, 1600]
        results = []

        for m_current in train_sizes:
            X_subset = X[:m_current]
            y_subset_ohe = y_ohe[:m_current]

            # Train model explicitly on current subset
            final_run_subset = train_network(
                X_subset,
                y_subset_ohe,
                X_test,
                y_test_ohe,
                best_architecture_parameters["layer_sizes"],
                best_architecture_parameters["best_lr"],
                max_iters=1000,
                out_enc=out_enc,
                debug=DEBUG,
            )

            results.append(
                {
                    "m": m_current,
                    "train_error": final_run_subset["train_err_curve"][-1, 1],
                    "test_error": final_run_subset["test_err_curve"][-1, 1],
                    "train_loss": final_run_subset["train_err_curve"][-1, 0],
                    "test_loss": final_run_subset["test_err_curve"][-1, 0],
                }
            )

        df_results = pd.DataFrame(results)

        plt.figure(figsize=(10, 5))
        plt.plot(
            df_results["m"], df_results["train_error"], "o-", label="Training Error"
        )
        plt.plot(df_results["m"], df_results["test_error"], "s-", label="Test Error")
        plt.title("Learning Curve (Misclassification Error)")
        plt.xlabel("Number of Training Examples (m)")
        plt.ylabel("Misclassification Error")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("learning_curve_misclassification_error.png")
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(df_results["m"], df_results["train_loss"], "o-", label="Training Loss")
        plt.plot(df_results["m"], df_results["test_loss"], "s-", label="Test Loss")
        plt.title("Learning Curve (Proxy Error/Loss)")
        plt.xlabel("Number of Training Examples (m)")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("learning_curve_proxy_error.png")
        plt.show()

        # --------------------------------------#
        #   Weight Parameter Initialization     #
        # --------------------------------------#

        plot_output_weights(final_run)

    hidden_units = random.sample(range(H), 10)
    for h in hidden_units:
        w = (
            final_run["trained_model"].layer[0].W[h, 1:]
        )  # or layer[1] if that’s the first hidden
        plt.imshow(w.reshape(32, 32), cmap="gray")
        plt.title(f"Hidden Unit {h}")
        plt.show()
