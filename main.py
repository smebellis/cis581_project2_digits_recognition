import os
import pickle
import time
import warnings
import pandas as pd
from itertools import product

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer

import LearnNet
from plots import plot_cv_errors, plot_cv_errors_by_arch
from helper import (
    print_cv_summary,
    tabulate_cv_errors,
    run_experiment,
    get_best_params_and_final_runs,
)
from train import train_cv

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="overflow encountered in exp"
)


CV_RESULTS_FILE = "results/best_model_results.pkl"
FINAL_RESULTS_FILE = "results/final_model_results.pkl"

# Change to False for final run
# True means it runs on subset.
DEBUG = False

if __name__ == "__main__":

    # ---------------------------#
    #   Data Loading & Setup     #
    # ---------------------------#

    # Load datasets
    dataset_train = np.loadtxt("data/optdigits_train.dat")
    dataset_test = np.loadtxt("data/optdigits_test.dat")
    dataset_trial = np.loadtxt("data/optdigits_trial.dat")

    # Get number of samples and features
    m, n = (
        dataset_train.shape[0],
        dataset_train.shape[1] - 1,
    )

    # Extract features
    X = dataset_train[:, :-1].reshape(m, n)
    y = dataset_train[:, -1].reshape(m, 1)

    # One Hot Encoding
    out_enc = LabelBinarizer()
    y_ohe = out_enc.fit_transform(y)

    # Unique classes
    K = y_ohe.shape[1]

    # Get number of samples in the test dataset
    m_test = dataset_test.shape[0]

    # Extract features
    X_test = dataset_test[:, :-1].reshape(m_test, n)
    y_test = dataset_test[:, -1].reshape(m_test, 1)

    # One-hot encode test labels
    y_test_ohe = out_enc.transform(y_test)

    # Get number of samples in the trial dataset
    m_trial = dataset_trial.shape[0]
    n_trial = dataset_trial.shape[1] - 1

    # Extract features
    X_trial = dataset_trial[:, :-1].reshape(m_trial, n_trial)
    y_trial = dataset_trial[:, -1].reshape(m_trial, 1)

    # One-hot encode
    y_trial_ohe = out_enc.transform(y_trial)

    # -----------------------------#
    #   Cross-Validation Phase     #
    # -----------------------------#
    if os.path.exists(CV_RESULTS_FILE):
        print("\nSaved CV results found. Loading...")
        with open(CV_RESULTS_FILE, "rb") as f:
            saved_results = pickle.load(f)
        perceptron_cv = saved_results["cv_results_all"]["Perceptron"]
        multi_layer_cv = saved_results["cv_results_all"]["Multi-Layer NN"]
        two_layer_cv = saved_results["cv_results_all"]["Two-Layer NN"]

    else:
        print("\nNo saved CV results found. Starting CV training...")

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
        perceptron_cv = train_cv(
            X,
            y_ohe,
            kf,
            perceptron_configs,
            out_enc,
            arch_name="Perceptron",
            debug=DEBUG,
        )
        perceptron_duration = time.time() - start_time
        print_cv_summary("Perceptron", perceptron_cv, perceptron_duration)

        # ---------------------#
        #   Multi-Layer NN     #
        # ---------------------#

        # Cross-validation: Multi-Layer NN
        print("\n Starting Multi-Layer NN CV...")
        start_time = time.time()
        multi_layer_cv = train_cv(
            X,
            y_ohe,
            kf,
            multi_layer_configs,
            out_enc,
            arch_name="Multi-Layer NN",
            debug=DEBUG,
        )
        multi_duration = time.time() - start_time
        print_cv_summary("Multi-Layer NN", multi_layer_cv, multi_duration)

        # ---------------------#
        #   Two-Layer NN       #
        # ---------------------#

        # Cross-validation: Two-Layer NN
        print("\n Starting Two-Layer NN CV...")
        start_time = time.time()
        two_layer_cv = train_cv(
            X,
            y_ohe,
            kf,
            two_layer_configs,
            out_enc,
            arch_name="Two-Layer NN",
            debug=DEBUG,
        )
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
            "Multi-Layer NN": multi_layer_cv,
            "Two-Layer NN": two_layer_cv,
        }

        # df_cv = tabulate_cv_errors(
        #     cv_results_all,
        #     csv_save_path="results/cv_results_summary.csv",
        #     plot_save_path="results/cv_results_plot.png",
        # )

        dfs = []
        for arch, result in cv_results_all.items():
            # result should be the dictionary returned by train_cv for each architecture
            df_arch = result["summary_df"]
            dfs.append(df_arch)

        df_cv_all = pd.concat(dfs, ignore_index=True)

        plot_cv_errors_by_arch(df_cv_all, plot_save_path="results/cv_results_plot2.png")

        # Save CV results for future reuse
        saved_results = {
            "cv_results_all": cv_results_all,
            "out_enc": out_enc,
        }
        with open(CV_RESULTS_FILE, "wb") as f:
            pickle.dump(saved_results, f)
        print("\nCV results saved successfully!")

    # Bundle CV results into a list for each model
    architectures = [
        ("Perceptron", perceptron_cv),
        ("Multi-Layer NN", multi_layer_cv),
        ("Two-Layer NN", two_layer_cv),
    ]

    # ---------------------------------------------------#
    #   Final Training Run for Each Model's Best Params  #
    # ---------------------------------------------------#

    final_results = get_best_params_and_final_runs(
        architectures, X, y_ohe, X_test, y_test_ohe, out_enc, debug=DEBUG
    )

    # Save the final results for all models
    with open(FINAL_RESULTS_FILE, "wb") as f:
        pickle.dump(final_results, f)
    print("\nFinal results for all models saved successfully!")

    # ----------------------------#
    #   Evaluate Trial Dataset    #
    #   Learning Curve            #
    #   Weights Interpretation    #
    # ----------------------------#

    for model_name, result in final_results.items():
        print(f"\nEvaluating trial dataset for {model_name}:")
        df_results = run_experiment(
            final_run=result["final_model"],
            X_trial=X_trial,
            y_trial=y_trial,
            out_enc=out_enc,
            X=X,
            y_ohe=y_ohe,
            X_test=X_test,
            y_test_ohe=y_test_ohe,
            best_architecture_parameters=result["best_params"],
            model_name=model_name,
            debug=DEBUG,
        )
        if df_results is not None:
            print(df_results)
