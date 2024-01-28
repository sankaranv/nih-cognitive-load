from matplotlib import pyplot as plt
from models.mcmc_imputer import *
from utils.data import *
import pandas as pd
import os
import numpy as np
import seaborn as sns
from utils.globals import role_names, param_names
from utils.stats import get_correlation_coefficients


def plot_predictions(
    predictions, test_dataset, unimputed_test_dataset, seq_len, pred_len, plots_dir
):
    for case_idx, case_data in test_dataset.items():
        for param_name in param_names:
            for actor_idx, actor_name in enumerate(role_names):
                param_idx = param_indices[param_name]
                imputed_hrv = case_data[param_idx][actor_idx, 0, :]
                pred_hrv = predictions[case_idx][param_name]
                if len(pred_hrv) == 0:
                    print(
                        f"No data for case {case_idx} {actor_name} {param_name}, skipping plots"
                    )
                    continue
                true_hrv = unimputed_test_dataset[case_idx][param_idx][actor_idx, 0, :]
                num_timesteps = imputed_hrv.shape[-1]
                plt.figure(figsize=(10, 6))

                # Plot imputed HRV with dotted line
                # Mask out elements where true data is not nan
                imputed_hrv = np.ma.masked_where(~np.isnan(true_hrv), imputed_hrv)
                plt.plot(
                    np.arange(0, num_timesteps),
                    imputed_hrv,
                    label=f"Imputed",
                    color="black",
                    alpha=0.3,
                    linestyle="--",
                )

                # Plot true HRV with solid line
                plt.plot(
                    np.arange(0, num_timesteps),
                    true_hrv,
                    label=f"True",
                    color="black",
                    alpha=0.3,
                )

                # Plot predicted HRV with solid colored line
                plt.plot(
                    np.arange(seq_len, num_timesteps - pred_len),
                    pred_hrv,
                    label=f"Predicted",
                    color=f"C{actor_idx}",
                )
                plt.legend()
                plt.title(f"Case {case_idx} {actor_name} {param_name}")
                # Save plot
                if not os.path.exists(f"{plots_dir}/Case{case_idx:02d}"):
                    os.makedirs(f"{plots_dir}/Case{case_idx:02d}")
                plt.savefig(
                    f"{plots_dir}/Case{case_idx:02d}/{actor_name}_{param_name}.png"
                )
                plt.close()


def plot_feature_importances(model, plots_dir, input_window, param_name):
    # Create x axis names
    x_names = []
    for i in range(input_window, 0, -1):
        for actor_name in role_names:
            x_names.append(f"{actor_name} t-{i}")
    for phase in phases:
        x_names.append(f"Phase {phase+1}")
    x_names.append("No Phase")
    x_names.append("Time")

    importances = model.coef_

    # Plot coefficients for each actor

    indices = np.argsort(importances[0])[::-1]
    fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharey=True)
    fig.suptitle(f"Regression coefficients for {param_name}")
    for actor_idx, actor_name in enumerate(role_names):
        axs[actor_idx].bar(
            range(len(indices)),
            importances[actor_idx],
            color=f"C{actor_idx}",
            align="center",
            label=actor_name,
        )
        axs[actor_idx].set_xticks(range(len(indices)))
        axs[actor_idx].set_xticklabels(x_names, rotation=90)
        axs[actor_idx].set_xlim([-1, len(indices)])
        axs[actor_idx].set_ylabel("Coefficient")
        axs[actor_idx].set_title(actor_name)

    plt.subplots_adjust(hspace=0.5)
    if not os.path.exists(f"{plots_dir}/feature_importances"):
        os.makedirs(f"{plots_dir}/feature_importances")
    plt.savefig(f"{plots_dir}/feature_importances/feature_importances_{param_name}.png")
    plt.close()


def generate_scatterplots(model, predictions, test_dataset, plots_dir):
    for case_idx, case_data in test_dataset.items():
        for param_name in model.param_names:
            for actor_idx, actor_name in enumerate(model.role_names):
                param_idx = model.param_indices[param_name]
                true_hrv = case_data[param_idx][actor_idx, 0, :]
                pred_hrv = predictions[case_idx][param_name][:, actor_idx]
                num_timesteps = true_hrv.shape[-1]

                # Make sure both true and predicted HRV are same length (hack)
                diff = np.abs(true_hrv.shape[-1] - pred_hrv.shape[-1])
                if diff > 0:
                    if true_hrv.shape[-1] > pred_hrv.shape[-1]:
                        true_hrv = true_hrv[diff:]
                    else:
                        pred_hrv = pred_hrv[diff:]

                # Remove NaNs
                pred_hrv = pred_hrv[~np.isnan(true_hrv)]
                true_hrv = true_hrv[~np.isnan(true_hrv)]

                if len(true_hrv) == 0:
                    print(
                        f"No data for case {case_idx} {actor_name} {param_name}, skipping plots"
                    )
                else:
                    # Compute correlation coefficient between predicted and true HRV
                    # Use pandas since it is NaN sensitive
                    corr = (
                        pd.DataFrame({"true": true_hrv, "pred": pred_hrv})
                        .corr()
                        .iloc[0, 1]
                    )
                    # Find mean square error
                    mse = np.mean((true_hrv - pred_hrv) ** 2)
                    # Find angle between line of best fit of the points and the diagonal
                    # Pruning points more than 3 standard deviations away from the mean
                    # For each prediction, compute squared distance from true value
                    # Then remove points more than 3 standard deviations away from the mean
                    distances = (true_hrv - pred_hrv) ** 2
                    dist_idx = np.where(distances < 3 * np.std(distances))
                    true_hrv_pruned = true_hrv[dist_idx]
                    pred_hrv_pruned = pred_hrv[dist_idx]
                    coefficients = np.polyfit(true_hrv_pruned, pred_hrv_pruned, 1)
                    slope = coefficients[0]
                    angle = np.degrees(np.arctan(slope) - np.pi / 4)
                    # Dotted line along the diagonal
                    plt.figure(figsize=(6, 6))
                    plt.plot([-6, 6], [-6, 6], color="black", linestyle="--")
                    # Faint dotted line along the line of best fit
                    plt.plot(
                        [-6, 6],
                        [-6 * slope, 6 * slope],
                        color="black",
                        linestyle="--",
                        alpha=0.3,
                    )
                    # Scatterplot of predicted vs true HRV on unimputed data
                    plt.scatter(
                        true_hrv, pred_hrv, label=f"Predicted", color=f"C{actor_idx}"
                    )
                    plt.title(
                        f"Case {case_idx} {actor_name} {param_name}: Corr={corr:.2f}, MSE={mse:.2f}, Angle={angle:.2f}"
                    )
                    plt.xlim([-6, 6])
                    plt.ylim([-6, 6])
                    plt.xlabel("True HRV")
                    plt.ylabel("Predicted HRV")
                    if not os.path.exists(f"{plots_dir}/Case{case_idx:02d}"):
                        os.makedirs(f"{plots_dir}/Case{case_idx:02d}")
                    plt.savefig(
                        f"{plots_dir}/Case{case_idx:02d}/{actor_name}_{param_name}_scatter.png"
                    )
                    plt.close()


def correlation_density_plots(model, predictions, test_dataset, plots_dir):
    correlations = {}
    for param in param_names:
        correlations[param] = {}
        for actor in role_names:
            correlations[param][actor] = []
    for case_idx, case_data in test_dataset.items():
        for param_name in model.param_names:
            for actor_idx, actor_name in enumerate(model.role_names):
                param_idx = model.param_indices[param_name]
                true_hrv = case_data[param_idx][actor_idx, 0, :]
                pred_hrv = predictions[case_idx][param_name][:, actor_idx]
                num_timesteps = true_hrv.shape[-1]

                # Make sure both true and predicted HRV are same length (hack)
                diff = np.abs(true_hrv.shape[-1] - pred_hrv.shape[-1])
                if diff > 0:
                    if true_hrv.shape[-1] > pred_hrv.shape[-1]:
                        true_hrv = true_hrv[diff:]
                    else:
                        pred_hrv = pred_hrv[diff:]

                # Remove NaNs
                pred_hrv = pred_hrv[~np.isnan(true_hrv)]
                true_hrv = true_hrv[~np.isnan(true_hrv)]

                if len(true_hrv) == 0:
                    print(
                        f"No data for case {case_idx} {actor_name} {param_name}, skipping plots"
                    )
                else:
                    # Compute correlation coefficient between predicted and true HRV
                    # Use pandas since it is NaN sensitive
                    corr = (
                        pd.DataFrame({"true": true_hrv, "pred": pred_hrv})
                        .corr()
                        .iloc[0, 1]
                    )
                    correlations[param_name][actor_name].append(corr)

    for param in param_names:
        sns.set_style("whitegrid")
        for actor in role_names:
            sns.kdeplot(correlations[param][actor], bw_method=0.5, label=actor)
        plt.title(f"Density of {param} correlation coefficients for {model.name} model")
        plt.xlabel(f"{param}")
        plt.ylabel("Density")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        plt.savefig(f"{plots_dir}/{param}_density.png")
        plt.close()

    return correlations
