import numpy as np

from utils.data import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.config import config
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.ticker as ticker


def plot_roc_curve(model_name, trace, test_dataset, seq_len, plots_dir):
    # Plot ROC curves per case
    sns.set_style("whitegrid")
    for case_idx, case_data in tqdm(test_dataset.items()):
        for param_name in trace:
            case_predictions = trace[param_name]["predictions"][case_idx]
            for actor_idx, actor_name in enumerate(config.role_names):
                actor_color = f"C{config.role_colors[actor_name]}"
                param_idx = config.param_indices[param_name]
                imputed_hrv = case_data[param_idx][actor_idx, 0, :]
                num_timesteps = imputed_hrv.shape[-1]
                plt.figure(figsize=(6, 6))

                # Get predicted HRV
                if len(case_predictions.shape) > 1:
                    pred_hrv = case_predictions[:, actor_idx]
                else:
                    pred_hrv = case_predictions

                # Cut off first seq_len elements from true data
                true_hrv = imputed_hrv[seq_len:]

                # Plot ROC curve
                fpr, tpr, thresholds = roc_curve(
                    true_hrv, pred_hrv, drop_intermediate=False
                )
                plt.plot(fpr, tpr, label=f"{actor_name}", color=actor_color)
                plt.plot([0, 1], [0, 1], color="black", linestyle="--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")

                plt.legend()
                accuracy = trace[param_name]["accuracy"][case_idx]
                plt.title(
                    f"{model_name}: Case {case_idx} {actor_name} {param_name} Acc={accuracy:.2f}"
                )

                # Save plot
                if not os.path.exists(
                    f"{plots_dir}/roc/{model_name}/Case{case_idx:02d}"
                ):
                    os.makedirs(f"{plots_dir}/roc/{model_name}/Case{case_idx:02d}")
                plt.savefig(
                    f"{plots_dir}/roc/{model_name}/Case{case_idx:02d}/{actor_name}_{param_name}.png"
                )
                plt.close()


def eval_metric_density_plots(
    model_name,
    trace,
    test_dataset,
    metric_name,
    plots_dir,
):
    # Get metric values from trace
    first_param = list(trace.keys())[0]
    if metric_name not in trace[first_param]:
        print(f"Metric {metric_name} not found in trace, only contains {trace.keys()}")
        return
    if metric_name == "corr_coef":
        metric_title = "Correlation Coefficient"
    elif metric_name == "mean_squared_error":
        metric_title = "Mean Squared Error"
    elif metric_name == "rms_error":
        metric_title = "RMS Error"
    elif metric_name == "mean_absolute_error":
        metric_title = "Mean Absolute Error"
    elif metric_name == "r_squared":
        metric_title = "R Squared"
    elif metric_name == "accuracy":
        metric_title = "Accuracy"
    elif metric_name == "precision":
        metric_title = "Precision"
    elif metric_name == "recall":
        metric_title = "Recall"
    elif metric_name == "f1_score":
        metric_title = "F1 Score"
    else:
        print(f"Metric {metric_name} not supported")
        return

    # Get metrics from trace
    metric = {param_name: [] for param_name in trace}
    for case_idx, case_data in test_dataset.items():
        for param_name in trace:
            metric[param_name].append(trace[param_name][metric_name][case_idx])
    for param_name in metric:
        metric[param_name] = np.array(metric[param_name])

    sns.set_style("whitegrid")
    for param_name in metric:
        if metric_name == "corr_coef":
            for actor in config.role_names:
                actor_idx = config.role_indices[actor]
                actor_color = f"C{config.role_colors[actor]}"
                sns.kdeplot(
                    metric[param_name][:, actor_idx],
                    bw_method=0.5,
                    label=actor,
                    color=actor_color,
                )
        else:
            sns.kdeplot(
                metric[param_name],
                bw_method=0.5,
                label=param_name,
            )
        plt.title(f"Density of {param_name} {metric_title} for {model_name}")
        plt.xlabel(metric_title)
        plt.ylabel("Density")
        plt.legend()
        if not os.path.exists(f"{plots_dir}/density_plots/{model_name}/{metric_name}"):
            os.makedirs(f"{plots_dir}/density_plots/{model_name}/{metric_name}")
        plt.savefig(
            f"{plots_dir}/density_plots/{model_name}/{metric_name}/{metric_name}_{param_name}_density.png"
        )
        plt.close()


def plot_predictions(
    model_name,
    trace,
    test_dataset,
    unimputed_test_dataset,
    seq_len,
    pred_len,
    plots_dir,
):
    # Plot predictions for each case
    sns.set_style("whitegrid")
    for case_idx, case_data in tqdm(test_dataset.items()):
        for param_name in trace:
            case_predictions = trace[param_name]["predictions"][case_idx]
            for actor_idx, actor_name in enumerate(config.role_names):
                actor_color = f"C{config.role_colors[actor_name]}"
                param_idx = config.param_indices[param_name]
                imputed_hrv = case_data[param_idx][actor_idx, 0, :]
                true_hrv = unimputed_test_dataset[case_idx][param_idx][actor_idx, 0, :]
                num_timesteps = imputed_hrv.shape[-1]
                plt.figure(figsize=(10, 6))

                # Get predicted HRV
                if len(case_predictions.shape) > 1:
                    pred_hrv = case_predictions[:, actor_idx]
                else:
                    pred_hrv = case_predictions

                # Pad first seq_len elements with NaNs
                pred_hrv = np.concatenate([np.full(seq_len, np.nan), pred_hrv], axis=0)

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
                    np.arange(0, num_timesteps),
                    pred_hrv,
                    label=f"Predicted",
                    color=actor_color,
                )
                plt.legend()
                mean_abs_error = trace[param_name]["mean_absolute_error"][case_idx]
                plt.title(
                    f"{model_name}: Case {case_idx} {actor_name} {param_name} MAE={mean_abs_error:.2f}"
                )

                # Save plot
                if not os.path.exists(
                    f"{plots_dir}/predictions/{model_name}/Case{case_idx:02d}"
                ):
                    os.makedirs(
                        f"{plots_dir}/predictions/{model_name}/Case{case_idx:02d}"
                    )
                plt.savefig(
                    f"{plots_dir}/predictions/{model_name}/Case{case_idx:02d}/{actor_name}_{param_name}.png"
                )
                plt.close()


def generate_scatterplots(model, trace, test_dataset, seq_len, pred_len, plots_dir):
    sns.set_style("whitegrid")
    for case_idx, case_data in test_dataset.items():
        for param_name in trace:
            case_predictions = trace[param_name]["predictions"][case_idx]
            for actor_idx, actor_name in enumerate(config.role_names):
                actor_color = f"C{config.role_colors[actor_name]}"
                param_idx = config.param_indices[param_name]
                true_hrv = case_data[param_idx][actor_idx, 0, :]
                num_timesteps = true_hrv.shape[-1]

                # Get predicted HRV
                if len(case_predictions.shape) > 1:
                    pred_hrv = case_predictions[:, actor_idx]
                else:
                    pred_hrv = case_predictions

                # Trim off first seq_len elements
                if len(pred_hrv) != num_timesteps:
                    true_hrv = true_hrv[seq_len:]
                    num_timesteps = num_timesteps - seq_len

                # # Make sure both true and predicted HRV are same length (hack)
                # diff = np.abs(true_hrv.shape[-1] - pred_hrv.shape[-1])
                # if diff > 0:
                #     if true_hrv.shape[-1] > pred_hrv.shape[-1]:
                #         true_hrv = true_hrv[diff:]
                #     else:
                #         pred_hrv = pred_hrv[diff:]

                # Remove NaNs
                pred_hrv = pred_hrv[~np.isnan(true_hrv)]
                true_hrv = true_hrv[~np.isnan(true_hrv)]

                if len(true_hrv) == 0:
                    print(
                        f"No data for case {case_idx} {actor_name} {param_name}, skipping plots"
                    )
                else:
                    # Setup plot
                    plt.figure(figsize=(6, 6))
                    # Set scatterplot range so x and y axes are the same
                    plot_range = np.max(np.abs(true_hrv))

                    # Dotted line along the diagonal
                    plt.plot(
                        [-plot_range, plot_range],
                        [-plot_range, plot_range],
                        color="black",
                        linestyle="--",
                    )
                    # Compute correlation coefficient between predicted and true HRV
                    # Use pandas since it is NaN sensitive
                    corr = (
                        pd.DataFrame({"true": true_hrv, "pred": pred_hrv})
                        .corr()
                        .iloc[0, 1]
                    )

                    # Find angle between line of best fit of the points and the diagonal
                    # Pruning points more than 3 standard deviations away from the mean
                    # For each prediction, compute squared distance from true value
                    # Then remove points more than 3 standard deviations away from the mean
                    distances = (true_hrv - pred_hrv) ** 2
                    # Find mean square error
                    mse = np.mean(distances)
                    dist_idx = np.where(distances < (3 * np.std(distances)))
                    if len(dist_idx) > 1:
                        true_hrv_pruned = true_hrv[dist_idx]
                        pred_hrv_pruned = pred_hrv[dist_idx]
                        coefficients = np.polyfit(true_hrv_pruned, pred_hrv_pruned, 1)
                        slope = coefficients[0]
                        angle = np.degrees(np.arctan(slope) - np.pi / 4)

                        # Faint dotted line along the line of best fit
                        plt.plot(
                            [-plot_range, plot_range],
                            [-plot_range * slope, plot_range * slope],
                            color="black",
                            linestyle="--",
                            alpha=0.3,
                        )
                        plt.title(
                            f"Case {case_idx} {actor_name} {param_name}: Corr={corr:.2f}, MSE={mse:.2f}, Angle={angle:.2f}"
                        )
                    else:
                        plt.title(
                            f"{model.model_name} {actor_name} {param_name}: Corr={corr:.2f}, MSE={mse:.2f}"
                        )
                    # Scatterplot of predicted vs true HRV on unimputed data
                    plt.scatter(
                        true_hrv, pred_hrv, label=f"Predicted", color=actor_color
                    )
                    plt.xlim([-plot_range, plot_range])
                    plt.ylim([-plot_range, plot_range])
                    plt.xlabel("True HRV")
                    plt.ylabel("Predicted HRV")
                    model_name = model.model_name
                    if not os.path.exists(
                        f"{plots_dir}/scatterplots/{model_name}/Case{case_idx:02d}"
                    ):
                        os.makedirs(
                            f"{plots_dir}/scatterplots/{model_name}/Case{case_idx:02d}"
                        )
                    plt.savefig(
                        f"{plots_dir}/scatterplots/{model_name}/Case{case_idx:02d}/{actor_name}_{param_name}_scatter.png"
                    )
                    plt.close()


def plot_loss_curves(model_name, trace, plots_dir):
    sns.set_style("whitegrid")
    for param_name, param_trace in trace.items():
        if isinstance(param_trace["train_loss"], dict):
            for k in param_trace["train_loss"].keys():
                # Cross validation, we need one plot per CV split
                train_loss = param_trace["train_loss"][k]
                val_loss = param_trace["val_loss"][k]
                test_loss = param_trace["test_loss"][k]
                plt.figure(figsize=(10, 6))
                plt.plot(train_loss, label="train loss")
                plt.plot(val_loss, label="val loss")
                plt.title(f"{param_name} loss for {model_name}: {test_loss}")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                if not os.path.exists(f"{plots_dir}/loss_curves/{model_name}"):
                    os.makedirs(f"{plots_dir}/loss_curves/{model_name}")
                plt.savefig(
                    f"{plots_dir}/loss_curves/{model_name}/{param_name}_loss_cv{k}.png"
                )
                plt.close()
        elif isinstance(param_trace["train_loss"], list):
            # Single run, we only need one plot
            train_loss = param_trace["train_loss"]
            val_loss = param_trace["val_loss"]
            test_loss = param_trace["test_loss"]
            plt.figure(figsize=(10, 6))
            plt.plot(train_loss, label="train loss")
            plt.plot(val_loss, label="val loss")
            plt.title(f"{param_name} loss for {model_name}: {test_loss}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            if not os.path.exists(f"{plots_dir}/loss_curves/{model_name}"):
                os.makedirs(f"{plots_dir}/loss_curves/{model_name}")
            plt.savefig(f"{plots_dir}/loss_curves/{model_name}/{param_name}_loss.png")
            plt.close()


def plot_feature_importances(model, seq_len, plots_dir):
    sns.set_style("whitegrid")
    if model.__class__.__name__ != "JointLinearRegressor":
        print("Model is not a JointLinearModel, skipping feature importances")
        return
    for param_name in config.param_names:
        # Create x axis names
        x_names = []
        for i in range(seq_len, 0, -1):
            for actor_name in config.role_names:
                x_names.append(f"{actor_name} t-{i}")
        for p in config.phases:
            x_names.append(f"Phase {p}")
        x_names.append("No Phase")
        x_names.append("Time")

        importances = model.models[param_name].coef_

        # Plot coefficients for each actor

        indices = np.argsort(importances[0])[::-1]
        fig, axs = plt.subplots(
            len(config.role_names), 1, figsize=(10, 15), sharey=True
        )
        fig.suptitle(f"Regression coefficients for {param_name}")
        for actor_idx, actor_name in enumerate(config.role_names):
            actor_color = f"C{config.role_colors[actor_name]}"
            if len(config.role_names) == 1:
                axs.bar(
                    range(len(indices)),
                    importances[actor_idx],
                    color=actor_color,
                    align="center",
                    label=actor_name,
                )
                axs.set_xticks(range(len(indices)))
                axs.set_xticklabels(x_names, rotation=90)
                axs.set_xlim([-1, len(indices)])
                axs.set_ylabel("Coefficient")
                axs.set_title(actor_name)
            else:
                axs[actor_idx].bar(
                    range(len(indices)),
                    importances[actor_idx],
                    color=actor_color,
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
        plt.savefig(
            f"{plots_dir}/feature_importances/{param_name}_feature_importance.png"
        )
        plt.close()


def binning_line_plot(
    normalized_original_dataset,
    discretized_dataset,
    num_bins,
    plots_dir,
    cmap="viridis",
):
    sns.set_style("whitegrid")
    for case_idx, case_data in tqdm(discretized_dataset.items()):
        for param_idx, param_data in enumerate(case_data):
            param_name = config.param_names[param_idx]
            discretized_samples = param_data[:, 0, :]
            original_samples = (
                normalized_original_dataset[case_idx][param_idx, :, 0, :] * num_bins
            )
            # Rescale normalized samples by number of bins
            for actor_idx, actor_name in enumerate(config.role_names):
                actor_color = f"C{config.role_colors[actor_name]}"
                plt.figure(figsize=(10, 6))

                # Plot discretized HRV with solid line
                plt.scatter(
                    np.arange(0, discretized_samples.shape[-1]),
                    original_samples[actor_idx],
                    c=discretized_samples[actor_idx],
                    cmap=cmap,
                )
                # Plot a line connecting the points
                plt.plot(
                    np.arange(0, discretized_samples.shape[-1]),
                    original_samples[actor_idx],
                    color="gray",
                    linewidth=0.5,
                )
                # Add colorbar
                cbar = plt.colorbar()
                cbar.set_label("Values")

                # Set formatter for colorbar ticks to display only integers
                cbar.ax.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, _: f"{int(x)}")
                )

                plt.title(
                    f"Case {case_idx} {actor_name} {param_name} with discretization"
                )
                plt.xlabel("Time")
                plt.ylabel("Normalized HRV parameter value")

                # Save plot
                if not os.path.exists(f"{plots_dir}/discretization/Case{case_idx:02d}"):
                    os.makedirs(f"{plots_dir}/discretization/Case{case_idx:02d}")
                plt.savefig(
                    f"{plots_dir}/discretization/Case{case_idx:02d}/{actor_name}_{param_name}.png"
                )
                plt.close()


def plot_anomalies(dataset, all_probs, model_name, seq_len, plots_dir):
    sns.set_style("whitegrid")

    # Get the maximum and minimum probabilities in the all_probs dict
    prob_ranges = {param: [0, 0] for param in config.param_names}
    for case_idx, case_data in dataset.items():
        for param_idx, param_data in enumerate(case_data):
            param_name = config.param_names[param_idx]
            current_max = np.max(all_probs[case_idx][param_name])
            current_min = np.min(all_probs[case_idx][param_name])
            prob_ranges[param_name][0] = np.min(
                [prob_ranges[param_name][0], current_min]
            )
            prob_ranges[param_name][1] = np.max(
                [prob_ranges[param_name][1], current_max]
            )

    for case_idx, case_data in tqdm(dataset.items()):
        for param_idx, param_data in enumerate(case_data):
            param_name = config.param_names[param_idx]

            # Plot anomalies
            for actor_idx, actor_name in enumerate(config.role_names):
                plt.figure(figsize=(10, 6))
                samples = case_data[param_idx, actor_idx, 0, :]
                probs = all_probs[case_idx][param_name][actor_idx]
                # Pad first seq_len elements with NaNs
                probs = np.concatenate([np.full(seq_len, np.nan), probs], axis=0)

                # Rescale probabilities used for colors such that min and max are taken from prob_ranges
                probs_rescaled = (probs - prob_ranges[param_name][0]) / (
                    prob_ranges[param_name][1] - prob_ranges[param_name][0]
                )

                # Plot discretized HRV with solid line
                plt.scatter(
                    np.arange(0, samples.shape[-1]),
                    samples,
                    c=probs_rescaled,
                    cmap="coolwarm_r",
                )
                # Plot a line connecting the points
                plt.plot(
                    np.arange(0, samples.shape[-1]),
                    samples,
                    color="gray",
                    linewidth=0.5,
                )
                # Add colorbar with min and max using prob_ranges
                cbar = plt.colorbar()
                cbar.set_label("Probability")

                plt.title(
                    f"Case {case_idx} {actor_name} {param_name} with predicted probabilities"
                )
                plt.xlabel("Time")
                plt.ylabel("Normalized HRV parameter value")

                # Save plot
                if not os.path.exists(
                    f"{plots_dir}/anomaly/eval/{model_name}/Case{case_idx:02d}"
                ):
                    os.makedirs(
                        f"{plots_dir}/anomaly/eval/{model_name}/Case{case_idx:02d}"
                    )
                plt.savefig(
                    f"{plots_dir}/anomaly/eval/{model_name}/Case{case_idx:02d}/{actor_name}_{param_name}.png"
                )
                plt.close()


def plot_anomaly_roc(test_probs, predictions, model_name, plots_dir):
    # Plot ROC curves using the test probabilities and the predictions
    for case_idx in test_probs.keys():
        for param_idx, param_name in enumerate(config.param_names):
            for actor_idx, actor_name in enumerate(config.role_names):
                y_score = test_probs[case_idx][param_name][actor_idx]
                y_true = predictions[case_idx][param_idx]
                fpr, tpr, thresholds = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                plt.figure()
                plt.title(
                    f"Case {case_idx} {actor_name} {param_name} ROC: AUC={roc_auc:.2f}"
                )
                plt.plot(fpr, tpr, color="darkorange", lw=2)
                plt.plot([0, 1], [0, 1], color="black", lw=2, alpha=0.2, linestyle="--")
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                if not os.path.exists(
                    f"{plots_dir}/anomaly/roc/{model_name}/Case{case_idx:02d}"
                ):
                    os.makedirs(
                        f"{plots_dir}/anomaly/roc/{model_name}/Case{case_idx:02d}"
                    )
                plt.savefig(
                    f"{plots_dir}/anomaly/roc/{model_name}/Case{case_idx:02d}/{case_idx}_{actor_name}_{param_name}.png"
                )
                plt.close()
