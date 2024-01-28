from utils.data import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.config import config
from tqdm import tqdm


def eval_metric_density_plots(
    model_name,
    trace,
    test_dataset,
    metric_name,
    plots_dir,
):
    # Get metric values from trace
    first_key = list(trace.keys())[0]
    first_param = list(trace[first_key].keys())[0]
    if metric_name not in trace[first_key][first_param]:
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
    else:
        print(f"Metric {metric_name} not supported")
        return

    # Get correlation coefficients from trace
    metric = {param_name: [] for param_name in trace[first_key]}
    for case_idx, case_data in test_dataset.items():
        for param_name in trace[case_idx]:
            metric[param_name].append(trace[case_idx][param_name][metric_name])
    for param_name in metric:
        metric[param_name] = np.array(metric[param_name])

    sns.set_style("whitegrid")
    for param_name in metric:
        for actor in config.role_names:
            actor_idx = config.role_indices[actor]
            sns.kdeplot(metric[param_name][actor_idx], bw_method=0.5, label=actor)
        plt.title(f"Density of {param_name} {metric_title} for {model_name}")
        plt.xlabel(metric_title)
        plt.ylabel("Density")
        if not os.path.exists(f"{plots_dir}/density_plots/{model_name}"):
            os.makedirs(f"{plots_dir}/density_plots/{model_name}")
        plt.savefig(
            f"{plots_dir}/density_plots/{model_name}/{metric_name}_{param_name}_density.png"
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
        for param_name in trace[case_idx]:
            case_predictions = trace[case_idx][param_name]["predictions"]
            for actor_idx, actor_name in enumerate(config.role_names):
                param_idx = config.param_indices[param_name]
                imputed_hrv = case_data[param_idx][actor_idx, 0, :]
                true_hrv = unimputed_test_dataset[case_idx][param_idx][actor_idx, 0, :]
                num_timesteps = imputed_hrv.shape[-1]
                plt.figure(figsize=(10, 6))

                # Take first prediction in window and append last prediction to end
                pred_hrv = case_predictions[:, actor_idx]
                # pred_hrv = np.concatenate(
                #     (pred_hrv, case_predictions[-1, actor_idx].reshape(1))
                # )

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
                    np.arange(seq_len, num_timesteps),
                    pred_hrv,
                    label=f"Predicted",
                    color=f"C{actor_idx}",
                )
                plt.legend()
                plt.title(f"{model_name}: Case {case_idx} {actor_name} {param_name}")

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


def generate_scatterplots(model, trace, test_dataset, plots_dir):
    sns.set_style("whitegrid")
    for case_idx, case_data in test_dataset.items():
        for param_name in trace[case_idx]:
            case_predictions = trace[case_idx][param_name]["predictions"]
            for actor_idx, actor_name in enumerate(config.role_names):
                param_idx = config.param_indices[param_name]
                true_hrv = case_data[param_idx][actor_idx, 0, :]
                pred_hrv = case_predictions[:, actor_idx]
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
                    # Set scatterplot range so x and y axes are the same
                    plot_range = np.max(np.abs(true_hrv))
                    plt.plot(
                        [-plot_range, plot_range],
                        [-plot_range, plot_range],
                        color="black",
                        linestyle="--",
                    )
                    # Faint dotted line along the line of best fit
                    plt.plot(
                        [-plot_range, plot_range],
                        [-plot_range * slope, plot_range * slope],
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


def plot_loss_curves(model_name, train_loss, val_loss, test_loss, plots_dir, param):
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.title(f"{param} loss for {model_name}: {test_loss}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if not os.path.exists(f"{plots_dir}/loss_curves/{model_name}"):
        os.makedirs(f"{plots_dir}/loss_curves/{model_name}")
    plt.savefig(f"{plots_dir}/loss_curves/{model_name}/{param}_loss.png")
    plt.close()


def plot_feature_importances(model, seq_len, plots_dir):
    sns.set_style("whitegrid")
    if model.__class__.__name__ != "JointLinearModel":
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
        fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharey=True)
        fig.suptitle(f"Regression coefficients for {param_name}")
        for actor_idx, actor_name in enumerate(config.role_names):
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
        plt.savefig(
            f"{plots_dir}/feature_importances/{param_name}_feature_importance.png"
        )
        plt.close()
