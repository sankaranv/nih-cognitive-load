from utils.data import *
from models.linear import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.globals import *
from utils.stats import get_correlation_coefficients


def plot_predictions(
    model,
    case_predictions,
    test_dataset,
    unimputed_test_dataset,
    seq_len,
    pred_len,
    plots_dir,
    param,
):
    # Plot predictions for each case
    for case_idx, case_data in test_dataset.items():
        for actor_idx, actor_name in enumerate(role_names):
            param_idx = param_indices[param]
            imputed_hrv = case_data[param_idx][actor_idx, 0, :]
            true_hrv = unimputed_test_dataset[case_idx][param_idx][actor_idx, 0, :]
            num_timesteps = imputed_hrv.shape[-1]
            plt.figure(figsize=(10, 6))

            # Take first prediction in window and append last prediction to end
            pred_hrv = case_predictions[case_idx][:, actor_idx, 0]
            pred_hrv = np.concatenate(
                (pred_hrv, case_predictions[case_idx][-1, actor_idx, :])
            )

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
            plt.title(f"Case {case_idx} {actor_name} {param}")
            # Save plot
            if not os.path.exists(f"{plots_dir}/Case{case_idx:02d}"):
                os.makedirs(f"{plots_dir}/Case{case_idx:02d}")
            plt.savefig(f"{plots_dir}/Case{case_idx:02d}/{actor_name}_{param}.png")
            plt.close()


def plot_weights(model, seq_len, pred_len, plots_dir, param, num_features=11):
    # Plot linear seasonal weights for all actors in 2x2 grid
    fig, ax = plt.subplots(4, 3, figsize=(10, 10))
    xticks = np.arange(0, seq_len)
    yticks = np.arange(0, num_features)
    for i, actor_name in enumerate(role_names):
        seasonal = model.linear_seasonal[i].weight.detach().numpy()
        trend = model.linear_trend[i].weight.detach().numpy()
        decoder = model.linear_decoder[i].weight.detach().numpy()
        seasonal = seasonal.reshape(num_features, seq_len)
        trend = trend.reshape(num_features, seq_len)
        decoder = decoder.reshape(num_features, seq_len)
        ax[i, 0].imshow(seasonal)
        ax[i, 0].set_title(f"{actor_name} Seasonal")
        ax[i, 0].set_xlabel("Features")
        ax[i, 0].set_ylabel("Input Timesteps")
        ax[i, 0].set_xticks(xticks)
        ax[i, 0].set_yticks(yticks)

        ax[i, 1].imshow(trend)
        ax[i, 1].set_title(f"{actor_name} Trend")
        ax[i, 1].set_xlabel("Features")
        ax[i, 1].set_ylabel("Input Timesteps")
        ax[i, 1].set_xticks(xticks)
        ax[i, 1].set_yticks(yticks)

        ax[i, 2].imshow(decoder)
        ax[i, 2].set_title(f"{actor_name} Decoder")
        ax[i, 2].set_xlabel("Features")
        ax[i, 2].set_ylabel("Input Timesteps")
        ax[i, 2].set_xticks(xticks)
        ax[i, 2].set_yticks(yticks)

    # Add one colorbar to the right of all plots indicating magnitude of weights
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(ax[0, 0].imshow(seasonal), cax=cbar_ax)
    plt.title(f"{param} Weights")
    plt.tight_layout()
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(f"{plots_dir}/{param}_weights.png")
    plt.close()


def generate_scatterplots(
    model, param_name, seq_len, pred_len, predictions, test_dataset, plots_dir
):
    # Split predictions by case
    case_ids = list(test_dataset.keys())
    case_predictions = {}
    idx = 0
    for case_idx in case_ids:
        num_samples = test_dataset[case_idx].shape[-1]
        offset = seq_len + pred_len
        case_predictions[case_idx] = predictions[idx : idx + num_samples - offset]
        idx += num_samples - offset

        # param_indices = {"PNS index": 0, "SNS index": 1, "Mean RR": 2, "RMSSD": 3, "LF-HF": 4}
        param_idx = param_indices[param_name]

        for actor_idx, actor_name in enumerate(role_names):
            true_hrv = test_dataset[case_idx][param_idx][actor_idx, 0, :]
            pred_hrv = case_predictions[case_idx][:, actor_idx, 0].detach().numpy()

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
                    pd.DataFrame({"true": true_hrv, "pred": pred_hrv}).corr().iloc[0, 1]
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


def torch_correlation_density_plots(
    model_name,
    param_name,
    predictions,
    test_dataset,
    plots_dir,
):
    # Get correlation coefficients
    correlations = get_correlation_coefficients(param_name, predictions, test_dataset)

    for param in param_names:
        sns.set_style("whitegrid")
        for actor in role_names:
            sns.kdeplot(correlations[actor], bw_method=0.5, label=actor)
        plt.title(f"Density of {param} correlation coefficients for {model_name} model")
        plt.xlabel(f"Correlation Coefficient")
        plt.ylabel("Density")
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        plt.savefig(f"{plots_dir}/{param}_density.png")
        plt.close()


def plot_loss_curves(
    train_loss, val_loss, test_loss, plots_dir, param, model_name, hidden_dims
):
    shortened_name = model_name.split("_")[0]
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.title(
        f"{param} Loss for {shortened_name}: {hidden_dims} Test MSE loss: {test_loss}"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(f"{plots_dir}/{param}_loss.png")
    plt.close()
