from matplotlib import pyplot as plt
from matplotlib import ticker
from models.mcmc_imputer import *
from utils.data import *
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def imputation_line_plot(
    unimputed_dataset,
    imputed_dataset,
    time_interval=5,
    data_dir="./data",
    plots_dir="./plots",
    first_phase_removed=True,
):
    cases_summary = pd.read_excel(f"{data_dir}/NIH-OR-cases-summary.xlsx").iloc[1:, :]
    phase_ids = get_phase_ids(time_interval=time_interval)
    limits = {
        "PNS index": [-4, 5],
        "SNS index": [-2, 12],
        "Mean RR": [400, 1200],
        "RMSSD": [0, 160],
        "LF-HF": [0, 70],
    }
    for case_id, unimputed_case_data in unimputed_dataset.items():
        for param_id, unimputed_param_data in enumerate(unimputed_case_data):
            # Get corresponding data from imputed dataset
            assert case_id in imputed_dataset
            imputed_param_data = imputed_dataset[case_id][param_id][:, 0, :]
            unimputed_param_data = unimputed_dataset[case_id][param_id][:, 0, :]
            assert unimputed_param_data.shape == imputed_param_data.shape

            plt.figure(figsize=(10, 4))
            n_samples = unimputed_param_data.shape[1]
            x_axis = np.linspace(0, n_samples - 1, n_samples)
            for actor_id, unimputed_role_data in enumerate(unimputed_param_data):
                imputed_role_data = imputed_param_data[actor_id]
                role = config.role_names[actor_id]
                actor_name = cases_summary.loc[cases_summary["Case"] == case_id][
                    role
                ].values[0]
                # actor_param_mean = means[param_id, actor_id]
                # plt.axhline(
                #     y=actor_param_mean, color=f"C{actor_id}", linestyle="--", alpha=0.4
                # )
                role_color = f"C{config.role_colors[config.role_names[actor_id]]}"
                # Plot imputed data
                plt.plot(
                    x_axis,
                    imputed_role_data,
                    color=role_color,
                    linestyle="--",
                    alpha=0.6,
                )
                # Superimpose unimputed data
                plt.plot(
                    x_axis,
                    unimputed_role_data,
                    label=f"{role} {actor_name}",
                    color=role_color,
                )

            # Add phase ID lines
            minor_ticks = []
            minor_labels = []
            offset = phase_ids[case_id][config.phases[0]][0]
            for phase in config.phases:
                interval = phase_ids[case_id][phase]
                if interval[0] is not None and interval[1] is not None:
                    plt.axvspan(
                        interval[0] - offset,
                        interval[1] - offset,
                        color="black",
                        alpha=0.1,
                    )
                    midpoint = (
                        int(interval[0] + (interval[1] - interval[0]) / 2) - offset
                    )
                    minor_ticks.append(midpoint)
                    minor_labels.append(f"P{phase}")

            ax = plt.gca()
            ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            ax.xaxis.set_minor_formatter(ticker.FixedFormatter(minor_labels))
            ax.set_xticks(minor_ticks, minor=True)
            ax.tick_params(
                axis="x",
                which="minor",
                direction="out",
                top=True,
                labeltop=True,
                bottom=False,
                labelbottom=False,
            )

            plt.title(
                f"Imputed HRV for case {case_id}: {config.param_names[param_id]} ({time_interval}min)"
            )
            plt.xlabel("Timestep")
            plt.ylabel(config.param_names[param_id])
            # plt.ylim(limits[param_names[param_id]])
            plt.legend()

            # Write plots to file
            if not os.path.exists(f"{plots_dir}/imputation/Case{case_id:02d}"):
                os.makedirs(f"{plots_dir}/imputation/Case{case_id:02d}")
            plt.savefig(
                f"{plots_dir}/imputation/Case{case_id:02d}/{config.param_names[param_id]}.png"
            )

            plt.close()


def mcmc_plot(
    samples,
    original_dataset,
    imputed_dataset,
    time_interval=5,
    data_dir="./data",
    lag_length=3,
    plots_dir="./plots",
    first_phase_removed=True,
):
    cases_summary = pd.read_excel(f"{data_dir}/NIH-OR-cases-summary.xlsx").iloc[1:, :]
    phase_ids = get_phase_ids(time_interval=time_interval)
    limits = {
        "PNS index": [-4, 5],
        "SNS index": [-2, 12],
        "Mean RR": [400, 1200],
        "RMSSD": [0, 160],
        "LF-HF": [0, 70],
    }
    for case_id, per_case_samples in tqdm(samples.items()):
        for param_name, param_samples in per_case_samples.items():
            # Shape of the data is (num_actors, seq_len - lag_length, num_iterations)
            for actor_idx in range(param_samples.shape[0]):
                # Create line plot by taking the mean of the samples at every timestep
                actor_name = config.actor_names[actor_idx]
                param_idx = config.param_names.index(param_name)
                actor_samples = param_samples[actor_idx]
                plt.figure(figsize=(10, 4))
                n_samples = actor_samples.shape[0]
                # Collect sample means and stddevs per timestep
                original_samples = original_dataset[case_id][param_idx, actor_idx, 0, :]
                imputed_samples = imputed_dataset[case_id][param_idx, actor_idx, 0, :]
                seq_len = original_samples.shape[0]
                n_samples = actor_samples.shape[0]
                start_idx = original_samples.shape[0] - n_samples
                sample_means = np.mean(actor_samples, axis=1)
                sample_stddevs = np.std(actor_samples, axis=1)

                # Add phase ID lines
                minor_ticks = []
                minor_labels = []
                offset = 0
                for phase, interval in enumerate(phase_ids[case_id]):
                    if first_phase_removed and phase == 0:
                        offset = phase_ids[case_id][phase][1]
                        continue
                    if interval[0] is not None and interval[1] is not None:
                        plt.axvspan(
                            interval[0] - offset,
                            interval[1] - offset,
                            color="black",
                            alpha=0.1,
                        )
                        midpoint = (
                            int(interval[0] + (interval[1] - interval[0]) / 2) - offset
                        )
                        minor_ticks.append(midpoint)
                        minor_labels.append(f"P{phase+1}")

                # Add x-axis markers for phase IDs
                ax = plt.gca()
                ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
                ax.xaxis.set_minor_formatter(ticker.FixedFormatter(minor_labels))
                ax.set_xticks(minor_ticks, minor=True)
                ax.tick_params(
                    axis="x",
                    which="minor",
                    direction="out",
                    top=True,
                    labeltop=True,
                    bottom=False,
                    labelbottom=False,
                )

                # Plot sample means as a line and stddevs as an errorbar
                # Starting from start_idx
                x_axis = np.linspace(start_idx, seq_len - 1, n_samples)
                plt.plot(
                    x_axis,
                    sample_means,
                    label="sample mean",
                    color=f"C{actor_idx}",
                )
                plt.fill_between(
                    x_axis,
                    sample_means - sample_stddevs,
                    sample_means + sample_stddevs,
                    alpha=0.2,
                    color=f"C{actor_idx}",
                )

                x_axis = np.linspace(0, seq_len - 1, seq_len)

                # Plot imputed data
                plt.plot(
                    x_axis,
                    imputed_samples,
                    label=f"imputed",
                    color=f"C{actor_idx}",
                    linestyle="--",
                    alpha=0.4,
                )
                # Plot original data
                plt.plot(
                    x_axis,
                    original_samples,
                    label=f"original",
                    color=f"C{actor_idx}",
                    alpha=0.6,
                )

                # Save plot
                plt.title(
                    f"MCMC samples for case {case_id}: {param_name} ({time_interval}min)"
                )
                plt.xlabel("Timestep")
                plt.ylabel(param_name)
                plt.legend()

                # Write plots to file
                if not os.path.exists(f"{plots_dir}/mcmc/Case{case_id:02d}"):
                    os.makedirs(f"{plots_dir}/mcmc/Case{case_id:02d}")
                plt.savefig(
                    f"{plots_dir}/mcmc/Case{case_id:02d}/{param_name}_{actor_name}.png"
                )

                plt.close()
