import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
import matplotlib as mpl

param_names = ["PNS index", "SNS index", "Mean RR", "RMSSD", "LF-HF"]
role_names = ["Anes", "Nurs", "Perf", "Surg"]
cases_summary = pd.read_excel("data/NIH-OR-cases-summary.xlsx").iloc[1:, :]


def plot_params(
    dataset, means, phase_ids, case_id, plots_dir, latex_dir, time_interval="5min"
):
    limits = {
        "PNS index": [-4, 5],
        "SNS index": [-2, 12],
        "Mean RR": [400, 1200],
        "RMSSD": [0, 160],
        "LF-HF": [0, 70],
    }
    for case_data in dataset:
        for param_id, param_data in enumerate(case_data):
            plt.figure(figsize=(10, 4))
            n_samples = param_data.shape[1]
            x_axis = np.linspace(0, n_samples - 1, n_samples)
            for actor_id, role_data in enumerate(param_data):
                role = role_names[actor_id]
                actor_name = cases_summary.loc[cases_summary["Case"] == case_id][
                    role
                ].values[0]
                actor_param_mean = means[param_id, actor_id]
                plt.axhline(
                    y=actor_param_mean, color=f"C{actor_id}", linestyle="--", alpha=0.4
                )
                plt.plot(x_axis, role_data, label=f"{role} {actor_name}")

            # Add phase ID lines
            minor_ticks = []
            minor_labels = []
            for phase, interval in enumerate(phase_ids[case_id]):
                plt.axvspan(interval[0], interval[1], color="black", alpha=0.1)
                if interval[0] is not None and interval[1] is not None:
                    midpoint = int(interval[0] + (interval[1] - interval[0]) / 2)
                    minor_ticks.append(midpoint)
                    minor_labels.append(f"P{phase+1}")

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
                f"Heart rate variability for case {case_id}: {param_names[param_id]} ({time_interval})"
            )
            plt.xlabel("Timestep")
            plt.ylabel(param_names[param_id])
            plt.ylim(limits[param_names[param_id]])
            plt.legend()

            # Write plots to file
            # if not os.path.isdir(f'{plots_dir}/line_plots/Case{case_id:02d}'):
            #     os.mkdir(f'{plots_dir}/line_plots/Case{case_id:02d}')
            plt.savefig(
                f"{plots_dir}/{time_interval}/line_plots/Case{case_id:02d}/{param_names[param_id]}.png"
            )

            # Push plots directly to Overleaf
            if latex_dir is not None:
                plt.savefig(
                    f"{latex_dir}/plots/{time_interval}/line_plots/Case{case_id:02d}/{param_names[param_id]}.png"
                )

            plt.close()


def generate_line_plots(plots_dir="plots", latex_dir=None, time_interval="5min"):
    means = get_means(time_interval)
    phase_ids = get_phase_ids(time_interval=time_interval)
    print(means)
    for i in tqdm(range(1, 41)):
        if i not in [5, 9, 14, 16, 24, 39]:
            try:
                dataset = import_case_data(case_id=i, time_interval=time_interval)
                plot_params(
                    dataset, means, phase_ids, i, plots_dir, latex_dir, time_interval
                )
            except Exception as e:
                print(e)


def plot_densities_by_role(latex_dir=None, time_interval="5min"):
    # Collect data by actor ID
    dataset_by_actor = {"Anes": {}, "Nurs": {}, "Perf": {}, "Surg": {}}
    for i in range(1, 41):
        if i not in [5, 9, 14, 16, 24, 39]:
            dataset = import_case_data(case_id=i, time_interval=time_interval)
            for param_id, param_data in enumerate(dataset[0]):
                for actor_id, role_data in enumerate(param_data):
                    samples = role_data[np.where(~np.isnan(role_data))]
                    role = role_names[actor_id]
                    param = param_names[param_id]
                    key = cases_summary.loc[cases_summary["Case"] == i][role].values[0]
                    if param not in dataset_by_actor[role]:
                        dataset_by_actor[role][param] = {}
                    if key in dataset_by_actor[role][param]:
                        dataset_by_actor[role][param][key] = np.hstack(
                            (dataset_by_actor[role][param][key], samples)
                        )
                    elif len(samples) > 0:
                        dataset_by_actor[role][param][key] = role_data
    for role in role_names:
        for param in param_names:
            if not os.path.isdir(f"plots/{time_interval}/density_plots/{role}"):
                os.mkdir(f"plots/{time_interval}/density_plots/{role}")
            plt.figure(figsize=(10, 4))
            for actor, samples in dataset_by_actor[role][param].items():
                sns.set_style("whitegrid")
                sns.kdeplot(samples, bw_method=0.5, label=actor)
                plt.xlabel(f"{param}")
                plt.ylabel("Density")
                plt.title(f"Density of {param} ({time_interval}) for {role}")
                plt.legend()
                print(f"{role} {param} {actor}")
            plt.savefig(f"plots/{time_interval}/density_plots/{role}/{param}.png")
            if latex_dir is not None:
                plt.savefig(
                    f"{latex_dir}/plots/{time_interval}/density_plots/{role}/{param}.png"
                )
            plt.close()


def generate_per_phase_density_plots(latex_dir=None, time_interval="5min"):
    per_phase_normalized_samples = get_per_phase_normalized_samples(time_interval)
    for role in role_names:
        for param in param_names:
            for phase_id, per_phase_samples in per_phase_normalized_samples[param][
                role
            ].items():
                samples = per_phase_samples[~np.isnan(per_phase_samples)]
                sns.set_style("whitegrid")
                sns.kdeplot(
                    per_phase_samples,
                    bw_method=0.5,
                    label=f"Phase {phase_id} ({len(per_phase_samples)} samples)",
                )
                plt.xlabel(f"Normalized {param}")
                plt.ylabel("Density")
                plt.title(
                    f"Per-phase density of normalized {param} ({time_interval}) for {role}"
                )
            plt.legend()
            plt.savefig(
                f"plots/{time_interval}/density_plots/per_phase/{param}/{role}.png"
            )
            if latex_dir is not None:
                plt.savefig(
                    f"{latex_dir}/plots/{time_interval}/density_plots/per_phase/{param}/{role}.png"
                )
            plt.close()


def generate_scatterplots(latex_dir=None, time_interval="5min"):
    means = get_means(time_interval)

    # Collect correlation coefficients for each parameter
    param_corr_coef_samples = {
        "PNS index": [],
        "SNS index": [],
        "Mean RR": [],
        "RMSSD": [],
        "LF-HF": [],
    }

    for i in tqdm(range(1, 41)):
        if i not in [5, 9, 14, 16, 24, 39]:
            try:
                dataset = import_case_data(case_id=i, time_interval=time_interval)[0]
                # Ignore cases with missing per-step data
                if dataset.shape[-1] > 1:
                    means = np.nanmean(dataset, axis=-1)
                    std = np.nanstd(dataset, axis=-1)
                    dataset = (dataset - means[:, :, None]) / std[:, :, None]
                    for param_id, param_name in enumerate(param_names):
                        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
                        for ax_counter, (idx_1, idx_2) in enumerate(
                            combinations([0, 1, 2, 3], 2)
                        ):
                            samples_x = dataset[param_id, idx_1, :]
                            samples_y = dataset[param_id, idx_2, :]

                            # # Set ranges for scatterplots
                            # if (not np.isnan(samples_x).all()) and (not np.isnan(samples_y).all()):
                            #     range_high = ceil(max(max(samples_x), max(samples_y)))
                            #     range_low = ceil(min(min(samples_x), min(samples_y)))
                            #     range_equal = max(abs(range_high), abs(range_low))
                            # else:
                            #     range_equal = 3
                            range_equal = 6

                            # Correlation coefficient that is NaN-sensitive
                            corr_coef = np.ma.corrcoef(
                                np.ma.masked_invalid(samples_x),
                                np.ma.masked_invalid(samples_y),
                            )
                            if corr_coef[0, 1].data == 0:
                                corr_coef = corr_coef[0, 1]
                            else:
                                corr_coef = corr_coef[0, 1]
                                param_corr_coef_samples[param_name].append(corr_coef)

                            # Plot
                            axs.flat[ax_counter].scatter(samples_x, samples_y)
                            axs.flat[ax_counter].set_xlabel(
                                f"{role_names[idx_1]} {param_name}"
                            )
                            axs.flat[ax_counter].set_ylabel(
                                f"{role_names[idx_2]} {param_name}"
                            )
                            axs.flat[ax_counter].set_xlim(-range_equal, range_equal)
                            axs.flat[ax_counter].set_ylim(-range_equal, range_equal)
                            axs.flat[ax_counter].set_title(f"rho = {corr_coef:04f}")

                        # Save overall plot
                        fig.suptitle(f"Standardized {param_name} for Case {i:02d}")
                        fig.savefig(
                            f"plots/{time_interval}/scatterplots/{param_name}/Case{i:02d}.png"
                        )
                        if latex_dir is not None:
                            fig.savefig(
                                f"{latex_dir}/plots/{time_interval}/scatterplots/{param_name}/Case{i:02d}.png"
                            )
                        plt.close()
            except Exception as e:
                print(e)

    # Density plot of correlation coefficients
    for param_name in param_names:
        sns.set_style("whitegrid")
        sns.kdeplot(
            param_corr_coef_samples[param_name], bw_method=0.5, label=param_name
        )
        plt.xlabel(f"Pearson Correlation Coefficient")
        plt.ylabel("Density")
        plt.title(
            f"Density of correlation coefficient ({time_interval}) between pairs of actors"
        )
    plt.legend()
    plt.savefig(f"plots/{time_interval}/density_plots/corr_coef.png")
    if latex_dir is not None:
        plt.savefig(f"{latex_dir}/plots/{time_interval}/density_plots/corr_coef.png")
    plt.close()


def scatterplots_per_phase(latex_dir=None, time_interval="5min", color_by="x"):
    per_phase_normalized_samples = get_per_phase_normalized_samples(time_interval)

    # Convert per phase actor names to indices for coloring scatterplots
    per_phase_actor_ids = get_per_phase_actor_ids(time_interval)
    unique_actor_names = {}
    for role_name, actor_names_by_phase in per_phase_actor_ids.items():
        unique_actor_names[role_name] = []
        # First get unique actor names
        for _, names_per_phase in actor_names_by_phase.items():
            unique_actor_names[role_name] = np.union1d(
                unique_actor_names[role_name], np.unique(names_per_phase)
            )

    phase_colors = {}
    for role_name, actor_names_by_phase in per_phase_actor_ids.items():
        phase_colors[role_name] = {}
        num_colors = len(unique_actor_names[role_name])
        for phase, names_per_phase in actor_names_by_phase.items():
            phase_colors[role_name][phase] = np.zeros(len(names_per_phase))

            # Is this vectorizable or has a numpy command?
            for color_idx, color_key in enumerate(unique_actor_names[role_name]):
                phase_colors[role_name][phase][
                    np.where(names_per_phase == color_key)
                ] = color_idx

    for role_id_1, role_id_2 in combinations([0, 1, 2, 3], 2):
        # For each pair of roles, generate one figure with [num_phases] scatterplots for each parameter
        role_1 = role_names[role_id_1]
        role_2 = role_names[role_id_2]

        # Obtain colors for samples based on actor ID
        role_1_color_keys, role_1_color_idx = np.unique(
            per_phase_actor_ids[role_1][0], return_inverse=True
        )
        role_2_color_keys, role_2_color_idx = np.unique(
            per_phase_actor_ids[role_2][0], return_inverse=True
        )
        print(role_1, role_2)

        for param_name, param_data in per_phase_normalized_samples.items():
            num_phases = len(param_data[role_1].keys())
            fig, axs = plt.subplots(2, 4, figsize=(20, 10))
            sns.set_style("whitegrid")
            if color_by == "x":
                cmap = mpl.colors.ListedColormap(
                    [f"C{i}" for i in range(len(unique_actor_names[role_1]))]
                )
            else:
                cmap = mpl.colors.ListedColormap(
                    [f"C{i}" for i in range(len(unique_actor_names[role_2]))]
                )
            for phase in range(num_phases):
                current_ax = axs.flat[phase]
                samples_x = param_data[role_1][phase]
                samples_y = param_data[role_2][phase]

                # Plot
                range_equal = 6
                if color_by == "x":
                    sc = current_ax.scatter(
                        samples_x,
                        samples_y,
                        alpha=0.4,
                        c=phase_colors[role_1][phase],
                        cmap=cmap,
                        label=unique_actor_names[role_1],
                    )
                else:
                    sc = current_ax.scatter(
                        samples_x,
                        samples_y,
                        alpha=0.4,
                        c=phase_colors[role_2][phase],
                        cmap=cmap,
                        label=unique_actor_names[role_2],
                    )
                current_ax.set_xlabel(f"{role_1} {param_name}")
                current_ax.set_ylabel(f"{role_2} {param_name}")
                current_ax.set_xlim(-range_equal, range_equal)
                current_ax.set_ylim(-range_equal, range_equal)
                current_ax.set_title(f"Phase {phase}")

            # Create legend for colours
            if color_by == "x":
                color_tick_labels = unique_actor_names[role_1]
            else:
                color_tick_labels = unique_actor_names[role_2]
            norm = mpl.colors.Normalize(0, len(color_tick_labels))
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            # print(color_tick_labels, np.unique(phase_colors[role_1][phase], return_counts=True))
            cbar_ax = fig.add_axes(
                [0.92, 0.1, 0.02, 0.8]
            )  # [left, bottom, width, height]
            cbar = plt.colorbar(sm, cax=cbar_ax)
            cbar.set_ticklabels(color_tick_labels)
            cbar.ax.set_yticks(np.arange(len(color_tick_labels)) + 0.5)
            cbar.ax.set_yticklabels(color_tick_labels)

            # Save overall plot
            fig.suptitle(f"Standardized {param_name} for {role_1} vs. {role_2}")
            plt.savefig(
                f"plots/{time_interval}/per_phase/{param_name}/{role_1}-{role_2}-{color_by}.png"
            )
            if latex_dir is not None:
                fig.savefig(
                    f"{latex_dir}/plots/{time_interval}/per_phase/{param_name}/{role_1}-{role_2}-{color_by}.png"
                )
            plt.close()


def serial_correlation_plot(dataset):
    actor_names = ["Anes", "Nurs", "Perf", "Surg"]
    for actor_idx, actor_name in enumerate(actor_names):
        samples_0 = []
        samples_1 = []
        for case_idx, case_data in dataset.items():
            print(case_data.shape)
            for param_idx, param_data in enumerate(case_data):
                actor_data = param_data[actor_idx]
                seq_len = actor_data.shape[1]
                for i in range(seq_len - 1):
                    samples_0.append(actor_data[0, i])
                    samples_1.append(actor_data[0, i + 1])
        # Plot
        plt.scatter(samples_0, samples_1)
        plt.xlabel("x(t)")
        plt.ylabel("x(t+1)")
        plt.title(f"Serial correlation for {actor_name}")
        # Check if dir exists
        if not os.path.isdir(f"plots/serial_correlation"):
            os.mkdir(f"plots/serial_correlation")
        plt.savefig(f"plots/serial_correlation/{actor_name}.png")
