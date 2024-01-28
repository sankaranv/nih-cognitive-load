import numpy as np
import pandas as pd
from utils.config import config


def compute_metrics(y, predictions):
    metrics = {}
    # Compute mean squared error
    metrics["mean_squared_error"] = np.mean((y - predictions) ** 2)
    # Compute root mean squared error
    metrics["rms_error"] = np.sqrt(metrics["mean_squared_error"])
    # Compute mean absolute error
    metrics["mean_absolute_error"] = np.mean(np.abs(y - predictions))
    # Compute R^2 score
    metrics["r_squared"] = 1 - (
        np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2)
    )
    # Compute correlation coefficient for each actor
    num_actors = y.shape[1]
    metrics["corr_coef"] = np.zeros(num_actors)
    for actor_idx in range(num_actors):
        metrics["corr_coef"][actor_idx] = np.corrcoef(
            y[:, actor_idx], predictions[:, actor_idx]
        )[0, 1]
    return metrics


def hms_to_min(s):
    t = 0
    for u in s.split(":"):
        t = 60 * t + int(u)
    # Round to the minute
    if t % 60 >= 30:
        return int(t / 60) + 1
    else:
        return int(t / 60)


def get_means(dataset):
    means = np.zeros((5, 4))
    num_samples = np.zeros((5, 4))
    for _, case_data in dataset.items():
        if len(case_data.shape) == 3:
            data = case_data
        elif case_data.shape[-2] > 1:
            # If temporal or static features are present, ignore them
            data = case_data[:, :, 0, :]
        num_samples += np.sum(~np.isnan(data), axis=-1)
        means += np.sum(np.nan_to_num(data), axis=-1)
    return means / num_samples


def get_stddevs(dataset):
    samples = {}
    std_devs = np.zeros((5, 4))
    for _, data in dataset.items():
        for x in range(5):
            for y in range(4):
                if (x, y) not in samples:
                    samples[(x, y)] = np.array([])
                samples[(x, y)] = np.concatenate(
                    (samples[(x, y)], data[x][y][~np.isnan(data[x][y])])
                )
    for x in range(5):
        for y in range(4):
            std_devs[x, y] = np.std(samples[(x, y)])
    return std_devs


def get_max_len(dataset):
    max_len = 0
    for case_id in dataset.keys():
        if dataset[case_id].shape[-1] > max_len:
            max_len = dataset[case_id].shape[-1]
    return max_len


def get_lengths(dataset):
    lengths = {}
    for case_id in dataset.keys():
        lengths[case_id] = dataset[case_id].shape[-1]
    return lengths


def get_per_phase_actor_ids(dataset, time_interval=5, data_dir="./data"):
    per_phase_actor_ids = {actor: {} for actor in config.role_names}
    phase_ids = get_phase_ids(time_interval=time_interval)
    cases_summary = pd.read_excel(f"{data_dir}/NIH-OR-cases-summary.xlsx").iloc[1:, :]
    print("Getting per phase actor IDs")
    for i in config.valid_cases:
        try:
            # Ignore cases with missing per-step data
            if dataset.shape[-1] > 1:
                # We assume all parameters have the same number of measurements in the dataset
                param_id = 0
                for actor_id, role_name in enumerate(config.role_names):
                    for phase, interval in enumerate(phase_ids[i]):
                        if interval[0] is not None and interval[1] is not None:
                            per_case_samples = dataset[
                                param_id, actor_id, interval[0] : interval[1]
                            ]
                            # Log actor names in case needed for coloring scatterplots
                            # Obtain actor name for the given case
                            case_actor_name = cases_summary.loc[
                                cases_summary["Case"] == i
                            ][role_name].values[0]
                            if phase not in per_phase_actor_ids[role_name]:
                                per_phase_actor_ids[role_name][phase] = np.full(
                                    len(per_case_samples), case_actor_name
                                )
                            else:
                                per_phase_actor_ids[role_name][phase] = np.concatenate(
                                    (
                                        per_phase_actor_ids[role_name][phase],
                                        np.full(len(per_case_samples), case_actor_name),
                                    )
                                )
        except Exception as e:
            print(e)
    return per_phase_actor_ids


def get_phase_ids(data_dir="./data", time_interval=5):
    phase_ids = {}
    for i in config.valid_cases:
        phase_ids[i] = []
        file_name = f"{data_dir}/Case{i:02d}/3296_{i:02d}-abstractedPhases.csv"
        with open(file_name, "r") as f:
            df = pd.read_csv(file_name, header=0)
            df = df.dropna(axis=0, how="all")
            for phase, row in df.iterrows():
                start_time = row["Onset_Time"]
                stop_time = row["Offset_Time"]
                if not isinstance(start_time, str):
                    phase_ids[i].append([None, None])
                else:
                    start_idx = hms_to_min(start_time) // time_interval
                    stop_idx = hms_to_min(stop_time) // time_interval
                    phase_ids[i].append([start_idx, stop_idx])
    return phase_ids


def get_correlation_coefficients(param_name, predictions, test_dataset):
    # Store correlation coefficients
    correlations = {}
    for actor in config.role_names:
        correlations[actor] = []

    for case_idx, case_predictions in predictions[param_name].items():
        param_idx = config.param_indices[param_name]

        for actor_idx, actor_name in enumerate(config.role_names):
            true_hrv = test_dataset[case_idx][param_idx][actor_idx, 0, :]
            pred_hrv = case_predictions[actor_idx, 0, :]

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
                correlations[actor_name].append(corr)

    return correlations
