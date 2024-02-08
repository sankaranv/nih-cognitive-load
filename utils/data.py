import csv
import itertools
import numpy as np
import os
import json
from tqdm import tqdm
from math import ceil
import random
import torch
import pickle
from typing import Union

from utils.features import add_temporal_features, add_static_features
from utils.stats import get_phase_ids, get_means, get_stddevs
from utils.config import config


def load_dataset(data_dir="./data/processed", normalized=False, pad_phase_on=False):
    if normalized:
        try:
            if pad_phase_on:
                dataset = pickle.load(
                    open(
                        f"{data_dir}/pad_phase_on/normalized_imputed_dataset.pkl",
                        "rb",
                    )
                )
            else:
                dataset = pickle.load(
                    open(
                        f"{data_dir}/pad_phase_off/normalized_imputed_dataset.pkl",
                        "rb",
                    )
                )
        except ValueError:
            print("Normalized imputed dataset not found")

        try:
            if pad_phase_on:
                unimputed_dataset = pickle.load(
                    open(
                        f"{data_dir}/pad_phase_on/normalized_original_dataset.pkl",
                        "rb",
                    )
                )
            else:
                unimputed_dataset = pickle.load(
                    open(
                        f"{data_dir}/pad_phase_off/normalized_original_dataset.pkl",
                        "rb",
                    )
                )
        except ValueError:
            print("Normalized unimputed dataset not found")
    else:
        try:
            if pad_phase_on:
                dataset = pickle.load(
                    open(
                        f"{data_dir}/pad_phase_on/imputed_dataset.pkl",
                        "rb",
                    )
                )
            else:
                dataset = pickle.load(
                    open(
                        f"{data_dir}/pad_phase_off/imputed_dataset.pkl",
                        "rb",
                    )
                )
        except ValueError:
            print("Imputed dataset not found")

        try:
            if pad_phase_on:
                unimputed_dataset = pickle.load(
                    open(
                        f"{data_dir}/pad_phase_on/original_dataset.pkl",
                        "rb",
                    )
                )
            else:
                unimputed_dataset = pickle.load(
                    open(
                        f"{data_dir}/pad_phase_off/original_dataset.pkl",
                        "rb",
                    )
                )
        except ValueError:
            print("Unimputed dataset not found")

    return dataset, unimputed_dataset


def import_case_data(
    data_dir="./data", case_id=3, time_interval=5, pad_to_max_len=False, max_length=None
):
    relevant_lines = [66, 67, 72, 78, 111]
    if time_interval == 5:
        phase_name = "cognitiveLoad-phases-5min"
    elif time_interval == 1:
        phase_name = "cognitiveLoad-phases-1min"
    else:
        raise ValueError(
            f"Data is only available for intervals of 1min or 5min, not {time_interval}"
        )
    dataset = []
    max_num_samples = 0
    for role_name in config.role_names:
        if time_interval == 5:
            file_name = f"{data_dir}/Case{case_id:02d}/{phase_name}/3296_{case_id:02d}_{role_name}_hrv.csv"
        else:
            file_name = f"{data_dir}/Case{case_id:02d}/{phase_name}/3296_{case_id:02d}_{role_name}_hrv-1min.csv"
        if not os.path.isfile(file_name):
            if max_num_samples > 0:
                empty_role_data = np.full((5, max_num_samples), np.nan)
            else:
                empty_role_data = np.full((5, 1), np.nan)
            dataset.append(empty_role_data)
        else:
            role_data = []
            with open(file_name, "r") as f:
                r = csv.reader(f)
                for i in itertools.count(start=1):
                    if i > relevant_lines[-1]:
                        break
                    elif i not in relevant_lines:
                        next(r)
                    else:
                        try:
                            row = next(r)
                            row = [x.replace(" ", "") for x in row]
                            row = [x for x in row if x != ""][1:]
                            row = [float(x) if x != "NaN" else np.nan for x in row]
                            role_data.append(row)
                        except StopIteration as e:
                            print("End of file reached")
            role_data[-1] = role_data[-1][::2]
            role_data = np.array(role_data)
            dataset.append(role_data)
            if role_data.shape[1] > max_num_samples:
                max_num_samples = role_data.shape[1]

    # Add padding to the data for missing samples
    # This assumes all measurements start at the same time and just cut off early for some roles!
    for i, role_data in enumerate(dataset):
        if pad_to_max_len:
            if max_length is None:
                raise ValueError(
                    "Must provide max_length if padding to max length for all cases"
                )
            padding_length = max_length
        else:
            padding_length = max_num_samples
        if role_data.shape[1] < padding_length:
            pad_length = max_num_samples - role_data.shape[1]
            empty_data = np.full((5, pad_length), np.nan)
            dataset[i] = np.hstack((role_data, empty_data))

    dataset = np.array(dataset)
    dataset = np.expand_dims(dataset, axis=0)
    dataset = np.swapaxes(dataset, 1, 2)
    return dataset


def get_per_phase_normalized_samples(time_interval="5min"):
    print("Getting per phase normalized samples")
    per_phase_normalized_samples = {
        "PNS index": {},
        "SNS index": {},
        "Mean RR": {},
        "RMSSD": {},
        "LF-HF": {},
    }
    phase_ids = get_phase_ids(time_interval=time_interval)
    for i in config.valid_cases:
        try:
            dataset = import_case_data(case_id=i, time_interval=time_interval)[0]
            # Ignore cases with missing per-step data
            if dataset.shape[-1] > 1:
                means = np.nanmean(dataset, axis=-1)
                std = np.nanstd(dataset, axis=-1)
                dataset = (dataset - means[:, :, None]) / std[:, :, None]
                for param_id, param_name in enumerate(config.param_names):
                    for actor_id, role_name in enumerate(config.role_names):
                        if (
                            role_name
                            not in per_phase_normalized_samples[param_name].keys()
                        ):
                            per_phase_normalized_samples[param_name][role_name] = {}
                        for phase, interval in enumerate(phase_ids[i]):
                            if interval[0] is not None and interval[1] is not None:
                                per_case_samples = dataset[
                                    param_id, actor_id, interval[0] : interval[1]
                                ]

                                # Select out data for the interval of each phase
                                if (
                                    phase
                                    not in per_phase_normalized_samples[param_name][
                                        role_name
                                    ].keys()
                                ):
                                    per_phase_normalized_samples[param_name][role_name][
                                        phase
                                    ] = per_case_samples
                                else:
                                    per_phase_normalized_samples[param_name][role_name][
                                        phase
                                    ] = np.concatenate(
                                        (
                                            per_phase_normalized_samples[param_name][
                                                role_name
                                            ][phase],
                                            per_case_samples,
                                        )
                                    )
        except Exception as e:
            print(e)
    return per_phase_normalized_samples


def make_dataset_from_file(
    data_dir="./data",
    time_interval=5,
    param_id=None,
    standardize=False,
    max_length=None,
    pad_to_max_len=False,
    temporal_features=False,
    static_features=False,
    pad_phase_on=False,
):
    # Shape of the data for each case is (5, 4, num_samples) or (4, num_samples)
    dataset = {}
    for i in config.valid_cases:
        dataset[i] = import_case_data(
            data_dir=data_dir,
            case_id=i,
            time_interval=time_interval,
            max_length=max_length,
            pad_to_max_len=pad_to_max_len,
        )[0]

    # Standardize dataset
    if standardize:
        means = get_means(dataset)
        std_devs = get_stddevs(dataset)
        for case_id in dataset.keys():
            for x in range(5):
                for y in range(4):
                    dataset[case_id][x, y] -= means[x, y]
                    dataset[case_id][x, y] /= std_devs[x, y]

    if param_id is not None:
        for case_id in dataset.keys():
            dataset[case_id] = dataset[case_id][param_id]

    # Add temporal features if requested
    if temporal_features:
        dataset = add_temporal_features(dataset, pad_phase_on=pad_phase_on)

    # Add static features if requested
    if static_features:
        dataset = add_static_features(dataset)

    # Prune out phases that are not in the config
    dataset = prune_phases(dataset, start=config.phases[0], end=config.phases[-1])
    return dataset


def get_nan_mask(data):
    if isinstance(data, np.ndarray):
        return np.isnan(data)
    elif isinstance(data, torch.Tensor):
        return torch.isnan(data)


def make_train_test_split(
    dataset,
    train_split: float = 0.6,
    val_split: float = 0.2,
    test_split: float = 0.2,
    seed=None,
):
    # Set random seed
    if seed is not None:
        random.seed(seed)

    cases = list(dataset.keys())
    random.shuffle(cases)
    if train_split + val_split + test_split != 1:
        raise ValueError(
            "Train, validation, and test splits must sum to 1.0. "
            f"Current splits sum to {train_split + val_split + test_split}"
        )
    train_idx = int(train_split * len(cases))
    val_idx = train_idx + int(val_split * len(cases))
    train_cases = cases[:train_idx]
    val_cases = cases[train_idx:val_idx]
    test_cases = cases[val_idx:]

    train_dataset = {case: dataset[case] for case in train_cases}
    val_dataset = {case: dataset[case] for case in val_cases}
    test_dataset = {case: dataset[case] for case in test_cases}

    return (train_dataset, val_dataset, test_dataset)


def cv_split(
    dataset,
    train_split: float = 0.8,
    num_folds=5,
):
    cases = list(dataset.keys())

    # Generate cross validation splits in the specified ratio
    cv_splits = []
    num_test_cases_per_fold = ceil(len(cases) / num_folds)
    for i in range(num_folds):
        test_case_ids = cases[
            i * num_test_cases_per_fold : (i + 1) * num_test_cases_per_fold
        ]
        rem_case_ids = [case_id for case_id in cases if case_id not in test_case_ids]
        random.shuffle(rem_case_ids)
        train_idx = int(train_split * len(rem_case_ids))
        # val_split = 1 - train_split
        # val_idx = train_idx + int(val_split * len(rem_case_ids))
        train_cases = rem_case_ids[:train_idx]
        val_cases = rem_case_ids[train_idx:]
        test_cases = test_case_ids
        train_dataset = {case: dataset[case] for case in train_cases}
        val_dataset = {case: dataset[case] for case in val_cases}
        test_dataset = {case: dataset[case] for case in test_cases}
        cv_splits.append((train_dataset, val_dataset, test_dataset))

    return cv_splits


def extract_fully_observed_sequences(dataset):
    """
    Extract fully observed sequences from dataset
    """
    num_cases = 0
    fully_observed_sequences = None
    for case_idx, case_data in dataset.items():
        usable_rows_per_case = None
        num_cases += case_data.shape[-1]
        for param_data in case_data:
            param_data = np.transpose(param_data, (2, 0, 1))
            mask = np.all(~np.isnan(param_data), axis=(1, 2))
            usable_rows_per_param = param_data[mask]
            usable_rows_per_param = np.transpose(usable_rows_per_param, (1, 2, 0))
            usable_rows_per_param = usable_rows_per_param[np.newaxis, :, :, :]
            if usable_rows_per_case is None:
                usable_rows_per_case = usable_rows_per_param
            else:
                usable_rows_per_case = np.concatenate(
                    (usable_rows_per_case, usable_rows_per_param), axis=0
                )
        if fully_observed_sequences is None:
            fully_observed_sequences = usable_rows_per_case
        else:
            fully_observed_sequences = np.concatenate(
                (fully_observed_sequences, usable_rows_per_case), axis=3
            )

    # Report percentage of usable cases
    print(
        f"{fully_observed_sequences.shape[-1]/num_cases * 100 :.2f}% of the data is fully observed"
    )
    print(f"Shape of fully observed sequence data: {fully_observed_sequences.shape}")
    return fully_observed_sequences


def extract_all_sequences(dataset):
    sequences = None
    for case_idx, case_data in dataset.items():
        print(case_data.shape)
        if sequences is None:
            sequences = case_data
        else:
            sequences = np.concatenate((sequences, case_data), axis=3)
    return sequences


def drop_edge_phases(dataset, drop_first=True, drop_last=False, time_interval=5):
    """Drop all timesteps from the first phase and/or last phase of each case
       Hopefully this gets rid of the long empty sequences at the start of prediction window

    Args:
        dataset (dict): dataset of HRV parameters
    """
    phase_ids = get_phase_ids(time_interval=time_interval)
    first_key = next(iter(phase_ids))
    first_phase_id = 0
    last_phase_id = len(phase_ids[first_key]) - 1
    new_dataset = {}
    for case_id in dataset.keys():
        # Case data has shape (num_params, num_actors, num_features, num_samples)
        case_data = dataset[case_id]
        first_phase_end_idx = phase_ids[case_id][first_phase_id][1]
        last_phase_start_idx = phase_ids[case_id][last_phase_id][0]
        if drop_first and first_phase_end_idx is not None:
            new_dataset[case_id] = case_data[:, :, :, first_phase_end_idx:]
        if drop_last and last_phase_start_idx is not None:
            new_dataset[case_id] = case_data[:, :, :, :last_phase_start_idx]
    return new_dataset


def normalize_per_case(dataset):
    """Normalize each case to have mean 0 and std 1"""
    new_dataset = {}
    for case_id in dataset.keys():
        case_data = dataset[case_id]
        # Take out only HRV values
        hrv_case_data = case_data[:, :, 0, :]
        temporal_features_case_data = case_data[:, :, 1:, :]
        # Normalize only HRV values
        means = np.nanmean(hrv_case_data, axis=-1)
        stds = np.nanstd(hrv_case_data, axis=-1)
        hrv_case_data = (hrv_case_data - means[..., np.newaxis]) / stds[..., np.newaxis]
        # Reassemble case data
        case_data = np.concatenate(
            (hrv_case_data[:, :, np.newaxis, :], temporal_features_case_data), axis=2
        )
        new_dataset[case_id] = case_data
    return new_dataset


def residual_dataset(dataset):
    """
    For each sample in the dataset, subtract the previous HRV value
    This is the first feature
    Shape of case data is (num_params, num_actors, num_features, num_samples)
    Args:
        dataset:

    Returns:
    """

    new_dataset = {}
    for case_id in dataset.keys():
        case_data = dataset[case_id]
        case_data = np.diff(case_data, axis=-1)
        new_dataset[case_id] = case_data
    return new_dataset


def prune_phases(dataset, start=2, end=6, time_interval=5):
    """
    Remove all data except between phases 2 and 6
    Args:
        dataset:
        time_interval:

    Returns:

    """
    phase_ids = get_phase_ids(time_interval=time_interval)
    new_dataset = {}
    for case_id in dataset.keys():
        case_data = dataset[case_id]
        start_idx = phase_ids[case_id][start][0]
        end_idx = phase_ids[case_id][end][1]
        new_dataset[case_id] = case_data[:, :, :, start_idx:end_idx]

    # Update global config
    config.phases = list(range(start, end + 1))
    config.num_phases = len(config.phases)
    config.update_config()

    return new_dataset


def prune_actors(dataset, actors_to_keep=["Surg"]):
    """
    Shape of case data is (num_params, num_actors, num_features, num_samples)
    Only keep indices of actors to keep
    Args:
        dataset:
        actors_to_remove:

    Returns:

    """

    # Update global variable of actor names
    actor_indices = [config.role_colors[actor] for actor in actors_to_keep]

    # Update globals config
    config.role_names = actors_to_keep
    config.num_actors = len(actors_to_keep)
    config.role_indices = {actor: i for i, actor in enumerate(actors_to_keep)}
    config.update_config()

    # Slice out data for relevant actors
    new_dataset = {}
    for case_id in dataset.keys():
        case_data = dataset[case_id]
        print(case_data.shape)
        case_data = case_data[:, actor_indices, :, :]
        if len(case_data.shape) == 3:
            case_data = case_data[:, np.newaxis, :, :]
        new_dataset[case_id] = case_data
    return new_dataset


def standardize_dataset(dataset):
    """
    For each parameter in the dataset, make the mean 0 and std 1 across all cases
    Return the mean and std for each parameter so it can be inverted at test time
    Args:
        dataset:

    Returns:

    """
    samples = {param: None for param in config.param_names}
    # Collect samples for each parameter
    for case_idx, case_data in dataset.items():
        for param_idx, param_data in enumerate(case_data):
            param_name = config.param_names[param_idx]
            # Concatenate samples across last axis
            if samples[param_name] is None:
                samples[param_name] = param_data
            else:
                samples[param_name] = np.concatenate(
                    (samples[param_name], param_data), axis=-1
                )
    # Compute dataset mean and variance for HRV only, which is first element in third dimension
    # We want separate means and variances for each actor and parameter
    # Shape of means should be (num_params, num_actors)
    dataset_means = {param: None for param in config.param_names}
    dataset_stds = {param: None for param in config.param_names}
    for param_name, param_data in samples.items():
        dataset_means[param_name] = np.nanmean(param_data[:, 0, :], axis=-1)
        dataset_stds[param_name] = np.nanstd(param_data[:, 0, :], axis=-1)

    # Standardize dataset
    new_dataset = {}
    for case_idx, case_data in dataset.items():
        new_case_data = {}
        for param_idx, param_data in enumerate(case_data):
            # Standardize HRV data only, which is first element in third dimension
            # For each param, mean and std are of shape (num_actors,)
            param_name = config.param_names[param_idx]
            new_case_data[param_name] = param_data
            # Transpose to make it easier to broadcast (4, 8, 49) -> (8, 49, 4)
            new_case_data[param_name] = np.transpose(
                new_case_data[param_name], (1, 2, 0)
            )
            # Subtract mean from every row
            new_case_data[param_name][0] -= dataset_means[param_name]
            new_case_data[param_name][0] /= dataset_stds[param_name]
            # Transpose back
            new_case_data[param_name] = np.transpose(
                new_case_data[param_name], (2, 0, 1)
            )
        # Concatenate samples across all params where each is (num_actors, num_features, num_samples)
        # Result is (num_params, num_actors, num_features, num_samples)
        all_param_data = np.array(
            [new_case_data[param] for param in config.param_names]
        )
        new_dataset[case_idx] = all_param_data
    return new_dataset, dataset_means, dataset_stds


def rescale_standardized_predictions(trace, dataset_means, dataset_stds):
    for param_name, param_trace in trace.items():
        for case_idx, case_predictions in param_trace["predictions"].items():
            case_predictions = np.array(case_predictions)
            case_predictions = (
                case_predictions * dataset_stds[param_name] + dataset_means[param_name]
            )
            trace[param_name]["predictions"][case_idx] = case_predictions
    return trace
