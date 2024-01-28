import pandas as pd
import numpy as np
import torch
from utils.stats import get_lengths, get_phase_ids
from typing import Union
from utils.config import config


def make_static_features(
    data_dir="./data", return_type="np", unroll_through_time=False, lengths=None
):
    """Make static features for time-series models
    We will use procedure type, number of vessels, 30-day mortality,
    180-day mortality, 30-day morbidity, 30-day SSI, and the IDs of the
    anesthesiologist, perfusionist, surgeon, and nurse

    Args:
        data_dir (str): the directory containing the data
        return_type (str): Specifies whether to return a numpy array or torch Tensor. Defaults to "np".

    Returns:
        dict: a dictionary of static features
    """
    static_features = {}
    surg_procedure_encoding = {"CABG": 0, "AVR": 1, "min. inv. AVR": 2, "AVR/CABG": 3}
    metadata = pd.read_excel(
        f"{data_dir}/metadata-for-statisticians-2022-10-04.xlsx", header=1
    )
    for idx, row in metadata.iterrows():
        case_id = int(row["Case ID"].split("_")[1])
        if case_id in config.valid_cases:
            procedure_type = surg_procedure_encoding[row["Procedure Type"]]
            no_vessels = 0 if pd.isna(row["No. Vessels"]) else int(row["No. Vessels"])
            day_mort_30 = round(row["30 Day Mort."] * 100, 2)
            day_mort_180 = round(row["180 Day Mort."] * 100, 2)
            day_morb_30 = round(row["30 Day Morb."] * 100, 2)
            day_ssi_30 = round(row["30 Day SSI"] * 100, 2)
            anes_id = int(row["Anesthesia Code"][-2:])
            perf_id = int(row["Perfusionist Code"][-2:])
            surg_id = int(row["Surgeon Code"][-2:])
            nurs_id = int(row["Nurse Code"][-2:])
            features = (
                one_hot(procedure_type, 4)
                + [
                    no_vessels,
                    day_mort_30,
                    day_mort_180,
                    day_morb_30,
                    day_ssi_30,
                ]
                + one_hot(anes_id, 5, zero_based=False)
                + one_hot(perf_id, 5, zero_based=False)
                + one_hot(surg_id, 3, zero_based=False)
                # + one_hot(nurs_id, 18) # 18 categories is very long and some are never in the data, skipping this feature for now
            )
            if return_type == "np":
                static_features[case_id] = np.array(features)
            elif return_type == "torch":
                static_features[case_id] = torch.Tensor(features)
            else:
                raise ValueError(
                    f"Invalid return type {return_type}. Must be 'np' or 'torch'"
                )

    if unroll_through_time:
        if lengths is None:
            raise ValueError(
                "Lengths must be provided if unrolling static features through time"
            )
        return unroll_static_features(static_features, lengths, return_type)
    else:
        return static_features


def unroll_static_features(static_features, lengths, return_type="np"):
    static_feature_dict = {}
    for case_id in lengths.keys():
        if return_type == "np":
            static_feature_dict[case_id] = np.tile(
                static_features[case_id], (lengths[case_id], 1)
            ).transpose()
        elif return_type == "torch":
            static_feature_dict[case_id] = torch.tile(
                static_features[case_id], (lengths[case_id], 1)
            ).transpose()
        else:
            raise ValueError(
                f"Invalid return type {return_type}. Must be 'np' or 'torch'"
            )
    return static_feature_dict


def add_temporal_features(dataset, pad_phase_on=False):
    # Look at the first key in the dataset to determine the return type
    if isinstance(dataset[list(dataset.keys())[0]], np.ndarray):
        return_type = "np"
    elif isinstance(dataset[list(dataset.keys())[0]], torch.Tensor):
        return_type = "torch"
    else:
        raise ValueError(
            f"Invalid dataset type {type(dataset)}. Must be np.ndarray or torch.Tensor"
        )

    # Make the temporal features
    temporal_features = make_temporal_features(
        dataset, return_type=return_type, pad_phase_on=pad_phase_on
    )

    # Temporal features are identical for every parameter and actor
    # Dataset shape is (num_params, num_actors, num_samples)
    # Temporal features shape is (num_temporal_features, num_samples)
    # The returned dataset should have shape (num_params, num_actors, num_temporal_features + 1, num_samples)
    new_dataset = {}

    for case_id in dataset.keys():
        case_data = dataset[case_id]

        # If dataset shape is (num_params, num_actors, num_samples)
        # Add an axis such that the shape is now (num_params, num_actors, 1, num_samples)
        if len(case_data.shape) == 3:
            if return_type == "np":
                case_data = np.expand_dims(case_data, axis=2)
            else:
                case_data = case_data.unsqueeze(2)

        # Dataset shape should be (num_params, num_actors, num_features, num_samples)
        if return_type == "np":
            new_dataset[case_id] = np.concatenate(
                (
                    case_data,
                    np.tile(
                        temporal_features[case_id],
                        (case_data.shape[0], case_data.shape[1], 1, 1),
                    ),
                ),
                axis=2,
            )
        elif return_type == "torch":
            new_dataset[case_id] = torch.cat(
                (
                    case_data,
                    torch.tile(
                        temporal_features[case_id],
                        (case_data.shape[0], case_data.shape[1], 1, 1),
                    ),
                ),
                dim=2,
            )

    return new_dataset


def add_static_features(dataset):
    # Look at the first key in the dataset to determine the return type
    if isinstance(dataset[list(dataset.keys())[0]], np.ndarray):
        return_type = "np"
    elif isinstance(dataset[list(dataset.keys())[0]], torch.Tensor):
        return_type = "torch"
    else:
        raise ValueError(
            f"Invalid dataset type {type(dataset)}. Must be np.ndarray or torch.Tensor"
        )

    # Make the static features
    static_features = make_static_features(
        return_type=return_type, unroll_through_time=True, lengths=get_lengths(dataset)
    )

    # Static features are identical for every parameter and actor
    # Dataset shape can be (num_params, num_actors, num_samples) or (num_params, num_actors, num_features, num_samples)
    # Temporal features shape is (num_static_features, num_samples)
    # The returned dataset should have shape (num_params, num_actors, num_features + num_static_features, num_samples)

    new_dataset = {}

    for case_id in dataset.keys():
        case_data = dataset[case_id]

        # If dataset shape is (num_params, num_actors, num_samples)
        # Add an axis such that the shape is now (num_params, num_actors, 1, num_samples)
        if len(case_data.shape) == 3:
            if return_type == "np":
                case_data = np.expand_dims(case_data, axis=2)
            else:
                case_data = case_data.unsqueeze(2)

        # Dataset shape should be (num_params, num_actors, num_features, num_samples)
        if return_type == "np":
            new_dataset[case_id] = np.concatenate(
                (
                    case_data,
                    np.tile(
                        static_features[case_id],
                        (case_data.shape[0], case_data.shape[1], 1, 1),
                    ),
                ),
                axis=2,
            )
        elif return_type == "torch":
            new_dataset[case_id] = torch.cat(
                (
                    case_data,
                    torch.tile(
                        static_features[case_id],
                        (case_data.shape[0], case_data.shape[1], 1, 1),
                    ),
                ),
                dim=2,
            )

    return new_dataset


def combine_feature_sets(feature_dicts: tuple, keys: list, return_type="np"):
    """Combine feature vectors from the different dictionaries of feature sets
    Used to append any combination of static, temporal, and HRV features

    Args:
        feature_dicts (tuple): a tuple of dictionaries of feature vectors
    """
    output_dict = {}
    for feature_dict in feature_dicts:
        if not isinstance(feature_dict, dict):
            raise ValueError(
                f"Expected a dictionary of features but got {type(feature_dict)}"
            )
        for key in keys:
            if key not in feature_dict:
                raise ValueError(f"Key {key} not found in feature dictionary")
            if key not in output_dict:
                output_dict[key] = []
            output_dict[key].append(feature_dict[key])

    for key in keys:
        if return_type == "np":
            output_dict[key] = np.hstack(output_dict[key])
        elif return_type == "torch":
            output_dict[key] = torch.cat(output_dict[key], dim=0)
        else:
            raise ValueError(
                f"Invalid return type {return_type}. Must be 'np' or 'torch'"
            )

    return output_dict


def make_temporal_features(
    dataset, time_interval=5, return_type="np", pad_phase_on=False
):
    """Make temporal features for time-series models
    We will use phase ID and time within the surgery as features

    Args:
        dataset (dict): a dictionary of HRV parameter values
        time_interval (str): the time interval to use for temporal features
        num_phases (int): the number of phases to use for temporal features
        return_type (str): Specifies whether to return a numpy array or torch tensor. Defaults to "np".
    """

    phase_ids = get_phase_ids(time_interval=time_interval)
    temporal_features = {}
    for case in dataset.keys():
        # For every case, create a vector of features for each time step
        # Set the last feature to be the time within the surgery
        if return_type == "np":
            features = np.zeros((config.num_phases + 2, dataset[case].shape[-1]))
            features[-1, :] = np.array([j for j in range(dataset[case].shape[-1])])
        else:
            features = torch.zeros((config.num_phases + 2, dataset[case].shape[-1]))
            features[-1, :] = torch.Tensor([j for j in range(dataset[case].shape[-1])])

        # The remaining features are a one-hot vector indicating which phase is active at each time step
        # The last phase indicates that no phase is active
        features[:-1, config.num_phases + 1] = 1

        for phase in range(config.num_phases):
            phase_start = phase_ids[case][phase][0]
            phase_end = phase_ids[case][phase][1]
            if phase_end is not None and phase_start is not None:
                if pad_phase_on:
                    # The feature is always 1 for any timestep after the phase starts
                    features[phase, phase_start:] = 1
                else:
                    # The feature is 1 only when the phase is active and 0 when it finishes
                    features[phase, phase_start:phase_end] = 1
                # A phase was active so set the no phase feature to 0
                features[config.num_phases, phase_start:phase_end] = 0
        temporal_features[case] = features
        # # Stack features for each of the four actors
        # if return_type == "np":
        #     temporal_features[case] = np.stack(
        #         [features] * dataset[case].shape[0], axis=0
        #     )
        # elif return_type == "torch":
        #     temporal_features[case] = torch.stack(
        #         [features] * dataset[case].shape[0], axis=0
        #     )

    return temporal_features


def one_hot(
    idx: Union[int, list],
    num_categories: int,
    zero_based=True,
    return_type: str = "list",
):
    """Create a one-hot vector for the given categorical feature
    Supports both zero-based and one-based indexing, and multiclass features

    Args:
        idx (int, list): the index or list of indices of the categories
        num_categories (int): the number of categories
        zero_based (bool): whether the index is zero-based or one-based
        return_type (str): Specifies whether to return a numpy array, torch tensor, or list. Defaults to "list".

    Returns:
        list: a one-hot vector
    """
    # Create vector of zeros
    if return_type == "list":
        one_hot = [0] * num_categories
    elif return_type == "np":
        one_hot = np.zeros(num_categories)
    elif return_type == "torch":
        one_hot = torch.zeros(num_categories)

    # Assign value 1 to the given index
    if isinstance(idx, int):
        if zero_based:
            one_hot[idx] = 1
        else:
            one_hot[idx - 1] = 1
        return one_hot
    elif (
        isinstance(idx, list)
        or isinstance(idx, np.ndarray)
        or isinstance(idx, torch.Tensor)
    ):
        for i in idx:
            if zero_based:
                one_hot[i] = 1
            else:
                one_hot[i - 1] = 1
        return one_hot

    else:
        raise ValueError(f"Indices should be of type int or list, got {type(idx)}")
