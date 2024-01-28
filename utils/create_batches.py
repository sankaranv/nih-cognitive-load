from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import numpy as np
import random
from utils.config import config


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

    return train_dataset, val_dataset, test_dataset


class HRVDataset(Dataset):
    """
    Dataset for HRV data for a single parameter
    """

    def __init__(self, dataset, seq_len, pred_len, param):
        self.dataset = dataset
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.param = param
        sequences = self.get_sequences_from_dict()
        self.input_output_pairs = self.create_input_output_pairs(sequences)

    def get_sequences_from_dict(self):
        """
        Get sequences for the given parameter from the dataset
        Returns:
            List of sequences of shape (num_actors, num_features, seq_len)
        """
        sequences = []
        for key in self.dataset.keys():
            case_param_data = self.dataset[key][config.param_indices[self.param]]
            sequences.append(torch.Tensor(case_param_data))

        return sequences

    def create_input_output_pairs(self, sequences):
        """
        Create input-output pairs for each sequence in the dataset
        Inputs are of shape (num_actors + num_temporal_features, seq_len)
        Outputs are of shape (num_actors, pred_len)
        Args:
            sequences: List of sequences of shape (4, num_features, seq_len)

        Returns:
            List of input-output pairs, where each pair is a tuple of (inputs, outputs)
        """
        input_output_pairs = []
        for sequence in sequences:
            case_seq_length = sequence.shape[-1]
            assert case_seq_length >= self.seq_len + self.pred_len, (
                f"Cannot create input-output pairs for sequence of length {case_seq_length} "
                f"using seq_len {self.seq_len} and pred_len {self.pred_len}"
            )
            for i in range(case_seq_length - self.seq_len - self.pred_len):
                input_features = sequence[:, :, i : i + self.seq_len]
                output_features = sequence[
                    :, :, i + self.seq_len : i + self.seq_len + self.pred_len
                ]
                # Input features are of shape (4, 11, seq_len)
                # First component of second dim is HRV value, remaining are temporal features
                input_hrv_values = input_features[:, 0, :].reshape(-1, self.seq_len)
                temporal_features = input_features[0, 1:, :]
                # Concatenate to get input of shape (14, seq_len)
                inputs = torch.cat((input_hrv_values, temporal_features), dim=0)

                # For outputs, we just want 4 HRV values
                target = output_features[:, 0, :].reshape(-1, self.pred_len)
                # Shape of inputs is now (14, 90)
                input_output_pairs.append((inputs, target))
        # Convert to torch tensors
        input_output_pairs = [
            (torch.Tensor(x), torch.Tensor(y)) for x, y in input_output_pairs
        ]
        return input_output_pairs

    def __len__(self):
        return len(self.input_output_pairs)

    def __getitem__(self, idx):
        return self.input_output_pairs[idx]


def create_torch_loader_from_dataset(
    dataset, seq_len, pred_len, param, batch_size, shuffle=True
):
    # Create train, test, val splits
    hrv_dataset = HRVDataset(dataset, seq_len, pred_len, param)
    data_loader = DataLoader(hrv_dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def create_input_output_pairs(dataset, lag_length):
    input_output_pairs = {}
    for key in config.param_names:
        input_output_pairs[key] = []
    for case_idx, case_data in dataset.items():
        # Shape of the data is (num_params, num_actors, num_features, num_timesteps)
        for param_idx, param_data in enumerate(case_data):
            # Shape of the data is (num_actors, num_features, num_timesteps)
            # Timestep axis is moved to the front for broadcasting
            num_timesteps = param_data.shape[-1]
            param_data = np.transpose(param_data, (2, 0, 1))
            # Check if there are any NaNs in any feature for any of the actors
            # Shape of the mask is (num_timesteps,)
            mask = np.all(~np.isnan(param_data), axis=(1, 2))
            param_data = np.transpose(param_data, (1, 2, 0))
            # Use the mask to check if there are any missing values in the last L + 1 timesteps
            for i in range(num_timesteps - lag_length - 1):
                if np.all(mask[i : i + lag_length + 1]):
                    # If there are no missing values, add to training set
                    # The first L steps are used as input and the last one is used as output
                    key = config.param_names[param_idx]
                    input_vector = param_data[:, :, i : i + lag_length]
                    output_vector = param_data[:, :, i + lag_length]
                    assert np.all(~np.isnan(output_vector))
                    input_output_pairs[key].append((input_vector, output_vector))
    return input_output_pairs


def create_data_pairs(dataset, lag_length, param):
    input_output_pairs = create_input_output_pairs(dataset, lag_length)
    # Create training data where X is of shape (num_samples, num_actors * lag_length + num_temporal_features)
    # and y is of shape (num_samples, num_actors)

    X = []
    y = []
    for input_vector, output_vector in input_output_pairs[param]:
        # Shape of input_vector is (num_actors, num_features, lag_length)
        # We only want the first feature from the second axis
        # Take one copy of remaining features from the last element along the third axis
        # Concat them together
        # Shape of output_vector is (num_actors, num_features)
        # We want to reshape input_vector to (num_actors * num_features, lag_length)
        # and output_vector to (num_actors * num_features,)
        hrv_features = input_vector[:, 0, :].reshape(-1)
        temporal_features = input_vector[:, 1:, -1].reshape(-1)
        new_input_vector = np.concatenate((hrv_features, temporal_features), axis=0)
        X.append(new_input_vector)
        y.append(output_vector[:, 0].reshape(-1))
    X = np.array(X)
    y = np.array(y)
    return X, y
