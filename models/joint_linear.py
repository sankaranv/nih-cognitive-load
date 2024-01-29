from sklearn.linear_model import Ridge
from utils.config import config
from utils.stats import compute_metrics
import numpy as np


class JointLinearModel:
    def __init__(self, model_config):
        self.base_model = Ridge
        self.model_name = model_config["model_name"]
        self.seq_len = model_config["seq_len"]
        self.pred_len = model_config["pred_len"]
        self.setup_models()

    def setup_models(self):
        self.models = {}
        for param in config.param_names:
            self.models[param] = self.base_model()

    def create_input_output_pairs(self, dataset, param):
        input_output_pairs = []
        for case_idx, case_data in dataset.items():
            # Shape of the data is (num_params, num_actors, num_features, num_timesteps)
            param_idx = config.param_indices[param]
            param_data = case_data[param_idx]
            # Shape of the data is (num_actors, num_features, num_timesteps)
            # Timestep axis is moved to the front for broadcasting
            num_timesteps = param_data.shape[-1]
            param_data = np.transpose(param_data, (2, 0, 1))
            # Check if there are any NaNs in any feature for any of the actors
            # Shape of the mask is (num_timesteps,)
            mask = np.all(~np.isnan(param_data), axis=(1, 2))
            param_data = np.transpose(param_data, (1, 2, 0))
            # Use the mask to check if there are any missing values in the last L + 1 timesteps
            for i in range(num_timesteps - self.seq_len - self.pred_len + 1):
                if np.all(mask[i : i + self.seq_len + 1]):
                    # If there are no missing values, add to training set
                    # The first L steps are used as input and the last one is used as output
                    key = config.param_names[param_idx]
                    input_vector = param_data[:, :, i : i + self.seq_len]
                    output_vector = param_data[:, :, i + self.seq_len]
                    assert np.all(~np.isnan(output_vector))
                    input_output_pairs.append((input_vector, output_vector))
        return input_output_pairs

    def create_train_dataset(self, dataset, param):
        # Concat HRV values for all actors across seq_len timesteps with temporal features for timestep t-1
        # Input-output pairs are (num_actors, (seq_len + num_temporal_features), num_params) and (num_actors, num_params)
        # Create vectors of length (num_actors * (seq_len + num_temporal_features)) and (num_actors) for the given param
        input_output_pairs = self.create_input_output_pairs(dataset, param)
        X = []
        y = []
        for sample in input_output_pairs:
            target = sample[1][:, 0].reshape(-1)
            input_vector = sample[0][:, 0].reshape(-1)
            # Add temporal features
            actor_idx = 0
            input_vector = np.append(
                input_vector, sample[0][actor_idx, 1:, -1].reshape(-1)
            )
            X.append(input_vector)
            y.append(target)
        return np.array(X), np.array(y)

    def create_test_dataset(self, case_data, param):
        input_output_pairs = []
        X = []
        y = []
        param_idx = config.param_indices[param]
        param_data = case_data[param_idx]
        # Shape of the data is (num_actors, num_features, num_timesteps)
        # Timestep axis is moved to the front for broadcasting
        num_timesteps = param_data.shape[-1]
        param_data = np.transpose(param_data, (2, 0, 1))
        # Check if there are any NaNs in any feature for any of the actors
        # Shape of the mask is (num_timesteps,)
        mask = np.all(~np.isnan(param_data), axis=(1, 2))
        param_data = np.transpose(param_data, (1, 2, 0))
        # Use the mask to check if there are any missing values in the last L + 1 timesteps
        for i in range(num_timesteps - self.seq_len - self.pred_len + 1):
            if np.all(mask[i : i + self.seq_len + 1]):
                # If there are no missing values, add to training set
                # The first L steps are used as input and the last one is used as output
                key = config.param_names[param_idx]
                input_vector = param_data[:, :, i : i + self.seq_len]
                output_vector = param_data[:, :, i + self.seq_len]
                assert np.all(~np.isnan(output_vector))
                input_output_pairs.append((input_vector, output_vector))
        for sample in input_output_pairs:
            target = sample[1][:, 0].reshape(-1)
            input_vector = sample[0][:, 0].reshape(-1)
            # Add temporal features
            actor_idx = 0
            input_vector = np.append(
                input_vector, sample[0][actor_idx, 1:, -1].reshape(-1)
            )
            X.append(input_vector)
            y.append(target)
        return np.array(X), np.array(y)

    def train(self, train_dataset, val_dataset=None, verbose=False):
        trace = {}
        # Train joint models for each parameter
        for param in config.param_names:
            print(f"Training {self.model_name} for {param}")
            X, y = self.create_train_dataset(train_dataset, param)
            self.models[param].fit(X, y)
        return trace

    def predict(self, test_dataset, trace = None):
        if trace is None or len(trace) == 0:
            trace = {param: {} for param in config.param_names}
        # Predict HRV for each case and each param
        for param in config.param_names:
            trace[param]["predictions"] =  {}
            trace[param]["mean_squared_error"] = {}
            trace[param]["rms_error"] = {}
            trace[param]["mean_absolute_error"] = {}
            trace[param]["r_squared"] = {}
            trace[param]["corr_coef"] = {}
            for case_idx, case_data in test_dataset.items():
                # Create input-output pairs for the given parameter
                X, y = self.create_test_dataset(case_data, param)
                # Check if no data was returned
                if len(X) == 0 or len(y) == 0:
                    continue
                # Predict HRV for each actor
                predictions = self.models[param].predict(X)
                # Check if there are any NaNs in the predictions
                assert np.all(
                    ~np.isnan(predictions)
                ), f"Predictions for {param} in case {case_idx} contain NaNs"
                trace[param]["predictions"][case_idx] = predictions
                # Get metrics for each case
                metrics = compute_metrics(y, predictions)
                trace[param]["mean_squared_error"][case_idx] = metrics[
                    "mean_squared_error"
                ]
                trace[param]["rms_error"][case_idx] = metrics["rms_error"]
                trace[param]["mean_absolute_error"][case_idx] = metrics[
                    "mean_absolute_error"
                ]
                trace[param]["r_squared"][case_idx] = metrics["r_squared"]
                trace[param]["corr_coef"][case_idx] = metrics["corr_coef"]
        return trace
