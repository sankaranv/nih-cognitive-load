import numpy as np
from utils.config import config
from utils.stats import (
    compute_regression_metrics,
    compute_classification_metrics,
    get_means,
    get_stddevs,
)


class ParameterFreeAutoregressiveModel:
    def __init__(self, model_config):
        self.seq_len = model_config["seq_len"]
        self.pred_len = model_config["pred_len"]
        self.model_name = "ParameterFreeAutoregressiveModel"

    def contains_missing_data(self, dataset):
        for case_idx, case_data in dataset.items():
            if np.isnan(case_data).any():
                return True
        return False

    def train(self, train_dataset, val_dataset=None, verbose=False):
        trace = {}
        return trace

    def predict(self, test_dataset, trace=None):
        if self.contains_missing_data(test_dataset):
            raise ValueError("Test dataset contains missing data")

        if trace is None or len(trace) == 0:
            trace = {param: {} for param in config.param_names}
        for param_name in config.param_names:
            trace[param_name]["predictions"] = {}
            trace[param_name]["mean_squared_error"] = {}
            trace[param_name]["rms_error"] = {}
            trace[param_name]["mean_absolute_error"] = {}
            trace[param_name]["r_squared"] = {}
            trace[param_name]["corr_coef"] = {}
            for case_idx, case_data in test_dataset.items():
                # Get HRV data for the given parameter
                param_idx = config.param_indices[param_name]
                param_data = case_data[param_idx, :, 0, :].transpose()
                # Use previous timestep to get the next timestep
                # Start making predictions from seq_len timestep onwards
                predictions = param_data[self.seq_len - self.pred_len : -self.pred_len]
                # Get ground truth
                y = param_data[self.seq_len :]
                trace[param_name]["predictions"][case_idx] = predictions
                # Compute metrics
                metrics = compute_regression_metrics(y, predictions)
                trace[param_name]["mean_squared_error"][case_idx] = metrics[
                    "mean_squared_error"
                ]
                trace[param_name]["rms_error"][case_idx] = metrics["rms_error"]
                trace[param_name]["mean_absolute_error"][case_idx] = metrics[
                    "mean_absolute_error"
                ]
                trace[param_name]["r_squared"][case_idx] = metrics["r_squared"]
                trace[param_name]["corr_coef"][case_idx] = metrics["corr_coef"]
        return trace


class RandomRegressionModel:
    def __init__(self, model_config):
        self.seq_len = model_config["seq_len"]
        self.pred_len = model_config["pred_len"]
        self.model_name = "RandomRegressor"

    def contains_missing_data(self, dataset):
        for case_idx, case_data in dataset.items():
            if np.isnan(case_data).any():
                return True
        return False

    def train(self, train_dataset, val_dataset=None, verbose=False):
        # Compute mean and std for the dataset
        self.means = get_means(train_dataset)
        self.stddevs = get_stddevs(train_dataset)
        trace = {}
        return trace

    def predict(self, test_dataset, trace=None):
        if self.contains_missing_data(test_dataset):
            raise ValueError("Test dataset contains missing data")

        if trace is None or len(trace) == 0:
            trace = {param: {} for param in config.param_names}
        for param_name in config.param_names:
            trace[param_name]["predictions"] = {}
            trace[param_name]["mean_squared_error"] = {}
            trace[param_name]["rms_error"] = {}
            trace[param_name]["mean_absolute_error"] = {}
            trace[param_name]["r_squared"] = {}
            trace[param_name]["corr_coef"] = {}
            for case_idx, case_data in test_dataset.items():
                # Get HRV data for the given parameter
                param_idx = config.param_indices[param_name]
                param_data = case_data[param_idx, :, 0, :].transpose()
                # Get ground truth
                y = param_data[self.seq_len :]
                # Make random predictions by sampling from normal with dataset mean and std
                param_mean = np.zeros(y.shape[-1])
                param_stddev = np.ones(y.shape[-1])
                predictions = np.random.normal(param_mean, param_stddev, y.shape)
                # Store predictions
                trace[param_name]["predictions"][case_idx] = predictions
                # Compute metrics
                metrics = compute_regression_metrics(y, predictions)
                trace[param_name]["mean_squared_error"][case_idx] = metrics[
                    "mean_squared_error"
                ]
                trace[param_name]["rms_error"][case_idx] = metrics["rms_error"]
                trace[param_name]["mean_absolute_error"][case_idx] = metrics[
                    "mean_absolute_error"
                ]
                trace[param_name]["r_squared"][case_idx] = metrics["r_squared"]
                trace[param_name]["corr_coef"][case_idx] = metrics["corr_coef"]
        return trace


class ParameterFreeAutoregressiveClassifier(ParameterFreeAutoregressiveModel):
    def __init__(self, model_config):
        super().__init__(model_config)
        self.model_name = "ParameterFreeAutoregressiveClassifier"

    def predict(self, test_dataset, trace=None):
        if self.contains_missing_data(test_dataset):
            raise ValueError("Test dataset contains missing data")

        if trace is None or len(trace) == 0:
            trace = {param: {} for param in config.param_names}
        for param_name in config.param_names:
            trace[param_name]["predictions"] = {}
            trace[param_name]["accuracy"] = {}
            trace[param_name]["corr_coef"] = {}
            for case_idx, case_data in test_dataset.items():
                # Get HRV data for the given parameter
                param_idx = config.param_indices[param_name]
                param_data = case_data[param_idx, :, 0, :].transpose()
                # Use previous timestep to get the next timestep
                # Start making predictions from seq_len timestep onwards
                predictions = param_data[self.seq_len - self.pred_len : -self.pred_len]
                # Get ground truth
                y = param_data[self.seq_len :]
                trace[param_name]["predictions"][case_idx] = predictions
                # Compute metrics
                metrics = compute_classification_metrics(y, predictions)
                trace[param_name]["accuracy"][case_idx] = metrics["accuracy"]
                trace[param_name]["corr_coef"][case_idx] = metrics["corr_coef"]
        return trace
