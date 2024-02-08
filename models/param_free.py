import numpy as np
from utils.config import config
from utils.stats import compute_metrics


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
                metrics = compute_metrics(y, predictions)
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
