import numpy as np
from utils.config import config
from utils.stats import compute_metrics


class ParameterFreeAutoregressiveModel:
    def __init__(self, model_config):
        self.seq_len = 1
        self.pred_len = 1
        self.model_name = "ParameterFreeAutoregressiveModel"

    def contains_missing_data(self, dataset):
        for case_idx, case_data in dataset.items():
            if np.isnan(case_data).any():
                return True
        return False

    def train(self, train_dataset, verbose=False):
        pass

    def predict(self, test_dataset):
        if self.contains_missing_data(test_dataset):
            raise ValueError("Test dataset contains missing data")

        trace = {}
        for case_idx, case_data in test_dataset.items():
            trace[case_idx] = {}
            for param_name in config.param_names:
                trace[case_idx][param_name] = {}
                # Get HRV data for the given parameter
                param_idx = config.param_indices[param_name]
                param_data = case_data[param_idx, :, 0, :].transpose()
                # Use previous timestep to get the next timestep
                y = param_data[1:]
                predictions = param_data[:-1]
                trace[case_idx][param_name]["predictions"] = predictions
                # Compute metrics
                metrics = compute_metrics(y, predictions)
                trace[case_idx][param_name]["mean_squared_error"] = metrics[
                    "mean_squared_error"
                ]
                trace[case_idx][param_name]["rms_error"] = metrics["rms_error"]
                trace[case_idx][param_name]["mean_absolute_error"] = metrics[
                    "mean_absolute_error"
                ]
                trace[case_idx][param_name]["r_squared"] = metrics["r_squared"]
                trace[case_idx][param_name]["corr_coef"] = metrics["corr_coef"]
        return trace
