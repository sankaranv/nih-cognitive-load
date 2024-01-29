from models.mlp import MLP
from utils.config import config
from utils.create_batches import create_torch_loader_from_dataset
import utils.training as train_utils
import torch
import torch.nn as nn
from utils.stats import compute_metrics
import numpy as np
class JointNNModel:

    def __init__(self, model_config):
        self.base_model = MLP
        self.model_name = model_config["model_name"]
        self.seq_len = model_config["seq_len"]
        self.pred_len = model_config["pred_len"]
        self.num_epochs = model_config["num_epochs"]
        self.model_config = model_config
        self.setup_models()

    def setup_models(self):
        self.models = {}
        self.optimizer = {}
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for param in config.param_names:
            self.models[param] = self.base_model(model_config=self.model_config).to(self.device)
            self.optimizer[param] = torch.optim.Adam(self.models[param].parameters(), lr=self.model_config["lr"])


    def train(self, train_dataset, val_dataset=None, verbose=False):
        # Train joint models for each parameter
        trace = {}
        for param in config.param_names:
            print(f"Training {self.model_name} for {param}")
            train_loader = create_torch_loader_from_dataset(train_dataset,
                                                            self.seq_len,
                                                            self.pred_len,
                                                            param,
                                                            self.model_config["batch_size"],
                                                            shuffle=True)
            if val_dataset is not None:
                val_loader = create_torch_loader_from_dataset(val_dataset,
                                                              self.seq_len,
                                                              self.pred_len,
                                                              param,
                                                              self.model_config["batch_size"],
                                                              shuffle=False)
            else:
                val_loader = None
            trace[param] = train_utils.train(self.models[param], train_loader, val_loader, self.num_epochs,
                                 self.optimizer[param], self.criterion, self.device)
        return trace

    def predict(self, test_dataset, trace=None):
        if trace is None or len(trace) == 0:
            trace = {param: {} for param in config.param_names}
        case_ids = list(test_dataset.keys())
        for param in config.param_names:
            param_idx = config.param_indices[param]
            test_loader = create_torch_loader_from_dataset(test_dataset,
                                                           self.seq_len,
                                                           self.pred_len,
                                                           param,
                                                           self.model_config["batch_size"],
                                                           shuffle=False)
            test_loss = train_utils.test(self.models[param], test_loader, self.optimizer[param], self.criterion, self.device)
            trace[param]["test_loss"] = test_loss
            predictions = train_utils.predict(self.models[param], test_loader, self.device)
            # Check if there are any NaNs in the predictions
            assert torch.all(
                ~torch.isnan(predictions)
            ), f"Predictions for {param} contain NaNs"
            # Split predictions by case ID to add to trace
            trace[param]["predictions"] = {}
            trace[param]["mean_squared_error"] = {}
            trace[param]["rms_error"] = {}
            trace[param]["mean_absolute_error"] = {}
            trace[param]["r_squared"] = {}
            trace[param]["corr_coef"] = {}
            idx = 0
            for case_idx in case_ids:
                num_samples = test_dataset[case_idx].shape[-1]
                offset = self.seq_len + self.pred_len
                case_predictions = predictions[idx: idx + num_samples - offset].squeeze().detach().cpu().numpy()
                idx += num_samples - offset
                trace[param]["predictions"][case_idx] = case_predictions
                y = test_dataset[case_idx][param_idx, :, 0, self.seq_len:-self.pred_len].transpose()
                # Get metrics for each case
                metrics = compute_metrics(y, case_predictions)
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