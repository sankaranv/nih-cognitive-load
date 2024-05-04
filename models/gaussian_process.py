import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
    IndependentMultitaskVariationalStrategy,
)
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from utils.config import config
import numpy as np
import pickle
import os
from utils.stats import compute_regression_metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau


class MultitaskVariationalGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([config.num_actors])
        )
        variational_strategy = IndependentMultitaskVariationalStrategy(
            VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=config.num_actors,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([config.num_actors])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([config.num_actors])),
            batch_shape=torch.Size([config.num_actors]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
        #     gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        # )
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class VariationalGP:
    def __init__(self, model_config):
        self.base_model = model_config["base_model"]
        self.model_name = model_config["model_name"]
        self.seq_len = model_config["seq_len"]
        self.pred_len = 1
        self.num_actors = len(config.role_names)
        self.model_config = model_config
        self.num_inducing_points = 50
        self.trained = False
        # Setup models
        self.models = {}
        self.setup_models()

    def setup_models(self):
        inducing_points = torch.rand(
            config.num_actors,
            self.num_inducing_points,
            config.num_actors * self.seq_len + config.num_phases + 2,
        )
        self.likelihoods = {}
        self.vi_params = {}
        self.optimizers = {}

        for param in config.param_names:
            self.models[param] = MultitaskVariationalGPModel(inducing_points)
            self.likelihoods[param] = MultitaskGaussianLikelihood(
                num_tasks=config.num_actors
            )
            self.vi_params[param] = [{"params": self.models[param].parameters()}]
            self.optimizers[param] = torch.optim.Adam(
                self.vi_params[param], lr=self.model_config["lr"]
            )

    def load(self, filename):
        self.models = pickle.load(open(filename, "rb"))
        self.trained = True

    def save(self, save_path):
        if not self.trained:
            raise ValueError("Dependency network is not trained, cannot save models")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(os.path.dirname(os.path.join(save_path, "anomaly"))):
            os.makedirs(os.path.dirname(os.path.join(save_path, "anomaly")))
        # Get first key from models
        param_key = list(self.models.keys())[0]
        role_key = list(self.models[param_key].keys())[0]
        model_name = self.models[param_key][role_key].__class__.__name__
        if not os.path.exists(os.path.join(save_path, "anomaly")):
            os.makedirs(os.path.join(save_path, "anomaly"))
        pickle.dump(
            self.models,
            open(f"{save_path}/anomaly/{model_name}_dependency_network.pkl", "wb"),
        )

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
        actors = [config.role_colors[actor] for actor in config.role_names]
        for sample in input_output_pairs:
            target = sample[1][actors, 0].reshape(-1)
            input_vector = sample[0][actors, 0].reshape(-1)
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
        actors = [config.role_colors[actor] for actor in config.role_names]
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
                input_vector = param_data[actors, :, i : i + self.seq_len]
                output_vector = param_data[actors, :, i + self.seq_len]
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
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            self.models[param].train()
            self.likelihoods[param].train()
            mll = gpytorch.mlls.VariationalELBO(
                self.likelihoods[param], self.models[param], num_data=X.shape[0]
            )
            scheduler = ReduceLROnPlateau(
                self.optimizers[param], "min", factor=0.5, patience=5, verbose=True
            )
            for i in range(self.model_config["num_epochs"]):
                self.optimizers[param].zero_grad()
                output = self.models[param](X)
                loss = -mll(output, y)
                loss.backward()
                self.optimizers[param].step()
                scheduler.step(loss)
                # if verbose:
                #     print(
                #         f"Epoch {i+1}/{self.model_config['num_epochs']}, Loss: {loss.item()}"
                #     )

    def predict(self, test_dataset, trace=None):
        if trace is None or len(trace) == 0:
            trace = {param: {} for param in config.param_names}
        case_ids = list(test_dataset.keys())
        for param in config.param_names:
            param_idx = config.param_indices[param]

            # Prepare test dataset for the given case
            X, y = self.create_test_dataset(test_dataset, param)
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

            # Set models to eval mode
            self.models[param].eval()
            self.likelihoods[param].eval()

            # Get predictions
            with torch.no_grad():
                predictions = self.likelihoods[param](self.models[param](X))

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
                offset = self.seq_len + self.pred_len - 1
                case_predictions = (
                    predictions[idx : idx + num_samples - offset]
                    .squeeze()
                    .detach()
                    .cpu()
                    .numpy()
                )
                idx += num_samples - offset
                trace[param]["predictions"][case_idx] = case_predictions
                y = test_dataset[case_idx][param_idx, :, 0, self.seq_len :].transpose()
                actors = [config.role_colors[actor] for actor in config.role_names]
                y = y[:, actors].reshape(-1, len(actors))
                case_predictions = case_predictions.reshape(-1, len(actors))

                # Get metrics for each case
                metrics = compute_regression_metrics(y, case_predictions)
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
