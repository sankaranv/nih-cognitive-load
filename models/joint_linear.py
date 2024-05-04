from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import (
    Matern,
    WhiteKernel,
    ConstantKernel,
    RBF,
    RationalQuadratic,
)
from xgboost import XGBRegressor, XGBClassifier
from sklearn.kernel_ridge import KernelRidge
from utils.config import config
from utils.stats import compute_regression_metrics, compute_classification_metrics
import numpy as np


class JointLinearModel:
    def __init__(self, model_config):
        self.base_model = model_config["base_model"]
        self.model_name = model_config["model_name"]
        self.seq_len = model_config["seq_len"]
        self.pred_len = model_config["pred_len"]
        self.model_config = model_config
        self.setup_models()

    def setup_models(self):
        self.models = {}
        for param in config.param_names:
            self.models[param] = Ridge(alpha=1.0)

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
            self.models[param].fit(X, y)


class JointLinearRegressor(JointLinearModel):
    def __init__(self, model_config):
        super().__init__(model_config)
        self.base_model = model_config["base_model"]
        self.model_name = model_config["model_name"]
        self.seq_len = model_config["seq_len"]
        self.pred_len = model_config["pred_len"]
        self.model_config = model_config
        self.setup_models()

    def setup_models(self):
        self.models = {}
        for param in config.param_names:
            if self.base_model == "Ridge":
                self.models[param] = Ridge(alpha=1.0)
            elif self.base_model == "XGBoost":
                self.models[param] = XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=1000,
                    learning_rate=0.01,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=1,
                    gamma=1,
                    reg_lambda=1,
                    n_jobs=-1,
                )
            elif self.base_model == "GaussianProcess":
                kernel = None
                if "kernel" in self.model_config:
                    if self.model_config["kernel"] == "Matern":
                        kernel = (
                            ConstantKernel()
                            + Matern(length_scale=1, nu=1.5)
                            + WhiteKernel(noise_level=1)
                        )
                    elif self.model_config["kernel"] == "RBF":
                        kernel = (
                            ConstantKernel()
                            + RBF(length_scale=1)
                            + WhiteKernel(noise_level=1)
                        )
                    elif self.model_config["kernel"] == "RationalQuadratic":
                        kernel = (
                            ConstantKernel()
                            + RationalQuadratic(length_scale=1, alpha=1)
                            + WhiteKernel(noise_level=1)
                        )
                # Make GP with the given kernel
                kernel = (
                    ConstantKernel()
                    + Matern(length_scale=1, nu=1.5)
                    + WhiteKernel(noise_level=1)
                )
                self.models[param] = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-10,
                    optimizer="fmin_l_bfgs_b",
                    n_restarts_optimizer=0,
                    normalize_y=False,
                    copy_X_train=True,
                    random_state=None,
                )
            elif self.base_model == "KernelRidge":
                self.models[param] = KernelRidge(
                    alpha=1.0,
                    kernel="rbf",
                    gamma=None,
                    degree=3,
                    coef0=1,
                    kernel_params=None,
                )

    def predict(self, test_dataset, trace=None):
        if trace is None or len(trace) == 0:
            trace = {param: {} for param in config.param_names}
        # Predict HRV for each case and each param
        for param in config.param_names:
            trace[param]["predictions"] = {}
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
                metrics = compute_regression_metrics(y, predictions)
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


class JointLinearClassifier(JointLinearModel):
    def __init__(self, model_config):
        super().__init__(model_config)
        self.base_model = model_config["base_model"]
        self.model_name = model_config["model_name"]
        self.seq_len = model_config["seq_len"]
        self.pred_len = model_config["pred_len"]
        self.model_config = model_config
        self.setup_models()

    def setup_models(self):
        self.models = {}
        for param in config.param_names:
            if self.base_model == "SVC":
                self.models[param] = SVC(kernel="linear", C=1, probability=True)
            elif self.base_model == "RandomForest":
                self.models[param] = RandomForestClassifier(
                    n_estimators=100, max_depth=2, random_state=0
                )
            elif self.base_model == "GaussianProcessClassifier":
                kernel = None
                if "kernel" in self.model_config:
                    if self.model_config["kernel"] == "Matern":
                        kernel = (
                            ConstantKernel()
                            + Matern(length_scale=1, nu=1.5)
                            + WhiteKernel(noise_level=1)
                        )
                    elif self.model_config["kernel"] == "RBF":
                        kernel = (
                            ConstantKernel()
                            + RBF(length_scale=1)
                            + WhiteKernel(noise_level=1)
                        )
                    elif self.model_config["kernel"] == "RationalQuadratic":
                        kernel = (
                            ConstantKernel()
                            + RationalQuadratic(length_scale=1, alpha=1)
                            + WhiteKernel(noise_level=1)
                        )
                # Make GP with the given kernel
                kernel = (
                    ConstantKernel()
                    + Matern(length_scale=1, nu=1.5)
                    + WhiteKernel(noise_level=1)
                )
                self.models[param] = GaussianProcessClassifier(
                    kernel=kernel,
                    alpha=1e-10,
                    optimizer="fmin_l_bfgs_b",
                    n_restarts_optimizer=0,
                    normalize_y=False,
                    copy_X_train=True,
                    random_state=None,
                )
            elif self.base_model == "LogisticRegression":
                self.models[param] = LogisticRegression()
            elif self.base_model == "XGBoost":
                self.models[param] = XGBClassifier(
                    objective="binary:logistic",
                    n_estimators=1000,
                    learning_rate=0.01,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=1,
                    gamma=1,
                    reg_lambda=1,
                    n_jobs=-1,
                )

    def predict(self, test_dataset, trace=None):
        if trace is None or len(trace) == 0:
            trace = {param: {} for param in config.param_names}
        # Predict HRV for each case and each param
        for param in config.param_names:
            trace[param]["predictions"] = {}
            trace[param]["accuracy"] = {}
            trace[param]["precision"] = {}
            trace[param]["recall"] = {}
            trace[param]["f1_score"] = {}
            trace[param]["corr_coef"] = {}
            trace[param]["shap_values"] = {}
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
                metrics = compute_classification_metrics(y, predictions)
                trace[param]["accuracy"][case_idx] = metrics["accuracy"]
                trace[param]["precision"][case_idx] = metrics["precision"]
                trace[param]["recall"][case_idx] = metrics["recall"]
                trace[param]["f1_score"][case_idx] = metrics["f1_score"]
                trace[param]["corr_coef"][case_idx] = metrics["corr_coef"]

        return trace
